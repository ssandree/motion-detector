"""Stage 1: 5fps → Gap1 Farneback @1/4 → 8×8×1 R-mean → P15 → T5.

Stores Gap1 base vectors/magnitudes on the 4×4 (16px) grid:
  U1 shape (T, Hb, Wb, 2)
  M1=‖U1‖ shape (T, Hb, Wb)

Stage2 loads this NPZ for Gap1 and applies the same 8×8×1 → P15 → T5
recipe to Gap 5/10/20/50 before fusion.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from motion_analyzer.opencv_cuda_bootstrap import bootstrap_opencv_cuda, reload_cv2_if_needed

bootstrap_opencv_cuda()
reload_cv2_if_needed()

import cv2
import numpy as np

from motion_analyzer.config import (
    APERTURE_ALPHA,
    APERTURE_GAMMA,
    APERTURE_TAU_EDGE,
    BASE_MOTION_TAG,
    FARNEBACK_POLY_N,
    FARNEBACK_POLY_SIGMA,
    FARNEBACK_USE_GAUSSIAN,
    FARNEBACK_WINSIZE,
    INPUT_SCALE,
    MAG_KEYS,
    ORIGINAL_CELL_PX,
    PRE_AGG_MAG_THRESHOLD,
    RESIZED_BASE_BLOCK,
    STAGE1_GAPS,
    STAGE1_P15_MIN,
    STAGE1_P15_WINDOW,
    STAGE1_SPATIAL_WIN,
    STAGE1_TEMPORAL_RADIUS,
    STRUCTURE_TENSOR_SIGMA,
    VEC_KEYS,
    PipelineConfig,
)
from motion_analyzer.farneback import (
    compute_dense_flow_original_px,
    farneback_device,
)
from motion_analyzer.structure_tensor import aperture_reliability_stack_for_flow
from motion_analyzer.video_io import resolve_video_path, sample_video_frames

logger = logging.getLogger("motion_map")

_P15_EPS = 1e-12


def suppress_flow_below_magnitude(
    flow: np.ndarray, *, threshold: float = PRE_AGG_MAG_THRESHOLD
) -> np.ndarray:
    """Zero vectors with ‖v‖ < threshold (copy-safe)."""
    out = np.asarray(flow, dtype=np.float32).copy()
    if threshold <= 0:
        return out
    mag = np.linalg.norm(out, axis=-1)
    out[mag < float(threshold)] = 0.0
    return out


def aggregate_mean_flow_vector(
    flow: np.ndarray,
    *,
    block_size: int,
    mag_threshold: float | None = PRE_AGG_MAG_THRESHOLD,
) -> np.ndarray:
    """Mean flow vector (dx, dy) per non-overlapping block; shape (H, W, 2)."""
    if mag_threshold is not None and float(mag_threshold) > 0:
        flow = suppress_flow_below_magnitude(flow, threshold=float(mag_threshold))
    height, width = flow.shape[:2]
    out_h = int(math.ceil(height / float(block_size)))
    out_w = int(math.ceil(width / float(block_size)))
    out = np.zeros((out_h, out_w, 2), dtype=np.float32)
    for row in range(out_h):
        y0 = row * block_size
        y1 = min((row + 1) * block_size, height)
        for col in range(out_w):
            x0 = col * block_size
            x1 = min((col + 1) * block_size, width)
            patch = flow[y0:y1, x0:x1].reshape(-1, 2)
            out[row, col] = np.mean(patch, axis=0).astype(np.float32)
    return out


def _metadata_arrays(meta_rows: list[dict]) -> dict[str, np.ndarray]:
    keys = sorted(meta_rows[0].keys())
    arrays: dict[str, np.ndarray] = {}
    for key in keys:
        values = [row[key] for row in meta_rows]
        if isinstance(values[0], float):
            arrays[key] = np.asarray(values, dtype=np.float32)
        else:
            arrays[key] = np.asarray(values, dtype=np.int32)
    return arrays


def gaps_tag(gaps: tuple[int, ...]) -> str:
    return "_".join(f"U{g}" for g in gaps)


def base_motion_npz_path(
    data_root: Path, video_id: str, *, tag: str | None = None
) -> Path:
    use_tag = str(tag) if tag is not None else str(BASE_MOTION_TAG)
    return data_root / video_id / f"{video_id}_base_motion_{use_tag}.npz"


def load_stage1_base_motion(data_root: Path, video_id: str) -> dict:
    """Load Stage1 Gap1 NPZ (U1/M1). Raises if missing — run Stage1 first."""
    path = base_motion_npz_path(Path(data_root), video_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage1 NPZ not found: {path}\n"
            f"Run Stage1 first:\n"
            f"  python scripts/1_base_motion.py --video_id {video_id}"
        )
    with np.load(path) as z:
        if "U1" not in z.files:
            raise KeyError(f"Stage1 NPZ missing U1: {path}")
        u1 = np.asarray(z["U1"], dtype=np.float32)
        if "M1" in z.files:
            m1 = np.asarray(z["M1"], dtype=np.float32)
        else:
            m1 = np.linalg.norm(u1, axis=-1).astype(np.float32)
        align_start = (
            int(z["align_start_sampled_index"])
            if "align_start_sampled_index" in z.files
            else 1
        )
        n_sampled = (
            int(z["num_sampled_frames"]) if "num_sampled_frames" in z.files else None
        )
    return {
        "path": path,
        "U1": u1,
        "M1": m1,
        "align_start": align_start,
        "num_sampled_frames": n_sampled,
    }


def slice_to_align(
    arr: np.ndarray, *, src_align: int, dst_align: int
) -> np.ndarray:
    """Trim a map aligned at ``src_align`` so it matches ``dst_align``."""
    delta = int(dst_align) - int(src_align)
    if delta < 0:
        raise ValueError(f"cannot align backward {src_align} → {dst_align}")
    if int(arr.shape[0]) <= delta:
        raise ValueError(
            f"map length {arr.shape[0]} too short to align {src_align}→{dst_align}"
        )
    return np.asarray(arr[delta:], dtype=np.float32)


def _time_prefix(sum_t: np.ndarray) -> np.ndarray:
    t_len = int(sum_t.shape[0])
    prefix = np.empty((t_len + 1,) + sum_t.shape[1:], dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(sum_t, axis=0, out=prefix[1:])
    return prefix


def _centered_window_bounds(t_len: int, window_len: int) -> tuple[np.ndarray, np.ndarray]:
    w = max(1, int(window_len))
    left = (w - 1) // 2
    right = w // 2
    t = np.arange(t_len, dtype=np.int32)
    t0 = np.maximum(0, t - left)
    t1 = np.minimum(t_len, t + right + 1)
    return t0, t1


def directional_persistence_window(
    v: np.ndarray, *, window_len: int, eps: float = _P15_EPS
) -> np.ndarray:
    """P_t = ||Σ_{k∈W(t)} v(k)|| / (Σ_{k∈W(t)} ||v(k)|| + eps), centered W."""
    stack = np.asarray(v, dtype=np.float32)
    if stack.ndim != 4 or stack.shape[-1] != 2:
        raise ValueError(f"expected (T,Hb,Wb,2), got {stack.shape}")
    t_len = stack.shape[0]
    mag = np.linalg.norm(stack, axis=-1).astype(np.float32)
    prefix_u = _time_prefix(stack[..., 0])
    prefix_v = _time_prefix(stack[..., 1])
    prefix_m = _time_prefix(mag)
    t0, t1 = _centered_window_bounds(t_len, window_len)
    sum_u = prefix_u[t1] - prefix_u[t0]
    sum_v = prefix_v[t1] - prefix_v[t0]
    sum_m = prefix_m[t1] - prefix_m[t0]
    num = np.hypot(sum_u, sum_v).astype(np.float32)
    p = (num / np.maximum(sum_m, eps)).astype(np.float32)
    p = np.clip(p, 0.0, 1.0)
    return np.where(sum_m > eps, p, 0.0).astype(np.float32)


def temporal_mean_cell_vectors(
    v: np.ndarray, *, temporal_radius: int = STAGE1_TEMPORAL_RADIUS
) -> np.ndarray:
    """Centered temporal mean of cell vectors. Window length 2R+1 (R=2 → T5)."""
    stack = np.asarray(v, dtype=np.float32)
    if stack.ndim != 4 or stack.shape[-1] != 2:
        raise ValueError(f"expected (T,Hb,Wb,2), got {stack.shape}")
    t_len = stack.shape[0]
    window_len = 2 * int(temporal_radius) + 1
    prefix_u = _time_prefix(stack[..., 0])
    prefix_v = _time_prefix(stack[..., 1])
    t0, t1 = _centered_window_bounds(t_len, window_len)
    n = np.maximum((t1 - t0).astype(np.float32)[:, None, None], 1.0)
    out = np.empty_like(stack)
    out[..., 0] = (prefix_u[t1] - prefix_u[t0]) / n
    out[..., 1] = (prefix_v[t1] - prefix_v[t0]) / n
    return out.astype(np.float32)


def apply_p15_t5(
    v1: np.ndarray,
    *,
    persist_window: int = STAGE1_P15_WINDOW,
    p_min: float = STAGE1_P15_MIN,
    temporal_radius: int = STAGE1_TEMPORAL_RADIUS,
) -> dict[str, np.ndarray | int]:
    """Official Stage1 after 8×8×1: keep if P15 ≥ τ_P, then T5."""
    v1 = np.asarray(v1, dtype=np.float32)
    if v1.ndim != 4 or v1.shape[-1] != 2:
        raise ValueError(f"expected (T,Hb,Wb,2), got {v1.shape}")
    p15 = directional_persistence_window(v1, window_len=int(persist_window))
    keep = p15 >= float(p_min)
    gated = v1.copy()
    gated[..., 0] *= keep.astype(np.float32)
    gated[..., 1] *= keep.astype(np.float32)
    v_t5 = temporal_mean_cell_vectors(gated, temporal_radius=int(temporal_radius))
    mag_t5 = np.linalg.norm(v_t5, axis=-1).astype(np.float32)
    return {
        "U": v_t5.astype(np.float32),
        "M": mag_t5,
        "keep": keep,
        "P15": p15.astype(np.float32),
        "n_keep": int(np.count_nonzero(keep)),
        "n_reject": int(np.count_nonzero(~keep)),
    }


def _sample_boxfilter_centers(img: np.ndarray, *, stride: int) -> np.ndarray:
    height, width = img.shape[:2]
    out_h = int(math.ceil(height / float(stride)))
    out_w = int(math.ceil(width / float(stride)))
    half = int(stride) // 2
    ys = np.minimum(np.arange(out_h) * int(stride) + half, height - 1)
    xs = np.minimum(np.arange(out_w) * int(stride) + half, width - 1)
    return img[np.ix_(ys, xs)]


def spatial_r_weighted_mean_stack(
    flow_stack: np.ndarray,
    reliability_stack: np.ndarray,
    *,
    stride: int,
    window: int,
) -> np.ndarray:
    """Per-frame 8×8 R-weighted mean on the stride grid (no temporal mix)."""
    stack = np.asarray(flow_stack, dtype=np.float32)
    rel = np.asarray(reliability_stack, dtype=np.float32)
    if stack.ndim != 4 or stack.shape[-1] != 2:
        raise ValueError(f"expected (T,H,W,2), got {stack.shape}")
    if rel.shape != stack.shape[:3]:
        raise ValueError(f"reliability shape {rel.shape} != flow {stack.shape[:3]}")
    ksize = (int(window), int(window))
    frames: list[np.ndarray] = []
    for t in range(stack.shape[0]):
        u = stack[t, ..., 0]
        v = stack[t, ..., 1]
        w = rel[t]
        su = cv2.boxFilter(
            u * w, ddepth=cv2.CV_64F, ksize=ksize, normalize=False
        )
        sv = cv2.boxFilter(
            v * w, ddepth=cv2.CV_64F, ksize=ksize, normalize=False
        )
        sw = cv2.boxFilter(w, ddepth=cv2.CV_64F, ksize=ksize, normalize=False)
        su_s = _sample_boxfilter_centers(su, stride=int(stride))
        sv_s = _sample_boxfilter_centers(sv, stride=int(stride))
        sw_s = _sample_boxfilter_centers(sw, stride=int(stride))
        out = np.zeros(su_s.shape + (2,), dtype=np.float32)
        valid = sw_s > 0
        w_safe = np.maximum(sw_s, 1e-12)
        out[..., 0] = np.where(valid, su_s / w_safe, 0.0).astype(np.float32)
        out[..., 1] = np.where(valid, sv_s / w_safe, 0.0).astype(np.float32)
        frames.append(out)
    return np.stack(frames, axis=0).astype(np.float32)


def _dense_flow_stack(
    gray: list[np.ndarray],
    *,
    gap: int,
    mag_threshold: float,
) -> np.ndarray:
    """Farneback stack for current indices [gap, T)."""
    flows: list[np.ndarray] = []
    for curr in range(int(gap), len(gray)):
        flow = compute_dense_flow_original_px(gray[curr - int(gap)], gray[curr])
        if float(mag_threshold) > 0:
            flow = suppress_flow_below_magnitude(flow, threshold=float(mag_threshold))
        flows.append(np.asarray(flow, dtype=np.float32))
    return np.stack(flows, axis=0)


def _cpu_gap_8x8x1_rmean(
    gray: list[np.ndarray],
    *,
    gap: int,
    block_size: int,
    spatial_win: int,
    mag_threshold: float,
    tensor_sigma: float,
) -> np.ndarray:
    dense = _dense_flow_stack(gray, gap=int(gap), mag_threshold=float(mag_threshold))
    rel, _ = aperture_reliability_stack_for_flow(
        gray, dense, gap=int(gap), tensor_sigma=float(tensor_sigma)
    )
    return spatial_r_weighted_mean_stack(
        dense, rel, stride=int(block_size), window=int(spatial_win)
    )


def _meta_rows_for_gaps(
    sampled, *, gaps: tuple[int, ...], start: int, n_aligned: int
) -> list[dict]:
    meta_rows: list[dict] = []
    for offset in range(n_aligned):
        current_pos = int(start) + offset
        current = sampled[current_pos]
        meta = {
            "sampled_index_curr": current.sampled_index,
            "frame_idx_curr": current.frame_idx,
            "timestamp_sec_curr": current.timestamp_sec,
        }
        for gap in gaps:
            previous = sampled[current_pos - int(gap)]
            meta[f"sampled_index_prev_gap{gap}"] = previous.sampled_index
            meta[f"frame_idx_prev_gap{gap}"] = previous.frame_idx
            meta[f"timestamp_sec_prev_gap{gap}"] = previous.timestamp_sec
        meta_rows.append(meta)
    return meta_rows


def compute_stage1_gap_stacks(
    gray: list,
    sampled,
    *,
    gaps: tuple[int, ...],
    align_start: int | None = None,
    block_size: int = RESIZED_BASE_BLOCK,
    spatial_win: int = STAGE1_SPATIAL_WIN,
    mag_threshold: float = PRE_AGG_MAG_THRESHOLD,
    tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
    use_gpu: bool = True,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    list[dict],
    int,
    bool,
    dict[int, dict],
]:
    """8×8×1 R_ap-mean → P15 → T5 per gap, then slice to a common timeline.

    ``u_full[0]`` for gap G corresponds to sampled index G. P15/T5 run on that
    full series before aligning to ``align_start`` (default max(gaps)).
    """
    gaps = tuple(int(g) for g in gaps)
    if not gaps:
        raise ValueError("gaps must be non-empty")
    if len(sampled) <= max(gaps):
        raise ValueError(
            f"need at least {max(gaps) + 1} sampled frames, got {len(sampled)}"
        )
    start = int(max(gaps) if align_start is None else align_start)
    if start < max(gaps):
        start = int(max(gaps))
    if int(spatial_win) < int(block_size):
        raise ValueError(
            f"spatial_win ({spatial_win}) must be >= block_size ({block_size})"
        )

    u_fulls: dict[int, np.ndarray] = {}
    used_gpu = False
    if bool(use_gpu):
        try:
            from motion_analyzer.stage1_gpu import (
                _compute_gap_mean_flow_from_quarter,
                upload_quarter_gray_sequence,
            )

            gray_q, full_h, full_w, scaled_h, scaled_w = upload_quarter_gray_sequence(
                gray
            )
            for gap in gaps:
                u_fulls[int(gap)] = _compute_gap_mean_flow_from_quarter(
                    gray_q,
                    gap=int(gap),
                    stride=int(block_size),
                    window=int(spatial_win),
                    temporal_radius=0,
                    mag_threshold=float(mag_threshold),
                    full_h=full_h,
                    full_w=full_w,
                    scaled_h=scaled_h,
                    scaled_w=scaled_w,
                    use_aperture_reliability=True,
                    tensor_sigma=float(tensor_sigma),
                )
            used_gpu = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("GPU 8×8×1 R-mean failed (%s); falling back to CPU", exc)
            u_fulls = {}

    if not used_gpu:
        for gap in gaps:
            u_fulls[int(gap)] = _cpu_gap_8x8x1_rmean(
                gray,
                gap=int(gap),
                block_size=int(block_size),
                spatial_win=int(spatial_win),
                mag_threshold=float(mag_threshold),
                tensor_sigma=float(tensor_sigma),
            )

    u_stacks: dict[int, np.ndarray] = {}
    m_stacks: dict[int, np.ndarray] = {}
    gate_by_gap: dict[int, dict] = {}
    for gap in gaps:
        gated = apply_p15_t5(u_fulls[int(gap)])
        idx0 = int(start) - int(gap)
        u_stacks[int(gap)] = np.asarray(gated["U"][idx0:], dtype=np.float32)
        m_stacks[int(gap)] = np.asarray(gated["M"][idx0:], dtype=np.float32)
        gate_by_gap[int(gap)] = {
            "keep": np.asarray(gated["keep"][idx0:]),
            "P15": np.asarray(gated["P15"][idx0:], dtype=np.float32),
            "n_keep": int(gated["n_keep"]),
            "n_reject": int(gated["n_reject"]),
        }

    n_aligned = int(u_stacks[gaps[0]].shape[0])
    expected = len(sampled) - int(start)
    if n_aligned != expected:
        raise RuntimeError(f"aligned length mismatch {n_aligned} vs {expected}")
    for gap in gaps:
        if int(u_stacks[gap].shape[0]) != n_aligned:
            raise RuntimeError(
                f"gap {gap} length {u_stacks[gap].shape[0]} != {n_aligned}"
            )

    meta_rows = _meta_rows_for_gaps(
        sampled, gaps=gaps, start=start, n_aligned=n_aligned
    )
    return u_stacks, m_stacks, meta_rows, start, used_gpu, gate_by_gap


def compute_video_base_motion(
    video_id: str,
    *,
    cfg: PipelineConfig,
    data_root: Path,
    max_seconds: float | None = None,
    block_size: int = RESIZED_BASE_BLOCK,
    spatial_win: int = STAGE1_SPATIAL_WIN,
    temporal_radius: int = STAGE1_TEMPORAL_RADIUS,
    gaps: tuple[int, ...] = STAGE1_GAPS,
    use_gpu: bool = True,
) -> dict:
    """Stage1: Gap1 Farneback + R_ap 8×8×1, then P15 keep, then T5."""
    gaps = tuple(int(g) for g in gaps)
    video_path = resolve_video_path(video_id, cfg.video_search_roots)
    sampled = sample_video_frames(video_path, cfg.sampling_fps, max_seconds=max_seconds)
    gray = [cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY) for frame in sampled]
    frame_h, frame_w = gray[0].shape[:2]

    u_stacks, m_stacks, meta_rows, align_start, used_gpu, gate_by_gap = (
        compute_stage1_gap_stacks(
            gray,
            sampled,
            gaps=gaps,
            align_start=max(gaps),
            block_size=int(block_size),
            spatial_win=int(spatial_win),
            use_gpu=bool(use_gpu),
        )
    )
    aggregation_name = (
        "stage1_gap1_gpu_8x8x1_P15_T5" if used_gpu else "stage1_gap1_8x8x1_P15_T5"
    )

    out_dir = data_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = gaps_tag(gaps)
    npz_path = base_motion_npz_path(data_root, video_id, tag=tag)
    save_kwargs = {VEC_KEYS[gap]: u_stacks[gap] for gap in gaps}
    save_kwargs.update({MAG_KEYS[gap]: m_stacks[gap] for gap in gaps})
    n_keep = 0
    n_reject = 0
    if 1 in gate_by_gap:
        save_kwargs["stage1_keep"] = gate_by_gap[1]["keep"]
        save_kwargs["P15"] = gate_by_gap[1]["P15"]
        n_keep = int(gate_by_gap[1]["n_keep"])
        n_reject = int(gate_by_gap[1]["n_reject"])

    np.savez_compressed(
        npz_path,
        **save_kwargs,
        **_metadata_arrays(meta_rows),
        gaps=np.asarray(gaps, dtype=np.int32),
        sampling_fps=np.asarray(cfg.sampling_fps, dtype=np.float32),
        input_scale=np.asarray(INPUT_SCALE, dtype=np.float32),
        farneback_winsize=np.asarray(FARNEBACK_WINSIZE, dtype=np.int32),
        farneback_poly_n=np.asarray(FARNEBACK_POLY_N, dtype=np.int32),
        farneback_poly_sigma=np.asarray(FARNEBACK_POLY_SIGMA, dtype=np.float32),
        farneback_use_gaussian=np.asarray(FARNEBACK_USE_GAUSSIAN, dtype=np.bool_),
        farneback_device=np.asarray(farneback_device()),
        pre_agg_mag_threshold=np.asarray(PRE_AGG_MAG_THRESHOLD, dtype=np.float32),
        aggregation=np.asarray(aggregation_name),
        spatial_win=np.asarray(int(spatial_win), dtype=np.int32),
        temporal_radius=np.asarray(int(temporal_radius), dtype=np.int32),
        resized_base_block=np.asarray(int(block_size), dtype=np.int32),
        original_cell_px=np.asarray(ORIGINAL_CELL_PX, dtype=np.int32),
        use_aperture_reliability=np.asarray(True, dtype=np.bool_),
        aperture_tau_edge=np.asarray(APERTURE_TAU_EDGE, dtype=np.float32),
        aperture_gamma=np.asarray(APERTURE_GAMMA, dtype=np.float32),
        aperture_alpha=np.asarray(APERTURE_ALPHA, dtype=np.float32),
        structure_tensor_sigma=np.asarray(float(STRUCTURE_TENSOR_SIGMA), dtype=np.float32),
        stage1_gpu=np.asarray(bool(used_gpu), dtype=np.bool_),
        p15_min=np.asarray(float(STAGE1_P15_MIN), dtype=np.float32),
        p15_window=np.asarray(int(STAGE1_P15_WINDOW), dtype=np.int32),
        video_width=np.asarray(frame_w, dtype=np.int32),
        video_height=np.asarray(frame_h, dtype=np.int32),
        align_start_sampled_index=np.asarray(align_start, dtype=np.int32),
        num_sampled_frames=np.asarray(len(sampled), dtype=np.int32),
    )
    g0 = gaps[0]
    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "base_motion_npz": str(npz_path),
        "num_sampled_frames": len(sampled),
        "num_aligned_maps": int(m_stacks[g0].shape[0]),
        "map_shape": list(m_stacks[g0].shape),
        "vector_shape": list(u_stacks[g0].shape),
        "align_start": align_start,
        "gaps": list(gaps),
        "aggregation": aggregation_name,
        "stage1_gpu": bool(used_gpu),
        "p15_n_keep": n_keep,
        "p15_n_reject": n_reject,
        "spatial_win": int(spatial_win),
        "temporal_radius": int(temporal_radius),
        "block_size": int(block_size),
    }

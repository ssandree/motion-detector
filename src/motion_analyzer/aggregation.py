"""Stage 2: Stage1 Gap1 + same-recipe extra gaps → normalize → fusion → 64px.

Pipeline:
  1) Load Stage1 U1/M1 (8×8×1 R-mean → P15 → T5)
  2) Compute Gap 5/10/20/50 with the same Stage1 recipe
  3) Align all gaps to max(gaps), per-gap normalize, temporal fusion (default RMS)
  4) 4×4 spatial max of fused magnitude → MU_fused @64px
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np

from motion_analyzer.config import (
    AGGREGATION_BLOCK,
    DEFAULT_DATA_ROOT,
    DEFAULT_FUSION,
    FUSION_CHOICES,
    GAP_NORM_DIV,
    GAPS,
    HEAT_VMAX,
    HEAT_VMIN,
    MAG_KEYS,
    ORIGINAL_CELL_PX,
    PRE_AGG_MAG_THRESHOLD,
    RESIZED_BASE_BLOCK,
    STAGE1_SPATIAL_WIN,
    STAGE1_TEMPORAL_RADIUS,
    UNIT_CELL_PX,
    VEC_KEYS,
    PipelineConfig,
)
from motion_analyzer.farneback import farneback_device
from motion_analyzer.motion_map import (
    _metadata_arrays,
    _meta_rows_for_gaps,
    aggregate_mean_flow_vector,
    compute_stage1_gap_stacks,
    load_stage1_base_motion,
    slice_to_align,
)
from motion_analyzer.video_io import resolve_video_path, sample_video_frames

logger = logging.getLogger("aggregation")


def fusion_npz_path(output_root: Path, video_id: str, *, fusion: str = DEFAULT_FUSION) -> Path:
    tag = str(fusion).lower()
    return output_root / video_id / f"{video_id}_gap_fusion_{tag}.npz"


def normalize_gap_stacks(
    stacks: dict[int, np.ndarray],
    *,
    gaps: tuple[int, ...] | None = None,
    gap_norm_div: dict[int, float] | None = None,
) -> dict[int, np.ndarray]:
    """Divide each gap stack by its GAP_NORM_DIV factor (Gap1 stays ×1)."""
    use_gaps = tuple(gaps) if gaps is not None else tuple(GAPS)
    divisors = gap_norm_div if gap_norm_div is not None else GAP_NORM_DIV
    out: dict[int, np.ndarray] = {}
    for gap in use_gaps:
        div = float(divisors.get(int(gap), 1.0))
        if div <= 0:
            raise ValueError(f"invalid gap_norm_div[{gap}]={div}")
        out[int(gap)] = (np.asarray(stacks[gap], dtype=np.float32) / div).astype(
            np.float32
        )
    return out


def fuse_gap_magnitudes(
    mag_stacks: dict[int, np.ndarray],
    *,
    fusion: str = DEFAULT_FUSION,
    gaps: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Fuse per-gap magnitude maps (T,H,W) → (T,H,W)."""
    method = str(fusion).lower()
    if method not in FUSION_CHOICES:
        raise ValueError(f"fusion must be one of {FUSION_CHOICES}, got {fusion!r}")
    use_gaps = tuple(gaps) if gaps is not None else tuple(GAPS)
    stacked = np.stack([mag_stacks[gap] for gap in use_gaps], axis=0).astype(np.float32)
    with np.errstate(all="ignore"):
        if method == "mean":
            out = np.nanmean(stacked, axis=0)
        elif method == "max":
            out = np.nanmax(stacked, axis=0)
        elif method == "median":
            out = np.nanmedian(stacked, axis=0)
        else:  # rms
            out = np.sqrt(np.nanmean(np.square(stacked), axis=0))
    return np.asarray(out, dtype=np.float32)


def fuse_gap_vectors(
    vec_stacks: dict[int, np.ndarray],
    *,
    fusion: str = DEFAULT_FUSION,
    gaps: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Fuse per-gap vector fields (T,H,W,2) → (T,H,W,2)."""
    method = str(fusion).lower()
    use_gaps = tuple(gaps) if gaps is not None else tuple(GAPS)
    stacked = np.stack([vec_stacks[gap] for gap in use_gaps], axis=0).astype(np.float32)
    with np.errstate(all="ignore"):
        if method == "mean":
            out = np.nanmean(stacked, axis=0)
        elif method == "max":
            mags = np.linalg.norm(stacked, axis=-1)
            idx = np.nanargmax(mags, axis=0)
            out = np.take_along_axis(
                stacked, idx[None, ..., None], axis=0
            ).squeeze(0)
        elif method == "median":
            out = np.nanmedian(stacked, axis=0)
        else:
            out = np.sqrt(np.nanmean(np.square(stacked), axis=0))
            signs = np.sign(np.nanmean(stacked, axis=0))
            signs[signs == 0] = 1.0
            out = out * signs
    return np.asarray(out, dtype=np.float32)


def aggregate_magnitude_blocks(mag: np.ndarray, *, block_size: int) -> np.ndarray:
    """Spatial max of magnitude over block_size×block_size → coarser (T,Hu,Wu)."""
    arr = np.asarray(mag, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected (T,H,W), got {arr.shape}")
    t, height, width = arr.shape
    out_h = int(math.ceil(height / float(block_size)))
    out_w = int(math.ceil(width / float(block_size)))
    out = np.zeros((t, out_h, out_w), dtype=np.float32)
    for row in range(out_h):
        y0 = row * block_size
        y1 = min((row + 1) * block_size, height)
        for col in range(out_w):
            x0 = col * block_size
            x1 = min((col + 1) * block_size, width)
            patch = arr[:, y0:y1, x0:x1]
            with np.errstate(all="ignore"):
                out[:, row, col] = np.nanmax(patch, axis=(1, 2))
    return out


def fuse_video(
    video_id: str,
    *,
    cfg: PipelineConfig,
    output_root: Path,
    fusion: str = DEFAULT_FUSION,
    unit_block: int = AGGREGATION_BLOCK,
    gaps: tuple[int, ...] = GAPS,
    gap_norm_div: dict[int, float] | None = None,
    block_size: int = RESIZED_BASE_BLOCK,
    spatial_win: int = STAGE1_SPATIAL_WIN,
    temporal_radius: int = STAGE1_TEMPORAL_RADIUS,
    max_seconds: float | None = None,
    data_root: Path | None = None,
    use_gpu: bool = True,
) -> dict:
    """Stage2: Stage1 Gap1 + Stage1-recipe extra gaps → gap-normalize → fuse → 4×4 max."""
    gaps = tuple(int(g) for g in gaps)
    method = str(fusion).lower()
    if method not in FUSION_CHOICES:
        raise ValueError(f"fusion must be one of {FUSION_CHOICES}, got {fusion!r}")
    if 1 not in gaps:
        raise ValueError("Stage2 requires Gap1 (loaded from Stage1 NPZ)")
    divisors = dict(GAP_NORM_DIV if gap_norm_div is None else gap_norm_div)
    cache_root = Path(DEFAULT_DATA_ROOT if data_root is None else data_root)

    stage1 = load_stage1_base_motion(cache_root, video_id)
    video_path = resolve_video_path(video_id, cfg.video_search_roots)
    sampled = sample_video_frames(video_path, cfg.sampling_fps, max_seconds=max_seconds)
    if (
        stage1["num_sampled_frames"] is not None
        and int(stage1["num_sampled_frames"]) != len(sampled)
    ):
        raise ValueError(
            f"Stage1 sampled {stage1['num_sampled_frames']} frames but Stage2 "
            f"sampled {len(sampled)}. Use the same video / --max_seconds."
        )
    gray = [cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY) for frame in sampled]
    frame_h, frame_w = gray[0].shape[:2]
    align_start = int(max(gaps))

    u1 = slice_to_align(
        stage1["U1"], src_align=int(stage1["align_start"]), dst_align=align_start
    )
    m1 = slice_to_align(
        stage1["M1"], src_align=int(stage1["align_start"]), dst_align=align_start
    )

    extra_gaps = tuple(g for g in gaps if g != 1)
    used_gpu = False
    u_stacks_raw: dict[int, np.ndarray] = {1: u1}
    m_stacks_raw: dict[int, np.ndarray] = {1: m1}
    if extra_gaps:
        extra_u, extra_m, _meta, extra_start, used_gpu, _gate = (
            compute_stage1_gap_stacks(
                gray,
                sampled,
                gaps=extra_gaps,
                align_start=align_start,
                block_size=int(block_size),
                spatial_win=int(spatial_win),
                mag_threshold=float(PRE_AGG_MAG_THRESHOLD),
                use_gpu=bool(use_gpu),
            )
        )
        if int(extra_start) != align_start:
            raise RuntimeError(
                f"extra-gap align_start {extra_start} != {align_start}"
            )
        u_stacks_raw.update(extra_u)
        m_stacks_raw.update(extra_m)

    n_aligned = int(u1.shape[0])
    expected = len(sampled) - align_start
    if n_aligned != expected:
        raise RuntimeError(
            f"Stage1 Gap1 length {n_aligned} != expected {expected} "
            f"(align_start={align_start})"
        )
    for gap in extra_gaps:
        if u_stacks_raw[gap].shape != u1.shape:
            raise RuntimeError(
                f"gap {gap} shape {u_stacks_raw[gap].shape} != Gap1 {u1.shape}"
            )

    meta_rows = _meta_rows_for_gaps(
        sampled, gaps=gaps, start=align_start, n_aligned=n_aligned
    )

    m_stacks = normalize_gap_stacks(
        m_stacks_raw, gaps=gaps, gap_norm_div=divisors
    )
    u_stacks = normalize_gap_stacks(
        u_stacks_raw, gaps=gaps, gap_norm_div=divisors
    )

    m_fused = fuse_gap_magnitudes(m_stacks, fusion=method, gaps=gaps)
    u_fused = fuse_gap_vectors(u_stacks, fusion=method, gaps=gaps)
    mu_fused = aggregate_magnitude_blocks(m_fused, block_size=int(unit_block))
    uu = np.stack(
        [
            aggregate_mean_flow_vector(
                u_fused[index], block_size=int(unit_block), mag_threshold=None
            )
            for index in range(u_fused.shape[0])
        ],
        axis=0,
    ).astype(np.float32)

    out_path = fusion_npz_path(output_root, video_id, fusion=method)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gap_norm_keys = sorted(int(k) for k in divisors)
    gap_norm_arr = np.asarray(
        [float(divisors[g]) for g in gap_norm_keys], dtype=np.float32
    )

    save_kwargs: dict = {
        "M_fused": m_fused,
        "MU_fused": mu_fused,
        "U_fused": u_fused,
        "UU_fused": uu,
        f"M_{method}": m_fused,
        f"MU_{method}": mu_fused,
        "M_max": m_fused,
        **{MAG_KEYS[gap]: m_stacks_raw[gap] for gap in gaps},
        **{VEC_KEYS[gap]: u_stacks_raw[gap] for gap in gaps},
        **{f"{MAG_KEYS[gap]}_norm": m_stacks[gap] for gap in gaps},
        **{f"{VEC_KEYS[gap]}_norm": u_stacks[gap] for gap in gaps},
        **_metadata_arrays(meta_rows),
    }
    aggregation_name = (
        f"stage2_stage1maps_gapnorm_fusion_{method}_then_4x4max"
    )
    np.savez_compressed(
        out_path,
        **save_kwargs,
        gaps=np.asarray(gaps, dtype=np.int32),
        fusion=np.asarray(method),
        aggregation=np.asarray(aggregation_name),
        sampling_fps=np.asarray(cfg.sampling_fps, dtype=np.float32),
        spatial_win=np.asarray(int(spatial_win), dtype=np.int32),
        temporal_radius=np.asarray(int(temporal_radius), dtype=np.int32),
        resized_base_block=np.asarray(int(block_size), dtype=np.int32),
        pre_agg_mag_threshold=np.asarray(PRE_AGG_MAG_THRESHOLD, dtype=np.float32),
        use_aperture_reliability=np.asarray(True, dtype=np.bool_),
        stage1_npz=np.asarray(str(stage1["path"])),
        stage2_gpu=np.asarray(bool(used_gpu), dtype=np.bool_),
        gap_norm_gaps=np.asarray(gap_norm_keys, dtype=np.int32),
        gap_norm_div=gap_norm_arr,
        heat_vmin=np.asarray(HEAT_VMIN, dtype=np.float32),
        heat_vmax=np.asarray(HEAT_VMAX, dtype=np.float32),
        farneback_device=np.asarray(farneback_device()),
        original_cell_px=np.asarray(ORIGINAL_CELL_PX, dtype=np.int32),
        unit_cell_px=np.asarray(UNIT_CELL_PX, dtype=np.int32),
        unit_block=np.asarray(int(unit_block), dtype=np.int32),
        video_width=np.asarray(frame_w, dtype=np.int32),
        video_height=np.asarray(frame_h, dtype=np.int32),
        align_start_sampled_index=np.asarray(align_start, dtype=np.int32),
        num_sampled_frames=np.asarray(len(sampled), dtype=np.int32),
        video_path=np.asarray(str(video_path)),
    )
    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "fusion_npz": str(out_path),
        "stage1_npz": str(stage1["path"]),
        "fusion": method,
        "gaps": list(gaps),
        "gap_norm_div": {str(k): float(divisors[k]) for k in gap_norm_keys},
        "heat_vmin": float(HEAT_VMIN),
        "heat_vmax": float(HEAT_VMAX),
        "align_start": align_start,
        "map_shape_base": list(m_fused.shape),
        "map_shape_unit": list(mu_fused.shape),
        "map_shape": list(mu_fused.shape),
        "stage2_gpu": bool(used_gpu),
        "aggregation": aggregation_name,
    }

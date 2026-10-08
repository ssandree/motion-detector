"""Stage 2: long gaps from filtered Stage1 Gap1 → fuse → 64px.

Pipeline:
  1) Load Stage1 U1/M1 (tangent-strip → 6×6 R-mean → P13 → T5 → cell mag floor)
  2) Gap G = sum of G consecutive filtered Gap1 vectors (no extra Farneback)
  3) Keep the Stage1 timeline (align_start = src_align, typically 1).
     Gap G is NaN until sampled index ≥ G; fusion skips those with nanmean/max/rms.
     Divide each available gap by √G
  4) Temporal fusion → M_fused @16px
  5) 4×4 spatial agg → MU_fused @64px
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np

from motion_analyzer.config import (
    AGGREGATION_BLOCK,
    DEFAULT_DATA_ROOT,
    DEFAULT_FUSION,
    DEFAULT_FUSION_SCHEME,
    DEFAULT_SPATIAL_AGG,
    FUSION_CHOICES,
    FUSION_SCHEME_CHOICES,
    GAP_NORM_DIV,
    GAPS,
    HEAT_VMAX,
    HEAT_VMIN,
    LONG_GAPS,
    MAG_KEYS,
    ORIGINAL_CELL_PX,
    PRE_AGG_MAG_THRESHOLD,
    RESIZED_BASE_BLOCK,
    SHORT_GAPS,
    STAGE2_CELL_MAG_FLOOR,
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
    load_stage1_base_motion,
    zero_cell_vectors_below,
)
from motion_analyzer.video_io import resolve_video_path, sample_video_frames

logger = logging.getLogger("aggregation")


def long_gaps_from_gap1(
    u1: np.ndarray,
    *,
    src_align: int,
    dst_align: int,
    gaps: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Build Gap-G maps by summing G consecutive filtered Gap1 vectors.

    ``u1[i]`` is the Stage1 vector at sampled index ``src_align + i``
    (flow from the previous sampled frame into that one). Gap G at sampled
    index s is ``sum(u1[s-G+1], …, u1[s])``, i.e. net displacement over G
    Gap1 steps. Outputs are sliced to start at ``dst_align``.

    If ``dst_align < G``, early timesteps cannot form a full Gap-G window
    and are filled with NaN so later fusion uses only the gaps that exist.
    """
    vec = np.asarray(u1, dtype=np.float32)
    if vec.ndim != 4 or vec.shape[-1] != 2:
        raise ValueError(f"expected (T,H,W,2), got {vec.shape}")
    src = int(src_align)
    dst = int(dst_align)
    if dst < src:
        raise ValueError(f"cannot align backward {src} → {dst}")
    t_len = int(vec.shape[0])
    delta = dst - src
    n_out = t_len - delta
    if n_out <= 0:
        raise ValueError(f"map length {t_len} too short to align {src}→{dst}")
    if src + t_len < dst + n_out:
        raise ValueError("Gap1 series does not cover the aligned window")

    prefix_u = np.empty((t_len + 1,) + vec.shape[1:3], dtype=np.float64)
    prefix_v = np.empty_like(prefix_u)
    prefix_u[0] = 0.0
    prefix_v[0] = 0.0
    np.cumsum(vec[..., 0], axis=0, out=prefix_u[1:])
    np.cumsum(vec[..., 1], axis=0, out=prefix_v[1:])

    k = np.arange(n_out, dtype=np.int32)
    i1 = delta + k + 1
    if int(i1.max()) > t_len:
        raise ValueError(f"gap window end out of range for align {src}→{dst}")
    u_out: dict[int, np.ndarray] = {}
    m_out: dict[int, np.ndarray] = {}
    for gap in gaps:
        g = int(gap)
        if g < 1:
            raise ValueError(f"gap must be >= 1, got {g}")
        i0 = i1 - g
        valid = i0 >= 0
        uu = np.full((n_out,) + vec.shape[1:], np.nan, dtype=np.float32)
        if np.any(valid):
            i0v = i0[valid]
            i1v = i1[valid]
            uu[valid, ..., 0] = (prefix_u[i1v] - prefix_u[i0v]).astype(np.float32)
            uu[valid, ..., 1] = (prefix_v[i1v] - prefix_v[i0v]).astype(np.float32)
        u_out[g] = uu
        m_out[g] = np.linalg.norm(uu, axis=-1).astype(np.float32)
    return u_out, m_out


def fusion_npz_path(output_root: Path, video_id: str, *, fusion: str = DEFAULT_FUSION) -> Path:
    tag = str(fusion).lower()
    return output_root / video_id / f"{video_id}_gap_fusion_{tag}.npz"


def normalize_gap_stacks(
    stacks: dict[int, np.ndarray],
    *,
    gaps: tuple[int, ...] | None = None,
    gap_norm_div: dict[int, float] | None = None,
) -> dict[int, np.ndarray]:
    """Divide each gap stack by its GAP_NORM_DIV factor (official: √G)."""
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
            finite = np.isfinite(mags)
            mags_safe = np.where(finite, mags, -np.inf)
            idx = np.argmax(mags_safe, axis=0)
            out = np.take_along_axis(
                stacked, idx[None, ..., None], axis=0
            ).squeeze(0)
            out[~np.any(finite, axis=0)] = np.nan
        elif method == "median":
            out = np.nanmedian(stacked, axis=0)
        else:
            out = np.sqrt(np.nanmean(np.square(stacked), axis=0))
            signs = np.sign(np.nanmean(stacked, axis=0))
            signs[signs == 0] = 1.0
            out = out * signs
    return np.asarray(out, dtype=np.float32)


def fuse_gap_magnitudes_short_long(
    mag_stacks: dict[int, np.ndarray],
    *,
    short_gaps: tuple[int, ...] = SHORT_GAPS,
    long_gaps: tuple[int, ...] = LONG_GAPS,
    fusion: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S_short = mean(short), S_long = mean(long), fused = mean(S_short, S_long)."""
    s_short = fuse_gap_magnitudes(mag_stacks, fusion=fusion, gaps=short_gaps)
    s_long = fuse_gap_magnitudes(mag_stacks, fusion=fusion, gaps=long_gaps)
    with np.errstate(all="ignore"):
        fused = np.nanmean(np.stack([s_short, s_long], axis=0), axis=0)
    return (
        np.asarray(fused, dtype=np.float32),
        np.asarray(s_short, dtype=np.float32),
        np.asarray(s_long, dtype=np.float32),
    )


def fuse_gap_vectors_short_long(
    vec_stacks: dict[int, np.ndarray],
    *,
    short_gaps: tuple[int, ...] = SHORT_GAPS,
    long_gaps: tuple[int, ...] = LONG_GAPS,
    fusion: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vector analogue of fuse_gap_magnitudes_short_long."""
    u_short = fuse_gap_vectors(vec_stacks, fusion=fusion, gaps=short_gaps)
    u_long = fuse_gap_vectors(vec_stacks, fusion=fusion, gaps=long_gaps)
    with np.errstate(all="ignore"):
        fused = np.nanmean(np.stack([u_short, u_long], axis=0), axis=0)
    return (
        np.asarray(fused, dtype=np.float32),
        np.asarray(u_short, dtype=np.float32),
        np.asarray(u_long, dtype=np.float32),
    )


def aggregate_magnitude_blocks(
    mag: np.ndarray, *, block_size: int, method: str = DEFAULT_SPATIAL_AGG
) -> np.ndarray:
    """Spatial reduce of magnitude over block_size×block_size → (T,Hu,Wu).

    Official Stage2 unit agg is 4×4 max. ``method`` is mean | max | median | rms.
    Partial edge blocks ignore padded NaNs.
    """
    arr = np.asarray(mag, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected (T,H,W), got {arr.shape}")
    method = str(method).lower()
    if method not in FUSION_CHOICES:
        raise ValueError(f"method must be one of {FUSION_CHOICES}, got {method!r}")
    t, height, width = arr.shape
    bs = int(block_size)
    if bs < 1:
        raise ValueError("block_size must be >= 1")
    out_h = int(math.ceil(height / float(bs)))
    out_w = int(math.ceil(width / float(bs)))
    pad_h = out_h * bs - height
    pad_w = out_w * bs - width
    if pad_h or pad_w:
        arr = np.pad(
            arr,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=np.nan,
        )
    blocks = arr.reshape(t, out_h, bs, out_w, bs)
    with np.errstate(all="ignore"):
        if method == "mean":
            out = np.nanmean(blocks, axis=(2, 4))
        elif method == "max":
            out = np.nanmax(blocks, axis=(2, 4))
        elif method == "median":
            out = np.nanmedian(blocks, axis=(2, 4))
        else:
            out = np.sqrt(np.nanmean(np.square(blocks), axis=(2, 4)))
    return np.asarray(out, dtype=np.float32)


def fuse_video(
    video_id: str,
    *,
    cfg: PipelineConfig,
    output_root: Path,
    fusion: str = DEFAULT_FUSION,
    fusion_scheme: str = DEFAULT_FUSION_SCHEME,
    spatial_agg: str = DEFAULT_SPATIAL_AGG,
    unit_block: int = AGGREGATION_BLOCK,
    gaps: tuple[int, ...] = GAPS,
    gap_norm_div: dict[int, float] | None = None,
    short_gaps: tuple[int, ...] = SHORT_GAPS,
    long_gaps: tuple[int, ...] = LONG_GAPS,
    block_size: int = RESIZED_BASE_BLOCK,
    spatial_win: int = STAGE1_SPATIAL_WIN,
    temporal_radius: int = STAGE1_TEMPORAL_RADIUS,
    max_seconds: float | None = None,
    data_root: Path | None = None,
    use_gpu: bool = True,
    cell_mag_floor: float | None = None,
) -> dict:
    """Stage2: integrate filtered Stage1 Gap1 to long gaps → fuse → 4×4.

    Extra gaps are not re-run Farneback. ``use_gpu`` is ignored.
    Default ``gap_norm_div`` is √G per gap. Stage2 does not re-zero Gap1
    cells (``STAGE2_CELL_MAG_FLOOR`` default 0). Timeline starts at Stage1
    ``align_start`` (typically 1); unavailable long gaps are NaN and skipped.

    ``fusion_scheme``:
      - ``flat``: fuse all gaps with ``fusion`` (official equal-weight mean)
      - ``short_long``: S_short=mean(short), S_long=mean(long), then mean of both
    """
    del use_gpu
    gaps = tuple(int(g) for g in gaps)
    short = tuple(int(g) for g in short_gaps)
    long = tuple(int(g) for g in long_gaps)
    method = str(fusion).lower()
    scheme = str(fusion_scheme).lower()
    spatial = str(spatial_agg).lower()
    if method not in FUSION_CHOICES:
        raise ValueError(f"fusion must be one of {FUSION_CHOICES}, got {fusion!r}")
    if scheme not in FUSION_SCHEME_CHOICES:
        raise ValueError(
            f"fusion_scheme must be one of {FUSION_SCHEME_CHOICES}, got {fusion_scheme!r}"
        )
    if spatial not in FUSION_CHOICES:
        raise ValueError(
            f"spatial_agg must be one of {FUSION_CHOICES}, got {spatial_agg!r}"
        )
    if 1 not in gaps:
        raise ValueError("Stage2 requires Gap1 (loaded from Stage1 NPZ)")
    if scheme == "short_long":
        needed = set(short) | set(long)
        missing = sorted(needed - set(gaps))
        if missing:
            raise ValueError(
                f"short_long scheme needs gaps {sorted(needed)}, missing {missing}"
            )
    if gap_norm_div is None:
        divisors = {
            int(g): float(GAP_NORM_DIV.get(int(g), float(g) ** 0.5)) for g in gaps
        }
    else:
        divisors = {int(k): float(v) for k, v in dict(gap_norm_div).items()}
        for g in gaps:
            divisors.setdefault(int(g), float(g) ** 0.5)
    floor = float(STAGE2_CELL_MAG_FLOOR if cell_mag_floor is None else cell_mag_floor)
    cache_root = Path(DEFAULT_DATA_ROOT if data_root is None else data_root)

    pipeline_t0 = time.perf_counter()
    stage1 = load_stage1_base_motion(cache_root, video_id)
    src_align = int(stage1["align_start"])
    # Keep Stage1's first flow frame (usually sampled index 1). Do not drop
    # the first max(gaps) seconds just so Gap50 is always defined.
    align_start = src_align

    u1_in = zero_cell_vectors_below(stage1["U1"], mag_floor=floor)
    u_stacks_raw, m_stacks_raw = long_gaps_from_gap1(
        u1_in,
        src_align=src_align,
        dst_align=align_start,
        gaps=gaps,
    )
    u1 = u_stacks_raw[1]
    n_aligned = int(u1.shape[0])
    n_sampled_stage1 = stage1["num_sampled_frames"]
    if n_sampled_stage1 is not None:
        expected = int(n_sampled_stage1) - align_start
        if n_aligned != expected:
            raise RuntimeError(
                f"Gap1-integrated length {n_aligned} != expected {expected} "
                f"(align_start={align_start})"
            )
    for gap in gaps:
        if u_stacks_raw[gap].shape != u1.shape:
            raise RuntimeError(
                f"gap {gap} shape {u_stacks_raw[gap].shape} != Gap1 {u1.shape}"
            )

    m_stacks = normalize_gap_stacks(
        m_stacks_raw, gaps=gaps, gap_norm_div=divisors
    )
    u_stacks = normalize_gap_stacks(
        u_stacks_raw, gaps=gaps, gap_norm_div=divisors
    )

    m_short = m_long = None
    u_short = u_long = None
    if scheme == "short_long":
        m_fused, m_short, m_long = fuse_gap_magnitudes_short_long(
            m_stacks, short_gaps=short, long_gaps=long, fusion=method
        )
        u_fused, u_short, u_long = fuse_gap_vectors_short_long(
            u_stacks, short_gaps=short, long_gaps=long, fusion=method
        )
    else:
        m_fused = fuse_gap_magnitudes(m_stacks, fusion=method, gaps=gaps)
        u_fused = fuse_gap_vectors(u_stacks, fusion=method, gaps=gaps)
    m_fused = np.nan_to_num(m_fused, nan=0.0)
    u_fused = np.nan_to_num(u_fused, nan=0.0)
    mu_fused = aggregate_magnitude_blocks(
        m_fused, block_size=int(unit_block), method=spatial
    )
    uu = np.stack(
        [
            aggregate_mean_flow_vector(
                u_fused[index], block_size=int(unit_block), mag_threshold=None
            )
            for index in range(u_fused.shape[0])
        ],
        axis=0,
    ).astype(np.float32)
    pipeline_sec = float(time.perf_counter() - pipeline_t0)

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
    if n_sampled_stage1 is None:
        expected = len(sampled) - align_start
        if n_aligned != expected:
            raise RuntimeError(
                f"Gap1-integrated length {n_aligned} != expected {expected} "
                f"(align_start={align_start})"
            )
    frame_h, frame_w = sampled[0].bgr.shape[:2]

    meta_rows = _meta_rows_for_gaps(
        sampled, gaps=gaps, start=align_start, n_aligned=n_aligned
    )

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
        **{f"M{gap}": m_stacks_raw[gap] for gap in gaps},
        **{f"U{gap}": u_stacks_raw[gap] for gap in gaps},
        **{f"M{gap}_norm": m_stacks[gap] for gap in gaps},
        **{f"U{gap}_norm": u_stacks[gap] for gap in gaps},
        **_metadata_arrays(meta_rows),
    }
    if m_short is not None and m_long is not None:
        save_kwargs["M_short"] = np.nan_to_num(m_short, nan=0.0)
        save_kwargs["M_long"] = np.nan_to_num(m_long, nan=0.0)
        save_kwargs["U_short"] = np.nan_to_num(u_short, nan=0.0)
        save_kwargs["U_long"] = np.nan_to_num(u_long, nan=0.0)
    if scheme == "short_long":
        aggregation_name = (
            f"stage2_gap1_integrate_divsqrtg_short_long_{method}_then_4x4{spatial}"
        )
    else:
        aggregation_name = (
            f"stage2_gap1_integrate_divsqrtg_fusion_{method}_then_4x4{spatial}"
        )
    np.savez_compressed(
        out_path,
        **save_kwargs,
        gaps=np.asarray(gaps, dtype=np.int32),
        short_gaps=np.asarray(short, dtype=np.int32),
        long_gaps=np.asarray(long, dtype=np.int32),
        fusion=np.asarray(method),
        fusion_scheme=np.asarray(scheme),
        spatial_agg=np.asarray(spatial),
        aggregation=np.asarray(aggregation_name),
        sampling_fps=np.asarray(cfg.sampling_fps, dtype=np.float32),
        spatial_win=np.asarray(int(spatial_win), dtype=np.int32),
        temporal_radius=np.asarray(int(temporal_radius), dtype=np.int32),
        resized_base_block=np.asarray(int(block_size), dtype=np.int32),
        pre_agg_mag_threshold=np.asarray(PRE_AGG_MAG_THRESHOLD, dtype=np.float32),
        cell_mag_floor=np.asarray(floor, dtype=np.float32),
        use_aperture_reliability=np.asarray(True, dtype=np.bool_),
        stage1_npz=np.asarray(str(stage1["path"])),
        stage2_gpu=np.asarray(False, dtype=np.bool_),
        long_gaps_from_gap1=np.asarray(True, dtype=np.bool_),
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
        "fusion_scheme": scheme,
        "short_gaps": list(short),
        "long_gaps": list(long),
        "spatial_agg": spatial,
        "gaps": list(gaps),
        "gap_norm_div": {str(k): float(divisors[k]) for k in gap_norm_keys},
        "heat_vmin": float(HEAT_VMIN),
        "heat_vmax": float(HEAT_VMAX),
        "align_start": align_start,
        "map_shape_base": list(m_fused.shape),
        "map_shape_unit": list(mu_fused.shape),
        "map_shape": list(mu_fused.shape),
        "stage2_gpu": False,
        "long_gaps_from_gap1": True,
        "cell_mag_floor": floor,
        "aggregation": aggregation_name,
        "pipeline_sec": round(pipeline_sec, 6),
    }

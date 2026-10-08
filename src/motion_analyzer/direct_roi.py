"""Stage1 → Stage2 → Stage3 in one process.

Keeps Gap1 flow and the fused unit maps in memory. Does not write NPZ,
overlay MP4, or 3D figures. The only artifact is ``<video_id>_roi_tracks.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from motion_analyzer.aggregation import (
    aggregate_magnitude_blocks,
    fuse_gap_magnitudes,
    fuse_gap_vectors,
    long_gaps_from_gap1,
    normalize_gap_stacks,
)
from motion_analyzer.config import (
    AGGREGATION_BLOCK,
    DEFAULT_FUSION,
    DEFAULT_SPATIAL_AGG,
    GAP_NORM_DIV,
    GAPS,
    STAGE1_GAPS,
    STAGE2_CELL_MAG_FLOOR,
    UNIT_CELL_PX,
    PipelineConfig,
)
from motion_analyzer.motion_map import (
    aggregate_mean_flow_vector,
    compute_stage1_gap_stacks,
    zero_cell_vectors_below,
)
from motion_analyzer.video_io import resolve_video_path, sample_video_frames
from stage3.v2.light_pipeline import build_light_roi_tracks


def fuse_u1_to_unit_maps(
    u1: np.ndarray,
    *,
    align_start: int,
    num_sampled_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Official Stage2 math: Gap1 vectors → unit magnitude ``MU`` and flow ``UU``.

    Same sequence as ``fuse_video`` with the locked defaults (flat mean, ÷√G,
    4×4 max). No video decode and no NPZ.
    """
    gaps = tuple(int(g) for g in GAPS)
    divisors = {
        int(g): float(GAP_NORM_DIV.get(int(g), float(g) ** 0.5)) for g in gaps
    }
    u1_in = zero_cell_vectors_below(u1, mag_floor=float(STAGE2_CELL_MAG_FLOOR))
    u_raw, m_raw = long_gaps_from_gap1(
        u1_in,
        src_align=int(align_start),
        dst_align=int(align_start),
        gaps=gaps,
    )
    n_aligned = int(u_raw[1].shape[0])
    expected = int(num_sampled_frames) - int(align_start)
    if n_aligned != expected:
        raise RuntimeError(
            f"Gap1-integrated length {n_aligned} != expected {expected} "
            f"(align_start={align_start})"
        )
    for gap in gaps:
        if u_raw[gap].shape != u_raw[1].shape:
            raise RuntimeError(
                f"gap {gap} shape {u_raw[gap].shape} != Gap1 {u_raw[1].shape}"
            )

    m_stacks = normalize_gap_stacks(m_raw, gaps=gaps, gap_norm_div=divisors)
    u_stacks = normalize_gap_stacks(u_raw, gaps=gaps, gap_norm_div=divisors)
    m_fused = np.nan_to_num(
        fuse_gap_magnitudes(m_stacks, fusion=DEFAULT_FUSION, gaps=gaps),
        nan=0.0,
    )
    u_fused = np.nan_to_num(
        fuse_gap_vectors(u_stacks, fusion=DEFAULT_FUSION, gaps=gaps),
        nan=0.0,
    )
    mu = aggregate_magnitude_blocks(
        m_fused,
        block_size=int(AGGREGATION_BLOCK),
        method=DEFAULT_SPATIAL_AGG,
    )
    uu = np.stack(
        [
            aggregate_mean_flow_vector(
                u_fused[index],
                block_size=int(AGGREGATION_BLOCK),
                mag_threshold=None,
            )
            for index in range(u_fused.shape[0])
        ],
        axis=0,
    ).astype(np.float32)
    return mu, uu


def run_video_roi_tracks(
    video_id: str,
    *,
    cfg: PipelineConfig,
    output_root: Path,
    use_gpu: bool = True,
) -> dict:
    """Run one clip and write ``<output_root>/<video_id>_roi_tracks.json``."""
    video_path = resolve_video_path(video_id, cfg.video_search_roots)
    t0 = time.perf_counter()
    sampled = sample_video_frames(video_path, cfg.sampling_fps)
    gray: list[np.ndarray] = []
    for frame in sampled:
        gray.append(cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY))
        frame.bgr = None  # type: ignore[assignment]
    if not gray:
        raise RuntimeError(f"{video_id}: no sampled frames")
    frame_h, frame_w = gray[0].shape[:2]
    gaps = tuple(int(g) for g in STAGE1_GAPS)
    u_stacks, _m_stacks, _meta_rows, align_start, used_gpu, _gate = (
        compute_stage1_gap_stacks(
            gray,
            sampled,
            gaps=gaps,
            align_start=max(gaps),
            use_gpu=bool(use_gpu),
        )
    )
    u1 = u_stacks[int(gaps[0])]
    num_sampled = len(sampled)
    stage1_sec = time.perf_counter() - t0
    del gray, sampled, u_stacks, _m_stacks, _meta_rows, _gate

    t1 = time.perf_counter()
    mag, flow = fuse_u1_to_unit_maps(
        u1,
        align_start=int(align_start),
        num_sampled_frames=num_sampled,
    )
    del u1
    stage2_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    built = build_light_roi_tracks(
        video_id,
        mag,
        flow,
        frame_width=int(frame_w),
        frame_height=int(frame_h),
        cell_px=int(UNIT_CELL_PX),
        score_key="MU_fused",
        fusion_npz=None,
    )
    stage3_sec = time.perf_counter() - t2
    tracks = built["tracks_json"]

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tracks_path = output_root / f"{video_id}_roi_tracks.json"
    tracks_path.write_text(json.dumps(tracks, indent=2) + "\n", encoding="utf-8")
    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "tracks_json": str(tracks_path),
        "stage1_gpu": bool(used_gpu),
        "num_sampled_frames": num_sampled,
        "map_shape": list(mag.shape),
        "num_block_events": tracks["num_block_events"],
        "num_split_components": tracks["num_split_components"],
        "num_composed_rois": tracks["num_composed_rois"],
        "num_final_rois": tracks["num_final_rois"],
        "stage1_sec": round(stage1_sec, 3),
        "stage2_sec": round(stage2_sec, 3),
        "stage3_sec": round(stage3_sec, 3),
        "elapsed_sec": round(stage1_sec + stage2_sec + stage3_sec, 3),
    }

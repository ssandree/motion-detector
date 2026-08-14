"""Stage 3: ROI tube — hysteresis block-events + 8-neigh time-overlap merge."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from motion_analyzer.aggregation import (
    aggregate_magnitude_blocks,
    fusion_npz_path,
)
from motion_analyzer.config import (
    AGGREGATION_BLOCK,
    DEFAULT_FUSION,
    HEAT_VMAX,
    HEAT_VMIN,
    ORIGINAL_CELL_PX,
    PipelineConfig,
    ROI_MAX_GAP,
    ROI_MERGE_SPATIAL_DIST,
    ROI_MERGE_TEMPORAL_GAP,
    ROI_MIN_BLOCK_EVENT,
    ROI_MIN_TUBE_CELLS,
    ROI_MIN_TUBE_DURATION,
    ROI_NEIGH_RADIUS,
    ROI_SUPPRESS_CONTAINED,
    ROI_TAU_HIGH,
    ROI_TAU_LOW,
    UNIT_CELL_PX,
)
from stage3.hysteresis_tube import (
    build_roi_tubes,
    tube_to_dict,
    tubes_to_frame_overlays,
)
from stage3.tube_3d_viz import render_tubes_3d
from motion_analyzer.video_io import resolve_video_path, sample_video_frames
from motion_analyzer.visualization import grid_bbox_to_pixels, heat_overlay

# Stage2/3 turbo heatmap absolute scale (from config; frozen at 0.8~3.5).
HEAT_MAX_ALPHA = 0.75

TRACK_COLORS = (
    (0, 220, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 200, 0),
    (0, 255, 255),
    (180, 105, 255),
    (50, 200, 50),
    (255, 100, 100),
    (255, 255, 0),
    (0, 128, 255),
)


def resolve_stage2_npz(
    fusion_root: Path,
    video_id: str,
    *,
    fusion: str = DEFAULT_FUSION,
) -> Path:
    """Locate Stage-2 NPZ: gap_fusion_* or stage2_agg_* (legacy naming)."""
    preferred = fusion_npz_path(fusion_root, video_id, fusion=fusion)
    if preferred.is_file():
        return preferred

    video_dir = fusion_root / video_id
    patterns = (
        f"{video_id}_gap_fusion_{str(fusion).lower()}.npz",
        f"{video_id}_gap_fusion_*.npz",
        f"{video_id}_stage2_agg_*.npz",
        f"{video_id}_stage3_agg_*.npz",
    )
    for pattern in patterns:
        matches = sorted(video_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(preferred)


def list_videos_in_fusion_root(fusion_root: Path) -> list[str]:
    ids: list[str] = []
    for path in sorted(fusion_root.iterdir()):
        if not path.is_dir():
            continue
        has_npz = (
            any(path.glob("*_gap_fusion_*.npz"))
            or any(path.glob("*_stage2_agg_*.npz"))
            or any(path.glob("*_stage3_agg_*.npz"))
        )
        if has_npz:
            ids.append(path.name)
    return ids


def load_stage2_unit_map(
    fusion_npz: Path,
    *,
    prefer_unit: bool = True,
    unit_block: int = AGGREGATION_BLOCK,
    fusion: str = DEFAULT_FUSION,
) -> tuple[np.ndarray, dict]:
    fusion_tag = str(fusion).lower()
    with np.load(fusion_npz) as data:
        key = None
        if prefer_unit:
            for cand in (
                "MU_fused",
                f"MU_{fusion_tag}",
                "MU_rms",
                "MU_mean",
                "MU_max",
                "MU_median",
            ):
                if cand in data.files:
                    key = cand
                    break
        if key is None:
            for cand in (
                "M_fused",
                f"M_{fusion_tag}",
                "M_rms",
                "M_max",
                "M_mean",
                "M_median",
            ):
                if cand in data.files:
                    key = cand
                    break
        if key is None:
            raise KeyError(f"{fusion_npz}: missing fused magnitude map")
        mag = np.asarray(data[key], dtype=np.float32)
        original_cell_px = int(
            np.asarray(data.get("original_cell_px", ORIGINAL_CELL_PX)).item()
        )
        unit_cell_px = int(np.asarray(data.get("unit_cell_px", UNIT_CELL_PX)).item())
        meta = {
            "key": key,
            "fusion": str(np.asarray(data["fusion"]).item())
            if "fusion" in data.files
            else fusion_tag,
            "unit_cell_px": unit_cell_px,
            "original_cell_px": original_cell_px,
            "video_width": int(np.asarray(data.get("video_width", 0)).item())
            if "video_width" in data.files
            else 0,
            "video_height": int(np.asarray(data.get("video_height", 0)).item())
            if "video_height" in data.files
            else 0,
            "sampled_index_curr": np.asarray(data["sampled_index_curr"], dtype=np.int32)
            if "sampled_index_curr" in data.files
            else None,
            "align_start_sampled_index": int(
                np.asarray(data.get("align_start_sampled_index", 0)).item()
            )
            if "align_start_sampled_index" in data.files
            else 0,
        }

    if mag.ndim != 3:
        raise ValueError(f"{fusion_npz}: expected (T,H,W), got {mag.shape}")

    if prefer_unit and str(meta["key"]).startswith("M_") and not str(meta["key"]).startswith(
        "MU_"
    ):
        source_key = str(meta["key"])
        mag = aggregate_magnitude_blocks(mag, block_size=int(unit_block))
        meta["key"] = "MU_" + source_key.split("_", 1)[1]
        meta["unit_cell_px"] = int(meta["original_cell_px"]) * int(unit_block)
        meta["aggregated_from"] = source_key

    return mag, meta


def _linear_heat_level(mag: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    """Same absolute turbo mapping as Stage2 viz3 (MU unit panel)."""
    arr = np.asarray(mag, dtype=np.float32)
    level = np.zeros(arr.shape, dtype=np.float32)
    valid = np.isfinite(arr) & (arr >= float(vmin))
    ceiling = max(float(vmax), 1e-6)
    level[valid] = np.clip(arr[valid] / ceiling, 0.0, 1.0)
    return level


def draw_tube_overlays(
    frame: np.ndarray,
    overlays: list[tuple[int, tuple[int, int, int, int], list[tuple[int, int]]]],
    *,
    cell_px: int,
    mag_frame: np.ndarray | None = None,
    heat_vmin: float = HEAT_VMIN,
    heat_vmax: float = HEAT_VMAX,
    heat_alpha: float = HEAT_MAX_ALPHA,
    tube_thickness: int = 3,
) -> np.ndarray:
    """Stage2 turbo MU heatmap + thick fixed tube bbox (no spatial drift)."""
    if mag_frame is not None and np.isfinite(mag_frame).any():
        out = heat_overlay(
            frame,
            _linear_heat_level(mag_frame, vmin=heat_vmin, vmax=heat_vmax),
            cell_px=int(cell_px),
            max_alpha=float(heat_alpha),
        )
    else:
        out = frame.copy()
    fh, fw = out.shape[:2]
    for tube_id, fixed_bbox, _cells in overlays:
        color = TRACK_COLORS[(tube_id - 1) % len(TRACK_COLORS)]
        # Final ROI tube: thick fixed spatial bbox for whole lifetime.
        pxb = grid_bbox_to_pixels(
            fixed_bbox,
            unit_pixel_size=cell_px,
            frame_width=fw,
            frame_height=fh,
        )
        cv2.rectangle(
            out,
            (pxb[0], pxb[1]),
            (pxb[2] - 1, pxb[3] - 1),
            color,
            int(tube_thickness),
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"T{tube_id}",
            (pxb[0] + 4, max(16, pxb[1] + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def process_video(
    video_id: str,
    *,
    cfg: PipelineConfig,
    fusion_root: Path,
    output_root: Path,
    fusion: str = DEFAULT_FUSION,
    prefer_unit: bool = True,
    tau_high: float = ROI_TAU_HIGH,
    tau_low: float = ROI_TAU_LOW,
    max_gap: int = ROI_MAX_GAP,
    min_block_event: int = ROI_MIN_BLOCK_EVENT,
    min_tube_cells: int = ROI_MIN_TUBE_CELLS,
    min_tube_duration: int = ROI_MIN_TUBE_DURATION,
    neigh_radius: int = ROI_NEIGH_RADIUS,
    merge_spatial_dist: int = ROI_MERGE_SPATIAL_DIST,
    merge_temporal_gap: int = ROI_MERGE_TEMPORAL_GAP,
    suppress_contained: bool = ROI_SUPPRESS_CONTAINED,
    write_3d: bool = True,
    viz_3d_root: Path | None = None,
    heat_vmin: float = HEAT_VMIN,
    heat_vmax: float = HEAT_VMAX,
    heat_alpha: float = HEAT_MAX_ALPHA,
    threshold: float | None = None,
    min_area: int = 1,
) -> dict:
    del threshold, min_area
    fusion_npz = resolve_stage2_npz(fusion_root, video_id, fusion=fusion)
    mag, meta = load_stage2_unit_map(
        fusion_npz, prefer_unit=prefer_unit, fusion=fusion
    )
    if prefer_unit and str(meta["key"]).startswith("MU"):
        cell_px = int(meta["unit_cell_px"])
    else:
        cell_px = int(meta.get("original_cell_px") or ORIGINAL_CELL_PX)

    block_events, tubes, suppressed_ids = build_roi_tubes(
        mag,
        tau_high=float(tau_high),
        tau_low=float(tau_low),
        max_gap=int(max_gap),
        min_block_event=int(min_block_event),
        neigh_radius=int(neigh_radius),
        merge_spatial_dist=int(merge_spatial_dist),
        merge_temporal_gap=int(merge_temporal_gap),
        min_tube_cells=int(min_tube_cells),
        min_tube_duration=int(min_tube_duration),
        suppress_contained=bool(suppress_contained),
    )
    by_frame = tubes_to_frame_overlays(tubes, num_frames=mag.shape[0])

    video_path = resolve_video_path(video_id, cfg.video_search_roots)
    sampled = sample_video_frames(video_path, cfg.sampling_fps)
    align_start = int(meta.get("align_start_sampled_index") or 0)
    if meta["sampled_index_curr"] is not None:
        curr_indices = [int(i) for i in meta["sampled_index_curr"].tolist()]
    else:
        curr_indices = list(range(align_start, align_start + mag.shape[0]))
    if len(curr_indices) != mag.shape[0]:
        raise ValueError(
            f"{video_id}: mag frames={mag.shape[0]} but indices={len(curr_indices)}"
        )

    out_dir = output_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"hyst_h{tau_high:g}_l{tau_low:g}_gap{max_gap}"
        f"_n{neigh_radius}_ms{merge_spatial_dist}_mt{merge_temporal_gap}"
        f"_min{min_block_event}_c{min_tube_cells}d{min_tube_duration}"
    )
    mp4_path = out_dir / f"{video_id}_roi_tube_{tag}.mp4"
    fh, fw = sampled[0].bgr.shape[:2]
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(cfg.sampling_fps),
        (fw, fh),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {mp4_path}")

    written = 0
    try:
        for t, samp_idx in enumerate(curr_indices):
            if samp_idx < 0 or samp_idx >= len(sampled):
                continue
            frame = sampled[samp_idx].bgr
            writer.write(
                draw_tube_overlays(
                    frame,
                    by_frame[t],
                    cell_px=cell_px,
                    mag_frame=mag[t],
                    heat_vmin=float(heat_vmin),
                    heat_vmax=float(heat_vmax),
                    heat_alpha=float(heat_alpha),
                )
            )
            written += 1
    finally:
        writer.release()

    params = {
        "tau_high": float(tau_high),
        "tau_low": float(tau_low),
        "max_gap": int(max_gap),
        "neigh_radius": int(neigh_radius),
        "merge_spatial_dist": int(merge_spatial_dist),
        "merge_temporal_gap": int(merge_temporal_gap),
        "min_block_event": int(min_block_event),
        "min_tube_cells": int(min_tube_cells),
        "min_tube_duration": int(min_tube_duration),
        "suppress_contained": bool(suppress_contained),
        "heat_vmin": float(heat_vmin),
        "heat_vmax": float(heat_vmax),
        "heat_alpha": float(heat_alpha),
        "heat_colormap": "turbo",
    }
    tracks_json = {
        "video_id": video_id,
        "variant": "hysteresis_block_tube_8cc_proximity_merge",
        "params": params,
        "fusion_npz": str(fusion_npz),
        "score_key": meta["key"],
        "cell_px": cell_px,
        "num_frames": int(mag.shape[0]),
        "num_block_events": len(block_events),
        "num_tracks": len(tubes),
        "suppressed_contained_tube_ids": list(suppressed_ids),
        "num_suppressed_contained": len(suppressed_ids),
        "tracks": [tube_to_dict(tube) for tube in tubes],
    }
    tracks_path = out_dir / "roi_tracks.json"
    tracks_path.write_text(json.dumps(tracks_json, indent=2) + "\n", encoding="utf-8")

    fig_3d_path = None
    if write_3d:
        fig_dir = Path(viz_3d_root) if viz_3d_root is not None else out_dir
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_3d_path = fig_dir / f"{video_id}_roi_tubes_3d_{tag}.png"
        render_tubes_3d(
            tubes,
            grid_h=int(mag.shape[1]),
            grid_w=int(mag.shape[2]),
            num_frames=int(mag.shape[0]),
            out_path=fig_3d_path,
            title=(
                f"{video_id} | tubes={len(tubes)} | "
                f"τH={tau_high:g} τL={tau_low:g} r={neigh_radius} "
                f"ms={merge_spatial_dist} mt={merge_temporal_gap}"
            ),
        )

    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "fusion_npz": str(fusion_npz),
        "tracks_json": str(tracks_path),
        "visualization_mp4": str(mp4_path),
        "visualization_3d": str(fig_3d_path) if fig_3d_path else None,
        "visualization_frames": written,
        "num_tracks": len(tubes),
        "num_block_events": len(block_events),
        "num_suppressed_contained": len(suppressed_ids),
        "suppressed_contained_tube_ids": list(suppressed_ids),
        "score_key": meta["key"],
        "cell_px": cell_px,
        "map_shape": list(mag.shape),
        "params": params,
    }

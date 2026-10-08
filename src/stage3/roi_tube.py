"""Stage 3 loaders and ROI overlay drawing (official clustering: ``stage3.v2``).
"""

from __future__ import annotations

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
    UNIT_CELL_PX,
)
from stage3.hysteresis_tube import RoiTube, tubes_to_frame_overlays
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
            f"E{tube_id}",
            (pxb[0] + 4, max(16, pxb[1] + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def write_roi_overlay_mp4(
    *,
    frames_bgr: list[np.ndarray],
    frame_indices: list[int],
    tubes: list[RoiTube],
    mag: np.ndarray,
    cell_px: int,
    out_path: Path,
    fps: float,
    heat_vmin: float = HEAT_VMIN,
    heat_vmax: float = HEAT_VMAX,
    heat_alpha: float = HEAT_MAX_ALPHA,
) -> int:
    """Lifetime-fixed ROI overlay MP4 with Stage2 turbo MU heatmap."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_bgr:
        return 0
    overlays = tubes_to_frame_overlays(tubes, num_frames=int(mag.shape[0]))
    fh, fw = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (fw, fh),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {out_path}")
    written = 0
    try:
        for t, samp_idx in enumerate(frame_indices):
            if samp_idx < 0 or samp_idx >= len(frames_bgr):
                continue
            mag_frame = mag[t] if 0 <= t < mag.shape[0] else None
            writer.write(
                draw_tube_overlays(
                    frames_bgr[samp_idx],
                    overlays.get(t, []),
                    cell_px=int(cell_px),
                    mag_frame=mag_frame,
                    heat_vmin=float(heat_vmin),
                    heat_vmax=float(heat_vmax),
                    heat_alpha=float(heat_alpha),
                )
            )
            written += 1
    finally:
        writer.release()
    return written


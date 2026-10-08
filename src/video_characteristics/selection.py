"""Pick the annotation targets out of the existing Stage3 ROI tube results.

Two selection sets, using the same definitions as the published ROI statistics:

- ``single_roi_area_lt_090``: ``num_roi_tubes == 1`` and ``mean_roi_area_ratio < 0.90``
- ``multi_roi``: ``max_concurrent_rois >= 2`` (two tubes alive in the same frame)

The two sets cannot overlap by construction, but a video that landed in both
would still be emitted once, with both selection flags set.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import REPO_ROOT, DatasetSpec

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_multi_roi import (  # noqa: E402
    LifetimeTube,
    box_area,
    clip_pixel_bbox,
    load_lifetime_tubes,
    load_npz_meta,
)
from motion_analyzer.visualization import grid_bbox_to_pixels  # noqa: E402

AREA_THRESHOLD = 0.90


@dataclass
class TubeInfo:
    """One Stage3 lifetime ROI tube, in units a prompt can state directly."""

    roi_id: str
    tube_id: int
    overlay_label: str
    t0_frame: int
    t1_frame: int
    t0_sec: float
    t1_sec: float
    pixel_bbox: tuple[int, int, int, int]
    area_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "roi_id": self.roi_id,
            "tube_id": self.tube_id,
            "overlay_label": self.overlay_label,
            "t0_frame": self.t0_frame,
            "t1_frame": self.t1_frame,
            "t0_sec": round(self.t0_sec, 2),
            "t1_sec": round(self.t1_sec, 2),
            "pixel_bbox": list(self.pixel_bbox),
            "area_ratio": round(self.area_ratio, 6),
        }


@dataclass
class VideoTarget:
    video_id: str
    dataset: str
    selection: dict[str, bool]
    num_roi_tubes: int
    max_concurrent_rois: int
    mean_roi_area_ratio: float
    sampling_fps: float
    num_sampled_frames: int
    duration_sec: float
    frame_width: int
    frame_height: int
    tubes: list[TubeInfo] = field(default_factory=list)
    concurrent_pairs: list[dict[str, Any]] = field(default_factory=list)
    raw_video: Path | None = None
    overlay_video: Path | None = None
    tracks_json: Path | None = None

    @property
    def is_multi_roi(self) -> bool:
        return bool(self.selection.get("multi_roi"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "dataset": self.dataset,
            "selection": self.selection,
            "stage3": {
                "num_roi_tubes": self.num_roi_tubes,
                "max_concurrent_rois": self.max_concurrent_rois,
                "mean_roi_area_ratio": round(self.mean_roi_area_ratio, 6),
                "sampling_fps": self.sampling_fps,
                "num_sampled_frames": self.num_sampled_frames,
                "duration_sec": round(self.duration_sec, 2),
                "frame_size": [self.frame_width, self.frame_height],
                "tubes": [tube.as_dict() for tube in self.tubes],
                "concurrent_pairs": self.concurrent_pairs,
            },
            "paths": {
                "raw_video": None if self.raw_video is None else str(self.raw_video),
                "overlay_video": None if self.overlay_video is None else str(self.overlay_video),
                "tracks_json": None if self.tracks_json is None else str(self.tracks_json),
            },
        }


def _tube_infos(
    tubes: list[LifetimeTube],
    *,
    cell_px: int,
    frame_width: int,
    frame_height: int,
    sampling_fps: float,
) -> list[TubeInfo]:
    frame_area = float(frame_width * frame_height)
    infos: list[TubeInfo] = []
    for tube in sorted(tubes, key=lambda t: (t.t0, t.tube_id)):
        box = clip_pixel_bbox(
            grid_bbox_to_pixels(
                tube.spatial_bbox_grid,
                unit_pixel_size=cell_px,
                frame_width=frame_width,
                frame_height=frame_height,
            ),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        infos.append(
            TubeInfo(
                # Keep the id tied to the Stage3 tube_id so it matches the
                # ``E<tube_id>`` label burned into the overlay MP4.
                roi_id=f"ROI_{int(tube.tube_id)}",
                tube_id=int(tube.tube_id),
                overlay_label=f"E{int(tube.tube_id)}",
                t0_frame=int(tube.t0),
                t1_frame=int(tube.t1),
                t0_sec=int(tube.t0) / sampling_fps if sampling_fps > 0 else float("nan"),
                t1_sec=(int(tube.t1) + 1) / sampling_fps if sampling_fps > 0 else float("nan"),
                pixel_bbox=box,
                area_ratio=box_area(box) / frame_area if frame_area > 0 else float("nan"),
            )
        )
    return infos


def _concurrent_pairs(tubes: list[TubeInfo], sampling_fps: float) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(tubes):
        for right in tubes[i + 1 :]:
            t0 = max(left.t0_frame, right.t0_frame)
            t1 = min(left.t1_frame, right.t1_frame)
            if t0 > t1:
                continue
            pairs.append(
                {
                    "roi_a": left.roi_id,
                    "roi_b": right.roi_id,
                    "overlap_t0_sec": round(t0 / sampling_fps, 2) if sampling_fps > 0 else None,
                    "overlap_t1_sec": (
                        round((t1 + 1) / sampling_fps, 2) if sampling_fps > 0 else None
                    ),
                    "overlap_sec": (
                        round((t1 - t0 + 1) / sampling_fps, 2) if sampling_fps > 0 else None
                    ),
                }
            )
    return pairs


def analyze_video(spec: DatasetSpec, video_id: str) -> VideoTarget:
    """Read one Stage3 tracks JSON plus its Stage2 NPZ metadata."""
    header, raw_tubes = load_lifetime_tubes(spec.tracks_path(video_id))
    meta = load_npz_meta(spec.npz_path(video_id))

    sampling_fps = float(meta["sampling_fps"] or 5.0)
    frame_width = int(meta["video_width"] or 0)
    frame_height = int(meta["video_height"] or 0)
    num_frames = int(header.get("num_frames") or meta["num_sampled_frames"] or 0)
    cell_px = int(header.get("cell_px") or 64)

    tubes = _tube_infos(
        raw_tubes,
        cell_px=cell_px,
        frame_width=frame_width,
        frame_height=frame_height,
        sampling_fps=sampling_fps,
    )

    active = np.zeros(max(num_frames, 0), dtype=np.int32)
    for tube in tubes:
        t0 = max(0, tube.t0_frame)
        t1 = min(num_frames - 1, tube.t1_frame)
        if t0 <= t1:
            active[t0 : t1 + 1] += 1
    max_concurrent = int(active.max()) if active.size else 0

    area_ratios = [tube.area_ratio for tube in tubes if np.isfinite(tube.area_ratio)]
    mean_area = float(np.mean(area_ratios)) if area_ratios else float("nan")

    selection = {
        "single_roi_area_lt_090": len(tubes) == 1 and mean_area < AREA_THRESHOLD,
        "multi_roi": max_concurrent >= 2,
    }

    return VideoTarget(
        video_id=video_id,
        dataset=spec.name,
        selection=selection,
        num_roi_tubes=len(tubes),
        max_concurrent_rois=max_concurrent,
        mean_roi_area_ratio=mean_area,
        sampling_fps=sampling_fps,
        num_sampled_frames=num_frames,
        duration_sec=num_frames / sampling_fps if sampling_fps > 0 else float("nan"),
        frame_width=frame_width,
        frame_height=frame_height,
        tubes=tubes,
        concurrent_pairs=_concurrent_pairs(tubes, sampling_fps),
        raw_video=spec.raw_video_path(video_id),
        overlay_video=(
            spec.overlay_path(video_id) if spec.overlay_path(video_id).is_file() else None
        ),
        tracks_json=spec.tracks_path(video_id),
    )


def select_targets(spec: DatasetSpec) -> list[VideoTarget]:
    """Every video in ``spec`` that matches either selection set, deduplicated."""
    targets: list[VideoTarget] = []
    for video_id in spec.video_ids():
        if not spec.tracks_path(video_id).is_file():
            continue
        target = analyze_video(spec, video_id)
        if any(target.selection.values()):
            targets.append(target)
    return targets

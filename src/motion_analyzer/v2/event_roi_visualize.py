"""Debug overlays for event-group ROI pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motion_analyzer.v1.blob_visualize import draw_blobs_on_frame
from motion_analyzer.v1.input_plan_visualize import (
    draw_input_plan_frame,
    draw_roi_legend,
    draw_selected_roi_tracks,
    overlay_timeline_frames,
    roi_inputs_from_plan,
)

logger = logging.getLogger(__name__)


def _bbox_for_candidate(cand: dict[str, Any], frame_idx: int) -> list[int] | None:
    for entry in cand.get("bbox_sequence", []):
        if int(entry["frame_idx"]) == frame_idx:
            return [int(v) for v in entry["bbox"]]
    before = [e for e in cand.get("bbox_sequence", []) if int(e["frame_idx"]) <= frame_idx]
    if before:
        return [int(v) for v in before[-1]["bbox"]]
    return None


_PATTERN_COLORS = {
    "event_motion": (0, 220, 0),
    "transit_motion": (0, 165, 255),
    "background_flow": (128, 128, 255),
    "background_jitter": (180, 180, 180),
    "ambiguous_motion": (255, 128, 0),
}


def draw_grouped_candidates(
    frame: Any,
    frame_idx: int,
    candidates: list[dict[str, Any]],
) -> Any:
    vis = frame.copy()
    for i, cand in enumerate(candidates):
        bbox = _bbox_for_candidate(cand, frame_idx)
        if not bbox:
            continue
        pattern = str(cand.get("motion_pattern", "ambiguous_motion"))
        color = _PATTERN_COLORS.get(pattern, (255, 255, 0))
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        rel = cand.get("event_relevance_score", cand.get("event_score", 0.0))
        label = f"G{cand.get('group_id', i+1)} {pattern[:6]} r={rel:.2f}"
        cv2.putText(
            vis, label, (x1 + 2, max(y1 - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2, cv2.LINE_AA,
        )
    return vis


def save_raw_blobs_overlay(
    video_path: str,
    frame_blob_map: dict[int, list[dict[str, Any]]],
    frame_indices: list[int],
    out_dir: Path,
    *,
    output_fps: float,
    filename: str = "raw_blobs_overlay.mp4",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(
        str(out_dir / filename),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    for fidx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if not ret:
            continue
        vis = draw_blobs_on_frame(frame, frame_blob_map.get(fidx, []), max_labels=40)
        writer.write(vis)
    cap.release()
    writer.release()
    return out_dir / filename


def save_grouped_blobs_overlay(
    video_path: str,
    candidates: list[dict[str, Any]],
    frame_indices: list[int],
    out_dir: Path,
    *,
    output_fps: float,
    filename: str = "grouped_blobs_overlay.mp4",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(
        str(out_dir / filename),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    for fidx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if not ret:
            continue
        vis = draw_grouped_candidates(frame, fidx, candidates)
        writer.write(vis)
    cap.release()
    writer.release()
    logger.info("Wrote %s", filename)
    return out_dir / filename


def _draw_dashed_rect(
    vis: Any,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    *,
    thickness: int = 1,
    dash_len: int = 10,
) -> None:
    segments = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (ax, ay), (bx, by) in segments:
        length = int(np.hypot(bx - ax, by - ay))
        if length <= 0:
            continue
        steps = max(1, length // dash_len)
        for i in range(0, steps, 2):
            t0 = i / steps
            t1 = min(1.0, (i + 1) / steps)
            px0 = int(ax + (bx - ax) * t0)
            py0 = int(ay + (by - ay) * t0)
            px1 = int(ax + (bx - ax) * t1)
            py1 = int(ay + (by - ay) * t1)
            cv2.line(vis, (px0, py0), (px1, py1), color, thickness, cv2.LINE_AA)


def draw_debug_rejected_candidates(
    frame: Any,
    frame_idx: int,
    rejected: list[dict[str, Any]],
) -> Any:
    vis = frame.copy()
    for cand in rejected:
        bbox = _bbox_for_candidate(cand, frame_idx)
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        color = (200, 200, 200)
        _draw_dashed_rect(vis, x1, y1, x2, y2, color, thickness=1)
        label = (
            f"G{cand.get('group_id')} rej r={cand.get('revised_event_score', 0):.2f}"
        )
        cv2.putText(
            vis, label, (x1 + 2, max(y1 - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA,
        )
    return vis


def save_final_roi_tracks_overlay(
    video_path: str,
    plan: dict[str, Any],
    frame_indices: list[int],
    out_dir: Path,
    *,
    output_fps: float,
    filename: str = "final_roi_tracks_overlay.mp4",
    debug_rejected_high_rank: list[dict[str, Any]] | None = None,
) -> Path:
    timeline = overlay_timeline_frames(plan, frame_indices, continuous=True)
    roi_inputs = roi_inputs_from_plan(plan)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(
        str(out_dir / filename),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    for fidx in timeline:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if not ret:
            continue
        vis = draw_selected_roi_tracks(frame, fidx, roi_inputs)
        if debug_rejected_high_rank:
            vis = draw_debug_rejected_candidates(vis, fidx, debug_rejected_high_rank)
        if roi_inputs:
            vis = draw_roi_legend(vis, show_roi=True)
        writer.write(vis)
    cap.release()
    writer.release()
    logger.info("Wrote %s (%d roi tracks)", filename, len(roi_inputs))
    return out_dir / filename

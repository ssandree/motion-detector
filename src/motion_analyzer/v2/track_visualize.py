"""Overlay for ByteTrack moving object tracks + motion fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motion_analyzer.v2.track_motion_features import DETECTOR_ONLY, OBSERVED_MOTION, WEAK_MOTION


def _track_color(importance: float, max_imp: float) -> tuple[int, int, int]:
    t = min(1.0, importance / max(max_imp, 1e-6))
    return (0, int(255 * (1 - t)), int(255 * t))


def _obs_tag(obs_type: str) -> str:
    if obs_type == OBSERVED_MOTION:
        return "motion"
    if obs_type == WEAK_MOTION:
        return "weak"
    return "det"


def draw_tracker_frame(
    frame: np.ndarray,
    *,
    moving_rois: list[dict[str, Any]],
    fallback_rois: list[dict[str, Any]],
    stationary_rois: list[dict[str, Any]] | None = None,
    detector_only: list[dict[str, Any]] | None = None,
    show_stationary_tracks: bool = False,
    show_detector_only: bool = False,
    base_image: np.ndarray | None = None,
) -> np.ndarray:
    vis = base_image.copy() if base_image is not None else frame.copy()
    max_imp = max((r.get("importance", 0) for r in moving_rois), default=1.0)

    for roi in moving_rois:
        x1, y1, x2, y2 = int(roi["x1"]), int(roi["y1"]), int(roi["x2"]), int(roi["y2"])
        imp = float(roi.get("importance", 0))
        color = _track_color(imp, max_imp)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        label = (
            f"T{roi.get('track_id', '?')} {roi.get('class_name', '?')} "
            f"imp={imp:.2f} obs={_obs_tag(roi.get('observation_type', ''))}"
        )
        cv2.putText(
            vis, label, (x1 + 2, max(y1 - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    for roi in fallback_rois:
        x1, y1, x2, y2 = int(roi["x1"]), int(roi["y1"]), int(roi["x2"]), int(roi["y2"])
        color = (255, 0, 255)
        label = f"F{roi.get('motion_tube_id', '?')} imp={roi.get('importance', 0):.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            vis, label, (x1 + 2, max(y1 - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    if show_stationary_tracks and stationary_rois:
        for roi in stationary_rois:
            x1, y1, x2, y2 = int(roi["x1"]), int(roi["y1"]), int(roi["x2"]), int(roi["y2"])
            color = (128, 128, 128)
            label = f"T{roi.get('track_id', '?')} {roi.get('class_name', '?')} stat"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                vis, label, (x1 + 2, max(y1 - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )

    if show_detector_only and detector_only:
        for d in detector_only:
            x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 80, 80), 1)

    return vis


from motion_analyzer.v1.input_plan_visualize import (
    draw_input_plan_frame,
    overlay_timeline_frames,
    roi_inputs_from_plan,
)


def save_tracker_motion_overlay(
    video_path: str,
    moving_tracks: list[dict[str, Any]],
    stationary_tracks: list[dict[str, Any]],
    track_obs_map: dict[int, list[dict]],
    fallback_tubes: list[dict[str, Any]],
    fallback_obs_map: dict[int, list[dict]],
    enriched_tracks: Any,
    out_dir: Path,
    *,
    output_fps: float = 5.0,
    filename: str = "tracker_motion_overlay.mp4",
    show_all_tracks: bool = False,
    show_stationary_tracks: bool = False,
    show_detector_only: bool = False,
    plan: dict[str, Any] | None = None,
    motion_type: str | None = None,
    debug: bool = False,
    frame_blob_map: dict[int, list[dict[str, Any]]] | None = None,
    flow_group_map: dict[int, dict[str, Any]] | None = None,
    sampled_frame_indices: list[int] | None = None,
    continuous: bool = True,
) -> Path:
    moving_ids = {t["track_id"] for t in moving_tracks}
    stationary_ids = {t["track_id"] for t in stationary_tracks}
    track_imp = {t["track_id"]: t["track_motion_importance"] for t in moving_tracks}
    tube_imp = {t["motion_tube_id"]: t.get("tube_importance", 0) for t in fallback_tubes}

    frame_moving: dict[int, list] = {}
    for tid in moving_ids:
        imp = track_imp.get(tid, 0)
        for obs in track_obs_map.get(tid, []):
            fi = int(obs["frame_idx"])
            frame_moving.setdefault(fi, []).append({**obs, "importance": imp})

    frame_stationary: dict[int, list] = {}
    if show_all_tracks or show_stationary_tracks:
        for tid in stationary_ids:
            for obs in track_obs_map.get(tid, []):
                fi = int(obs["frame_idx"])
                frame_stationary.setdefault(fi, []).append({**obs, "importance": 0.0})

    frame_fallback: dict[int, list] = {}
    for tube in fallback_tubes:
        tid = tube["motion_tube_id"]
        for obs in fallback_obs_map.get(tid, []):
            fi = int(obs["frame_idx"])
            frame_fallback.setdefault(fi, []).append({
                **obs, "motion_tube_id": tid, "importance": tube_imp.get(tid, 0),
            })

    frame_det_only: dict[int, list] = {}
    if show_detector_only and enriched_tracks is not None and not enriched_tracks.empty:
        det = enriched_tracks[enriched_tracks["observation_type"] == DETECTOR_ONLY]
        for _, row in det.iterrows():
            fi = int(row["frame_idx"])
            frame_det_only.setdefault(fi, []).append(row.to_dict())

    if plan is not None:
        timeline_frames = list(
            overlay_timeline_frames(plan, sampled_frame_indices, continuous=continuous)
        )
    else:
        timeline_frames = sorted(set(frame_moving) | set(frame_fallback))

    if plan and not continuous:
        extra: set[int] = set()
        for roi in roi_inputs_from_plan(plan):
            for entry in roi.get("bbox_sequence", []):
                extra.add(int(entry["frame_idx"]))
        timeline_frames = sorted(set(timeline_frames) | extra)

    if show_all_tracks or show_stationary_tracks:
        timeline_frames = sorted(set(timeline_frames) | set(frame_stationary))
    if show_detector_only:
        timeline_frames = sorted(set(timeline_frames) | set(frame_det_only))
    if debug and frame_blob_map:
        timeline_frames = sorted(set(timeline_frames) | set(frame_blob_map.keys()))

    if not timeline_frames:
        raise ValueError("No frames to visualize for tracker overlay")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out_dir / filename
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))

    for fi in timeline_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue
        fg = flow_group_map.get(fi, {}) if flow_group_map else {}
        if plan is not None:
            blobs = frame_blob_map.get(fi, []) if debug and frame_blob_map else None
            vis = draw_input_plan_frame(
                frame,
                fi,
                plan=plan,
                motion_type=motion_type,
                debug_blobs=blobs,
                flow_groups=fg.get("groups"),
                individual_blobs=fg.get("individual"),
            )
        else:
            vis = frame.copy()

        vis = draw_tracker_frame(
            frame,
            moving_rois=frame_moving.get(fi, []),
            fallback_rois=frame_fallback.get(fi, []),
            stationary_rois=frame_stationary.get(fi),
            detector_only=frame_det_only.get(fi),
            show_stationary_tracks=show_all_tracks or show_stationary_tracks,
            show_detector_only=show_detector_only,
            base_image=vis,
        )
        writer.write(vis)

    writer.release()
    cap.release()
    return out_path

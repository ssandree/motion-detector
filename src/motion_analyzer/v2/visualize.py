"""Motion-first overlay with observation-type styling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motion_analyzer.v2.gap_filling_tube_tracker import DETECTOR_SUPPORTED, INTERPOLATED, OBSERVED_MOTION
from motion_analyzer.v1.input_plan_visualize import (
    draw_input_plan_frame,
    roi_inputs_from_plan,
    sparse_global_frame_set,
)


def _motion_color(importance: float, max_imp: float) -> tuple[int, int, int]:
    t = min(1.0, importance / max(max_imp, 1e-6))
    return (0, int(255 * (1 - t)), int(255 * t))


def _draw_dashed_rect(vis: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple, thickness: int = 2) -> None:
    dash = 8
    for x in range(x1, x2, dash * 2):
        cv2.line(vis, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(vis, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(vis, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(vis, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def _draw_dotted_rect(vis: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple) -> None:
    step = 6
    for x in range(x1, x2, step):
        cv2.circle(vis, (x, y1), 2, color, -1)
        cv2.circle(vis, (x, y2), 2, color, -1)
    for y in range(y1, y2, step):
        cv2.circle(vis, (x1, y), 2, color, -1)
        cv2.circle(vis, (x2, y), 2, color, -1)


def _bbox_for_draw(roi: dict) -> tuple[int, int, int, int]:
    if roi.get("motion_bbox"):
        bb = roi["motion_bbox"]
        return int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
    if roi.get("predicted_bbox_coords"):
        pc = roi["predicted_bbox_coords"]
        return int(pc[0]), int(pc[1]), int(pc[2]), int(pc[3])
    if roi.get("predicted_bbox"):
        pb = roi["predicted_bbox"]
        return int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])
    return int(roi["x1"]), int(roi["y1"]), int(roi["x2"]), int(roi["y2"])


def _obs_label(obs_type: str) -> str:
    if obs_type == OBSERVED_MOTION:
        return "motion"
    if obs_type == DETECTOR_SUPPORTED:
        return "det"
    if obs_type == INTERPOLATED:
        return "interp"
    return "?"


def draw_detector_guided_frame(
    frame: np.ndarray,
    *,
    motion_rois: list[dict[str, Any]],
    detector_boxes: list[dict[str, Any]] | None = None,
    show_detector_boxes: bool = False,
    suppressed_blobs: list[dict[str, Any]] | None = None,
    base_image: np.ndarray | None = None,
) -> np.ndarray:
    vis = base_image.copy() if base_image is not None else frame.copy()
    overlay = vis.copy()

    if suppressed_blobs:
        for blob in suppressed_blobs:
            x1, y1, x2, y2 = int(blob["x1"]), int(blob["y1"]), int(blob["x2"]), int(blob["y2"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), -1)

    max_imp = max((r.get("importance", 0) for r in motion_rois), default=1.0)

    for roi in motion_rois:
        x1, y1, x2, y2 = _bbox_for_draw(roi)
        imp = float(roi.get("importance", 0))
        color = _motion_color(imp, max_imp)
        obs_type = roi.get("observation_type", OBSERVED_MOTION)

        if obs_type == OBSERVED_MOTION:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        elif obs_type == DETECTOR_SUPPORTED:
            _draw_dashed_rect(vis, x1, y1, x2, y2, color, 2)
        else:
            _draw_dotted_rect(vis, x1, y1, x2, y2, color)

        ctx_class = "none"
        if roi.get("associated_classes"):
            parts = str(roi["associated_classes"]).split(",")
            ctx_class = parts[0] if parts else "none"
        elif roi.get("nearest_detector_class"):
            ctx_class = roi["nearest_detector_class"]

        rank = roi.get("rank", "?")
        label = f"M{rank} imp={imp:.2f} ctx={ctx_class} obs={_obs_label(obs_type)}"
        cv2.putText(
            vis, label, (x1 + 2, max(y1 - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

        ctx_bb = roi.get("context_detector_bbox")
        if ctx_bb and len(ctx_bb) == 4:
            cx1, cy1, cx2, cy2 = map(int, ctx_bb)
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (200, 200, 0), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.08, vis, 0.92, 0, vis)

    if show_detector_boxes and detector_boxes:
        for det in detector_boxes:
            dx1, dy1, dx2, dy2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(
                vis, det.get("class_name", "?"), (dx1, dy2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA,
            )

    return vis


def save_detector_guided_overlay(
    video_path: str,
    guided_tubes: list[dict[str, Any]],
    guided_obs_map: dict[int, list[dict]],
    detections_df,
    blob_ctx_df,
    out_dir: Path,
    *,
    output_fps: float = 5.0,
    filename: str = "detector_guided_motion_overlay.mp4",
    show_detector_boxes: bool = False,
    plan: dict[str, Any] | None = None,
    motion_type: str | None = None,
    debug: bool = False,
    frame_blob_map: dict[int, list[dict[str, Any]]] | None = None,
) -> Path:
    """Write motion-first overlay MP4 with observation-type styling."""
    dets_by_frame: dict[int, list[dict]] = {}
    if detections_df is not None and not detections_df.empty:
        for fi in detections_df["frame_idx"].unique():
            dets_by_frame[int(fi)] = (
                detections_df[detections_df["frame_idx"] == fi].to_dict("records")
            )

    tube_rank = {
        t["motion_tube_id"]: i + 1
        for i, t in enumerate(
            sorted(guided_tubes, key=lambda x: x["detector_guided_motion_importance"], reverse=True)
        )
    }

    frame_rois: dict[int, list[dict]] = {}
    for tid, observations in guided_obs_map.items():
        rank = tube_rank.get(tid, 0)
        imp = next(
            (t["detector_guided_motion_importance"] for t in guided_tubes if t["motion_tube_id"] == tid),
            0.0,
        )
        for obs in observations:
            fi = int(obs["frame_idx"])
            frame_rois.setdefault(fi, []).append({
                **obs,
                "importance": imp,
                "rank": rank,
            })

    suppressed_by_frame: dict[int, list] = {}
    if blob_ctx_df is not None and not blob_ctx_df.empty:
        if "is_suppressed" in blob_ctx_df.columns:
            sup = blob_ctx_df[blob_ctx_df["is_suppressed"]]
            for _, row in sup.iterrows():
                fi = int(row["frame_idx"])
                suppressed_by_frame.setdefault(fi, []).append(row.to_dict())

    frame_indices = set(frame_rois.keys())
    if plan is not None:
        frame_indices |= sparse_global_frame_set(plan)
        for roi in roi_inputs_from_plan(plan):
            for entry in roi.get("bbox_sequence", []):
                frame_indices.add(int(entry["frame_idx"]))
    if debug and frame_blob_map:
        frame_indices |= set(frame_blob_map.keys())
    if not frame_indices:
        raise ValueError("No motion ROI frames to visualize")

    max_target = max(frame_indices)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out_dir / filename
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    frame_idx = 0
    while frame_idx <= max_target:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frame_indices:
            if plan is not None:
                blobs = frame_blob_map.get(frame_idx, []) if debug and frame_blob_map else None
                vis = draw_input_plan_frame(
                    frame,
                    frame_idx,
                    plan=plan,
                    motion_type=motion_type,
                    debug_blobs=blobs,
                )
                if debug:
                    vis = draw_detector_guided_frame(
                        frame,
                        motion_rois=frame_rois.get(frame_idx, []),
                        detector_boxes=dets_by_frame.get(frame_idx, []) if show_detector_boxes else None,
                        show_detector_boxes=show_detector_boxes,
                        suppressed_blobs=suppressed_by_frame.get(frame_idx),
                        base_image=vis,
                    )
            else:
                vis = draw_detector_guided_frame(
                    frame,
                    motion_rois=frame_rois.get(frame_idx, []),
                    detector_boxes=dets_by_frame.get(frame_idx, []) if show_detector_boxes else None,
                    show_detector_boxes=show_detector_boxes,
                    suppressed_blobs=suppressed_by_frame.get(frame_idx),
                )
            writer.write(vis)
        frame_idx += 1

    writer.release()
    cap.release()
    return out_path

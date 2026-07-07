"""Overlay visualization for motion_type and adaptive input plan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motion_analyzer.v1.blob_visualize import draw_blobs_on_frame
from motion_analyzer.v1.input_plan_builder import MAX_ROI_TRACKS

ROI_COLORS: dict[int, tuple[int, int, int]] = {
    1: (0, 255, 0),      # ROI1 green
    2: (0, 140, 255),    # ROI2 orange (distinct from ROI1)
}
ROI_LINE_THICK = 5
MOTION_EVIDENCE_LINE_THICK = 1
SPARSE_GLOBAL_COLOR = (0, 255, 255)
MOTION_BLOB_COLOR = (180, 220, 255)
FLOW_GROUP_COLOR = (160, 200, 240)
HUD_BG = (0, 0, 0)
HUD_FG = (255, 255, 255)
LEGEND_FG = (240, 240, 240)


def sparse_global_frame_set(plan: dict[str, Any]) -> set[int]:
    """Frame indices selected for sparse global sampling."""
    frames: set[int] = set()
    for entry in plan.get("global_inputs", []):
        if entry.get("type") != "sparse_global":
            continue
        frames.update(int(f) for f in entry.get("frame_indices", []))
    if not frames and plan.get("sparse_global_fps"):
        for entry in plan.get("frames", []):
            if entry.get("type") == "sparse_global":
                frames.add(int(entry["frame_idx"]))
    return frames


def plan_motion_type(plan: dict[str, Any]) -> str:
    if plan.get("motion_type"):
        return str(plan["motion_type"])
    legacy = str(plan.get("category", "unknown"))
    if legacy in ("Dense Global Motion", "Static / Sparse Local Motion", "Scene Change Dominant"):
        return "background_motion"
    return "event_motion"


def plan_roi_pattern(plan: dict[str, Any]) -> str | None:
    diag = plan.get("diagnostics") or {}
    return diag.get("roi_pattern")


def canonicalize_plan_roi_tracks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return plan['roi_inputs'] as the canonical ROI source for overlays and Input Builder.

    background_motion → always [].
    Finalized v2 plans (roi_source blob_group / none) → use roi_inputs as-is, never frames[].
    Legacy unfinalized plans → fall back to frames[] crop_bbox reconstruction.
    """
    mt = plan_motion_type(plan)
    if mt == "background_motion":
        plan["roi_inputs"] = []
        return []

    roi_source = str(plan.get("roi_source", ""))
    if roi_source in ("blob_group", "none", "explicit"):
        roi = list(plan.get("roi_inputs") or [])
        plan["roi_inputs"] = roi
        return roi

    if plan.get("roi_inputs"):
        return list(plan["roi_inputs"])

    roi_inputs = _legacy_roi_inputs_from_frames(plan)
    plan["roi_inputs"] = roi_inputs
    return roi_inputs


def roi_inputs_from_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return canonicalize_plan_roi_tracks(plan)


def _legacy_roi_inputs_from_frames(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Build ROI track dicts from legacy adaptive_input_plan ``frames`` entries."""
    by_key: dict[str, list[dict[str, Any]]] = {}
    key_importance: dict[str, float] = {}

    for entry in plan.get("frames", []):
        bbox = entry.get("crop_bbox")
        if not bbox:
            continue

        if entry.get("track_id") is not None:
            key = f"track:{int(entry['track_id'])}"
        elif entry.get("motion_tube_id") is not None:
            key = f"mtube:{int(entry['motion_tube_id'])}"
        elif entry.get("tube_id") is not None:
            key = f"tube:{int(entry['tube_id'])}"
        else:
            key = f"src:{entry.get('source', 'unknown')}"

        by_key.setdefault(key, []).append(
            {
                "frame_idx": int(entry["frame_idx"]),
                "timestamp_sec": float(entry["timestamp_sec"]),
                "bbox": [int(v) for v in bbox],
            }
        )
        imp = float(entry.get("importance", 0.0))
        key_importance[key] = max(key_importance.get(key, 0.0), imp)

    if not by_key:
        return []

    ranked = sorted(
        by_key.keys(),
        key=lambda k: (
            0 if k.startswith("track:") else 1 if k.startswith(("mtube:", "tube:")) else 2,
            -key_importance.get(k, 0.0),
        ),
    )
    roi_inputs: list[dict[str, Any]] = []
    for roi_id, key in enumerate(ranked[:MAX_ROI_TRACKS], start=1):
        seq = sorted(by_key[key], key=lambda e: e["frame_idx"])
        source_ids = key.split(":", 1)
        roi_inputs.append(
            {
                "roi_id": roi_id,
                "source": source_ids[0] if len(source_ids) == 2 else "unknown",
                "source_tube_ids": [int(source_ids[1])] if len(source_ids) == 2 else [],
                "source_track_ids": [int(source_ids[1])] if source_ids[0] == "track" else [],
                "merged_tube_count": len(seq),
                "bbox_sequence": seq,
                "start_sec": seq[0]["timestamp_sec"],
                "end_sec": seq[-1]["timestamp_sec"],
            }
        )
    return roi_inputs


def overlay_timeline_frames(
    plan: dict[str, Any],
    sampled_frame_indices: list[int] | None = None,
    *,
    continuous: bool = True,
) -> list[int]:
    """
    Ordered frame indices for overlay MP4 output.

    Bug fix: ``continuous=True`` without ``sampled_frame_indices`` used to fall back
    to sparse-global + ROI keyframes only (~1s clips). When continuous, always use
    the full sampled timeline.
    """
    if continuous:
        if not sampled_frame_indices:
            raise ValueError(
                "continuous overlay requires sampled_frame_indices "
                "(full analyzed clip timeline)"
            )
        return sorted(int(f) for f in sampled_frame_indices)

    sparse_frames = sparse_global_frame_set(plan)
    roi_frames: set[int] = set()
    for roi in roi_inputs_from_plan(plan):
        for entry in roi.get("bbox_sequence", []):
            roi_frames.add(int(entry["frame_idx"]))
    return sorted(sparse_frames | roi_frames)


def roi_bbox_at_frame(roi: dict[str, Any], frame_idx: int) -> list[int] | None:
    """Return ROI bbox only within the track's active frame range (no forward-fill)."""
    seq = sorted(roi.get("bbox_sequence", []), key=lambda e: int(e["frame_idx"]))
    if not seq:
        return None

    start_fi = int(seq[0]["frame_idx"])
    end_fi = int(seq[-1]["frame_idx"])
    if frame_idx < start_fi or frame_idx > end_fi:
        return None

    exact = [e for e in seq if int(e["frame_idx"]) == frame_idx]
    if exact:
        return [int(v) for v in exact[-1]["bbox"]]

    before = [e for e in seq if int(e["frame_idx"]) <= frame_idx]
    if before:
        return [int(v) for v in before[-1]["bbox"]]
    return None


def _roi_bbox_at_frame(roi: dict[str, Any], frame_idx: int) -> list[int] | None:
    return roi_bbox_at_frame(roi, frame_idx)


def draw_motion_type_hud(
    frame: np.ndarray,
    *,
    motion_type: str,
    roi_pattern: str | None = None,
) -> np.ndarray:
    vis = frame.copy()
    lines = [f"motion_type: {motion_type}"]
    if roi_pattern:
        lines.append(f"roi_pattern: {roi_pattern} (diagnostic)")
    y = 24
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (8, y - th - 6), (14 + tw, y + 4), HUD_BG, -1)
        cv2.putText(
            vis, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_FG, 2, cv2.LINE_AA,
        )
        y += th + 10
    return vis


def draw_sparse_global_marker(frame: np.ndarray, *, active: bool) -> np.ndarray:
    if not active:
        return frame
    vis = frame.copy()
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (4, 4), (min(80, w - 4), min(36, h - 4)), SPARSE_GLOBAL_COLOR, 2)
    cv2.putText(
        vis, "SG", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, SPARSE_GLOBAL_COLOR, 2, cv2.LINE_AA,
    )
    return vis


def draw_roi_legend(frame: np.ndarray, *, show_roi: bool = True) -> np.ndarray:
    """Legend: thin = motion/track evidence, thick = final VLM ROI tube."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    lines: list[tuple[str, tuple[int, int, int], int]] = [
        ("thin = motion / track evidence", MOTION_BLOB_COLOR, MOTION_EVIDENCE_LINE_THICK),
    ]
    if show_roi:
        lines.append(("thick = final VLM ROI tube", ROI_COLORS[1], ROI_LINE_THICK))
    box_w, box_h, pad = 280, 22, 8
    x0 = max(8, w - box_w - 12)
    y0 = max(8, h - pad - len(lines) * (box_h + 4) - 8)
    cv2.rectangle(vis, (x0 - 4, y0 - 6), (w - 8, h - 8), HUD_BG, -1)
    y = y0
    for text, color, thick in lines:
        lx, ly = x0, y + box_h // 2
        cv2.line(vis, (lx, ly), (lx + 36, ly), color, max(1, thick))
        cv2.putText(
            vis, text, (lx + 44, y + box_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, LEGEND_FG, 1, cv2.LINE_AA,
        )
        y += box_h + 4
    return vis


def draw_selected_roi_tracks(
    frame: np.ndarray,
    frame_idx: int,
    roi_inputs: list[dict[str, Any]],
) -> np.ndarray:
    """Draw final VLM ROI tubes (thick, labeled) within active frame range only."""
    vis = frame.copy()
    for roi in roi_inputs:
        roi_id = int(roi.get("roi_id", 0))
        bbox = _roi_bbox_at_frame(roi, frame_idx)
        if not bbox:
            continue
        color = ROI_COLORS.get(roi_id, (255, 255, 0))
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, ROI_LINE_THICK)
        label = f"ROI{roi_id}"
        cv2.putText(
            vis, label, (x1 + 4, max(y1 - 10, 22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA,
        )
    return vis


def draw_flow_groups(
    frame: np.ndarray,
    *,
    individual_blobs: list[dict[str, Any]] | None = None,
    flow_groups: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Draw motion evidence boxes (thin) — not final ROI tubes."""
    vis = frame.copy()
    if individual_blobs:
        for blob in individual_blobs:
            x1, y1, x2, y2 = int(blob["x1"]), int(blob["y1"]), int(blob["x2"]), int(blob["y2"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), MOTION_BLOB_COLOR, MOTION_EVIDENCE_LINE_THICK)
    if flow_groups:
        for grp in flow_groups:
            x1, y1, x2, y2 = grp["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), FLOW_GROUP_COLOR, MOTION_EVIDENCE_LINE_THICK)
    return vis


def draw_input_plan_frame(
    frame: np.ndarray,
    frame_idx: int,
    *,
    plan: dict[str, Any],
    motion_type: str | None = None,
    debug_blobs: list[dict[str, Any]] | None = None,
    flow_groups: list[dict[str, Any]] | None = None,
    individual_blobs: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Compose overlay: HUD, sparse global marker, selected ROIs, optional raw blobs."""
    mt = motion_type or plan_motion_type(plan)
    roi_pattern = plan_roi_pattern(plan)
    vis = draw_motion_type_hud(frame, motion_type=mt, roi_pattern=roi_pattern)
    vis = draw_sparse_global_marker(
        vis, active=frame_idx in sparse_global_frame_set(plan)
    )
    if flow_groups or individual_blobs:
        vis = draw_flow_groups(
            vis, individual_blobs=individual_blobs, flow_groups=flow_groups
        )
    if mt == "event_motion":
        roi_inputs = roi_inputs_from_plan(plan)
        if roi_inputs:
            vis = draw_selected_roi_tracks(vis, frame_idx, roi_inputs)
            vis = draw_roi_legend(vis, show_roi=True)
    elif debug_blobs or flow_groups or individual_blobs:
        vis = draw_roi_legend(vis, show_roi=False)
    if debug_blobs:
        vis = draw_blobs_on_frame(
            vis, debug_blobs, max_labels=40, fill_alpha=0.08, line_thickness=MOTION_EVIDENCE_LINE_THICK,
        )
    return vis


def save_input_plan_overlay(
    video_path: str,
    plan: dict[str, Any],
    out_dir: Path,
    *,
    motion_type: str | None = None,
    frame_blob_map: dict[int, list[dict[str, Any]]] | None = None,
    flow_group_map: dict[int, dict[str, Any]] | None = None,
    debug: bool = False,
    output_fps: float = 5.0,
    filename: str = "input_plan_overlay.mp4",
    sampled_frame_indices: list[int] | None = None,
    continuous: bool = True,
) -> Path:
    """Write MP4 overlay for adaptive input plan (motion_type + ROIs + sparse global)."""
    timeline_frames = overlay_timeline_frames(
        plan, sampled_frame_indices, continuous=continuous
    )
    if debug and frame_blob_map:
        extra = sorted(set(frame_blob_map.keys()) - set(timeline_frames))
        timeline_frames = sorted(set(timeline_frames) | set(extra))
    if not timeline_frames:
        raise ValueError("No frames to visualize for input plan overlay")

    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "input_plan_overlay: writing %d frames (continuous=%s, duration~%.1fs @ %.1ffps)",
        len(timeline_frames),
        continuous,
        len(timeline_frames) / max(output_fps, 1e-6),
        output_fps,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video writer: {out_path}")

    for fidx in timeline_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if not ret:
            logger.warning("input_plan_overlay: could not read frame %d", fidx)
            continue
        blobs = frame_blob_map.get(fidx, []) if debug and frame_blob_map else None
        fg = flow_group_map.get(fidx, {}) if flow_group_map else {}
        vis = draw_input_plan_frame(
            frame,
            fidx,
            plan=plan,
            motion_type=motion_type,
            debug_blobs=blobs,
            flow_groups=fg.get("groups"),
            individual_blobs=fg.get("individual"),
        )
        writer.write(vis)

    writer.release()
    cap.release()
    return out_path

"""Timeline visualization for adaptive input builder outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motion_analyzer.adaptive_input.frame_extractor import (
    ROI_COLORS,
    VideoFrameCache,
    load_frame_with_roi_overlay,
)
from motion_analyzer.adaptive_input.plan_resolver import ResolvedInputPlan
from motion_analyzer.v1.input_plan_visualize import SPARSE_GLOBAL_COLOR

logger = logging.getLogger(__name__)

EVENT_ANCHOR_COLOR = (255, 0, 255)
BG_COLOR = (28, 28, 28)
TEXT_COLOR = (240, 240, 240)
AXIS_COLOR = (90, 90, 90)
TIMELINE_MARGIN_LEFT = 80
TIMELINE_MARGIN_RIGHT = 40
TOP_ROW_Y = 70
TOP_THUMB_H = 130
TOP_THUMB_W = 180
BOTTOM_ROW_Y = 280
BOTTOM_SLOT_H = 170
CANVAS_H = 560
MIN_CANVAS_W = 1600
FPS = 2.0


@dataclass
class TimelineTopFrame:
    frame_idx: int
    timestamp_sec: float
    is_sparse_global: bool


@dataclass
class TimelineBottomItem:
    kind: str
    timestamp_sec: float
    frame_idx: int
    roi_id: int | None = None
    bbox: list[int] | None = None
    label: str | None = None


def _resize_thumb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image is None or image.size == 0:
        return np.full((height, width, 3), 40, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 40, dtype=np.uint8)
    ox = (width - new_w) // 2
    oy = (height - new_h) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized
    return canvas


def _put_centered_text(
    canvas: np.ndarray,
    text: str,
    center_x: int,
    y: int,
    *,
    color: tuple[int, int, int] = TEXT_COLOR,
    scale: float = 0.45,
) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(center_x - tw / 2)
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_border(
    image: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 4,
) -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), color, thickness)
    return vis


def _select_top_timeline_frames(
    resolved: ResolvedInputPlan,
    duration_sec: float,
    *,
    max_frames: int = 18,
) -> list[TimelineTopFrame]:
    sparse_set = {s.frame_idx for s in resolved.sparse_global}
    key_times = {s.timestamp_sec for s in resolved.sparse_global}
    key_times.update(r.timestamp_sec for r in resolved.roi_frames)
    key_times.update(a.timestamp_sec for a in resolved.event_anchors)

    if duration_sec <= 0:
        duration_sec = 1.0

    # Evenly sample across duration, merge with sparse-global timestamps.
    step = max(duration_sec / max(max_frames - 1, 1), 0.5)
    sample_times = [i * step for i in range(max_frames) if i * step <= duration_sec + 1e-6]
    sample_times = sorted(set(round(t, 3) for t in sample_times) | {round(t, 3) for t in key_times})

    sparse_by_ts = {s.timestamp_sec: s for s in resolved.sparse_global}
    sparse_sorted = sorted(resolved.sparse_global, key=lambda s: s.timestamp_sec)

    frames: list[TimelineTopFrame] = []
    for ts in sample_times:
        if ts in sparse_by_ts:
            sg = sparse_by_ts[ts]
            frames.append(
                TimelineTopFrame(
                    frame_idx=sg.frame_idx,
                    timestamp_sec=sg.timestamp_sec,
                    is_sparse_global=True,
                )
            )
            continue
        if not sparse_sorted:
            continue
        nearest = min(sparse_sorted, key=lambda s: abs(s.timestamp_sec - ts))
        frames.append(
            TimelineTopFrame(
                frame_idx=nearest.frame_idx,
                timestamp_sec=ts,
                is_sparse_global=nearest.frame_idx in sparse_set,
            )
        )

    # Deduplicate by frame_idx while preserving order.
    seen: set[int] = set()
    deduped: list[TimelineTopFrame] = []
    for item in frames:
        if item.frame_idx in seen:
            continue
        seen.add(item.frame_idx)
        deduped.append(item)
    deduped.sort(key=lambda f: f.timestamp_sec)
    return deduped[:max_frames]


def _bottom_timeline_items(resolved: ResolvedInputPlan) -> list[TimelineBottomItem]:
    items: list[TimelineBottomItem] = []
    for sg in resolved.sparse_global:
        items.append(
            TimelineBottomItem(
                kind="sparse_global",
                timestamp_sec=sg.timestamp_sec,
                frame_idx=sg.frame_idx,
                label="SG",
            )
        )
    for roi in resolved.roi_frames:
        items.append(
            TimelineBottomItem(
                kind=f"roi{roi.roi_id}",
                timestamp_sec=roi.timestamp_sec,
                frame_idx=roi.frame_idx,
                roi_id=roi.roi_id,
                bbox=roi.bbox,
                label=f"ROI{roi.roi_id}",
            )
        )
    for anchor in resolved.event_anchors:
        items.append(
            TimelineBottomItem(
                kind="event_anchor",
                timestamp_sec=anchor.timestamp_sec,
                frame_idx=anchor.frame_idx,
                label="EA",
            )
        )
    items.sort(key=lambda i: (i.timestamp_sec, i.kind, i.frame_idx))
    return items


def _time_to_x(timestamp_sec: float, duration_sec: float, plot_width: int) -> int:
    if duration_sec <= 0:
        return TIMELINE_MARGIN_LEFT
    ratio = max(0.0, min(1.0, timestamp_sec / duration_sec))
    return TIMELINE_MARGIN_LEFT + int(ratio * plot_width)


def _compose_timeline_canvas(
    *,
    resolved: ResolvedInputPlan,
    cache: VideoFrameCache,
    duration_sec: float,
    reveal_sec: float | None = None,
) -> np.ndarray:
    """Render timeline PNG frame; reveal_sec limits visible bottom-row items."""
    plot_width = max(int(duration_sec * 28), MIN_CANVAS_W - TIMELINE_MARGIN_LEFT - TIMELINE_MARGIN_RIGHT)
    canvas_w = plot_width + TIMELINE_MARGIN_LEFT + TIMELINE_MARGIN_RIGHT
    canvas = np.full((CANVAS_H, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    title = (
        "Sparse Global only"
        if resolved.motion_type == "background_motion"
        else "Sparse Global + ROI"
    )
    cv2.putText(
        canvas,
        f"Adaptive Input Builder — {title}",
        (TIMELINE_MARGIN_LEFT, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"motion_type: {resolved.motion_type}",
        (TIMELINE_MARGIN_LEFT, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    axis_y = TOP_ROW_Y + TOP_THUMB_H + 36
    cv2.line(
        canvas,
        (TIMELINE_MARGIN_LEFT, axis_y),
        (canvas_w - TIMELINE_MARGIN_RIGHT, axis_y),
        AXIS_COLOR,
        1,
    )
    cv2.putText(
        canvas,
        "Original timeline",
        (12, TOP_ROW_Y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "VLM inputs",
        (12, BOTTOM_ROW_Y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    for sec in range(0, int(duration_sec) + 1, max(1, int(duration_sec // 10) or 1)):
        x = _time_to_x(float(sec), duration_sec, plot_width)
        cv2.line(canvas, (x, axis_y - 4), (x, axis_y + 4), AXIS_COLOR, 1)
        cv2.putText(
            canvas,
            f"{sec}s",
            (x - 12, axis_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

    top_frames = _select_top_timeline_frames(resolved, duration_sec)
    for item in top_frames:
        frame = cache.get(item.frame_idx)
        if frame is None:
            continue
        thumb = _resize_thumb(frame, TOP_THUMB_W, TOP_THUMB_H)
        if item.is_sparse_global:
            thumb = _draw_border(thumb, SPARSE_GLOBAL_COLOR, thickness=4)
            cv2.putText(
                thumb,
                "SG",
                (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                SPARSE_GLOBAL_COLOR,
                2,
                cv2.LINE_AA,
            )
        x = _time_to_x(item.timestamp_sec, duration_sec, plot_width)
        x0 = x - TOP_THUMB_W // 2
        y0 = TOP_ROW_Y
        x0 = max(TIMELINE_MARGIN_LEFT, min(x0, canvas_w - TIMELINE_MARGIN_RIGHT - TOP_THUMB_W))
        canvas[y0 : y0 + TOP_THUMB_H, x0 : x0 + TOP_THUMB_W] = thumb
        _put_centered_text(
            canvas,
            f"{item.timestamp_sec:.1f}s",
            x0 + TOP_THUMB_W // 2,
            y0 + TOP_THUMB_H + 16,
            color=(180, 180, 180),
        )

    bottom_items = _bottom_timeline_items(resolved)
    if reveal_sec is not None:
        bottom_items = [i for i in bottom_items if i.timestamp_sec <= reveal_sec + 1e-6]

    slot_w = 220
    for item in bottom_items:
        x = _time_to_x(item.timestamp_sec, duration_sec, plot_width)
        x0 = max(TIMELINE_MARGIN_LEFT, min(x - slot_w // 2, canvas_w - TIMELINE_MARGIN_RIGHT - slot_w))
        y0 = BOTTOM_ROW_Y

        if item.kind == "sparse_global":
            frame = cache.get(item.frame_idx)
            thumb = _resize_thumb(frame, slot_w - 10, BOTTOM_SLOT_H - 30)
            thumb = _draw_border(thumb, SPARSE_GLOBAL_COLOR, thickness=3)
            canvas[y0 : y0 + BOTTOM_SLOT_H - 30, x0 + 5 : x0 + slot_w - 5] = thumb
            _put_centered_text(canvas, "SG", x + 0, y0 + BOTTOM_SLOT_H - 8, color=SPARSE_GLOBAL_COLOR)

        elif item.kind.startswith("roi") and item.roi_id is not None and item.bbox:
            from motion_analyzer.adaptive_input.plan_resolver import RoiFrameInput

            roi_input = RoiFrameInput(
                roi_id=item.roi_id,
                frame_idx=item.frame_idx,
                timestamp_sec=item.timestamp_sec,
                bbox=item.bbox,
            )
            overlay, crop = load_frame_with_roi_overlay(cache, roi_input)
            src_w = int(slot_w * 0.55)
            crop_w = slot_w - src_w - 8
            src_thumb = _resize_thumb(overlay, src_w, BOTTOM_SLOT_H - 30)
            crop_thumb = _resize_thumb(crop, crop_w, BOTTOM_SLOT_H - 30)
            color = ROI_COLORS.get(item.roi_id, (255, 255, 0))
            crop_thumb = _draw_border(crop_thumb, color, thickness=2)
            canvas[y0 : y0 + BOTTOM_SLOT_H - 30, x0 : x0 + src_w] = src_thumb
            canvas[y0 : y0 + BOTTOM_SLOT_H - 30, x0 + src_w + 4 : x0 + src_w + 4 + crop_w] = crop_thumb
            _put_centered_text(
                canvas,
                item.label or f"ROI{item.roi_id}",
                x,
                y0 + BOTTOM_SLOT_H - 8,
                color=color,
            )

        elif item.kind == "event_anchor":
            frame = cache.get(item.frame_idx)
            thumb = _resize_thumb(frame, slot_w - 10, BOTTOM_SLOT_H - 30)
            thumb = _draw_border(thumb, EVENT_ANCHOR_COLOR, thickness=3)
            canvas[y0 : y0 + BOTTOM_SLOT_H - 30, x0 + 5 : x0 + slot_w - 5] = thumb
            _put_centered_text(
                canvas,
                item.label or "EA",
                x,
                y0 + BOTTOM_SLOT_H - 8,
                color=EVENT_ANCHOR_COLOR,
            )

        _put_centered_text(
            canvas,
            f"{item.timestamp_sec:.1f}s",
            x,
            y0 - 8,
            color=(150, 150, 150),
            scale=0.4,
        )

    legend_y = CANVAS_H - 24
    cv2.rectangle(canvas, (TIMELINE_MARGIN_LEFT, legend_y - 14), (TIMELINE_MARGIN_LEFT + 18, legend_y + 2), SPARSE_GLOBAL_COLOR, 2)
    cv2.putText(canvas, "Sparse Global", (TIMELINE_MARGIN_LEFT + 24, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (TIMELINE_MARGIN_LEFT + 180, legend_y - 14), (TIMELINE_MARGIN_LEFT + 198, legend_y + 2), ROI_COLORS[1], 2)
    cv2.putText(canvas, "ROI1", (TIMELINE_MARGIN_LEFT + 204, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (TIMELINE_MARGIN_LEFT + 270, legend_y - 14), (TIMELINE_MARGIN_LEFT + 288, legend_y + 2), ROI_COLORS[2], 2)
    cv2.putText(canvas, "ROI2", (TIMELINE_MARGIN_LEFT + 294, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (TIMELINE_MARGIN_LEFT + 360, legend_y - 14), (TIMELINE_MARGIN_LEFT + 378, legend_y + 2), EVENT_ANCHOR_COLOR, 2)
    cv2.putText(canvas, "Event Anchor", (TIMELINE_MARGIN_LEFT + 384, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

    return canvas


def save_input_summary_png(
    video_path: str,
    resolved: ResolvedInputPlan,
    out_path: Path,
    *,
    duration_sec: float,
) -> Path:
    cache = VideoFrameCache(video_path)
    try:
        canvas = _compose_timeline_canvas(
            resolved=resolved,
            cache=cache,
            duration_sec=duration_sec,
            reveal_sec=None,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), canvas):
            raise RuntimeError(f"Failed to write summary PNG: {out_path}")
        return out_path
    finally:
        cache.close()


def save_input_builder_visualization_mp4(
    video_path: str,
    resolved: ResolvedInputPlan,
    out_path: Path,
    *,
    duration_sec: float,
    fps: float = FPS,
    hold_final_sec: float = 2.0,
) -> Path:
    """Write timeline MP4 that progressively reveals VLM inputs over time."""
    cache = VideoFrameCache(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        preview = _compose_timeline_canvas(
            resolved=resolved,
            cache=cache,
            duration_sec=duration_sec,
            reveal_sec=0.0,
        )
        h, w = preview.shape[:2]
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {out_path}")

        bottom_items = _bottom_timeline_items(resolved)
        reveal_times = sorted({i.timestamp_sec for i in bottom_items})
        if not reveal_times:
            reveal_times = [0.0]

        steps = [0.0]
        for t in reveal_times:
            if t > steps[-1]:
                steps.append(t)
        steps.append(duration_sec)

        for idx, reveal_sec in enumerate(steps):
            frame = _compose_timeline_canvas(
                resolved=resolved,
                cache=cache,
                duration_sec=duration_sec,
                reveal_sec=reveal_sec,
            )
            repeat = 1 if idx < len(steps) - 1 else int(hold_final_sec * fps)
            for _ in range(max(1, repeat)):
                writer.write(frame)

        writer.release()
        logger.info("Wrote input builder visualization: %s", out_path)
        return out_path
    finally:
        cache.close()


def build_report(
    resolved: ResolvedInputPlan,
    saved_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "motion_type": resolved.motion_type,
        "sparse_global_frame_count": resolved.sparse_global_frame_count,
        "roi_track_count": resolved.roi_track_count,
        "roi_frame_count": resolved.roi_frame_count,
        "event_anchor_count": resolved.event_anchor_count,
        "total_vlm_input_count": resolved.total_vlm_input_count,
        "inputs": saved_entries,
        "timestamps": {
            "sparse_global": [round(s.timestamp_sec, 4) for s in resolved.sparse_global],
            "roi": [round(r.timestamp_sec, 4) for r in resolved.roi_frames],
            "event_anchor": [round(a.timestamp_sec, 4) for a in resolved.event_anchors],
        "all": sorted(
            {
                round(s.timestamp_sec, 4)
                for s in resolved.sparse_global
            }
            | {
                round(r.timestamp_sec, 4)
                for r in resolved.roi_frames
            }
            | {
                round(a.timestamp_sec, 4)
                for a in resolved.event_anchors
            }
        ),
        },
    }

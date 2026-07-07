"""Visualize ground-truth motions/ROIs on source video."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motion_analyzer.v1.input_plan_visualize import (
    HUD_BG,
    HUD_FG,
    draw_motion_type_hud,
    roi_bbox_at_frame,
)

logger = logging.getLogger(__name__)

ROI_COLORS: dict[int, tuple[int, int, int]] = {
    1: (0, 255, 0),
    2: (0, 165, 255),
    3: (255, 128, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
}

OBJECT_BBOX_COLOR = (255, 255, 0)


def _resolve_video_path(video_id: str, repo_root: Path) -> str:
    summary = (
        repo_root
        / "Result"
        / "motion_analyzer_v2"
        / video_id
        / "_data"
        / "summary_pixel_v1.json"
    )
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        video_path = data.get("video_path", "")
        if video_path and Path(video_path).is_file():
            return video_path

    dataset_root = Path(r"C:\Datasets\VIRAT\video")
    if dataset_root.is_dir():
        matches = sorted(dataset_root.rglob(f"{video_id}.mp4"))
        if matches:
            return str(matches[0])

    raise FileNotFoundError(f"Video not found for {video_id}")


def _event_labels_at_frame(gt: dict[str, Any], frame_idx: int) -> list[str]:
    labels: list[str] = []
    for motion in gt.get("expected_motions", []):
        start_f = int(motion.get("start_frame_native", motion.get("start_frame_idx", -1)))
        end_f = int(motion.get("end_frame_native", motion.get("end_frame_idx", -1)))
        if start_f <= frame_idx <= end_f:
            label = motion.get("virat_event_type", "motion")
            event_id = motion.get("event_id")
            if event_id is not None:
                labels.append(f"E{event_id}: {label}")
            else:
                labels.append(str(label))
    return labels


def _object_bbox_at_frame(roi: dict[str, Any], frame_idx: int) -> list[int] | None:
    seq = sorted(roi.get("bbox_sequence", []), key=lambda e: int(e["frame_idx"]))
    if not seq:
        return None
    start_fi = int(seq[0]["frame_idx"])
    end_fi = int(seq[-1]["frame_idx"])
    if frame_idx < start_fi or frame_idx > end_fi:
        return None
    exact = [e for e in seq if int(e["frame_idx"]) == frame_idx]
    if exact:
        ob = exact[-1].get("object_bbox") or exact[-1].get("bbox")
        return [int(v) for v in ob] if ob else None
    before = [e for e in seq if int(e["frame_idx"]) <= frame_idx]
    if before:
        ob = before[-1].get("object_bbox") or before[-1].get("bbox")
        return [int(v) for v in ob] if ob else None
    return None


def _crop_bbox_at_frame(roi: dict[str, Any], frame_idx: int) -> list[int] | None:
    return roi_bbox_at_frame(roi, frame_idx)


def draw_ground_truth_frame(
    frame,
    frame_idx: int,
    gt: dict[str, Any],
) -> Any:
    vis = draw_motion_type_hud(
        frame,
        motion_type=str(gt.get("motion_type", "unknown")),
        roi_pattern=f"GT source: {gt.get('source', '?')}",
    )

    default_thin = gt.get("display_style") == "motion_bbox_thin"

    for roi in gt.get("roi_tracks", []):
        roi_id = int(roi.get("roi_id", 0))
        color = ROI_COLORS.get(roi_id, (255, 255, 0))
        display = roi.get("roi_display", "thin" if default_thin else "thick")

        motion_bbox = _object_bbox_at_frame(roi, frame_idx)
        crop_bbox = _crop_bbox_at_frame(roi, frame_idx)

        if display == "thick":
            if motion_bbox:
                x1, y1, x2, y2 = motion_bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), OBJECT_BBOX_COLOR, 1)
            if crop_bbox:
                cx1, cy1, cx2, cy2 = crop_bbox
                cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), color, 3)
                label_anchor = (cx1, cy1)
            elif motion_bbox:
                label_anchor = (motion_bbox[0], motion_bbox[1])
            else:
                continue
        else:
            if not motion_bbox:
                continue
            x1, y1, x2, y2 = motion_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label_anchor = (x1, y1)

        short = roi.get("label") or "motion"
        motion = next(
            (
                m
                for m in gt.get("expected_motions", [])
                if m.get("event_id") == roi.get("linked_event_id")
            ),
            None,
        )
        if motion:
            short = motion.get("virat_event_type", short)
        if len(short) > 36:
            short = short[:33] + "..."
        prefix = "R" if display == "thick" else "M"
        label = f"{prefix}{roi_id} {short}"
        cv2.putText(
            vis,
            label,
            (label_anchor[0] + 2, max(label_anchor[1] - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    active = _event_labels_at_frame(gt, frame_idx)
    if active:
        y = 52
        for line in active[:4]:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (8, y - th - 4), (14 + tw, y + 4), HUD_BG, -1)
            cv2.putText(
                vis,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 255, 180),
                1,
                cv2.LINE_AA,
            )
            y += th + 8

    ts = frame_idx / float(gt.get("native_fps", 30.0))
    ts_line = f"frame={frame_idx} t={ts:.2f}s"
    cv2.putText(
        vis,
        ts_line,
        (10, vis.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        HUD_FG,
        1,
        cv2.LINE_AA,
    )
    return vis


def save_ground_truth_overlay(
    video_path: str,
    gt: dict[str, Any],
    out_path: Path,
    *,
    output_fps: float | None = None,
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or gt.get("native_fps", 30.0))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = output_fps or native_fps

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video writer: {out_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vis = draw_ground_truth_frame(frame, frame_idx, gt)
        writer.write(vis)
        frame_idx += 1

    writer.release()
    cap.release()
    logger.info(
        "Wrote %s (%d frames, %.2ffps, %.1fs)",
        out_path,
        frame_idx,
        out_fps,
        frame_idx / max(out_fps, 1e-6),
    )
    return out_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Visualize ground-truth ROI overlay video.")
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("ground_truth"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ground_truth/overlays"),
    )
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="Output FPS (default: native video FPS)",
    )
    args = parser.parse_args()

    repo_root = PROJECT_ROOT
    gt_dir = (repo_root / args.gt_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(gt_dir.glob("*.ground_truth.json"))
    if args.video:
        wanted = set(args.video)
        gt_files = [p for p in gt_files if p.name.replace(".ground_truth.json", "") in wanted]

    if not gt_files:
        raise SystemExit(f"No ground truth JSON found in {gt_dir}")

    for gt_path in gt_files:
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        video_id = gt["video_id"]
        video_path = _resolve_video_path(video_id, repo_root)
        out_path = out_dir / f"{video_id}_ground_truth_overlay.mp4"
        save_ground_truth_overlay(
            video_path,
            gt,
            out_path,
            output_fps=args.output_fps,
        )
        print(f"{video_id} -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run Motion Analyzer v2 pipeline (ByteTrack + motion-first ROI)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motion_analyzer.v1.io import normalize_video_stem
from motion_analyzer.v2.io import DEFAULT_RESULT_ROOT_V2, DEFAULT_VIDEO_DIR_V2
from motion_analyzer.v2.object_detector import DEFAULT_STATIC_SUPPRESSION_CLASSES, DEFAULT_TARGET_CLASSES
from motion_analyzer.v2.pipeline import PipelineV2Config, process_video_v2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_class_list(value: str) -> list[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motion Analyzer v2: ByteTrack object tracking + motion-first ROI.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=DEFAULT_VIDEO_DIR_V2,
        help=f"Input video directory (default: {DEFAULT_VIDEO_DIR_V2})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_RESULT_ROOT_V2,
        help=f"Result root (default: {DEFAULT_RESULT_ROOT_V2})",
    )
    parser.add_argument(
        "--v1_output_dir",
        type=str,
        default="Result/motion_analyzer_v1",
        help="Optional v1 result root to reuse intermediate artifacts",
    )
    parser.add_argument("--video_name", type=str, required=True)
    parser.add_argument("--target_fps", type=float, default=5.0)
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort", "legacy"],
        help="Object tracker: bytetrack (default), botsort, or legacy detector-guided tubes",
    )
    parser.add_argument(
        "--tracker_config",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker config yaml (default: bytetrack.yaml)",
    )
    parser.add_argument(
        "--use_object_detector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy mode only: run frame-wise YOLO detector",
    )
    parser.add_argument("--detector_model", type=str, default="yolov8n.pt")
    parser.add_argument("--detector_conf", type=float, default=0.15)
    parser.add_argument(
        "--target_object_classes",
        type=str,
        default=",".join(DEFAULT_TARGET_CLASSES),
    )
    parser.add_argument(
        "--static_suppression_classes",
        type=str,
        default=",".join(DEFAULT_STATIC_SUPPRESSION_CLASSES),
    )
    parser.add_argument("--skip_pixel", action="store_true")
    parser.add_argument("--skip_box", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--skip_overlay", action="store_true")
    parser.add_argument(
        "--use_global_compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use_texture_suppression",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use_nearby_blob_merge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--nearby_merge_distance_ratio", type=float, default=0.06)
    parser.add_argument("--tube_max_gap", type=int, default=6, help="Legacy: max gap for tube linking")
    parser.add_argument("--gap_match_decay", type=float, default=0.85, help="Legacy: score decay per gap")
    parser.add_argument("--max_gap_seconds", type=float, default=1.2, help="Legacy: max gap seconds")
    parser.add_argument(
        "--show_all_tracks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug: draw all tracks including stationary",
    )
    parser.add_argument(
        "--show_stationary_tracks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug: draw stationary tracks in gray",
    )
    parser.add_argument(
        "--show_detector_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug: draw detector_only observations on overlay",
    )
    parser.add_argument(
        "--overlay_debug",
        action="store_true",
        help="Overlay raw motion blobs and detailed tracks on top of plan ROIs",
    )
    parser.add_argument(
        "--no_v1_reuse",
        action="store_true",
        help="Do not reuse artifacts from v1 output dir",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = PipelineV2Config(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        v1_output_dir=None if args.no_v1_reuse else args.v1_output_dir,
        target_fps=args.target_fps,
        use_object_detector=args.use_object_detector,
        detector_model=args.detector_model,
        detector_conf=args.detector_conf,
        tracker=args.tracker,
        tracker_config=args.tracker_config,
        target_object_classes=_parse_class_list(args.target_object_classes),
        static_suppression_classes=_parse_class_list(args.static_suppression_classes),
        skip_pixel=args.skip_pixel,
        skip_box=args.skip_box,
        skip_video=args.skip_video,
        skip_overlay=args.skip_overlay,
        show_all_tracks=args.show_all_tracks,
        show_stationary_tracks=args.show_stationary_tracks,
        show_detector_only=args.show_detector_only,
        overlay_debug=args.overlay_debug,
        use_global_compensation=args.use_global_compensation,
        use_texture_suppression=args.use_texture_suppression,
        use_nearby_blob_merge=args.use_nearby_blob_merge,
        nearby_merge_distance_ratio=args.nearby_merge_distance_ratio,
        tube_max_gap=args.tube_max_gap,
        gap_match_decay=args.gap_match_decay,
        max_gap_seconds=args.max_gap_seconds,
    )

    stem = normalize_video_stem(args.video_name)
    logger.info("Starting Motion Analyzer v2 (%s) for %s", cfg.tracker, stem)

    try:
        summary = process_video_v2(cfg, stem)
    except Exception:
        logger.exception("Pipeline failed for %s", stem)
        return 1

    overlay_key = (
        "tracker_motion_overlay"
        if cfg.tracker.lower() in ("bytetrack", "botsort", "bot-sort")
        else "detector_guided_motion_overlay"
    )
    logger.info(
        "Pipeline complete.\n"
        "  Plan overlay    : %s/%s/input_plan_overlay.mp4\n"
        "  Track overlay   : %s\n"
        "  Blob overlay    : %s/%s/motion_blobs_overlay.mp4 (debug only)\n"
        "  Data            : %s/%s/_data/",
        cfg.output_dir, stem,
        summary.get("outputs", {}).get(overlay_key),
        cfg.output_dir, stem, cfg.output_dir, stem,
    )
    logger.info("Summary: %s", summary.get("outputs", {}))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

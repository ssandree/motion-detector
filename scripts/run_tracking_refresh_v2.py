#!/usr/bin/env python3
"""Re-run v2 tracking + overlays without pixel/box/video stages."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motion_analyzer.v1.io import (
    data_dir,
    load_frame_features,
    load_motion_blobs,
    load_pixel_summary,
    normalize_video_stem,
    video_result_dir,
)
from motion_analyzer.v2.flow_group_merge import build_per_frame_flow_groups
from motion_analyzer.v2.io import DEFAULT_RESULT_ROOT_V2
from motion_analyzer.v2.pipeline import PipelineV2Config, ensure_base_stages, process_video_v2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh v2 tracking and overlays only.")
    p.add_argument("--output_dir", default=DEFAULT_RESULT_ROOT_V2)
    p.add_argument("--video_dir", default=r"C:/Datasets/VIRAT/video")
    p.add_argument("--video_name", required=True)
    p.add_argument("--force_retrack", action="store_true", help="Re-run YOLO+ByteTrack")
    p.add_argument("--overlay_debug", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stem = normalize_video_stem(args.video_name)
    cfg = PipelineV2Config(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        skip_pixel=True,
        skip_box=True,
        skip_video=True,
        force_retrack=args.force_retrack,
        overlay_debug=args.overlay_debug,
    )

    ensure_base_stages(cfg, stem)
    summary = process_video_v2(cfg, stem)

    # Both overlays + debug_roi_timeline.json are written inside process_video_v2
    # from the same adaptive_input_plan.json roi_inputs (no v1 tube rebuild).
    v2_dir = video_result_dir(cfg.output_dir, stem)
    ddir = data_dir(v2_dir)
    debug_path = ddir / "debug_roi_timeline.json"
    if debug_path.is_file():
        payload = json.loads(debug_path.read_text(encoding="utf-8"))
        logger.info(
            "debug_roi_timeline: %d frames, %d roi tracks",
            len(payload.get("timeline", [])),
            payload.get("num_roi_tracks", 0),
        )

    logger.info("Refresh complete: %s", summary.get("motion_type"))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

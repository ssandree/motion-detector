#!/usr/bin/env python3
"""Build VLM input frames and timeline visualization from adaptive_input_plan.json."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motion_analyzer.adaptive_input import build_adaptive_input
from motion_analyzer.v1.io import (
    list_result_videos,
    normalize_video_stem,
    save_json,
)
from motion_analyzer.v2.io import DEFAULT_RESULT_ROOT_V2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert adaptive_input_plan.json into VLM input frames "
            "and timeline visualization (no motion/tracking rerun)."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_RESULT_ROOT_V2,
        help=f"Result root containing per-video folders (default: {DEFAULT_RESULT_ROOT_V2})",
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
        help="Process a single video stem (with or without .mp4 suffix)",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Explicit path to a video result directory (overrides --output_dir/--video_name)",
    )
    parser.add_argument(
        "--plan_path",
        type=str,
        default=None,
        help="Optional path to adaptive_input_plan.json",
    )
    parser.add_argument("--max_videos", type=int, default=None)
    return parser.parse_args()


def _resolve_video_dirs(args: argparse.Namespace) -> list[Path]:
    if args.video_dir:
        return [Path(args.video_dir)]
    result_root = Path(args.output_dir)
    if args.video_name:
        return [result_root / normalize_video_stem(args.video_name)]
    video_dirs = list_result_videos(result_root, args.max_videos)
    return [
        v
        for v in video_dirs
        if (v / "_data" / "adaptive_input_plan.json").is_file()
    ]


def main() -> int:
    args = parse_args()
    video_dirs = _resolve_video_dirs(args)
    if not video_dirs:
        logger.error("No video directories found to process")
        return 1

    logger.info("Found %d video(s) to process", len(video_dirs))
    summaries = []
    ok = 0

    for i, video_dir in enumerate(video_dirs, 1):
        logger.info("[%d/%d] %s", i, len(video_dirs), video_dir.name)
        t0 = time.perf_counter()
        try:
            plan_path = Path(args.plan_path) if args.plan_path else None
            summary = build_adaptive_input(video_dir, plan_path=plan_path)
            summary["status"] = "ok"
            summary["elapsed_sec"] = round(time.perf_counter() - t0, 2)
            summaries.append(summary)
            ok += 1
        except Exception:
            logger.exception("Failed to build adaptive input for %s", video_dir.name)
            summaries.append(
                {
                    "video_name": video_dir.name,
                    "status": "error",
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                }
            )

    batch_path = Path(args.output_dir) / "batch_adaptive_input_summary.json"
    if len(summaries) > 1:
        save_json(batch_path, {"videos": summaries})
        logger.info("Batch summary: %s", batch_path)

    logger.info("Finished: %d/%d succeeded", ok, len(video_dirs))
    return 0 if ok == len(video_dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

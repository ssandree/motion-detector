#!/usr/bin/env python3
"""Run the full Motion Analyzer v1 pipeline (pixel → box → video level)."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motion_analyzer.v1.io import DEFAULT_RESULT_ROOT_V1, DEFAULT_VIDEO_DIR_V1, normalize_video_stem

DEFAULT_VIDEO_DIR = DEFAULT_VIDEO_DIR_V1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full motion analysis pipeline (unified per-video output)."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help=f"VIRAT root (default: {DEFAULT_VIDEO_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_RESULT_ROOT_V1,
        help=f"Result root (default: {DEFAULT_RESULT_ROOT_V1})",
    )
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--video_name", type=str, default=None)
    parser.add_argument("--target_fps", type=float, default=5.0)
    parser.add_argument("--skip_pixel", action="store_true")
    parser.add_argument("--skip_box", action="store_true")
    parser.add_argument("--skip_video", action="store_true", help="Skip Stage 3 (tubes/plan)")
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
    return parser.parse_args()


def _run(script: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *extra_args]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main() -> int:
    args = parse_args()
    out = ["--output_dir", args.output_dir]
    common: list[str] = list(out)

    if args.max_videos is not None:
        common += ["--max_videos", str(args.max_videos)]
    if args.video_name:
        common += ["--video_name", normalize_video_stem(args.video_name)]

    if not args.skip_pixel:
        pixel_args = ["--video_dir", args.video_dir, "--target_fps", str(args.target_fps)]
        pixel_args += out
        if args.max_videos is not None:
            pixel_args += ["--max_videos", str(args.max_videos)]
        if args.video_name:
            pixel_args += ["--video_name", normalize_video_stem(args.video_name)]
        if args.use_global_compensation:
            pixel_args.append("--use_global_compensation")
        else:
            pixel_args.append("--no-use_global_compensation")
        _run("run_motion_pixel_level_v1.py", pixel_args)

    if not args.skip_box:
        box_args = list(common)
        if args.use_texture_suppression:
            box_args.append("--use_texture_suppression")
        else:
            box_args.append("--no-use_texture_suppression")
        if args.use_nearby_blob_merge:
            box_args.append("--use_nearby_blob_merge")
        else:
            box_args.append("--no-use_nearby_blob_merge")
        box_args += [
            "--nearby_merge_distance_ratio",
            str(args.nearby_merge_distance_ratio),
        ]
        _run("run_motion_box_level_v1.py", box_args)

    if not args.skip_video:
        _run("run_motion_video_level_v1.py", common)

    stem = normalize_video_stem(args.video_name) if args.video_name else "{video_name}"
    logger.info(
        "Pipeline complete.\n"
        "  Human output : %s/%s/input_plan_overlay.mp4\n"
        "  Cache        : %s/%s/_cache/motion_maps.npz\n"
        "  Data         : %s/%s/_data/",
        args.output_dir, stem, args.output_dir, stem, args.output_dir, stem,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

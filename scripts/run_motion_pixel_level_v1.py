#!/usr/bin/env python3
"""Run pixel-level motion aggregation pipeline (Motion Analyzer v1)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motion_analyzer.v1.features import FeatureConfig, extract_all_features
from motion_analyzer.v1.io import (
    DEFAULT_RESULT_ROOT_V1,
    DEFAULT_VIDEO_DIR_V1,
    data_dir,
    get_video_metadata,
    list_videos,
    normalize_video_stem,
    resolve_video_path,
    save_frame_features,
    save_json,
    save_motion_map_layers,
    save_motion_maps,
    save_sampled_gray,
    video_result_dir,
)
from motion_analyzer.v1.motion_map_visualize import save_motion_layer_overlays
from motion_analyzer.v1.pixel_motion import MotionConfig, process_video_motion_maps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_VIDEO_DIR = DEFAULT_VIDEO_DIR_V1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pixel-level motion aggregation (writes to _cache/ and _data/)."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help=f"VIRAT root with videos-00/, videos-01/, … (default: {DEFAULT_VIDEO_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_RESULT_ROOT_V1,
        help=f"Result root (default: {DEFAULT_RESULT_ROOT_V1})",
    )
    parser.add_argument("--target_fps", type=float, default=5.0)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--video_name", type=str, default=None)
    parser.add_argument("--use_flow", action="store_true", default=False)
    parser.add_argument(
        "--use_global_compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate and subtract global camera motion (default: on)",
    )
    parser.add_argument(
        "--flow_scale",
        type=float,
        default=0.5,
        help="Optical-flow downscale factor for speed (1.0=full res, 0.5≈4× faster)",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--motion_threshold", type=float, default=0.05)
    parser.add_argument("--no_compress", action="store_true")
    return parser.parse_args()


def process_single_video(
    video_path: Path,
    result_root: Path,
    motion_config: MotionConfig,
    feature_config: FeatureConfig,
    *,
    compressed: bool = True,
) -> dict:
    """Process one video; cache motion maps under ``{stem}/_cache/``."""
    stem = video_path.stem
    video_dir = video_result_dir(result_root, stem)
    t0 = time.perf_counter()

    logger.info("Processing %s", video_path.name)
    metadata = get_video_metadata(video_path)

    motion_maps, frame_diff_means, frame_indices, timestamps, global_rows, sampled_gray, layer_maps = (
        process_video_motion_maps(str(video_path), motion_config)
    )

    if not motion_maps:
        logger.warning("No motion maps produced for %s (too few frames?)", video_path.name)
        return {"video": stem, "status": "skipped", "reason": "no_motion_maps"}

    features = extract_all_features(
        motion_maps,
        frame_indices,
        timestamps,
        frame_diff_means,
        feature_config,
        global_motion_rows=global_rows,
    )

    maps_array = np.stack(motion_maps, axis=0)
    maps_path = save_motion_maps(maps_array, video_dir, compressed=compressed)
    layers_path = save_motion_map_layers(layer_maps, video_dir, compressed=compressed)
    gray_path = save_sampled_gray(sampled_gray, video_dir)
    csv_path = save_frame_features(features, video_dir)

    overlay_paths: dict[str, str] = {}
    try:
        written = save_motion_layer_overlays(
            str(video_path),
            frame_indices,
            layer_maps,
            video_dir,
            output_fps=motion_config.target_fps,
        )
        overlay_paths = {k: str(v) for k, v in written.items()}
    except Exception:
        logger.exception("Failed to write motion compensation overlays for %s", stem)

    jitter_frames = sum(1 for r in global_rows if r.get("background_jitter_flag"))

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "pixel",
        "video": stem,
        "video_path": str(video_path),
        "status": "ok",
        "native_fps": metadata["native_fps"],
        "target_fps": motion_config.target_fps,
        "sampled_frame_pairs": len(motion_maps),
        "motion_map_shape": list(maps_array.shape),
        "use_flow": motion_config.use_flow,
        "use_global_compensation": motion_config.use_global_compensation,
        "use_jitter_suppression": motion_config.use_jitter_suppression,
        "flow_scale": motion_config.flow_scale,
        "alpha": motion_config.alpha,
        "beta": motion_config.beta,
        "motion_threshold": feature_config.motion_threshold,
        "background_jitter_frames": jitter_frames,
        "elapsed_sec": round(elapsed, 2),
        "outputs": {
            "motion_maps": str(maps_path),
            "motion_map_layers": str(layers_path) if layers_path else None,
            "sampled_gray": str(gray_path),
            "frame_features": str(csv_path),
            **{f"overlay_{k}": v for k, v in overlay_paths.items()},
        },
    }
    save_json(data_dir(video_dir) / "summary_pixel_v1.json", summary)
    logger.info(
        "Done %s: %d motion maps, %d jitter-suppressed frames in %.1fs",
        stem,
        len(motion_maps),
        jitter_frames,
        elapsed,
    )
    return summary


def main() -> int:
    args = parse_args()
    result_root = Path(args.output_dir)
    result_root.mkdir(parents=True, exist_ok=True)

    motion_config = MotionConfig(
        target_fps=args.target_fps,
        use_flow=args.use_flow,
        use_global_compensation=args.use_global_compensation,
        use_jitter_suppression=True,
        flow_scale=args.flow_scale,
        alpha=args.alpha,
        beta=args.beta,
    )
    feature_config = FeatureConfig(motion_threshold=args.motion_threshold)

    if args.video_name:
        stem = normalize_video_stem(args.video_name)
        try:
            candidate = resolve_video_path(stem, args.video_dir)
        except FileNotFoundError:
            logger.error("Video not found: %s under %s", stem, args.video_dir)
            return 1
        videos = [candidate]
    else:
        videos = list_videos(args.video_dir, args.max_videos)

    logger.info("Found %d video(s) to process", len(videos))

    all_summaries = []
    for i, video_path in enumerate(videos, 1):
        logger.info("[%d/%d] %s", i, len(videos), video_path.name)
        try:
            summary = process_single_video(
                video_path,
                result_root,
                motion_config,
                feature_config,
                compressed=not args.no_compress,
            )
            all_summaries.append(summary)
        except Exception:
            logger.exception("Failed to process %s", video_path.name)
            all_summaries.append({"video": video_path.stem, "status": "error"})

    ok_count = sum(1 for s in all_summaries if s.get("status") == "ok")
    logger.info("Finished: %d/%d videos processed successfully", ok_count, len(videos))

    save_json(result_root / "batch_summary_pixel_v1.json", {"videos": all_summaries})
    return 0 if ok_count == len(videos) else 1


if __name__ == "__main__":
    raise SystemExit(main())

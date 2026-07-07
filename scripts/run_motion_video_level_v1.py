#!/usr/bin/env python3
"""Run video-level motion tube aggregation and adaptive input planning."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from collections import defaultdict

from motion_analyzer.v1.input_plan_builder import build_adaptive_input_plan_v1
from motion_analyzer.v1.input_plan_visualize import canonicalize_plan_roi_tracks, save_input_plan_overlay
from motion_analyzer.v1.roi_debug import build_debug_roi_timeline, save_debug_roi_timeline
from motion_analyzer.v1.io import (
    DEFAULT_RESULT_ROOT_V1,
    data_dir,
    list_result_videos,
    load_frame_features,
    load_motion_blobs,
    load_pixel_summary,
    normalize_video_stem,
    save_json,
)
from motion_analyzer.v1.tube_builder import TubeMatchConfig, build_tubes
from motion_analyzer.v1.tube_features import tubes_to_feature_rows
from motion_analyzer.v1.video_features import classify_video_category, compute_video_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video-level tube aggregation (writes to _data/)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_RESULT_ROOT_V1,
        help=f"Result root (default: {DEFAULT_RESULT_ROOT_V1})",
    )
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--video_name", type=str, default=None)
    parser.add_argument("--max_gap", type=int, default=2)
    parser.add_argument("--match_threshold", type=float, default=0.25)
    parser.add_argument("--top_blobs_per_frame", type=int, default=50)
    parser.add_argument("--min_tube_length", type=int, default=2)
    parser.add_argument("--top_k_tubes", type=int, default=5)
    parser.add_argument("--skip_overlay", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Overlay raw motion blobs in addition to plan ROIs",
    )
    return parser.parse_args()


def _write_video_summary_md(
    path: Path,
    *,
    video_name: str,
    video_features: dict,
    category_info: dict,
    plan: dict,
    tube_count: int,
) -> None:
    lines = [
        f"# Video Summary: {video_name}",
        "",
        "## Motion Type",
        f"- **{category_info['motion_type']}** (confidence: {category_info['confidence']})",
        f"- Reason: {category_info['primary_reason']}",
        "",
        "## Video Features",
        f"- Duration: {video_features['duration_sec']:.1f}s",
        f"- Sampled frames: {video_features['num_sampled_frames']} @ {video_features['sampled_fps']} fps",
        f"- Global motion level: {video_features['global_motion_level']:.4f}",
        f"- Motion tubes: {tube_count}",
        f"- Local motion importance ratio: {video_features['local_motion_importance_ratio']:.4f}",
        f"- Scene changes: {video_features['scene_change_count']}",
        "",
        "## Adaptive Input Plan",
        f"- Motion type: {plan.get('motion_type', category_info['motion_type'])}",
        f"- Sparse global frames: {plan['diagnostics']['sparse_global_frame_count']}",
        f"- ROI tracks: {plan['diagnostics']['num_selected_rois']}",
        f"- ROI pattern: {plan['diagnostics']['roi_pattern']}",
        "",
        "## ROI Tracks",
    ]
    for roi in plan.get("roi_inputs", []):
        lines.append(
            f"- ROI {roi['roi_id']}: tubes={roi['source_tube_ids']}, "
            f"{roi['start_sec']:.1f}-{roi['end_sec']:.1f}s, "
            f"speed={roi['mean_motion_speed']:.1f}, fps={roi['sampling_fps']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_single_video(
    video_dir: Path,
    match_config: TubeMatchConfig,
    *,
    min_tube_length: int = 2,
    top_k_tubes: int = 5,
    skip_overlay: bool = False,
    debug: bool = False,
) -> dict:
    """Run video-level aggregation; all artifacts under ``_data/``."""
    video_name = video_dir.name
    ddir = data_dir(video_dir)
    t0 = time.perf_counter()

    logger.info("Building tubes for %s", video_name)

    blobs_df = load_motion_blobs(video_dir)
    frame_df = load_frame_features(video_dir)
    pixel_summary = load_pixel_summary(video_dir)

    frame_w = pixel_summary.get("motion_map_shape", [0, 720, 1280])[2]
    frame_h = pixel_summary.get("motion_map_shape", [0, 720, 1280])[1]
    sampled_fps = float(pixel_summary.get("target_fps", 5.0))
    duration_sec = float(frame_df["timestamp_sec"].max()) if len(frame_df) else 0.0

    tube_segments = build_tubes(
        blobs_df,
        frame_w=frame_w,
        frame_h=frame_h,
        config=match_config,
        min_tube_length=min_tube_length,
    )

    tube_obs_map = {t.tube_id: t.observations for t in tube_segments}
    tubes = tubes_to_feature_rows(
        tube_segments,
        total_sampled_frames=len(frame_df),
        video_duration_sec=duration_sec,
    )

    video_features = compute_video_features(
        frame_df, tubes, blobs_df, sampled_fps=sampled_fps
    )
    category_info = classify_video_category(video_features)

    plan = build_adaptive_input_plan_v1(
        category_info["motion_type"],
        frame_df,
        tubes,
        tube_obs_map,
        video_id=video_name,
        sampled_fps=sampled_fps,
        num_raw_blobs=len(blobs_df),
    )

    tubes_csv = ddir / "motion_tubes.csv"
    pd.DataFrame(tubes).to_csv(tubes_csv, index=False)

    vf_csv = ddir / "video_motion_features.csv"
    pd.DataFrame([video_features]).to_csv(vf_csv, index=False)

    cat_path = ddir / "video_category.json"
    save_json(cat_path, category_info)

    plan_path = ddir / "adaptive_input_plan.json"
    canonicalize_plan_roi_tracks(plan)
    save_json(plan_path, plan)

    sampled_indices = [int(r["frame_idx"]) for _, r in frame_df.iterrows()]
    save_debug_roi_timeline(
        ddir / "debug_roi_timeline.json",
        build_debug_roi_timeline(plan, frame_df, blobs_df),
    )

    summary_md = ddir / "video_summary.md"
    _write_video_summary_md(
        summary_md,
        video_name=video_name,
        video_features=video_features,
        category_info=category_info,
        plan=plan,
        tube_count=len(tubes),
    )

    overlay_path = None
    if not skip_overlay:
        video_path = pixel_summary.get("video_path", "")
        if video_path and Path(video_path).is_file():
            frame_blob_map: dict[int, list] = defaultdict(list)
            if debug:
                raw_csv = ddir / "motion_blobs_raw.csv"
                blob_src = raw_csv if raw_csv.is_file() else ddir / "motion_blobs.csv"
                if blob_src.is_file():
                    for row in pd.read_csv(blob_src).to_dict("records"):
                        frame_blob_map[int(row["frame_idx"])].append(row)
            try:
                overlay_path = save_input_plan_overlay(
                    video_path,
                    plan,
                    video_dir,
                    motion_type=category_info["motion_type"],
                    frame_blob_map=dict(frame_blob_map) if debug else None,
                    debug=debug,
                    output_fps=sampled_fps,
                    filename="input_plan_overlay.mp4",
                    sampled_frame_indices=sampled_indices,
                    continuous=True,
                )
            except Exception:
                logger.exception("Failed to write input plan overlay for %s", video_name)
        else:
            logger.warning("Video not found for overlay: %s", video_path)

    elapsed = time.perf_counter() - t0
    summary = {
        "stage": "video",
        "video_name": video_name,
        "status": "ok",
        "num_tubes": len(tubes),
        "motion_type": category_info["motion_type"],
        "local_motion_importance_ratio": video_features["local_motion_importance_ratio"],
        "elapsed_sec": round(elapsed, 2),
        "outputs": {
            "motion_tubes": str(tubes_csv),
            "video_motion_features": str(vf_csv),
            "video_category": str(cat_path),
            "adaptive_input_plan": str(plan_path),
            "video_summary": str(summary_md),
            "input_plan_overlay": str(overlay_path) if overlay_path else None,
        },
    }
    save_json(ddir / "summary_v1.json", summary)

    logger.info(
        "Done %s: %d tubes, motion_type=%s in %.1fs",
        video_name, len(tubes), category_info["motion_type"], elapsed,
    )
    return summary


def main() -> int:
    args = parse_args()
    result_root = Path(args.output_dir)
    result_root.mkdir(parents=True, exist_ok=True)

    match_config = TubeMatchConfig(
        max_gap=args.max_gap,
        match_threshold=args.match_threshold,
        top_blobs_per_frame=args.top_blobs_per_frame,
    )

    if args.video_name:
        video_dirs = [result_root / normalize_video_stem(args.video_name)]
    else:
        video_dirs = list_result_videos(result_root, args.max_videos, require_blobs=True)

    logger.info("Found %d video(s) to process", len(video_dirs))

    all_summaries = []
    for i, video_dir in enumerate(video_dirs, 1):
        logger.info("[%d/%d] %s", i, len(video_dirs), video_dir.name)
        try:
            summary = process_single_video(
                video_dir,
                match_config,
                min_tube_length=args.min_tube_length,
                top_k_tubes=args.top_k_tubes,
                skip_overlay=args.skip_overlay,
                debug=args.debug,
            )
            all_summaries.append(summary)
        except Exception:
            logger.exception("Failed to process %s", video_dir.name)
            all_summaries.append({"video_name": video_dir.name, "status": "error"})

    ok = sum(1 for s in all_summaries if s.get("status") == "ok")
    logger.info("Finished: %d/%d videos processed successfully", ok, len(video_dirs))

    save_json(result_root / "batch_summary_v1.json", {"videos": all_summaries})
    return 0 if ok == len(video_dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

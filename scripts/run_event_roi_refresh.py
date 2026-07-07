#!/usr/bin/env python3
"""
Rebuild event ROI tracks from motion blobs only (no tracking re-run).

Uses motion_blobs.csv + frame_motion_features.csv.
object_tracks.csv is optional — auxiliary bbox expansion only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motion_analyzer.v1.roi_debug import build_debug_roi_timeline, save_debug_roi_timeline
from motion_analyzer.v1.input_plan_visualize import (
    canonicalize_plan_roi_tracks,
    save_input_plan_overlay,
)
from motion_analyzer.v1.io import data_dir, load_frame_features, load_motion_blobs, load_pixel_summary, normalize_video_stem, save_json, video_result_dir
from motion_analyzer.v2.event_roi_grouping import (
    build_event_roi_pipeline,
    load_auxiliary_track_obs,
    save_event_roi_candidates_csv,
)
from motion_analyzer.v2.event_roi_visualize import save_final_roi_tracks_overlay, save_grouped_blobs_overlay
from motion_analyzer.v2.flow_group_merge import build_per_frame_flow_groups
from motion_analyzer.v2.plan_finalize import finalize_adaptive_input_plan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild event ROI from blobs (no tracking).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--video_name", required=True)
    p.add_argument("--write_overlays", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stem = normalize_video_stem(args.video_name)
    v2_dir = video_result_dir(args.output_dir, stem)
    ddir = data_dir(v2_dir)

    blobs_df = load_motion_blobs(v2_dir)
    frame_df = load_frame_features(v2_dir)
    pixel = load_pixel_summary(v2_dir)
    shape = pixel.get("motion_map_shape", [0, 720, 1280])
    frame_h, frame_w = int(shape[1]), int(shape[2])
    sampled_fps = float(pixel.get("target_fps", 5.0))
    video_path = pixel.get("video_path", "")

    motion_type = "event_motion"
    reason = ""
    cat_path = ddir / "video_category.json"
    if cat_path.is_file():
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        motion_type = cat.get("motion_type", motion_type)
        reason = str(cat.get("primary_reason", cat.get("reason", "")))

    aux_obs = load_auxiliary_track_obs(v2_dir)
    logger.info("Auxiliary track observations: %d tracks (expansion only)", len(aux_obs))

    result = build_event_roi_pipeline(
        blobs_df,
        frame_df,
        aux_obs,
        frame_w=frame_w,
        frame_h=frame_h,
        sampled_fps=sampled_fps,
        motion_type=motion_type,
    )

    save_json(ddir / "event_roi_candidates.json", {"candidates": result.candidates})
    save_event_roi_candidates_csv(result.candidates, ddir / "event_roi_candidates.csv")
    save_json(ddir / "event_roi_tracks.json", {"roi_tracks": result.selected_tracks})
    save_json(ddir / "event_roi_debug.json", result.debug_report)
    save_json(ddir / "background_flow_regions.json", {"regions": result.background_flow_regions})

    plan_path = ddir / "adaptive_input_plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        plan = {"motion_type": motion_type, "frames": []}

    event_tracks = result.selected_tracks if motion_type == "event_motion" else None
    plan_debug = finalize_adaptive_input_plan(
        plan,
        motion_type=motion_type,
        reason=reason,
        event_roi_tracks=event_tracks,
        event_roi_debug=result.debug_report,
    )
    save_json(ddir / "adaptive_input_plan_debug.json", plan_debug)
    canonicalize_plan_roi_tracks(plan)
    save_json(plan_path, plan)
    save_debug_roi_timeline(
        ddir / "debug_roi_timeline.json",
        build_debug_roi_timeline(
            plan,
            frame_df,
            blobs_df,
            flow_group_map=build_per_frame_flow_groups(blobs_df, frame_w=frame_w, frame_h=frame_h),
            event_roi_debug=result.debug_report,
        ),
    )

    logger.info(
        "event ROI: %d candidates, %d selected; plan roi_inputs=%d roi_source=%s",
        len(result.candidates),
        len(result.selected_tracks),
        plan_debug["num_roi_inputs"],
        plan_debug["roi_source"],
    )
    for roi in result.selected_tracks:
        logger.info(
            "  ROI%d group=%s duration=%.2fs frames=%d aux_tracks=%s",
            roi["roi_id"],
            roi["group_id"],
            roi.get("duration_sec", 0),
            roi.get("bbox_sequence_len", 0),
            roi.get("auxiliary_track_ids", []),
        )

    if args.write_overlays and video_path and Path(video_path).is_file():
        sampled = [int(r["frame_idx"]) for _, r in frame_df.iterrows()]
        flow_map = build_per_frame_flow_groups(blobs_df, frame_w=frame_w, frame_h=frame_h)
        save_grouped_blobs_overlay(
            video_path, result.candidates, sampled, v2_dir, output_fps=sampled_fps,
        )
        save_final_roi_tracks_overlay(
            video_path, plan, sampled, v2_dir, output_fps=sampled_fps,
            debug_rejected_high_rank=result.debug_rejected_high_rank,
        )
        save_input_plan_overlay(
            video_path,
            plan,
            v2_dir,
            motion_type=motion_type,
            flow_group_map=flow_map,
            output_fps=sampled_fps,
            sampled_frame_indices=sampled,
            continuous=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

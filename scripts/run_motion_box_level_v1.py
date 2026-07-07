#!/usr/bin/env python3
"""Run box-level motion blob aggregation (Motion Analyzer v1)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motion_analyzer.v1.blob_extractor import (
    BlobExtractConfig,
    extract_blobs_from_motion_map,
    load_gray_from_dict,
    read_gray_frames_cached,
)
from motion_analyzer.v1.blob_features import (
    BlobFeatureConfig,
    BlobHistory,
    add_importance_and_ranks,
    enrich_temporal_blob_features,
    filter_persistent_blobs,
    process_frame_blobs,
    select_top_k_blobs,
)
from motion_analyzer.v1.blob_merge import merge_nearby_raw_blobs
from motion_analyzer.v1.blob_visualize import save_blob_overlay_video
from motion_analyzer.v1.io import (
    DEFAULT_RESULT_ROOT_V1,
    data_dir,
    list_result_videos,
    load_frame_features,
    load_motion_maps,
    load_pixel_summary,
    load_sampled_gray,
    normalize_video_stem,
    overlay_video_path,
    save_json,
)
from motion_analyzer.v1.pixel_motion import compute_optical_flow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Box-level grid motion aggregation (unified output layout)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_RESULT_ROOT_V1,
        help=f"Result root (default: {DEFAULT_RESULT_ROOT_V1})",
    )
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--grid_rows", type=int, default=20)
    parser.add_argument("--grid_cols", type=int, default=20)
    parser.add_argument("--cell_motion_threshold", type=float, default=0.05)
    parser.add_argument("--no_flow_filter", action="store_true")
    parser.add_argument("--min_flow_magnitude", type=float, default=0.5)
    parser.add_argument("--min_direction_coherence", type=float, default=0.45)
    parser.add_argument("--small_blob_area_ratio", type=float, default=0.003)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--viz_max_blobs", type=int, default=50)
    parser.add_argument("--skip_overlay_video", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write motion_blobs_overlay.mp4 with raw blob bboxes (debug only)",
    )
    parser.add_argument(
        "--min_persistence_frames",
        type=int,
        default=2,
        help="Keep blobs only if they persist at least N sampled frames (default: 2)",
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
    parser.add_argument("--texture_flicker_penalty", type=float, default=0.20)
    parser.add_argument("--global_motion_penalty", type=float, default=0.15)
    parser.add_argument(
        "--flow_scale",
        type=float,
        default=None,
        help="Optical-flow downscale (default: use pixel summary or 0.5)",
    )
    parser.add_argument(
        "--overlay_filename",
        type=str,
        default="motion_blobs_overlay.mp4",
        help="Output overlay MP4 filename under video result dir",
    )
    parser.add_argument("--video_name", type=str, default=None)
    return parser.parse_args()


def _re_rank_all_frames(
    all_blobs: list[dict],
    config: BlobFeatureConfig,
) -> list[dict]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for blob in all_blobs:
        by_frame[int(blob["frame_idx"])].append(blob)

    ranked: list[dict] = []
    for frame_idx in sorted(by_frame):
        frame_blobs = add_importance_and_ranks(by_frame[frame_idx], config)
        for i, blob in enumerate(frame_blobs):
            blob["blob_id_in_frame"] = i
        ranked.extend(frame_blobs)
    return ranked


def process_single_video(
    video_dir: Path,
    extract_config: BlobExtractConfig,
    feature_config: BlobFeatureConfig,
    *,
    top_k: int = 10,
    viz_max_blobs: int = 50,
    skip_overlay_video: bool = False,
    min_persistence_frames: int = 2,
    use_nearby_blob_merge: bool = True,
    nearby_merge_distance_ratio: float = 0.06,
    overlay_filename: str = "motion_blobs_overlay.mp4",
    debug: bool = False,
) -> dict:
    """Run box-level aggregation; writes overlay mp4 + ``_data/`` artifacts."""
    video_name = video_dir.name
    t0 = time.perf_counter()
    ddir = data_dir(video_dir)

    logger.info("Processing blobs for %s", video_name)

    pixel_summary = load_pixel_summary(video_dir)
    if "flow_scale" in pixel_summary:
        extract_config.flow_scale = float(pixel_summary["flow_scale"])
    motion_maps = load_motion_maps(video_dir)
    frame_df = load_frame_features(video_dir)

    if len(motion_maps) != len(frame_df):
        raise ValueError(
            f"Motion map count ({len(motion_maps)}) != frame feature rows ({len(frame_df)})"
        )

    video_path = pixel_summary.get("video_path", "")
    sampled_fps = float(pixel_summary.get("target_fps", 5.0))
    history = BlobHistory(max_frames=feature_config.spatial_novelty_history_frames)

    all_raw: list[dict] = []
    all_merged: list[dict] = []

    frame_indices_list = [int(row["frame_idx"]) for _, row in frame_df.iterrows()]
    prev_indices = [
        frame_indices_list[i - 1] if i > 0 else None for i in range(len(frame_indices_list))
    ]

    gray_cache: dict[int, np.ndarray] = {}
    need_gray = extract_config.use_flow_filter or feature_config.use_texture_suppression
    if need_gray:
        needed = set(frame_indices_list)
        needed.update(idx for idx in prev_indices if idx is not None)
        cached = load_sampled_gray(video_dir)
        if cached:
            gray_cache = load_gray_from_dict(cached, needed)
            logger.info("Loaded %d/%d gray frames from _cache/sampled_gray.npz", len(gray_cache), len(needed))
        if len(gray_cache) < len(needed) and video_path and Path(video_path).exists():
            missing = needed - set(gray_cache.keys())
            logger.info("Reading %d gray frames from video (cache miss)", len(missing))
            gray_cache.update(read_gray_frames_cached(video_path, missing))

    for i in range(len(motion_maps)):
        row = frame_df.iloc[i]
        frame_idx = int(row["frame_idx"])
        timestamp_sec = float(row["timestamp_sec"])
        motion_map = motion_maps[i]
        camera_flag = bool(row.get("camera_motion_flag", False))
        gscore = float(row.get("global_motion_score", 0.0))

        threshold_override = None
        if camera_flag:
            threshold_override = extract_config.cell_motion_threshold * 2.0

        prev_idx = prev_indices[i]
        prev_gray = gray_cache.get(prev_idx) if prev_idx is not None else None
        curr_gray = gray_cache.get(frame_idx)

        flow = None
        if prev_gray is not None and curr_gray is not None and extract_config.use_flow_filter:
            flow = compute_optical_flow(
                prev_gray, curr_gray, scale=extract_config.flow_scale
            )

        raw_blobs = extract_blobs_from_motion_map(
            motion_map,
            extract_config,
            prev_gray=prev_gray,
            curr_gray=curr_gray,
            flow=flow,
            cell_motion_threshold_override=threshold_override,
        )

        raw_features = process_frame_blobs(
            motion_map,
            raw_blobs,
            video_name=video_name,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            history=history,
            config=feature_config,
            curr_gray=curr_gray,
            global_motion_score=gscore,
            camera_motion_flag=camera_flag,
            merge_stage="raw",
            compute_ranks=False,
        )
        all_raw.extend(raw_features)

        blobs_for_features = raw_blobs
        if use_nearby_blob_merge and len(raw_blobs) > 1:
            blobs_for_features = merge_nearby_raw_blobs(
                raw_blobs,
                motion_map,
                flow,
                extract_config,
                distance_ratio=nearby_merge_distance_ratio,
            )

        merged_features = process_frame_blobs(
            motion_map,
            blobs_for_features,
            video_name=video_name,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            history=history,
            config=feature_config,
            curr_gray=curr_gray,
            global_motion_score=gscore,
            camera_motion_flag=camera_flag,
            merge_stage="merged",
            compute_ranks=False,
        )
        all_merged.extend(merged_features)

    frame_h, frame_w = motion_maps[0].shape[:2]
    all_merged = enrich_temporal_blob_features(
        all_merged,
        frame_h=frame_h,
        frame_w=frame_w,
        config=feature_config,
    )
    all_merged = _re_rank_all_frames(all_merged, feature_config)

    blobs_before = len(all_merged)
    all_blobs, removed = filter_persistent_blobs(
        all_merged,
        frame_w=frame_w,
        frame_h=frame_h,
        min_frames=min_persistence_frames,
        config=feature_config,
    )
    frame_blob_map: dict[int, list[dict]] = defaultdict(list)
    for blob in all_blobs:
        frame_blob_map[int(blob["frame_idx"])].append(blob)

    logger.info(
        "Persistence filter (>=%d frames): %d -> %d blobs (removed %d)",
        min_persistence_frames,
        blobs_before,
        len(all_blobs),
        removed,
    )

    top_by_importance = select_top_k_blobs(all_blobs, top_k, rank_key="rank_by_importance")
    top_by_energy = select_top_k_blobs(all_blobs, top_k, rank_key="rank_by_motion_energy")

    raw_csv = ddir / "motion_blobs_raw.csv"
    merged_csv = ddir / "motion_blobs_merged.csv"
    blobs_csv = ddir / "motion_blobs.csv"
    top_csv = ddir / "frame_top_blobs.csv"

    pd.DataFrame(all_raw).to_csv(raw_csv, index=False)
    pd.DataFrame(all_merged).to_csv(merged_csv, index=False)
    pd.DataFrame(all_blobs).to_csv(blobs_csv, index=False)

    top_df = pd.DataFrame(top_by_importance)
    top_df["selection_criterion"] = "importance"
    top_energy_df = pd.DataFrame(top_by_energy)
    top_energy_df["selection_criterion"] = "motion_energy"
    pd.concat([top_df, top_energy_df], ignore_index=True).to_csv(top_csv, index=False)

    overlay_path: Path | None = None
    if debug and not skip_overlay_video and video_path and Path(video_path).exists():
        raw_blob_map: dict[int, list] = defaultdict(list)
        for blob in all_raw:
            raw_blob_map[int(blob["frame_idx"])].append(blob)
        overlay_path = save_blob_overlay_video(
            video_path,
            dict(raw_blob_map),
            video_dir,
            output_fps=sampled_fps,
            max_blobs_per_frame=viz_max_blobs,
            filename=overlay_filename,
            debug=True,
        )
    elif not skip_overlay_video and not debug:
        logger.info(
            "Skipping blob overlay for %s (use --debug to draw raw blobs)", video_name
        )
    elif not skip_overlay_video:
        logger.warning("Video not found for overlay viz: %s", video_path)

    elapsed = time.perf_counter() - t0
    small_count = sum(1 for b in all_blobs if b["small_blob_flag"])
    camera_frames = int(frame_df["camera_motion_flag"].sum()) if "camera_motion_flag" in frame_df else 0

    summary = {
        "stage": "box",
        "video_name": video_name,
        "status": "ok",
        "total_blobs": len(all_blobs),
        "total_raw_blobs": len(all_raw),
        "total_merged_blobs_before_persistence": len(all_merged),
        "total_frames": len(motion_maps),
        "avg_blobs_per_frame": round(len(all_blobs) / max(len(motion_maps), 1), 2),
        "small_blob_count": small_count,
        "camera_motion_frames": camera_frames,
        "grid_rows": extract_config.grid_rows,
        "grid_cols": extract_config.grid_cols,
        "cell_motion_threshold": extract_config.cell_motion_threshold,
        "use_flow_filter": extract_config.use_flow_filter,
        "use_texture_suppression": feature_config.use_texture_suppression,
        "use_nearby_blob_merge": use_nearby_blob_merge,
        "nearby_merge_distance_ratio": nearby_merge_distance_ratio,
        "min_persistence_frames": min_persistence_frames,
        "blobs_before_persistence_filter": blobs_before,
        "blobs_removed_single_frame": removed,
        "elapsed_sec": round(elapsed, 2),
        "outputs": {
            "motion_blobs_overlay": str(overlay_path or overlay_video_path(video_dir)),
            "motion_blobs_raw": str(raw_csv),
            "motion_blobs_merged": str(merged_csv),
            "motion_blobs": str(blobs_csv),
            "frame_top_blobs": str(top_csv),
        },
        "rank_comparison": _count_rank_disagreements(all_blobs, top_k),
    }
    save_json(ddir / "summary_box_v1.json", summary)

    logger.info(
        "Done %s: %d blobs across %d frames in %.1fs",
        video_name,
        len(all_blobs),
        len(motion_maps),
        elapsed,
    )
    return summary


def _count_rank_disagreements(all_blobs: list[dict], top_k: int) -> dict:
    frames: dict[int, list[dict]] = defaultdict(list)
    for b in all_blobs:
        frames[b["frame_idx"]].append(b)

    total_frames = 0
    disagreement_frames = 0
    small_in_importance_not_energy = 0

    for frame_blobs in frames.values():
        if not frame_blobs:
            continue
        total_frames += 1
        by_energy = sorted(frame_blobs, key=lambda b: b["rank_by_motion_energy"])[:top_k]
        by_importance = sorted(frame_blobs, key=lambda b: b["rank_by_importance"])[:top_k]

        if {b["blob_id_in_frame"] for b in by_energy} != {
            b["blob_id_in_frame"] for b in by_importance
        }:
            disagreement_frames += 1

        imp_small = {b["blob_id_in_frame"] for b in by_importance if b["small_blob_flag"]}
        eng_small = {b["blob_id_in_frame"] for b in by_energy if b["small_blob_flag"]}
        small_in_importance_not_energy += len(imp_small - eng_small)

    return {
        "frames_with_rank_disagreement": disagreement_frames,
        "total_frames_with_blobs": total_frames,
        "disagreement_ratio": round(disagreement_frames / max(total_frames, 1), 4),
        "small_blobs_in_importance_topk_not_energy_topk": small_in_importance_not_energy,
    }


def main() -> int:
    args = parse_args()
    result_root = Path(args.output_dir)
    result_root.mkdir(parents=True, exist_ok=True)

    extract_config = BlobExtractConfig(
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        cell_motion_threshold=args.cell_motion_threshold,
        use_flow_filter=not args.no_flow_filter,
        min_flow_magnitude=args.min_flow_magnitude,
        min_direction_coherence=args.min_direction_coherence,
        flow_scale=args.flow_scale if args.flow_scale is not None else 0.5,
    )
    feature_config = BlobFeatureConfig(
        small_blob_area_ratio_threshold=args.small_blob_area_ratio,
        use_texture_suppression=args.use_texture_suppression,
        texture_flicker_penalty=args.texture_flicker_penalty,
        global_motion_penalty=args.global_motion_penalty,
    )

    if args.video_name:
        video_dirs = [result_root / normalize_video_stem(args.video_name)]
        if not video_dirs[0].is_dir():
            logger.error("Video result dir not found: %s", video_dirs[0])
            return 1
    else:
        video_dirs = list_result_videos(result_root, args.max_videos, require_cache=True)

    logger.info("Found %d video(s) to process", len(video_dirs))

    all_summaries = []
    for i, video_dir in enumerate(video_dirs, 1):
        logger.info("[%d/%d] %s", i, len(video_dirs), video_dir.name)
        try:
            summary = process_single_video(
                video_dir,
                extract_config,
                feature_config,
                top_k=args.top_k,
                viz_max_blobs=args.viz_max_blobs,
                skip_overlay_video=args.skip_overlay_video,
                min_persistence_frames=args.min_persistence_frames,
                use_nearby_blob_merge=args.use_nearby_blob_merge,
                nearby_merge_distance_ratio=args.nearby_merge_distance_ratio,
                overlay_filename=args.overlay_filename,
                debug=args.debug,
            )
            all_summaries.append(summary)
        except Exception:
            logger.exception("Failed to process %s", video_dir.name)
            all_summaries.append({"video_name": video_dir.name, "status": "error"})

    ok_count = sum(1 for s in all_summaries if s.get("status") == "ok")
    logger.info("Finished: %d/%d videos processed successfully", ok_count, len(video_dirs))

    save_json(result_root / "batch_summary_box_v1.json", {"videos": all_summaries})
    return 0 if ok_count == len(video_dirs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

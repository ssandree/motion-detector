"""Motion Analyzer v2 pipeline — ByteTrack + motion-first ROI."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from motion_analyzer.v1.io import (
    data_dir,
    load_motion_blobs,
    load_motion_maps,
    normalize_video_stem,
    save_json,
    video_result_dir,
)
from motion_analyzer.v1.video_features import (
    classify_video_category,
    compute_video_features,
    legacy_category_for_input_plan,
)
from motion_analyzer.v2.blob_detector_context import (
    annotate_blobs_with_detector_context,
    merge_person_body_blobs,
)
from motion_analyzer.v2.detector_guided_tubes import (
    build_detector_guided_motion_tubes,
    tubes_context_summary,
)
from motion_analyzer.v2.gap_filling_tube_tracker import GapFillingTubeConfig
from motion_analyzer.v2.event_roi_grouping import (
    EventRoiBuildResult,
    build_event_roi_pipeline,
    load_auxiliary_track_obs,
    save_event_roi_candidates_csv,
)
from motion_analyzer.v2.flow_group_merge import build_per_frame_flow_groups
from motion_analyzer.v2.false_motion import apply_false_motion_suppression
from motion_analyzer.v2.input_plan import build_adaptive_input_plan_v2
from motion_analyzer.v2.io import (
    DEFAULT_RESULT_ROOT_V2,
    SUMMARY_BOX_V1,
    SUMMARY_PIXEL_V1,
    ensure_v1_artifacts,
    get_video_context,
    has_artifact,
    load_motion_tubes,
    save_dataframe,
)
from motion_analyzer.v2.object_detector import (
    DEFAULT_STATIC_SUPPRESSION_CLASSES,
    DEFAULT_TARGET_CLASSES,
    DetectorConfig,
    run_object_detector,
)
from motion_analyzer.v2.object_tracker import TrackerConfig, run_object_tracking
from motion_analyzer.v2.plan_finalize import finalize_adaptive_input_plan
from motion_analyzer.v2.track_input_plan import build_track_adaptive_input_plan
from motion_analyzer.v2.track_motion_features import (
    aggregate_track_features,
    enrich_tracks_with_motion,
    export_moving_object_tracks,
    export_stationary_object_tracks,
)
from motion_analyzer.v1.input_plan_visualize import (
    canonicalize_plan_roi_tracks,
    save_input_plan_overlay,
)
from motion_analyzer.v1.roi_debug import build_debug_roi_timeline, save_debug_roi_timeline
from motion_analyzer.v2.event_roi_visualize import (
    save_final_roi_tracks_overlay,
    save_grouped_blobs_overlay,
    save_raw_blobs_overlay,
)
from motion_analyzer.v2.track_visualize import save_tracker_motion_overlay
from motion_analyzer.v2.visualize import save_detector_guided_overlay

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = PROJECT_ROOT / "scripts"


def _debug_blob_map(video_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """Load per-frame raw blobs for debug overlay."""
    from collections import defaultdict

    ddir = data_dir(video_dir)
    raw_csv = ddir / "motion_blobs_raw.csv"
    blob_csv = raw_csv if raw_csv.is_file() else ddir / "motion_blobs.csv"
    if not blob_csv.is_file():
        return {}
    frame_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pd.read_csv(blob_csv).to_dict("records"):
        frame_map[int(row["frame_idx"])].append(row)
    return dict(frame_map)


def _finalize_plan_and_overlays(
    cfg: PipelineV2Config,
    *,
    plan: dict[str, Any],
    category_info: dict[str, Any],
    v2_dir: Path,
    ctx: dict[str, Any],
    blobs_df: pd.DataFrame,
    flow_group_map: dict[int, dict[str, Any]],
    sampled_indices: list[int],
    tracker_overlay_fn,
    event_roi_tracks: list[dict[str, Any]] | None = None,
    event_roi_candidates: list[dict[str, Any]] | None = None,
    event_roi_debug: dict[str, Any] | None = None,
    debug_rejected_high_rank: list[dict[str, Any]] | None = None,
    **tracker_overlay_kwargs,
) -> tuple[Path | None, Path | None, Path | None]:
    """
    Canonical ROI source + both overlays from the same adaptive_input_plan.json.

    Root cause fix: input_plan_overlay was rebuilt from v1 motion tubes while
    tracker_motion_overlay used v2 track plan frames — different ROI tracks.
    """
    ddir = data_dir(v2_dir)
    motion_type = category_info["motion_type"]
    tracks_for_finalize = event_roi_tracks if motion_type == "event_motion" else None
    plan_debug = finalize_adaptive_input_plan(
        plan,
        motion_type=motion_type,
        reason=str(category_info.get("primary_reason", "")),
        event_roi_tracks=tracks_for_finalize,
        event_roi_debug=event_roi_debug,
    )
    save_json(ddir / "adaptive_input_plan_debug.json", plan_debug)
    save_json(
        ddir / "video_category.json",
        {
            **category_info,
            "final_motion_type": plan_debug["final_motion_type"],
            "roi_source": plan_debug["roi_source"],
            "num_roi_inputs": plan_debug["num_roi_inputs"],
        },
    )
    canonicalize_plan_roi_tracks(plan)
    save_json(ddir / "adaptive_input_plan.json", plan)

    debug_path = save_debug_roi_timeline(
        ddir / "debug_roi_timeline.json",
        build_debug_roi_timeline(
            plan,
            ctx["frame_df"],
            blobs_df,
            flow_group_map=flow_group_map,
            event_roi_debug=event_roi_debug,
        ),
    )
    logger.info("debug_roi_timeline: %s (%d frames)", debug_path, len(sampled_indices))

    tracker_overlay_path = None
    input_overlay_path = None
    final_roi_overlay_path = None
    video_path = ctx.get("video_path")
    if not cfg.skip_overlay and video_path and Path(video_path).is_file():
        tracker_overlay_path = tracker_overlay_fn(
            video_path,
            tracker_overlay_kwargs.pop("moving_tracks"),
            tracker_overlay_kwargs.pop("stationary_tracks"),
            tracker_overlay_kwargs.pop("track_obs_map"),
            tracker_overlay_kwargs.pop("fallback_tubes"),
            tracker_overlay_kwargs.pop("fallback_obs_map"),
            tracker_overlay_kwargs.pop("enriched_tracks"),
            v2_dir,
            output_fps=ctx["sampled_fps"],
            plan=plan,
            motion_type=category_info["motion_type"],
            debug=cfg.overlay_debug,
            frame_blob_map=_debug_blob_map(v2_dir) if cfg.overlay_debug else None,
            flow_group_map=flow_group_map,
            sampled_frame_indices=sampled_indices,
            continuous=True,
            **tracker_overlay_kwargs,
        )
        input_overlay_path = save_input_plan_overlay(
            video_path,
            plan,
            v2_dir,
            motion_type=category_info["motion_type"],
            flow_group_map=flow_group_map,
            output_fps=ctx["sampled_fps"],
            sampled_frame_indices=sampled_indices,
            continuous=True,
            debug=cfg.overlay_debug,
            frame_blob_map=_debug_blob_map(v2_dir) if cfg.overlay_debug else None,
        )
        final_roi_overlay_path = save_final_roi_tracks_overlay(
            video_path,
            plan,
            sampled_indices,
            v2_dir,
            output_fps=ctx["sampled_fps"],
            debug_rejected_high_rank=debug_rejected_high_rank,
        )
        if event_roi_candidates is not None:
            blob_map = _debug_blob_map(v2_dir)
            if blob_map:
                save_raw_blobs_overlay(
                    video_path, blob_map, sampled_indices, v2_dir,
                    output_fps=ctx["sampled_fps"],
                )
            save_grouped_blobs_overlay(
                video_path, event_roi_candidates, sampled_indices, v2_dir,
                output_fps=ctx["sampled_fps"],
            )
        logger.info(
            "Overlays from shared roi_inputs (%d tracks): tracker=%s input_plan=%s final_roi=%s",
            len(plan.get("roi_inputs", [])),
            tracker_overlay_path,
            input_overlay_path,
            final_roi_overlay_path,
        )
    return tracker_overlay_path, input_overlay_path, final_roi_overlay_path


@dataclass
class PipelineV2Config:
    video_dir: str = r"C:/Datasets/VIRAT/video"
    output_dir: str = DEFAULT_RESULT_ROOT_V2
    v1_output_dir: str | None = "Result/motion_analyzer_v1"
    target_fps: float = 5.0
    use_object_detector: bool = True
    detector_model: str = "yolov8n.pt"
    detector_conf: float = 0.15
    tracker: str = "bytetrack"
    tracker_config: str = "bytetrack.yaml"
    target_object_classes: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_CLASSES))
    static_suppression_classes: list[str] = field(
        default_factory=lambda: list(DEFAULT_STATIC_SUPPRESSION_CLASSES)
    )
    skip_pixel: bool = False
    skip_box: bool = False
    skip_video: bool = False
    skip_overlay: bool = False
    overlay_debug: bool = False
    force_retrack: bool = False
    show_all_tracks: bool = False
    show_stationary_tracks: bool = False
    show_detector_only: bool = False
    use_global_compensation: bool = True
    use_texture_suppression: bool = True
    use_nearby_blob_merge: bool = True
    nearby_merge_distance_ratio: float = 0.06
    tube_max_gap: int = 6
    gap_match_decay: float = 0.85
    max_gap_seconds: float = 1.2


def _run_script(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def ensure_base_stages(cfg: PipelineV2Config, video_stem: str) -> Path:
    """Ensure v1 pixel/box/video stages exist under v2 output dir."""
    v2_dir = video_result_dir(cfg.output_dir, video_stem)
    v1_root = Path(cfg.v1_output_dir) if cfg.v1_output_dir else None

    required_pixel = [
        "_cache/motion_maps.npz",
        "_data/frame_motion_features.csv",
        SUMMARY_PIXEL_V1,
    ]
    required_box = ["_data/motion_blobs.csv", SUMMARY_BOX_V1]
    required_video = ["_data/motion_tubes.csv"]

    ensure_v1_artifacts(v2_dir, v1_root, required_pixel + required_box + required_video)
    common = ["--output_dir", cfg.output_dir, "--video_name", video_stem]

    if not cfg.skip_pixel and not all(has_artifact(v2_dir, r) for r in required_pixel):
        pixel_args = [
            "--video_dir", cfg.video_dir,
            "--target_fps", str(cfg.target_fps),
            *common,
        ]
        pixel_args.append(
            "--use_global_compensation" if cfg.use_global_compensation
            else "--no-use_global_compensation"
        )
        _run_script("run_motion_pixel_level_v1.py", pixel_args)

    if not cfg.skip_box and not all(has_artifact(v2_dir, r) for r in required_box):
        box_args = list(common)
        box_args.append(
            "--use_texture_suppression" if cfg.use_texture_suppression
            else "--no-use_texture_suppression"
        )
        box_args.append(
            "--use_nearby_blob_merge" if cfg.use_nearby_blob_merge
            else "--no-use_nearby_blob_merge"
        )
        box_args += ["--nearby_merge_distance_ratio", str(cfg.nearby_merge_distance_ratio)]
        _run_script("run_motion_box_level_v1.py", box_args)

    if not cfg.skip_video and not all(has_artifact(v2_dir, r) for r in required_video):
        _run_script("run_motion_video_level_v1.py", common)

    if not has_artifact(v2_dir, "motion_blobs_overlay.mp4") and v1_root:
        ensure_v1_artifacts(v2_dir, v1_root, ["motion_blobs_overlay.mp4"])

    return v2_dir


def _write_video_summary_bytetrack(
    path: Path,
    *,
    video_name: str,
    moving_tracks: list[dict],
    fallback_tubes: list[dict],
    plan: dict,
    num_tracks: int,
) -> None:
    lines = [
        f"# Video Summary (v2 ByteTrack + motion-first): {video_name}",
        "",
        "## Object Tracking",
        f"- Tracked detections: {num_tracks}",
        f"- Moving object tracks: {len(moving_tracks)}",
        f"- Motion fallback tubes: {len(fallback_tubes)}",
        "",
        "## Adaptive Input Plan",
        f"- Strategy: {', '.join(plan.get('strategy', []))}",
        f"- Entries by source: {plan.get('entries_by_source', {})}",
        f"- Total planned frames: {plan['estimated_total_frame_count']}",
        "",
        "## Top Moving Object Tracks",
    ]
    for t in plan.get("top_moving_tracks", []):
        lines.append(
            f"- Track {t['track_id']}: class={t.get('class_name','')}, "
            f"importance={t['track_motion_importance']:.3f}, "
            f"obs_ratio={t.get('observed_motion_ratio',0):.3f}, "
            f"disp={t.get('normalized_track_displacement',0):.3f}, "
            f"obs_frames={t.get('num_observed_motion_frames',0)}"
        )
    lines.append("")
    lines.append("## Motion Fallback Tubes")
    for t in plan.get("fallback_tubes", []):
        lines.append(f"- Tube {t['motion_tube_id']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_video_summary_legacy(
    path: Path,
    *,
    video_name: str,
    guided_tubes: list[dict],
    fallback_tubes: list[dict],
    plan: dict,
    detections_count: int,
) -> None:
    lines = [
        f"# Video Summary (v2 legacy detector-guided): {video_name}",
        "",
        "## Detector-Guided Motion",
        f"- Detections (auxiliary): {detections_count}",
        f"- Detector-guided motion tubes: {len(guided_tubes)}",
        f"- Fallback motion tubes: {len(fallback_tubes)}",
        "",
        "## Adaptive Input Plan",
        f"- Strategy: {', '.join(plan.get('strategy', []))}",
        f"- Entries by source: {plan.get('entries_by_source', {})}",
        f"- Total planned frames: {plan['estimated_total_frame_count']}",
        "",
        "## Top Detector-Guided Motion Tubes",
    ]
    for t in plan.get("top_detector_guided_tubes", []):
        lines.append(
            f"- Tube {t['motion_tube_id']}: classes={t.get('associated_classes','')}, "
            f"importance={t['detector_guided_motion_importance']:.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _detections_by_frame(detections_df: pd.DataFrame) -> dict[int, list[dict]]:
    if detections_df.empty:
        return {}
    out: dict[int, list[dict]] = {}
    for fi in detections_df["frame_idx"].unique():
        out[int(fi)] = detections_df[detections_df["frame_idx"] == fi].to_dict("records")
    return out


def _process_bytetrack(
    cfg: PipelineV2Config,
    video_stem: str,
    v2_dir: Path,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """ByteTrack + motion evidence pipeline."""
    ddir = data_dir(v2_dir)
    blobs_df = load_motion_blobs(v2_dir)
    frame_df = ctx["frame_df"]
    frame_w, frame_h = ctx["frame_w"], ctx["frame_h"]
    video_path = ctx["video_path"]

    if not video_path or not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found for tracker: {video_path}")

    tracks_csv = ddir / "object_tracks.csv"
    if cfg.force_retrack and tracks_csv.is_file():
        tracks_csv.unlink()
        logger.info("Removed cached tracks for re-tracking: %s", tracks_csv)
    if tracks_csv.is_file():
        tracks_df = pd.read_csv(tracks_csv)
        logger.info("Loading cached object tracks: %s", tracks_csv)
    else:
        tracker_cfg = TrackerConfig(
            model_name=cfg.detector_model,
            conf_threshold=cfg.detector_conf,
            tracker=cfg.tracker,
            tracker_config=cfg.tracker_config,
            target_classes=cfg.target_object_classes,
        )
        tracks_df = run_object_tracking(
            video_path, frame_df, config=tracker_cfg,
            video_name=video_stem, frame_w=frame_w, frame_h=frame_h,
        )
        save_dataframe(tracks_df, v2_dir, "object_tracks.csv")

    motion_maps = load_motion_maps(v2_dir)
    enriched_tracks, track_obs_map = enrich_tracks_with_motion(
        tracks_df, motion_maps, frame_df, blobs_df,
        frame_w=frame_w, frame_h=frame_h,
    )

    all_features, moving_tracks, stationary_tracks = aggregate_track_features(
        enriched_tracks, track_obs_map, frame_w=frame_w, frame_h=frame_h,
    )
    save_dataframe(
        pd.DataFrame(all_features) if all_features else pd.DataFrame(),
        v2_dir,
        "object_track_motion_features.csv",
    )

    moving_df = export_moving_object_tracks(enriched_tracks, moving_tracks)
    save_dataframe(moving_df, v2_dir, "moving_object_tracks.csv")

    stationary_df = export_stationary_object_tracks(all_features)
    save_dataframe(stationary_df, v2_dir, "stationary_object_tracks.csv")

    total_sampled = len(frame_df)
    fallback_tubes, fallback_obs_map = select_motion_fallback_tubes_from_tracks(
        blobs_df,
        moving_tracks,
        track_obs_map,
        frame_w=frame_w,
        frame_h=frame_h,
        total_sampled_frames=total_sampled,
        duration_sec=ctx["duration_sec"],
    )
    save_dataframe(
        pd.DataFrame(fallback_tubes) if fallback_tubes else pd.DataFrame(columns=[
            "motion_tube_id", "source", "fallback_reason", "tube_importance",
        ]),
        v2_dir,
        "motion_fallback_tubes.csv",
    )

    motion_tubes_df = load_motion_tubes(v2_dir)
    motion_tubes_list = motion_tubes_df.to_dict("records")
    video_features = compute_video_features(
        frame_df, motion_tubes_list, blobs_df, sampled_fps=ctx["sampled_fps"],
    )
    category_info = classify_video_category(video_features)
    flow_group_map = build_per_frame_flow_groups(
        blobs_df, frame_w=frame_w, frame_h=frame_h,
    )
    sampled_indices = [int(r["frame_idx"]) for _, r in frame_df.iterrows()]
    legacy_category = legacy_category_for_input_plan(
        category_info["motion_type"], video_features
    )

    aux_track_obs = load_auxiliary_track_obs(v2_dir)
    roi_result: EventRoiBuildResult = build_event_roi_pipeline(
        blobs_df,
        frame_df,
        aux_track_obs,
        frame_w=frame_w,
        frame_h=frame_h,
        sampled_fps=ctx["sampled_fps"],
        motion_type=category_info["motion_type"],
    )
    event_candidates = roi_result.candidates
    event_tracks = roi_result.selected_tracks
    save_json(ddir / "event_roi_candidates.json", {"candidates": event_candidates})
    save_event_roi_candidates_csv(event_candidates, ddir / "event_roi_candidates.csv")
    save_json(ddir / "event_roi_tracks.json", {"roi_tracks": event_tracks})
    save_json(ddir / "event_roi_debug.json", roi_result.debug_report)
    save_json(
        ddir / "background_flow_regions.json",
        {"regions": roi_result.background_flow_regions},
    )

    plan = build_track_adaptive_input_plan(
        legacy_category,
        frame_df,
        moving_tracks,
        track_obs_map,
        fallback_tubes,
        fallback_obs_map,
        sampled_fps=ctx["sampled_fps"],
        frame_w=frame_w,
        frame_h=frame_h,
    )

    overlay_path, input_overlay_path, final_roi_overlay_path = _finalize_plan_and_overlays(
        cfg,
        plan=plan,
        category_info=category_info,
        v2_dir=v2_dir,
        ctx=ctx,
        blobs_df=blobs_df,
        flow_group_map=flow_group_map,
        sampled_indices=sampled_indices,
        event_roi_tracks=event_tracks,
        event_roi_candidates=event_candidates,
        event_roi_debug=roi_result.debug_report,
        debug_rejected_high_rank=roi_result.debug_rejected_high_rank,
        tracker_overlay_fn=save_tracker_motion_overlay,
        moving_tracks=moving_tracks,
        stationary_tracks=stationary_tracks,
        track_obs_map=track_obs_map,
        fallback_tubes=fallback_tubes,
        fallback_obs_map=fallback_obs_map,
        enriched_tracks=enriched_tracks,
        show_all_tracks=cfg.show_all_tracks,
        show_stationary_tracks=cfg.show_stationary_tracks,
        show_detector_only=cfg.show_detector_only,
    )
    logger.info(
        "Event-group ROI: %d candidates, %d selected → plan roi_inputs=%d (%s)",
        len(event_candidates),
        len(event_tracks),
        len(plan.get("roi_inputs", [])),
        plan.get("roi_source"),
    )

    summary_md = ddir / "video_summary.md"
    _write_video_summary_bytetrack(
        summary_md,
        video_name=video_stem,
        moving_tracks=moving_tracks,
        fallback_tubes=fallback_tubes,
        plan=plan,
        num_tracks=len(tracks_df),
    )

    return {
        "num_tracks": len(tracks_df),
        "num_moving_tracks": len(moving_tracks),
        "num_stationary_tracks": len(stationary_tracks),
        "num_fallback_tubes": len(fallback_tubes),
        "motion_type": category_info["motion_type"],
        "plan": plan,
        "overlay_path": overlay_path,
        "input_overlay_path": input_overlay_path,
        "final_roi_overlay_path": final_roi_overlay_path,
        "event_roi_candidates": len(event_candidates),
        "event_roi_tracks": len(event_tracks),
        "moving_tracks": moving_tracks,
    }


def _process_legacy(
    cfg: PipelineV2Config,
    video_stem: str,
    v2_dir: Path,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Legacy detector-guided tube pipeline."""
    ddir = data_dir(v2_dir)
    blobs_df = load_motion_blobs(v2_dir)
    frame_df = ctx["frame_df"]
    frame_w, frame_h = ctx["frame_w"], ctx["frame_h"]
    video_path = ctx["video_path"]

    detections_df = pd.DataFrame()
    if cfg.use_object_detector:
        det_csv = ddir / "object_detections.csv"
        if det_csv.is_file():
            detections_df = pd.read_csv(det_csv)
        else:
            if not video_path or not Path(video_path).is_file():
                raise FileNotFoundError(f"Video not found for detector: {video_path}")
            det_config = DetectorConfig(
                model_name=cfg.detector_model,
                conf_threshold=cfg.detector_conf,
                target_classes=cfg.target_object_classes,
                static_suppression_classes=cfg.static_suppression_classes,
            )
            detections_df = run_object_detector(
                video_path, frame_df, config=det_config,
                video_name=video_stem, frame_w=frame_w, frame_h=frame_h,
            )
            save_dataframe(detections_df, v2_dir, "object_detections.csv")

    blobs_ctx = annotate_blobs_with_detector_context(
        blobs_df, detections_df, frame_w=frame_w, frame_h=frame_h,
    )
    blobs_ctx["motion_only_candidate"] = blobs_ctx["associated_det_ids"].apply(
        lambda s: not str(s).strip()
    )
    blobs_ctx = apply_false_motion_suppression(blobs_ctx, detections_df)
    save_dataframe(blobs_ctx, v2_dir, "motion_blobs_detector_context.csv")

    merged_person = merge_person_body_blobs(blobs_ctx, detections_df)
    if not merged_person.empty:
        save_dataframe(merged_person, v2_dir, "merged_person_motion_regions.csv")

    guided_tubes, guided_segments, guided_obs_map = build_detector_guided_motion_tubes(
        blobs_ctx,
        frame_df,
        detections_df,
        frame_w=frame_w,
        frame_h=frame_h,
        duration_sec=ctx["duration_sec"],
        gap_config=GapFillingTubeConfig(
            tube_max_gap=cfg.tube_max_gap,
            gap_match_decay=cfg.gap_match_decay,
            max_gap_seconds=cfg.max_gap_seconds,
            sampled_fps=ctx["sampled_fps"],
        ),
    )
    save_dataframe(guided_tubes, v2_dir, "detector_guided_motion_tubes.csv")

    motion_tubes_df = load_motion_tubes(v2_dir)
    tubes_ctx = tubes_context_summary(guided_tubes, motion_tubes_df)
    save_dataframe(tubes_ctx, v2_dir, "motion_tubes_detector_context.csv")

    fallback_tubes = select_motion_fallback_tubes(guided_tubes, guided_obs_map)
    fallback_obs_map = {
        t["motion_tube_id"]: guided_obs_map[t["motion_tube_id"]]
        for t in fallback_tubes
        if t["motion_tube_id"] in guided_obs_map
    }
    save_dataframe(
        fallback_tubes if fallback_tubes else pd.DataFrame(),
        v2_dir,
        "motion_only_fallback_tubes.csv",
    )

    motion_tubes_list = motion_tubes_df.to_dict("records")
    video_features = compute_video_features(
        frame_df, motion_tubes_list, blobs_df, sampled_fps=ctx["sampled_fps"],
    )
    category_info = classify_video_category(video_features)
    legacy_category = legacy_category_for_input_plan(
        category_info["motion_type"], video_features
    )

    dets_by_frame = _detections_by_frame(detections_df)
    plan = build_adaptive_input_plan_v2(
        legacy_category,
        frame_df,
        guided_tubes,
        guided_obs_map,
        fallback_tubes,
        fallback_obs_map,
        dets_by_frame,
        sampled_fps=ctx["sampled_fps"],
        frame_w=frame_w,
        frame_h=frame_h,
    )

    flow_group_map = build_per_frame_flow_groups(
        blobs_df, frame_w=frame_w, frame_h=frame_h,
    )
    sampled_indices = [int(r["frame_idx"]) for _, r in frame_df.iterrows()]

    plan_debug = finalize_adaptive_input_plan(
        plan,
        motion_type=category_info["motion_type"],
        reason=str(category_info.get("primary_reason", "")),
        event_roi_tracks=None,
    )
    save_json(ddir / "adaptive_input_plan_debug.json", plan_debug)
    save_json(
        ddir / "video_category.json",
        {
            **category_info,
            "final_motion_type": plan_debug["final_motion_type"],
            "roi_source": plan_debug["roi_source"],
            "num_roi_inputs": plan_debug["num_roi_inputs"],
        },
    )
    canonicalize_plan_roi_tracks(plan)
    save_json(ddir / "adaptive_input_plan.json", plan)

    save_debug_roi_timeline(
        ddir / "debug_roi_timeline.json",
        build_debug_roi_timeline(
            plan, frame_df, blobs_df, flow_group_map=flow_group_map,
        ),
    )

    overlay_path = None
    input_overlay_path = None
    if not cfg.skip_overlay and video_path and Path(video_path).is_file():
        overlay_path = save_detector_guided_overlay(
            video_path,
            guided_tubes,
            guided_obs_map,
            detections_df,
            blobs_ctx,
            v2_dir,
            output_fps=ctx["sampled_fps"],
            show_detector_boxes=cfg.show_detector_only,
            plan=plan,
            motion_type=category_info["motion_type"],
            debug=cfg.overlay_debug,
            frame_blob_map=_debug_blob_map(v2_dir) if cfg.overlay_debug else None,
        )
        input_overlay_path = save_input_plan_overlay(
            video_path,
            plan,
            v2_dir,
            motion_type=category_info["motion_type"],
            flow_group_map=flow_group_map,
            output_fps=ctx["sampled_fps"],
            sampled_frame_indices=sampled_indices,
            continuous=True,
            debug=cfg.overlay_debug,
            frame_blob_map=_debug_blob_map(v2_dir) if cfg.overlay_debug else None,
        )

    summary_md = ddir / "video_summary.md"
    _write_video_summary_legacy(
        summary_md,
        video_name=video_stem,
        guided_tubes=guided_tubes,
        fallback_tubes=fallback_tubes,
        plan=plan,
        detections_count=len(detections_df),
    )

    return {
        "num_detections": len(detections_df),
        "num_guided_tubes": len(guided_tubes),
        "num_fallback_tubes": len(fallback_tubes),
        "motion_type": category_info["motion_type"],
        "plan": plan,
        "overlay_path": overlay_path,
        "input_overlay_path": input_overlay_path,
    }


def process_video_v2(cfg: PipelineV2Config, video_stem: str) -> dict[str, Any]:
    """Run motion-first v2 pipeline for one video."""
    t0 = time.perf_counter()
    video_stem = normalize_video_stem(video_stem)
    v2_dir = ensure_base_stages(cfg, video_stem)
    ddir = data_dir(v2_dir)
    ctx = get_video_context(v2_dir)

    use_bytetrack = cfg.tracker.lower() in ("bytetrack", "botsort", "bot-sort")

    if use_bytetrack:
        result = _process_bytetrack(cfg, video_stem, v2_dir, ctx)
        elapsed = time.perf_counter() - t0
        summary = {
            "stage": "v2-bytetrack-motion-first",
            "video_name": video_stem,
            "status": "ok",
            "tracker": cfg.tracker,
            "num_tracks": result["num_tracks"],
            "num_moving_tracks": result["num_moving_tracks"],
            "num_stationary_tracks": result["num_stationary_tracks"],
            "num_fallback_tubes": result["num_fallback_tubes"],
            "motion_type": result["motion_type"],
            "elapsed_sec": round(elapsed, 2),
            "outputs": {
                "object_tracks": str(ddir / "object_tracks.csv"),
                "object_track_motion_features": str(ddir / "object_track_motion_features.csv"),
                "moving_object_tracks": str(ddir / "moving_object_tracks.csv"),
                "stationary_object_tracks": str(ddir / "stationary_object_tracks.csv"),
                "motion_fallback_tubes": str(ddir / "motion_fallback_tubes.csv"),
                "adaptive_input_plan": str(ddir / "adaptive_input_plan.json"),
                "adaptive_input_plan_debug": str(ddir / "adaptive_input_plan_debug.json"),
                "debug_roi_timeline": str(ddir / "debug_roi_timeline.json"),
                "tracker_motion_overlay": str(result["overlay_path"]) if result["overlay_path"] else None,
                "input_plan_overlay": str(result.get("input_overlay_path")) if result.get("input_overlay_path") else None,
                "final_roi_tracks_overlay": str(result.get("final_roi_overlay_path")) if result.get("final_roi_overlay_path") else None,
                "event_roi_candidates": str(ddir / "event_roi_candidates.json"),
                "event_roi_tracks": str(ddir / "event_roi_tracks.json"),
                "motion_blobs_overlay": str(v2_dir / "motion_blobs_overlay.mp4"),
                "video_summary": str(ddir / "video_summary.md"),
            },
        }
        logger.info(
            "v2 ByteTrack done %s: %d moving, %d stationary, %d fallback in %.1fs",
            video_stem, result["num_moving_tracks"], result["num_stationary_tracks"],
            result["num_fallback_tubes"], elapsed,
        )
    else:
        result = _process_legacy(cfg, video_stem, v2_dir, ctx)
        elapsed = time.perf_counter() - t0
        summary = {
            "stage": "v2-legacy-detector-guided",
            "video_name": video_stem,
            "status": "ok",
            "num_detections": result.get("num_detections", 0),
            "num_guided_tubes": result.get("num_guided_tubes", 0),
            "num_fallback_tubes": result.get("num_fallback_tubes", 0),
            "motion_type": result["motion_type"],
            "elapsed_sec": round(elapsed, 2),
            "outputs": {
                "detector_guided_motion_tubes": str(ddir / "detector_guided_motion_tubes.csv"),
                "motion_only_fallback_tubes": str(ddir / "motion_only_fallback_tubes.csv"),
                "adaptive_input_plan": str(ddir / "adaptive_input_plan.json"),
                "detector_guided_motion_overlay": str(result["overlay_path"]) if result["overlay_path"] else None,
                "video_summary": str(ddir / "video_summary.md"),
            },
        }

    save_json(ddir / "summary_v2.json", summary)
    return summary

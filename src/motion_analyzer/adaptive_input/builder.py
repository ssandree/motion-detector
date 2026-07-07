"""Orchestrate adaptive input building from adaptive_input_plan.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from motion_analyzer.adaptive_input.frame_extractor import (
    extract_vlm_frames,
    saved_inputs_to_report_entries,
)
from motion_analyzer.adaptive_input.plan_resolver import resolve_input_plan
from motion_analyzer.adaptive_input.visualize import (
    build_report,
    save_input_builder_visualization_mp4,
    save_input_summary_png,
)
from motion_analyzer.v1.io import data_dir, save_json
from motion_analyzer.v2.io import get_video_context

logger = logging.getLogger(__name__)


def load_adaptive_input_plan(plan_path: Path) -> dict[str, Any]:
    with plan_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_adaptive_input(
    video_dir: Path,
    *,
    plan_path: Path | None = None,
    video_path: str | None = None,
    duration_sec: float | None = None,
) -> dict[str, Any]:
    """
    Build VLM input frames and timeline artifacts under ``{video_dir}/adaptive_input/``.

    Consumes only ``adaptive_input_plan.json`` (no motion recompute, no tracking rerun).
    """
    video_dir = Path(video_dir)
    plan_path = plan_path or (data_dir(video_dir) / "adaptive_input_plan.json")
    if not plan_path.is_file():
        raise FileNotFoundError(f"Adaptive input plan not found: {plan_path}")

    plan = load_adaptive_input_plan(plan_path)
    resolved = resolve_input_plan(plan)

    ctx = get_video_context(video_dir)
    resolved_video_path = video_path or str(ctx["video_path"])
    if not resolved_video_path or not Path(resolved_video_path).is_file():
        raise FileNotFoundError(f"Source video not found: {resolved_video_path}")

    clip_duration = duration_sec
    if clip_duration is None:
        clip_duration = float(ctx["duration_sec"])
    if clip_duration <= 0 and resolved.sparse_global:
        clip_duration = resolved.sparse_global[-1].timestamp_sec + 1.0

    out_dir = video_dir / "adaptive_input"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building adaptive input for %s: motion_type=%s, sg=%d, roi_tracks=%d, roi_frames=%d, anchors=%d",
        video_dir.name,
        resolved.motion_type,
        resolved.sparse_global_frame_count,
        resolved.roi_track_count,
        resolved.roi_frame_count,
        resolved.event_anchor_count,
    )

    saved = extract_vlm_frames(resolved_video_path, resolved, out_dir)
    saved_entries = saved_inputs_to_report_entries(saved)

    summary_png = save_input_summary_png(
        resolved_video_path,
        resolved,
        out_dir / "input_summary.png",
        duration_sec=clip_duration,
    )
    viz_mp4 = save_input_builder_visualization_mp4(
        resolved_video_path,
        resolved,
        out_dir / "input_builder_visualization.mp4",
        duration_sec=clip_duration,
    )

    report = build_report(resolved, saved_entries)
    if plan.get("finalization"):
        report["finalization"] = plan["finalization"]
    report_path = save_json(out_dir / "input_builder_report.json", report)

    finalization = plan.get("finalization") or {}
    result = {
        "video_name": video_dir.name,
        "motion_type": resolved.motion_type,
        "final_motion_type": finalization.get("final_motion_type", resolved.motion_type),
        "reason": finalization.get("reason"),
        "roi_source": finalization.get("roi_source"),
        "num_roi_inputs": finalization.get("num_roi_inputs", resolved.roi_track_count),
        "sparse_global_count": finalization.get(
            "sparse_global_count", resolved.sparse_global_frame_count
        ),
        "output_dir": str(out_dir),
        "sparse_global_frame_count": resolved.sparse_global_frame_count,
        "roi_track_count": resolved.roi_track_count,
        "roi_frame_count": resolved.roi_frame_count,
        "event_anchor_count": resolved.event_anchor_count,
        "total_vlm_input_count": resolved.total_vlm_input_count,
        "outputs": {
            "sparse_global_dir": str(out_dir / "sparse_global"),
            "roi1_dir": str(out_dir / "roi1") if resolved.roi_track_count >= 1 else None,
            "roi2_dir": str(out_dir / "roi2") if resolved.roi_track_count >= 2 else None,
            "event_anchor_dir": str(out_dir / "event_anchor")
            if resolved.event_anchor_count
            else None,
            "input_summary_png": str(summary_png),
            "input_builder_visualization_mp4": str(viz_mp4),
            "input_builder_report_json": str(report_path),
        },
    }
    logger.info(
        "Adaptive input complete: %d total VLM frames → %s",
        resolved.total_vlm_input_count,
        out_dir,
    )
    return result

"""Finalize adaptive_input_plan.json — motion_type, roi_inputs, debug summary."""

from __future__ import annotations

from typing import Any

from motion_analyzer.adaptive_input.plan_resolver import _collect_sparse_global
from motion_analyzer.v1.input_plan_builder import MAX_ROI_TRACKS


def finalize_adaptive_input_plan(
    plan: dict[str, Any],
    *,
    motion_type: str,
    reason: str,
    event_roi_tracks: list[dict[str, Any]] | None = None,
    gate_summary: dict[str, Any] | None = None,
    event_roi_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Apply final motion_type and ROI policy to adaptive_input_plan.json.

    background_motion → roi_inputs = [] (sparse global only)
    event_motion      → roi_inputs from event_roi_tracks when provided (blob_group);
                        otherwise keep existing plan roi_inputs (legacy tube plans).
    Never rebuild ROI from plan["frames"] track/fallback crop entries here.
    """
    sparse_global_count = len(_collect_sparse_global(plan))

    if motion_type == "background_motion":
        plan["roi_inputs"] = []
        roi_source = "none"
        num_roi_inputs = 0
    elif motion_type == "event_motion":
        if event_roi_tracks is not None:
            plan["roi_inputs"] = list(event_roi_tracks)[:MAX_ROI_TRACKS]
            roi_source = "blob_group" if plan["roi_inputs"] else "none"
        else:
            plan["roi_inputs"] = list(plan.get("roi_inputs") or [])
            roi_source = str(plan.get("roi_source") or ("legacy_tube" if plan["roi_inputs"] else "none"))
        num_roi_inputs = len(plan["roi_inputs"])
    else:
        raise ValueError(f"Invalid motion_type: {motion_type!r} (expected background_motion or event_motion)")

    plan["motion_type"] = motion_type
    plan["roi_source"] = roi_source

    debug = {
        "final_motion_type": motion_type,
        "reason": reason,
        "num_roi_inputs": num_roi_inputs,
        "roi_source": roi_source,
        "sparse_global_count": sparse_global_count,
    }

    if gate_summary is None and event_roi_debug:
        gate_summary = event_roi_debug.get("gate_summary") or event_roi_debug.get("summary")

    if gate_summary:
        for key in (
            "num_candidates_by_pattern",
            "selected_roi_count",
            "rejected_transit_count",
            "rejected_background_count",
        ):
            if key in gate_summary:
                debug[key] = gate_summary[key]
        if num_roi_inputs == 0 and motion_type == "event_motion":
            debug["no_roi_reason"] = gate_summary.get("no_roi_reason") or "no_event_motion_candidates"
    elif num_roi_inputs == 0 and motion_type == "event_motion":
        debug["no_roi_reason"] = "no_event_motion_candidates"

    plan["finalization"] = debug
    return debug

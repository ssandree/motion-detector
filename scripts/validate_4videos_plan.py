#!/usr/bin/env python3
"""Validate finalized adaptive input plan for representative videos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2

from motion_analyzer.v1.input_plan_visualize import plan_motion_type, roi_inputs_from_plan
from motion_analyzer.v1.io import data_dir, load_frame_features, load_pixel_summary, video_result_dir

VIDEOS = [
    "VIRAT_S_050201_06_001168_001240",
    "VIRAT_S_010207_08_001308_001332",
    "VIRAT_S_050300_07_001623_001690",
    "VIRAT_S_000201_06_001354_001397",
]
OUTPUT_ROOT = PROJECT_ROOT / "Result" / "motion_analyzer_v2"


def _video_duration_sec(video_dir: Path) -> float:
    pixel = load_pixel_summary(video_dir)
    fps = float(pixel.get("target_fps", 5.0))
    frame_df = load_frame_features(video_dir)
    if frame_df.empty:
        return float(pixel.get("duration_sec", 0.0))
    return len(frame_df) / fps


def _mp4_duration_sec(path: Path) -> float | None:
    if not path.is_file():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return None
    return float(frames) / float(fps)


def validate_video(stem: str) -> dict:
    video_dir = video_result_dir(str(OUTPUT_ROOT), stem)
    ddir = data_dir(video_dir)
    failures: list[str] = []

    debug_path = ddir / "adaptive_input_plan_debug.json"
    plan_path = ddir / "adaptive_input_plan.json"
    overlay_path = video_dir / "input_plan_overlay.mp4"
    builder_viz = video_dir / "adaptive_input" / "input_builder_visualization.mp4"

    debug = json.loads(debug_path.read_text(encoding="utf-8")) if debug_path.is_file() else {}
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else {}

    expected_duration = _video_duration_sec(video_dir)
    overlay_duration = _mp4_duration_sec(overlay_path)
    builder_duration = _mp4_duration_sec(builder_viz)

    mt = debug.get("final_motion_type", plan_motion_type(plan))
    num_roi = int(debug.get("num_roi_inputs", len(roi_inputs_from_plan(plan))))
    roi_source = str(debug.get("roi_source", plan.get("roi_source", "")))
    sg_count = int(debug.get("sparse_global_count", 0))
    reason = str(debug.get("reason", ""))

    # Policy checks
    if mt == "background_motion":
        if num_roi != 0:
            failures.append(f"background_motion but num_roi_inputs={num_roi}")
        if roi_inputs_from_plan(plan):
            failures.append("background_motion but plan roi_inputs non-empty")
        if roi_source != "none":
            failures.append(f"background_motion but roi_source={roi_source}")
        roi1_dir = video_dir / "adaptive_input" / "roi1"
        if roi1_dir.is_dir() and any(roi1_dir.iterdir()):
            failures.append("background_motion but adaptive_input/roi1 has files")
    elif mt == "event_motion":
        if num_roi > 0 and roi_source != "blob_group":
            failures.append(f"event_motion with ROI but roi_source={roi_source}")
        if num_roi != len(roi_inputs_from_plan(plan)):
            failures.append(
                f"num_roi_inputs mismatch: debug={num_roi} plan={len(roi_inputs_from_plan(plan))}"
            )
    else:
        failures.append(f"invalid final_motion_type: {mt}")

    overlay_exists = overlay_path.is_file()
    builder_exists = builder_viz.is_file()

    duration_tol = 1.5  # seconds
    overlay_full = (
        overlay_exists
        and overlay_duration is not None
        and abs(overlay_duration - expected_duration) <= duration_tol
    )
    if overlay_exists and not overlay_full:
        failures.append(
            f"input_plan_overlay duration {overlay_duration:.1f}s != expected {expected_duration:.1f}s"
        )
    if not overlay_exists:
        failures.append("input_plan_overlay.mp4 missing")
    if not builder_exists:
        failures.append("input_builder_visualization.mp4 missing")

    required_debug = {
        "final_motion_type",
        "reason",
        "num_roi_inputs",
        "roi_source",
        "sparse_global_count",
    }
    if not required_debug.issubset(debug.keys()):
        failures.append(f"adaptive_input_plan_debug missing: {required_debug - debug.keys()}")

    return {
        "video": stem,
        "final_motion_type": mt,
        "reason": reason[:80] + ("…" if len(reason) > 80 else ""),
        "num_roi_inputs": num_roi,
        "roi_source": roi_source,
        "sparse_global_count": sg_count,
        "expected_duration_sec": round(expected_duration, 1),
        "overlay_duration_sec": round(overlay_duration, 1) if overlay_duration else None,
        "overlay_exists": overlay_exists,
        "overlay_full_duration": overlay_full,
        "builder_viz_exists": builder_exists,
        "builder_duration_sec": round(builder_duration, 1) if builder_duration else None,
        "roi_policy_ok": len([f for f in failures if "roi" in f.lower() or "background" in f.lower()]) == 0,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def main() -> int:
    rows = [validate_video(v) for v in VIDEOS]
    out_md = OUTPUT_ROOT / "validation_summary_4videos.md"

    lines = [
        "# Adaptive Input Plan Validation — 4 Representative Videos",
        "",
        f"Generated after `run_event_roi_refresh.py --write_overlays` + `run_adaptive_input_builder.py`.",
        "",
        "## Summary Table",
        "",
        "| Video | final_motion_type | num_roi_inputs | roi_source | sparse_global | overlay full | builder viz | Status |",
        "|-------|-------------------|----------------|------------|---------------|--------------|-------------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['video']} | {r['final_motion_type']} | {r['num_roi_inputs']} | {r['roi_source']} | "
            f"{r['sparse_global_count']} | "
            f"{'yes' if r['overlay_full_duration'] else 'no'} "
            f"({r['overlay_duration_sec']}s/{r['expected_duration_sec']}s) | "
            f"{'yes' if r['builder_viz_exists'] else 'no'} | **{r['status']}** |"
        )

    lines.extend(["", "## Per-Video Details", ""])
    for r in rows:
        lines.append(f"### {r['video']}")
        lines.append("")
        lines.append(f"- **final_motion_type**: `{r['final_motion_type']}`")
        lines.append(f"- **reason**: {r['reason']}")
        lines.append(f"- **num_roi_inputs**: {r['num_roi_inputs']}")
        lines.append(f"- **roi_source**: `{r['roi_source']}`")
        lines.append(f"- **sparse_global_count**: {r['sparse_global_count']}")
        lines.append(
            f"- **input_plan_overlay.mp4**: "
            f"{'exists' if r['overlay_exists'] else 'MISSING'}, "
            f"duration {r['overlay_duration_sec']}s (expected {r['expected_duration_sec']}s), "
            f"full duration: **{'OK' if r['overlay_full_duration'] else 'FAIL'}**"
        )
        lines.append(
            f"- **input_builder_visualization.mp4**: "
            f"{'exists' if r['builder_viz_exists'] else 'MISSING'}"
            + (f", duration {r['builder_duration_sec']}s" if r['builder_duration_sec'] else "")
        )
        roi_note = (
            "No ROI (correct for background_motion)"
            if r["final_motion_type"] == "background_motion"
            else (
                f"ROI from roi_inputs only ({r['num_roi_inputs']} tracks, source={r['roi_source']})"
                if r["num_roi_inputs"] > 0
                else "event_motion with 0 ROI tracks (sparse global only)"
            )
        )
        lines.append(f"- **ROI policy**: {roi_note}")
        if r["failures"]:
            lines.append("- **Failures**:")
            for f in r["failures"]:
                lines.append(f"  - {f}")
        else:
            lines.append("- **Failures**: none")
        lines.append("")

    all_pass = all(r["status"] == "PASS" for r in rows)
    lines.extend([
        "## Overall",
        "",
        f"**{'ALL PASS' if all_pass else 'SOME FAILURES'}** — "
        f"{sum(1 for r in rows if r['status'] == 'PASS')}/{len(rows)} videos passed validation.",
        "",
    ])

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")
    for r in rows:
        print(f"{r['status']} {r['video']}: {r['final_motion_type']} roi={r['num_roi_inputs']}")
        for f in r["failures"]:
            print(f"  FAIL: {f}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

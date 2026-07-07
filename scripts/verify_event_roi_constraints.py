#!/usr/bin/env python3
"""Quick validation for event ROI rebuild output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motion_analyzer.v1.io import data_dir, normalize_video_stem, video_result_dir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--video_name", required=True)
    args = p.parse_args()

    ddir = data_dir(video_result_dir(args.output_dir, normalize_video_stem(args.video_name)))
    tracks = json.loads((ddir / "event_roi_tracks.json").read_text(encoding="utf-8"))["roi_tracks"]
    dbg = json.loads((ddir / "event_roi_debug.json").read_text(encoding="utf-8"))
    cands = json.loads((ddir / "event_roi_candidates.json").read_text(encoding="utf-8"))["candidates"]

    errors: list[str] = []
    for t in tracks:
        if t.get("primary_source") != "blob_group":
            errors.append(f"ROI{t['roi_id']}: primary_source not blob_group")
        if t.get("bbox_sequence_len", 0) < 4:
            errors.append(f"ROI{t['roi_id']}: bbox_sequence_len < 4")
        if t.get("duration_sec", 0) < 2.0:
            errors.append(f"ROI{t['roi_id']}: duration_sec < 2.0")
        seq = t.get("bbox_sequence", [])
        if len(seq) < 4:
            errors.append(f"ROI{t['roi_id']}: sequence too short")
        sf, ef = seq[0]["frame_idx"], seq[-1]["frame_idx"]
        if any(not (sf <= e["frame_idx"] <= ef) for e in seq):
            errors.append(f"ROI{t['roi_id']}: frame outside start/end")

    required_dbg = {"selected_rois", "build_rejected", "selection_rejected", "rules"}
    if not required_dbg.issubset(dbg.keys()):
        errors.append(f"debug missing keys: {required_dbg - dbg.keys()}")

    moto = [c for c in cands if c["start_sec"] <= 34 and c["end_sec"] >= 32]
    print(f"selected={len(tracks)} candidates={len(cands)} moto_window={len(moto)}")
    for t in tracks:
        print(
            f"  ROI{t['roi_id']} group={t['group_id']} "
            f"{t['start_sec']:.1f}-{t['end_sec']:.1f}s len={t['bbox_sequence_len']}"
        )
    for c in sorted(moto, key=lambda x: -x["event_score"])[:3]:
        print(
            f"  moto cand g{c['group_id']} {c['start_sec']:.1f}-{c['end_sec']:.1f}s "
            f"active={c['num_blob_active_frames']} score={c['event_score']}"
        )

    if errors:
        print("FAIL:", *errors, sep="\n  ")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

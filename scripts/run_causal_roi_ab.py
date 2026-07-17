#!/usr/bin/env python3
"""Run Approach A (temporal groups) and/or B (block-peak 192x192) ROIs.

B is the simplified post-block_score path (recommended).
A keeps the previous temporal-grouping pipeline.

Outputs per video/method subdirectory:
  method_A/ or method_B/
    temporal_groups.json (or block_peak_events.json)
    frame_fused_roi.json
    sliding_window_overlay.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from motion_analyzer.temporal.frame_window_fusion import (  # noqa: E402
    save_fused_frame_roi_json,
    write_full_video_fused_overlay,
)
from motion_analyzer.temporal.temporal_group_roi import (  # noqa: E402
    build_temporal_group_rois,
    save_temporal_groups_json,
)
from motion_analyzer.temporal.block_peak_roi import (  # noqa: E402
    build_block_peak_rois,
    save_block_peak_events_json,
    write_context_overlay_video,
)
from motion_analyzer.temporal.sliding_window_regions import (  # noqa: E402
    SlidingWindowConfig,
    build_motion_regions,
    load_active_mask_stack,
    load_pooled_rms_stack,
)
from run_sliding_window_regions import (  # noqa: E402
    DEFAULT_COMPONENTS_SUBDIR,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    discover_videos,
    load_hints,
    resolve_video_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_causal_roi_ab")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--components_subdir", type=str, default=DEFAULT_COMPONENTS_SUBDIR)
    p.add_argument("--video_names", nargs="+", default=None)
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["A", "B", "a", "b"],
        default=["A", "B"],
        help="Which approach(es) to run",
    )
    p.add_argument("--window_size", type=int, default=10)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max_window_candidates", type=int, default=12)
    p.add_argument("--max_rois", type=int, default=2)
    p.add_argument("--skip_overlay", action="store_true")
    return p.parse_args()


def build_windows(video_name: str, input_root: Path, args: argparse.Namespace):
    video_dir = input_root / video_name
    agg_dir = video_dir / args.components_subdir
    hints = load_hints(video_dir, agg_dir)
    video_path = resolve_video_path(
        video_name,
        [Path(hints["source_video_path"]) if hints.get("source_video_path") else None],
    )
    if video_path is None:
        raise FileNotFoundError(f"Could not resolve source video for {video_name}")

    active_stack = load_active_mask_stack(agg_dir)
    pooled_rms = load_pooled_rms_stack(agg_dir)
    n_frames = int(hints.get("num_frames") or active_stack.shape[0])
    frames_meta = hints.get("frames") or []
    if frames_meta:
        frame_w = int(hints.get("frame_width") or frames_meta[0].get("frame_width", 1280))
        frame_h = int(hints.get("frame_height") or frames_meta[0].get("frame_height", 720))
        frame_pairs = [
            (int(fr.get("frame_index", i)), int(fr["frame_idx"]))
            for i, fr in enumerate(frames_meta)
        ]
        base_rows = int(frames_meta[0].get("base_grid_rows", active_stack.shape[1] * 12))
        base_cols = int(frames_meta[0].get("base_grid_cols", active_stack.shape[2] * 12))
    else:
        frame_w = int(hints.get("frame_width", 1280))
        frame_h = int(hints.get("frame_height", 720))
        frame_pairs = [(i, i) for i in range(n_frames)]
        base_rows = int(hints.get("base_rows") or (active_stack.shape[1] * 12))
        base_cols = int(hints.get("base_cols") or (active_stack.shape[2] * 12))

    cfg = SlidingWindowConfig(
        window_size=int(args.window_size),
        stride=int(args.stride),
        base_rows=base_rows,
        base_cols=base_cols,
        frame_w=frame_w,
        frame_h=frame_h,
        top_k_regions=int(args.max_window_candidates),
        max_blocks_per_roi=1,  # seed only → 192x192 cell
    )
    windows, stats, baseline = build_motion_regions(
        active_stack, cfg, n_frames=n_frames, pooled_rms=pooled_rms
    )
    frame_idx_map = {fi: fidx for fi, fidx in frame_pairs}
    return {
        "video_name": video_name,
        "video_path": video_path,
        "agg_dir": agg_dir,
        "windows": windows,
        "stats": stats,
        "baseline": baseline,
        "cfg": cfg,
        "n_frames": n_frames,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "frame_idx_map": frame_idx_map,
        "frame_pairs": frame_pairs,
    }


def run_method_a(ctx: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fused, groups, fusion_stats = build_temporal_group_rois(
        ctx["windows"],
        n_frames=ctx["n_frames"],
        cfg=ctx["cfg"],
        frame_idx_map=ctx["frame_idx_map"],
        max_rois=int(args.max_rois),
        max_rep_blocks=1,  # force single 192x192 block
        block_frequency=0.3,
        max_area_ratio=0.28,
        score_active_frac=0.2,
    )
    save_temporal_groups_json(
        out_dir / "temporal_groups.json",
        groups,
        meta={
            "approach": "A",
            "video_name": ctx["video_name"],
            "roi_policy": "max_rep_blocks=1 (192x192)",
            **{k: fusion_stats[k] for k in ("n_final_rois", "n_groups_before_topk", "method") if k in fusion_stats},
        },
    )
    save_fused_frame_roi_json(out_dir / "frame_fused_roi.json", fused, meta={"approach": "A"})
    summary = {"approach": "A", "n_groups": len(groups), **fusion_stats}
    if not args.skip_overlay:
        overlay = out_dir / "sliding_window_overlay.mp4"
        fused_by_sample = {fr.frame_index: fr for fr in fused}
        info = write_full_video_fused_overlay(
            video_path=ctx["video_path"],
            fused_by_sample=fused_by_sample,
            frame_pairs=ctx["frame_pairs"],
            output_mp4=overlay,
        )
        summary.update(info)
        summary["overlay"] = str(overlay)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_method_b(ctx: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # B+: hop1 neighbor growth + longer span + context overlay (why labels)
    fused, events, stats = build_block_peak_rois(
        ctx["windows"],
        n_frames=ctx["n_frames"],
        cfg=ctx["cfg"],
        frame_idx_map=ctx["frame_idx_map"],
        max_rois=int(args.max_rois),
        min_peak=0.05,
        top_onset_peak=0.05,
        spatial_mode="hop1",
        neighbor_ratio=0.22,
        max_cells=8,
        span_frac=0.12,
        span_pad=2,
        min_persist=2,
    )
    save_block_peak_events_json(
        out_dir / "temporal_groups.json",
        events,
        meta={
            "approach": "B_plus",
            "video_name": ctx["video_name"],
            "frame_area": float(ctx["frame_w"] * ctx["frame_h"]),
            "roi_policy": "hop1_neighbor_growth_max8_cells",
            **{
                k: stats[k]
                for k in (
                    "n_raw_events",
                    "n_final_rois",
                    "method",
                    "spatial_mode",
                    "neighbor_ratio",
                    "max_cells",
                    "span_frac",
                    "span_pad",
                )
                if k in stats
            },
        },
    )
    save_fused_frame_roi_json(out_dir / "frame_fused_roi.json", fused, meta={"approach": "B_plus"})
    summary = {"approach": "B_plus", "n_events": len(events), **stats}
    if not args.skip_overlay:
        overlay = out_dir / "sliding_window_overlay.mp4"
        info = write_context_overlay_video(
            video_path=ctx["video_path"],
            events=events,
            frame_pairs=ctx["frame_pairs"],
            output_mp4=overlay,
            variant_name="B_plus_hop1_long",
            cfg=ctx["cfg"],
        )
        summary.update(info)
        summary["overlay"] = str(overlay)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    methods = sorted({m.upper() for m in args.methods})
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    videos = discover_videos(input_root, args.components_subdir, args.video_names)
    logger.info("Videos=%s methods=%s", videos, methods)

    batch: list[dict[str, Any]] = []
    for name in videos:
        t0 = time.time()
        ctx = build_windows(name, input_root, args)
        video_out = output_root / name
        row: dict[str, Any] = {"video_name": name, "methods": {}}
        if "A" in methods:
            sa = run_method_a(ctx, video_out / "method_A", args)
            row["methods"]["A"] = sa
            logger.info("A %s groups=%s", name, sa.get("n_groups"))
        if "B" in methods:
            sb = run_method_b(ctx, video_out / "method_B", args)
            row["methods"]["B"] = sb
            logger.info("B %s events=%s", name, sb.get("n_events"))
        # Convenience: copy B overlay to the video root as primary
        if "B" in methods and not args.skip_overlay:
            src = video_out / "method_B" / "sliding_window_overlay.mp4"
            dst = video_out / "sliding_window_overlay_B.mp4"
            if src.is_file():
                dst.write_bytes(src.read_bytes())
            ag = video_out / "method_A" / "sliding_window_overlay.mp4"
            if ag.is_file():
                (video_out / "sliding_window_overlay_A.mp4").write_bytes(ag.read_bytes())
        row["elapsed_sec"] = round(time.time() - t0, 2)
        batch.append(row)
        logger.info("Done %s in %.1fs", name, row["elapsed_sec"])

    out = output_root / "batch_summary_causal_roi_ab.json"
    out.write_text(json.dumps({"videos": batch}, indent=2), encoding="utf-8")
    logger.info("Batch summary → %s", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sliding-window candidates → temporal-group ROIs → fixed-bbox overlay.

Window stage keeps up to max_window_candidates (default 10) novelty seed ROIs.
Final max_rois=2 is applied only after:
  adjacent-window temporal grouping → persistence>=2 → representative blocks
  (frequency>=0.5) → group_score ranking.

Representative bbox is fixed (grid-aligned) over each group's valid span.
No window Top-2, no frame-level union, no center interpolation.

Outputs (per video folder):
  spatial_baseline_*.npy / spatial_baseline_summary.json
  motion_regions.csv / .json / sliding_window_summary.json
  temporal_groups.json
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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from motion_analyzer.temporal.frame_window_fusion import (  # noqa: E402
    save_fused_frame_roi_json,
    write_full_video_fused_overlay,
)
from motion_analyzer.temporal.temporal_group_roi import (  # noqa: E402
    build_temporal_group_rois,
    save_temporal_groups_json,
)
from motion_analyzer.temporal.sliding_window_regions import (  # noqa: E402
    SlidingWindowConfig,
    build_motion_regions,
    load_active_mask_stack,
    load_pooled_rms_stack,
    save_motion_region_outputs,
    save_spatial_baseline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sliding_window_regions")

DEFAULT_INPUT_ROOT = Path(
    "/data1/vailab02_dir/vlm_motion/motion-detector/Result/"
    "residual_motion_120s_block16_4video"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT
DEFAULT_COMPONENTS_SUBDIR = "sub_block_agg_12x12"
DEFAULT_VIDEO_DIRS = [
    Path("/data1/vailab02_dir/Classification_DB/VIRAT/videos-00"),
    Path("/data1/vailab02_dir/Classification_DB/VIRAT/videos-01"),
    Path("/data1/vailab02_dir/Classification_DB/VIRAT/videos-04"),
    Path("/data1/vailab02_dir/Classification_DB/VIRAT/videos-05"),
    Path("/data1/vailab02_dir/Classification_DB/VIRAT"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--components_subdir", type=str, default=DEFAULT_COMPONENTS_SUBDIR)
    p.add_argument("--video_names", nargs="+", default=None)
    p.add_argument("--window_size", type=int, default=10)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--min_active_frames", type=int, default=2)
    p.add_argument("--connectivity", type=int, default=8)
    p.add_argument(
        "--max_window_candidates",
        type=int,
        default=12,
        help="per-window candidate cap (not final Top-2)",
    )
    p.add_argument(
        "--top_k_regions",
        type=int,
        default=None,
        help="alias for --max_window_candidates",
    )
    p.add_argument("--max_rois", type=int, default=2, help="final temporal groups")
    p.add_argument("--max_frame_rois", type=int, default=None, help="alias for --max_rois")
    p.add_argument("--min_window_persistence", type=int, default=2)
    p.add_argument("--block_frequency", type=float, default=0.3)
    p.add_argument("--max_rep_blocks", type=int, default=6)
    p.add_argument("--max_area_ratio", type=float, default=0.28)
    p.add_argument("--persist_cap", type=int, default=8)
    p.add_argument("--min_center_sep", type=float, default=1.5)
    p.add_argument("--score_active_frac", type=float, default=0.2)
    p.add_argument("--max_blocks_per_roi", type=int, default=4)
    p.add_argument("--large_area_ratio", type=float, default=0.60)
    p.add_argument("--skip_overlay", action="store_true")
    args = p.parse_args()
    if args.top_k_regions is not None:
        args.max_window_candidates = int(args.top_k_regions)
    if args.max_frame_rois is not None:
        args.max_rois = int(args.max_frame_rois)
    args.top_k_regions = int(args.max_window_candidates)
    args.max_frame_rois = int(args.max_rois)
    return args


def discover_videos(input_root: Path, subdir: str, names: list[str] | None) -> list[str]:
    if names:
        return list(names)
    found: list[str] = []
    for d in sorted(input_root.iterdir()):
        if d.is_dir() and (d / subdir / "active_mask.npy").is_file():
            found.append(d.name)
    return found


def resolve_video_path(stem: str, hints: list[Path | None]) -> Path | None:
    for h in hints:
        if h is not None and Path(h).is_file():
            return Path(h)
    seen: set[Path] = set()
    for root in DEFAULT_VIDEO_DIRS:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        direct = root / f"{stem}.mp4"
        if direct.is_file():
            return direct
        matches = sorted(root.rglob(f"{stem}.mp4"))
        if matches:
            return matches[0]
    return None


def load_hints(video_dir: Path, agg_dir: Path) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    for path in (
        video_dir / "temporal_linking_summary.json",
        video_dir / "config.json",
        agg_dir / "aggregation_metadata.json",
    ):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "source_video_path" in data:
            hints["source_video_path"] = data["source_video_path"]
        frames = data.get("frames") or []
        if frames:
            hints["frames"] = frames
            hints["num_frames"] = int(data.get("num_frames") or len(frames))
            f0 = frames[0]
            if "frame_width" in f0:
                hints["frame_width"] = int(f0["frame_width"])
                hints["frame_height"] = int(f0["frame_height"])
        if "base_block_px" in data:
            hints["base_block_px"] = int(data["base_block_px"])
        if "agg_factor" in data:
            hints["agg_factor"] = int(data["agg_factor"])
        bg = data.get("base_grid")
        if isinstance(bg, dict):
            if "rows" in bg:
                hints["base_rows"] = int(bg["rows"])
            if "cols" in bg:
                hints["base_cols"] = int(bg["cols"])
        elif isinstance(bg, (list, tuple)) and len(bg) >= 2:
            hints["base_rows"] = int(bg[0])
            hints["base_cols"] = int(bg[1])
        if frames and "base_grid_rows" in frames[0]:
            hints["base_rows"] = int(frames[0]["base_grid_rows"])
            hints["base_cols"] = int(frames[0]["base_grid_cols"])
        rep = data.get("representation_metadata") or {}
        if isinstance(rep, dict):
            fr = rep.get("frames") or []
            if fr and "frame_width" not in hints:
                hints["frame_width"] = int(fr[0].get("frame_width", 1280))
                hints["frame_height"] = int(fr[0].get("frame_height", 720))
                if "base_rows" not in hints and "grid_rows" in fr[0]:
                    hints["base_rows"] = int(fr[0]["grid_rows"])
                    hints["base_cols"] = int(fr[0]["grid_cols"])
    return hints


def process_video(
    video_name: str,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    video_dir = input_root / video_name
    agg_dir = video_dir / args.components_subdir
    active_path = agg_dir / "active_mask.npy"
    if not active_path.is_file():
        raise FileNotFoundError(f"Missing active_mask.npy: {active_path}")

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
    else:
        frame_w = int(hints.get("frame_width", 1280))
        frame_h = int(hints.get("frame_height", 720))
        frame_pairs = [(i, i) for i in range(n_frames)]

    base_rows = int(hints.get("base_rows") or (active_stack.shape[1] * 12))
    base_cols = int(hints.get("base_cols") or (active_stack.shape[2] * 12))
    if frames_meta and "base_grid_rows" in frames_meta[0]:
        base_rows = int(frames_meta[0]["base_grid_rows"])
        base_cols = int(frames_meta[0]["base_grid_cols"])

    agg_meta_path = agg_dir / "aggregation_metadata.json"
    if agg_meta_path.is_file():
        agg_meta = json.loads(agg_meta_path.read_text(encoding="utf-8"))
        bg = agg_meta.get("base_grid")
        if isinstance(bg, dict) and "rows" in bg and "cols" in bg:
            base_rows, base_cols = int(bg["rows"]), int(bg["cols"])
        elif isinstance(bg, (list, tuple)) and len(bg) >= 2:
            base_rows, base_cols = int(bg[0]), int(bg[1])
        if frames_meta and "base_grid_rows" in frames_meta[0]:
            base_rows = int(frames_meta[0]["base_grid_rows"])
            base_cols = int(frames_meta[0]["base_grid_cols"])

    cfg = SlidingWindowConfig(
        window_size=int(args.window_size),
        stride=int(args.stride),
        min_active_frames=int(args.min_active_frames),
        connectivity=int(args.connectivity),
        base_block_px=int(hints.get("base_block_px", 16)),
        agg_factor=int(hints.get("agg_factor", 12)),
        base_rows=base_rows,
        base_cols=base_cols,
        frame_w=frame_w,
        frame_h=frame_h,
        top_k_regions=int(args.max_window_candidates),
        large_area_ratio=float(args.large_area_ratio),
        max_blocks_per_roi=int(args.max_blocks_per_roi),
    )

    t0 = time.time()
    windows, stats, baseline = build_motion_regions(
        active_stack,
        cfg,
        n_frames=n_frames,
        pooled_rms=pooled_rms,
    )

    all_regions = [r for w in windows for r in w.regions]
    frame_area = float(frame_w * frame_h)
    areas = [r.area for r in all_regions]
    area_ratios = [a / frame_area for a in areas] if frame_area > 0 else []
    stats = {
        **stats,
        "total_regions_after_ios_merge": int(len(all_regions)),
        "total_global_motion_regions": 0,
        "windows_with_global_motion": 0,
        "mean_regions_per_window": float(
            np.mean([len(w.regions) for w in windows]) if windows else 0.0
        ),
        "mean_bbox_area_ratio": float(np.mean(area_ratios)) if area_ratios else 0.0,
        "mean_union_bbox_area": float(np.mean(areas)) if areas else 0.0,
        "stride": int(args.stride),
        "max_window_candidates": int(args.max_window_candidates),
        "max_rois": int(args.max_rois),
        "min_window_persistence": int(args.min_window_persistence),
        "block_frequency": float(args.block_frequency),
        "cross_window_matching": "adjacent_overlap_or_center<=1",
        "stabilization": "fixed_representative_bbox",
    }

    out_dir = output_root / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_paths = save_spatial_baseline(out_dir, baseline)

    summary: dict[str, Any] = {
        "video_name": video_name,
        "active_mask_npy": str(active_path),
        "aggregated_rms_mag_npy": str(agg_dir / "aggregated_rms_mag.npy"),
        "source_video_path": str(video_path),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "note": (
            "window candidates (max10) → adjacent-window temporal grouping → "
            "persistence>=2 → freq>=0.5 representative blocks → group_score Top-2 → "
            "fixed grid bbox over group span. No window Top-2 / frame union / center lerp. "
            "Farneback/16x16/12x12 RMS not recomputed."
        ),
        **stats,
    }
    paths = save_motion_region_outputs(out_dir, windows, summary)
    paths.update(baseline_paths)

    frame_idx_map = {fi: fidx for fi, fidx in frame_pairs}
    fused_frames, groups, fusion_stats = build_temporal_group_rois(
        windows,
        n_frames=n_frames,
        cfg=cfg,
        frame_idx_map=frame_idx_map,
        min_window_persistence=int(args.min_window_persistence),
        block_frequency=float(args.block_frequency),
        max_rois=int(args.max_rois),
        max_rep_blocks=int(args.max_rep_blocks),
        max_area_ratio=float(args.max_area_ratio),
        persist_cap=int(args.persist_cap),
        min_center_sep=float(args.min_center_sep),
        score_active_frac=float(args.score_active_frac),
    )
    rois_total = sum(len(fr.rois) for fr in fused_frames)
    groups_json = save_temporal_groups_json(
        out_dir / "temporal_groups.json",
        groups,
        meta={
            "video_name": video_name,
            "n_sampled_frames": n_frames,
            "max_window_candidates": int(args.max_window_candidates),
            "max_rois": int(args.max_rois),
            "min_window_persistence": int(args.min_window_persistence),
            "block_frequency": float(args.block_frequency),
            **{k: fusion_stats[k] for k in (
                "n_window_candidates",
                "n_groups_before_topk",
                "n_final_rois",
                "n_dropped_persistence",
            ) if k in fusion_stats},
        },
    )
    paths["temporal_groups_json"] = groups_json
    fused_json = save_fused_frame_roi_json(
        out_dir / "frame_fused_roi.json",
        fused_frames,
        meta={
            "video_name": video_name,
            "n_sampled_frames": n_frames,
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "min_active_frames": int(args.min_active_frames),
            "max_window_candidates": int(args.max_window_candidates),
            "max_rois": int(args.max_rois),
            "min_window_persistence": int(args.min_window_persistence),
            "block_frequency": float(args.block_frequency),
            "block_px": int(cfg.block_px),
            "roi_instances_total": rois_total,
            **fusion_stats,
        },
    )
    paths["frame_fused_roi_json"] = fused_json

    summary.update(fusion_stats)
    summary["roi_instances_total"] = rois_total
    summary["fused_sampled_frames"] = len(fused_frames)
    paths["sliding_window_summary"].write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if not args.skip_overlay:
        overlay_path = out_dir / "sliding_window_overlay.mp4"
        fused_by_sample = {fr.frame_index: fr for fr in fused_frames}
        overlay_info = write_full_video_fused_overlay(
            video_path=video_path,
            fused_by_sample=fused_by_sample,
            frame_pairs=frame_pairs,
            output_mp4=overlay_path,
        )
        summary.update(overlay_info)
        paths["sliding_window_summary"].write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        paths["sliding_window_overlay"] = overlay_path

    elapsed = time.time() - t0
    logger.info(
        "Wrote %s windows=%d win_cands=%d groups=%d frames_with_roi=%d "
        "overlay=%s in %.1fs → %s",
        video_name,
        summary["num_windows"],
        len(all_regions),
        fusion_stats.get("n_final_rois", 0),
        fusion_stats.get("frames_with_roi", 0),
        summary.get("overlay_frames"),
        elapsed,
        out_dir,
    )
    return {
        "video_name": video_name,
        "output_dir": str(out_dir),
        **{k: str(v) for k, v in paths.items()},
        "summary": {
            "num_windows": summary["num_windows"],
            "mean_regions_per_window": summary["mean_regions_per_window"],
            "n_window_candidates": fusion_stats.get("n_window_candidates"),
            "n_groups_before_topk": fusion_stats.get("n_groups_before_topk"),
            "n_final_rois": fusion_stats.get("n_final_rois"),
            "frames_with_roi": fusion_stats.get("frames_with_roi"),
            "mean_rois_per_frame": fusion_stats.get("mean_rois_per_frame"),
            "overlay_frames": summary.get("overlay_frames"),
            "overlay_matches_source_length": summary.get(
                "overlay_matches_source_length"
            ),
            "cross_window_matching": fusion_stats.get("cross_window_matching"),
            "stabilization": fusion_stats.get("stabilization"),
        },
        "elapsed_sec": elapsed,
    }


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    videos = discover_videos(input_root, args.components_subdir, args.video_names)
    if not videos:
        raise SystemExit(
            f"No videos with {args.components_subdir}/active_mask.npy under {input_root}"
        )

    logger.info("Input root: %s", input_root)
    logger.info("Output root: %s", output_root)
    logger.info(
        "Videos (%d): %s | window=%d stride=%d win_cands=%d max_rois=%d "
        "persist>=%d freq>=%.2f temporal-groups",
        len(videos),
        videos,
        args.window_size,
        args.stride,
        args.max_window_candidates,
        args.max_rois,
        args.min_window_persistence,
        args.block_frequency,
    )

    batch = [process_video(name, input_root, output_root, args) for name in videos]
    batch_path = output_root / "batch_summary_linear_temporal_roi.json"
    batch_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    logger.info("Batch summary → %s", batch_path)


if __name__ == "__main__":
    main()

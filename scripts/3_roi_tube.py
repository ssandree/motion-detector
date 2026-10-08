#!/usr/bin/env python3
"""Stage 3 — official ROI tubes (attractive Union-Find → AABB composition).

  1. Block events          tau_high=0.2, max_gap=5, 64px
  2. Attractive Union-Find c>0 merge, hashed Chebyshev≤2
  3. ROI composition       compactness≤1.5, tgap≤15, max 3 ROI/frame
  4. Filter + fixed lifetime bbox

Outputs:
  videos/json → outputs/stage3/3_roi_tube/<stamp>/
  3D PNGs    → outputs/stage3/3_roi_tube_3dviz/<stamp>/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motion_analyzer.config import (  # noqa: E402
    DEFAULT_FUSION,
    FUSION_CHOICES,
    HEAT_VMAX,
    HEAT_VMIN,
    ROI_MAX_GAP,
    ROI_MIN_BLOCK_EVENT,
    ROI_MIN_TUBE_CELLS,
    ROI_MIN_TUBE_DURATION,
    ROI_TAU_HIGH,
    ROI_TAU_LOW,
    PipelineConfig,
    load_target_video_ids,
)
from stage3.v2.roi_composition import (  # noqa: E402
    COMP_COMPACTNESS_FACTOR,
    COMP_MAX_ROI_PER_FRAME,
    COMP_MAX_TEMPORAL_GAP,
)
from stage3 import list_videos_in_fusion_root, process_video  # noqa: E402

logger = logging.getLogger("3_roi_tube")


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fusion_root",
        type=Path,
        required=True,
        help="Stage-2 root with */*_gap_fusion_*.npz or */*_stage2_agg_*.npz",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "stage3" / "3_roi_tube" / stamp,
        help="MP4 / motion_events / roi_tracks root.",
    )
    parser.add_argument(
        "--viz_3d_root",
        type=Path,
        default=None,
        help="3D PNG root (default: outputs/stage3/3_roi_tube_3dviz/<same-stamp>).",
    )
    parser.add_argument("--events_root", type=Path, default=None)
    parser.add_argument("--fusion", type=str, default=DEFAULT_FUSION, choices=list(FUSION_CHOICES))
    parser.add_argument("--tau_high", type=float, default=ROI_TAU_HIGH)
    parser.add_argument("--tau_low", type=float, default=ROI_TAU_LOW)
    parser.add_argument("--max_gap", type=int, default=ROI_MAX_GAP)
    parser.add_argument("--min_block_event", type=int, default=ROI_MIN_BLOCK_EVENT)
    parser.add_argument("--min_tube_cells", type=int, default=ROI_MIN_TUBE_CELLS)
    parser.add_argument("--min_tube_duration", type=int, default=ROI_MIN_TUBE_DURATION)
    parser.add_argument(
        "--comp_compactness_factor",
        type=float,
        default=COMP_COMPACTNESS_FACTOR,
    )
    parser.add_argument("--comp_max_temporal_gap", type=int, default=COMP_MAX_TEMPORAL_GAP)
    parser.add_argument("--comp_max_rois", type=int, default=COMP_MAX_ROI_PER_FRAME)
    parser.add_argument(
        "--no_suppress_contained",
        action="store_true",
        help="Keep tubes even if fully inside another tube's spatio-temporal AABB.",
    )
    parser.add_argument(
        "--use_base_map",
        action="store_true",
        help="Use 16px M_* instead of 64px MU_*.",
    )
    parser.add_argument("--no_3d", action="store_true", help="Skip 3D tube figures.")
    parser.add_argument("--heat_vmin", type=float, default=HEAT_VMIN)
    parser.add_argument("--heat_vmax", type=float, default=HEAT_VMAX)
    parser.add_argument("--sampling_fps", type=float, default=5.0)
    parser.add_argument("--video_id", type=str, default=None)
    parser.add_argument("--video_list", type=Path, default=None)
    parser.add_argument("--video_ids", nargs="+", default=None)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    if float(args.tau_low) > float(args.tau_high):
        raise SystemExit("tau_low must be ≤ tau_high")

    cfg = PipelineConfig(
        sampling_fps=float(args.sampling_fps),
        roi_threshold=float(args.tau_high),
        fusion=str(args.fusion),
    )
    cfg.validate()

    fusion_root = args.fusion_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.viz_3d_root is not None:
        viz_3d_root = args.viz_3d_root.resolve()
    else:
        viz_3d_root = (
            REPO_ROOT / "outputs" / "stage3" / "3_roi_tube_3dviz" / output_root.name
        ).resolve()
    if not args.no_3d:
        viz_3d_root.mkdir(parents=True, exist_ok=True)

    if args.video_ids:
        video_ids = list(args.video_ids)
    elif args.video_id:
        video_ids = [args.video_id]
    elif args.video_list is not None:
        video_ids = load_target_video_ids(args.video_list)
    else:
        video_ids = list_videos_in_fusion_root(fusion_root)
        if not video_ids:
            video_ids = load_target_video_ids(None)

    rows = []
    for index, video_id in enumerate(video_ids, start=1):
        logger.info("[%d/%d] %s", index, len(video_ids), video_id)
        started = time.time()
        try:
            info = process_video(
                video_id,
                cfg=cfg,
                fusion_root=fusion_root,
                output_root=output_root,
                events_root=args.events_root.resolve() if args.events_root else None,
                fusion=str(args.fusion),
                prefer_unit=not bool(args.use_base_map),
                tau_high=float(args.tau_high),
                tau_low=float(args.tau_low),
                max_gap=int(args.max_gap),
                min_block_event=int(args.min_block_event),
                min_tube_cells=int(args.min_tube_cells),
                min_tube_duration=int(args.min_tube_duration),
                suppress_contained=not bool(args.no_suppress_contained),
                comp_compactness_factor=float(args.comp_compactness_factor),
                comp_max_temporal_gap=int(args.comp_max_temporal_gap),
                comp_max_rois=int(args.comp_max_rois),
                write_3d=not bool(args.no_3d),
                viz_3d_root=None if args.no_3d else viz_3d_root,
                heat_vmin=float(args.heat_vmin),
                heat_vmax=float(args.heat_vmax),
            )
            info["elapsed_sec"] = round(time.time() - started, 3)
            rows.append(info)
            (output_root / f"{video_id}_result.json").write_text(
                json.dumps(info, indent=2) + "\n", encoding="utf-8"
            )
            logger.info(
                "  tubes=%d events=%d uf=%s frames=%d -> %s | 3d=%s "
                "(pipeline %.3fs, wall %.1fs)",
                info.get("num_tubes", 0),
                info.get("num_events", 0),
                info.get("num_split_components", info.get("num_graph_partitions")),
                info["visualization_frames"],
                info["visualization_mp4"],
                info.get("visualization_3d"),
                float(info.get("pipeline_sec") or 0.0),
                info["elapsed_sec"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", video_id, exc)
            rows.append({"video_id": video_id, "error": str(exc)})

    summary = {
        "stage": 3,
        "variant": "stage3_light_uf_compose",
        "params": {
            "tau_high": float(args.tau_high),
            "tau_low": float(args.tau_low),
            "max_gap": int(args.max_gap),
            "min_block_event": int(args.min_block_event),
            "min_tube_cells": int(args.min_tube_cells),
            "min_tube_duration": int(args.min_tube_duration),
            "suppress_contained": not bool(args.no_suppress_contained),
            "comp_compactness_factor": float(args.comp_compactness_factor),
            "comp_max_temporal_gap": int(args.comp_max_temporal_gap),
            "comp_max_rois": int(args.comp_max_rois),
            "heat_vmin": float(args.heat_vmin),
            "heat_vmax": float(args.heat_vmax),
            "heat_colormap": "turbo",
        },
        "fusion": str(args.fusion),
        "prefer_unit": not bool(args.use_base_map),
        "fusion_root": str(fusion_root),
        "output_root": str(output_root),
        "viz_3d_root": str(viz_3d_root) if not args.no_3d else None,
        "num_videos": len(rows),
        "num_ok": sum(1 for r in rows if "error" not in r),
        "num_failed": sum(1 for r in rows if "error" in r),
        "num_events": sum(
            int(r.get("num_events", 0)) for r in rows if "error" not in r
        ),
        "videos": rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if not args.no_3d:
        (viz_3d_root / "summary.json").write_text(
            json.dumps(
                {
                    "stage": 3,
                    "artifact": "roi_tube_3dviz",
                    "params": summary["params"],
                    "viz_3d_root": str(viz_3d_root),
                    "video_output_root": str(output_root),
                    "num_ok": summary["num_ok"],
                    "num_failed": summary["num_failed"],
                    "images": [
                        r.get("visualization_3d")
                        for r in rows
                        if "error" not in r and r.get("visualization_3d")
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    logger.info(
        "Done: ok=%d failed=%d | videos=%s | 3d=%s",
        summary["num_ok"],
        summary["num_failed"],
        output_root,
        None if args.no_3d else viz_3d_root,
    )
    return 0 if summary["num_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

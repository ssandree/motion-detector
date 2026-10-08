#!/usr/bin/env python3
"""Stage 2 — filtered Gap1 integrals → mean fusion → 4×4 max.

Requires Stage1 NPZ (run scripts/1_base_motion.py first).

1) Load Stage1 U1 (6×6×1 R_ap-mean → P13 → T5). No extra cell-mag floor.
2) Gap G = sum of G consecutive Gap1 vectors (no extra Farneback / P15)
3) Per-gap ÷√G before fusion
4) Temporal fusion (locked mean) → M_fused @16px
5) 4×4 magnitude max → MU_fused @64px
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

from motion_analyzer.aggregation import fuse_video  # noqa: E402
from motion_analyzer.config import (  # noqa: E402
    AGGREGATION_BLOCK,
    DEFAULT_DATA_ROOT,
    DEFAULT_FUSION,
    DEFAULT_FUSION_SCHEME,
    DEFAULT_SPATIAL_AGG,
    FUSION_CHOICES,
    FUSION_SCHEME_CHOICES,
    GAP_NORM_DIV,
    GAPS,
    HEAT_VMAX,
    HEAT_VMIN,
    LONG_GAPS,
    RESIZED_BASE_BLOCK,
    SHORT_GAPS,
    STAGE1_SPATIAL_WIN,
    STAGE1_TEMPORAL_RADIUS,
    STAGE2_CELL_MAG_FLOOR,
    PipelineConfig,
    load_target_video_ids,
)

logger = logging.getLogger("2_gap_fusion")


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Stage1 NPZ root (data/cache).",
    )
    parser.add_argument(
        "--fusion_root",
        type=Path,
        default=REPO_ROOT / "data" / "stage2" / "2_gap_fusion" / stamp,
        help="Stage2 NPZ root (not outputs/).",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "stage2" / "2_gap_fusion" / stamp,
        help="Summary JSON root.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default=DEFAULT_FUSION,
        choices=list(FUSION_CHOICES),
        help="Temporal fusion (official: mean).",
    )
    parser.add_argument(
        "--fusion_scheme",
        type=str,
        default=DEFAULT_FUSION_SCHEME,
        choices=list(FUSION_SCHEME_CHOICES),
        help=(
            "flat=equal-weight over all gaps; "
            "short_long=mean(S_short,S_long) with S_short=mean(1,5), "
            "S_long=mean(10,20,50)."
        ),
    )
    parser.add_argument(
        "--spatial_agg",
        type=str,
        default=DEFAULT_SPATIAL_AGG,
        choices=list(FUSION_CHOICES),
        help="4×4 spatial aggregation (official: max).",
    )
    parser.add_argument("--unit_block", type=int, default=AGGREGATION_BLOCK)
    parser.add_argument("--sampling_fps", type=float, default=5.0)
    parser.add_argument("--block_size", type=int, default=RESIZED_BASE_BLOCK)
    parser.add_argument("--spatial_win", type=int, default=STAGE1_SPATIAL_WIN)
    parser.add_argument("--temporal_radius", type=int, default=STAGE1_TEMPORAL_RADIUS)
    parser.add_argument("--video_id", type=str, default=None)
    parser.add_argument("--video_list", type=Path, default=None)
    parser.add_argument("--max_seconds", type=float, default=None)
    parser.add_argument(
        "--gaps",
        type=str,
        default=None,
        help="Comma-separated temporal gaps (default: config GAPS). Example: 1,5,10,20",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Unused (long gaps come from Stage1 Gap1; kept for CLI compatibility).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    cfg = PipelineConfig(
        sampling_fps=float(args.sampling_fps),
        fusion=str(args.fusion),
    )
    cfg.validate()

    fusion_root = args.fusion_root.resolve()
    output_root = args.output_root.resolve()
    fusion_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Stage2 Gap1-integrate | temporal=%s scheme=%s spatial=%s | ÷√g | heat=[%.3g,%.3g]",
        args.fusion,
        args.fusion_scheme,
        args.spatial_agg,
        HEAT_VMIN,
        HEAT_VMAX,
    )

    if args.gaps:
        gaps = tuple(int(x.strip()) for x in str(args.gaps).split(",") if x.strip())
        if not gaps:
            raise SystemExit("--gaps must list at least one integer")
    else:
        gaps = tuple(int(g) for g in GAPS)
    gap_norm_div = {
        int(g): float(GAP_NORM_DIV.get(int(g), float(g) ** 0.5)) for g in gaps
    }
    logger.info("gaps=%s | gap_norm_div=%s", list(gaps), gap_norm_div)

    video_ids = [args.video_id] if args.video_id else load_target_video_ids(args.video_list)
    rows = []
    for index, video_id in enumerate(video_ids, start=1):
        logger.info("[%d/%d] %s", index, len(video_ids), video_id)
        started = time.time()
        try:
            info = fuse_video(
                video_id,
                cfg=cfg,
                output_root=fusion_root,
                fusion=str(args.fusion),
                fusion_scheme=str(args.fusion_scheme),
                spatial_agg=str(args.spatial_agg),
                unit_block=int(args.unit_block),
                gaps=gaps,
                gap_norm_div=gap_norm_div,
                short_gaps=tuple(int(g) for g in SHORT_GAPS),
                long_gaps=tuple(int(g) for g in LONG_GAPS),
                block_size=int(args.block_size),
                spatial_win=int(args.spatial_win),
                temporal_radius=int(args.temporal_radius),
                max_seconds=args.max_seconds,
                data_root=args.data_root.resolve(),
                use_gpu=not bool(args.no_gpu),
            )
            info["elapsed_sec"] = round(time.time() - started, 3)
            rows.append(info)
            logger.info(
                "  base=%s unit=%s temporal=%s spatial=%s gaps=%s -> %s "
                "(pipeline %.3fs, wall %.1fs)",
                info["map_shape_base"],
                info["map_shape_unit"],
                info["fusion"],
                info["spatial_agg"],
                info["gaps"],
                info["fusion_npz"],
                float(info.get("pipeline_sec") or 0.0),
                info["elapsed_sec"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", video_id, exc)
            rows.append({"video_id": video_id, "error": str(exc)})

    if str(args.fusion_scheme) == "short_long":
        variant = (
            f"gap1_integrate_divsqrtg_short_long_{args.fusion}_then_4x4{args.spatial_agg}"
        )
    else:
        variant = (
            f"gap1_integrate_divsqrtg_then_fusion_{args.fusion}_then_4x4{args.spatial_agg}"
        )
    summary = {
        "stage": 2,
        "variant": variant,
        "fusion": str(args.fusion),
        "fusion_scheme": str(args.fusion_scheme),
        "short_gaps": list(SHORT_GAPS),
        "long_gaps": list(LONG_GAPS),
        "spatial_agg": str(args.spatial_agg),
        "stage2_gpu": False,
        "long_gaps_from_gap1": True,
        "cell_mag_floor": float(STAGE2_CELL_MAG_FLOOR),
        "gaps": list(gaps),
        "gap_norm_div": {str(k): float(v) for k, v in gap_norm_div.items()},
        "heat_vmin": float(HEAT_VMIN),
        "heat_vmax": float(HEAT_VMAX),
        "unit_block": int(args.unit_block),
        "block_size": int(args.block_size),
        "spatial_win": int(args.spatial_win),
        "temporal_radius": int(args.temporal_radius),
        "data_root": str(args.data_root.resolve()) if args.data_root else None,
        "fusion_root": str(fusion_root),
        "output_root": str(output_root),
        "num_videos": len(rows),
        "num_ok": sum(1 for r in rows if "error" not in r),
        "num_failed": sum(1 for r in rows if "error" in r),
        "videos": rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Done: ok=%d failed=%d | npz=%s | summary=%s",
        summary["num_ok"],
        summary["num_failed"],
        fusion_root,
        output_root,
    )
    return 0 if summary["num_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

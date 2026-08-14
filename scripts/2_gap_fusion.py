#!/usr/bin/env python3
"""Stage 2 — Stage1 Gap1 + same-recipe extra gaps → gap-normalize → fusion → 4×4 unit agg.

Requires Stage1 NPZ (run scripts/1_base_motion.py first).

1) Load Stage1 U1/M1 (8×8×1 R_ap-mean → P15 → T5)
2) Compute Gap 5/10/20/50 with the same Stage1 recipe
3) Per-gap normalize: Gap5÷3.0, Gap10÷3.5, Gap20÷3.8, Gap50÷4.2 (Gap1÷1)
4) Temporal fusion across gaps (default RMS) → M_fused @16px
5) 4×4 magnitude max → MU_fused @64px

Heatmap viz scale (Stage2/3 overlays): HEAT_VMIN=0.8, HEAT_VMAX=3.5.
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
    FUSION_CHOICES,
    GAP_NORM_DIV,
    GAPS,
    HEAT_VMAX,
    HEAT_VMIN,
    RESIZED_BASE_BLOCK,
    STAGE1_SPATIAL_WIN,
    STAGE1_TEMPORAL_RADIUS,
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
        "--output_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "stage2" / "2_gap_fusion" / stamp,
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default=DEFAULT_FUSION,
        choices=list(FUSION_CHOICES),
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
        "--no_gpu",
        action="store_true",
        help="Disable GPU 8×8×1 R-mean for extra gaps (CPU path).",
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

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Stage2 defaults | fusion=%s | gap_norm_div=%s | heat=[%.3g,%.3g]",
        args.fusion,
        {int(k): float(v) for k, v in GAP_NORM_DIV.items()},
        HEAT_VMIN,
        HEAT_VMAX,
    )

    video_ids = [args.video_id] if args.video_id else load_target_video_ids(args.video_list)
    rows = []
    for index, video_id in enumerate(video_ids, start=1):
        logger.info("[%d/%d] %s", index, len(video_ids), video_id)
        started = time.time()
        try:
            info = fuse_video(
                video_id,
                cfg=cfg,
                output_root=output_root,
                fusion=str(args.fusion),
                unit_block=int(args.unit_block),
                gaps=GAPS,
                gap_norm_div=GAP_NORM_DIV,
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
                "  base=%s unit=%s fusion=%s gaps=%s -> %s (%.1fs)",
                info["map_shape_base"],
                info["map_shape_unit"],
                info["fusion"],
                info["gaps"],
                info["fusion_npz"],
                info["elapsed_sec"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", video_id, exc)
            rows.append({"video_id": video_id, "error": str(exc)})

    summary = {
        "stage": 2,
        "variant": f"stage1maps_gapnorm_then_fusion_{args.fusion}_then_4x4max",
        "fusion": str(args.fusion),
        "stage2_gpu": not bool(args.no_gpu),
        "gaps": list(GAPS),
        "gap_norm_div": {str(k): float(v) for k, v in GAP_NORM_DIV.items()},
        "heat_vmin": float(HEAT_VMIN),
        "heat_vmax": float(HEAT_VMAX),
        "unit_block": int(args.unit_block),
        "block_size": int(args.block_size),
        "spatial_win": int(args.spatial_win),
        "temporal_radius": int(args.temporal_radius),
        "data_root": str(args.data_root.resolve()) if args.data_root else None,
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
        "Done: ok=%d failed=%d | outputs=%s",
        summary["num_ok"],
        summary["num_failed"],
        output_root,
    )
    return 0 if summary["num_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

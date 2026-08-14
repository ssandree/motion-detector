#!/usr/bin/env python3
"""Stage 1 — 5fps → Gap1 Farneback@1/4 → 8×8×1 R-mean → P15 → T5 @4×4 (16px).

Optical flow is Gap1 only. Official Stage1:
  v1 = Σ(R_ap · v_i) / Σ(R_ap) over 8×8 spatial (no temporal mix)
  keep = P15 ≥ 0.85
  U1 = T5(v1 × keep)

Default GPU path: upload gray once → keep ¼ → Farneback + Sobel R_ap + spatial
8×8 mean on GPU → download 16px grid → CPU P15 + T5. --no_gpu uses CPU 8×8×1.
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
    DEFAULT_DATA_ROOT,
    RESIZED_BASE_BLOCK,
    STAGE1_GAPS,
    STAGE1_SPATIAL_WIN,
    STAGE1_TEMPORAL_RADIUS,
    PipelineConfig,
    load_target_video_ids,
)
from motion_analyzer.motion_map import compute_video_base_motion  # noqa: E402

logger = logging.getLogger("1_base_motion")


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "outputs" / "stage1" / "1_base_motion" / stamp,
        help="Summary JSON root (NPZ under --data_root).",
    )
    parser.add_argument("--sampling_fps", type=float, default=5.0)
    parser.add_argument(
        "--block_size",
        type=int,
        default=RESIZED_BASE_BLOCK,
        help="Output stride / base-block size on 1/4-scale flow (default 4).",
    )
    parser.add_argument(
        "--spatial_win",
        type=int,
        default=STAGE1_SPATIAL_WIN,
        help="Spatial mean window in dense pixels (default 8).",
    )
    parser.add_argument(
        "--temporal_radius",
        type=int,
        default=STAGE1_TEMPORAL_RADIUS,
        help="Temporal radius R → frames [f−R, f+R] (default 2 → 5 frames).",
    )
    parser.add_argument("--video_id", type=str, default=None)
    parser.add_argument("--max_seconds", type=float, default=None)
    parser.add_argument("--video_list", type=Path, default=None)
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable Stage1 GPU keep/aggregate path (CPU 8×8×1 then P15+T5).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    if int(args.block_size) < 1:
        raise SystemExit("--block_size must be >= 1")
    if int(args.spatial_win) < int(args.block_size):
        raise SystemExit("--spatial_win must be >= --block_size")
    if int(args.temporal_radius) < 0:
        raise SystemExit("--temporal_radius must be >= 0")
    cfg = PipelineConfig(sampling_fps=float(args.sampling_fps))
    cfg.validate()

    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    video_ids = [args.video_id] if args.video_id else load_target_video_ids(args.video_list)
    rows = []
    for index, video_id in enumerate(video_ids, start=1):
        logger.info("[%d/%d] %s", index, len(video_ids), video_id)
        started = time.time()
        try:
            info = compute_video_base_motion(
                video_id,
                cfg=cfg,
                data_root=data_root,
                max_seconds=args.max_seconds,
                block_size=int(args.block_size),
                spatial_win=int(args.spatial_win),
                temporal_radius=int(args.temporal_radius),
                gaps=STAGE1_GAPS,
                use_gpu=not bool(args.no_gpu),
            )
            info["elapsed_sec"] = round(time.time() - started, 3)
            rows.append(info)
            logger.info(
                "  maps=%s shape=%s gaps=%s -> %s (%.1fs)",
                info["num_aligned_maps"],
                info["map_shape"],
                info["gaps"],
                info["base_motion_npz"],
                info["elapsed_sec"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", video_id, exc)
            rows.append({"video_id": video_id, "error": str(exc)})

    summary = {
        "stage": 1,
        "variant": "base_motion_gap1_8x8x1_P15_T5",
        "stage1_gpu": not bool(args.no_gpu),
        "gaps": list(STAGE1_GAPS),
        "block_size": int(args.block_size),
        "spatial_win": int(args.spatial_win),
        "temporal_radius": int(args.temporal_radius),
        "data_root": str(data_root),
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
        "Done: ok=%d failed=%d | data=%s",
        summary["num_ok"],
        summary["num_failed"],
        data_root,
    )
    return 0 if summary["num_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

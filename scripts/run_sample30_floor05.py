#!/usr/bin/env python3
"""sample30 rerun: Stage1 floor 0.5 + T5 zeros → Stage2 no floor → Stage3 UF.

Pipeline-only timing excludes NPZ writes and visualization (heat/MP4/3D).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motion_analyzer.opencv_cuda_bootstrap import (  # noqa: E402
    bootstrap_opencv_cuda,
    reload_cv2_if_needed,
)

bootstrap_opencv_cuda()
reload_cv2_if_needed()

from motion_analyzer.aggregation import fuse_video  # noqa: E402
from motion_analyzer.config import (  # noqa: E402
    DEFAULT_FUSION,
    DEFAULT_SPATIAL_AGG,
    ROI_MAX_GAP,
    ROI_TAU_HIGH,
    ROI_TAU_LOW,
    STAGE1_CELL_MAG_FLOOR,
    STAGE1_T5_ACTIVE_ONLY,
    STAGE2_CELL_MAG_FLOOR,
    PipelineConfig,
    load_target_video_ids,
)
from motion_analyzer.motion_map import compute_video_base_motion  # noqa: E402
from stage3 import process_video  # noqa: E402

logger = logging.getLogger("run_sample30_floor05")

SAMPLE30 = REPO_ROOT / "configs" / "virat_videos_le60s_sample30.txt"
STAGE1_NPZ = REPO_ROOT / "data" / "p15_t5_sample30_tanstrip_floor05"
STAGE2_NPZ = REPO_ROOT / "data" / "stage2" / "p15_t5_sample30_tanstrip_floor05"
STAGE1_SUM = REPO_ROOT / "outputs" / "stage1" / "p15_t5_sample30_tanstrip_floor05"
STAGE2_SUM = REPO_ROOT / "outputs" / "stage2" / "p15_t5_sample30_tanstrip_floor05"
STAGE3_ROOT = REPO_ROOT / "outputs" / "stage3" / "3_roi_tube" / "floor05_thr02_g10_c2"
STAGE3_3D = REPO_ROOT / "outputs" / "stage3" / "3_roi_tube_3dviz" / "floor05_thr02_g10_c2"
TIMING_PATH = REPO_ROOT / "outputs" / "p15_t5_sample30_tanstrip_floor05_timings.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_list", type=Path, default=SAMPLE30)
    p.add_argument("--video_id", type=str, nargs="*", default=None)
    p.add_argument("--skip-pipeline", action="store_true")
    p.add_argument("--no_gpu", action="store_true")
    return p.parse_args()


def _sum_ok(rows: list[dict], key: str = "pipeline_sec") -> float:
    return float(sum(float(r[key]) for r in rows if "error" not in r and key in r))




def run_pipeline(video_ids: list[str], *, use_gpu: bool) -> dict:
    cfg = PipelineConfig(sampling_fps=5.0, fusion=DEFAULT_FUSION)
    cfg.validate()
    for path in (
        STAGE1_NPZ,
        STAGE2_NPZ,
        STAGE1_SUM,
        STAGE2_SUM,
        STAGE3_ROOT,
        STAGE3_3D,
    ):
        path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Stage1 floor=%.3g t5_active_only=%s | Stage2 floor=%.3g fusion=%s spatial=%s",
        float(STAGE1_CELL_MAG_FLOOR),
        bool(STAGE1_T5_ACTIVE_ONLY),
        float(STAGE2_CELL_MAG_FLOOR),
        DEFAULT_FUSION,
        DEFAULT_SPATIAL_AGG,
    )
    if bool(STAGE1_T5_ACTIVE_ONLY) or float(STAGE1_CELL_MAG_FLOOR) != 0.5:
        raise SystemExit(
            "expected STAGE1_CELL_MAG_FLOOR=0.5 and STAGE1_T5_ACTIVE_ONLY=False"
        )
    if float(STAGE2_CELL_MAG_FLOOR) != 0.0:
        raise SystemExit("expected STAGE2_CELL_MAG_FLOOR=0")

    s1_rows: list[dict] = []
    for i, vid in enumerate(video_ids, start=1):
        logger.info("[stage1 %d/%d] %s", i, len(video_ids), vid)
        try:
            info = compute_video_base_motion(
                vid,
                cfg=cfg,
                data_root=STAGE1_NPZ,
                use_gpu=use_gpu,
            )
            s1_rows.append(
                {
                    "video_id": vid,
                    "pipeline_sec": float(info["pipeline_sec"]),
                    "map_shape": info["map_shape"],
                    "stage1_gpu": bool(info.get("stage1_gpu")),
                    "npz": info["base_motion_npz"],
                }
            )
            logger.info("  pipeline %.3fs gpu=%s", info["pipeline_sec"], info.get("stage1_gpu"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", vid, exc)
            s1_rows.append({"video_id": vid, "error": str(exc)})

    s2_rows: list[dict] = []
    for i, vid in enumerate(video_ids, start=1):
        logger.info("[stage2 %d/%d] %s", i, len(video_ids), vid)
        try:
            info = fuse_video(
                vid,
                cfg=cfg,
                output_root=STAGE2_NPZ,
                fusion=DEFAULT_FUSION,
                spatial_agg=DEFAULT_SPATIAL_AGG,
                data_root=STAGE1_NPZ,
            )
            s2_rows.append(
                {
                    "video_id": vid,
                    "pipeline_sec": float(info["pipeline_sec"]),
                    "map_shape_unit": info["map_shape_unit"],
                    "npz": info["fusion_npz"],
                }
            )
            logger.info("  pipeline %.3fs", info["pipeline_sec"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", vid, exc)
            s2_rows.append({"video_id": vid, "error": str(exc)})

    s3_rows: list[dict] = []
    for i, vid in enumerate(video_ids, start=1):
        logger.info("[stage3 %d/%d] %s", i, len(video_ids), vid)
        try:
            info = process_video(
                vid,
                cfg=cfg,
                fusion_root=STAGE2_NPZ,
                output_root=STAGE3_ROOT,
                fusion=DEFAULT_FUSION,
                tau_high=float(ROI_TAU_HIGH),
                tau_low=float(ROI_TAU_LOW),
                max_gap=int(ROI_MAX_GAP),
                write_3d=True,
                viz_3d_root=STAGE3_3D,
            )
            s3_rows.append(
                {
                    "video_id": vid,
                    "pipeline_sec": float(info["pipeline_sec"]),
                    "num_tracks": info["num_tracks"],
                    "num_tubes": info.get("num_tubes", info["num_tracks"]),
                    "mp4": info["visualization_mp4"],
                    "fig_3d": info.get("visualization_3d"),
                }
            )
            logger.info(
                "  pipeline %.3fs tubes=%s",
                info["pipeline_sec"],
                info.get("num_tubes", info["num_tracks"]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", vid, exc)
            s3_rows.append({"video_id": vid, "error": str(exc)})

    totals = {
        "stage1_sec": round(_sum_ok(s1_rows), 6),
        "stage2_sec": round(_sum_ok(s2_rows), 6),
        "stage3_sec": round(_sum_ok(s3_rows), 6),
    }
    totals["all_stages_sec"] = round(sum(totals.values()), 6)
    payload = {
        "note": (
            "pipeline_sec excludes NPZ save, video decode used only for viz "
            "metadata (Stage2 sample after compute), and MP4/3D/heat overlay. "
            "Stage3 writes official coarse-group → conservative GraphCut → ROI tubes."
        ),
        "stage1": {
            "cell_mag_floor": float(STAGE1_CELL_MAG_FLOOR),
            "t5_active_only": bool(STAGE1_T5_ACTIVE_ONLY),
            "npz_root": str(STAGE1_NPZ),
        },
        "stage2": {
            "cell_mag_floor": float(STAGE2_CELL_MAG_FLOOR),
            "fusion": DEFAULT_FUSION,
            "spatial_agg": DEFAULT_SPATIAL_AGG,
            "npz_root": str(STAGE2_NPZ),
        },
        "stage3": {
            "variant": "coarse_gc_compose_v2c",
            "thr": float(ROI_TAU_HIGH),
            "tau_high": float(ROI_TAU_HIGH),
            "tau_low": float(ROI_TAU_LOW),
            "max_gap": int(ROI_MAX_GAP),
            "output_root": str(STAGE3_ROOT),
        },
        "totals": totals,
        "num_videos": len(video_ids),
        "videos": {
            "stage1": s1_rows,
            "stage2": s2_rows,
            "stage3": s3_rows,
        },
    }
    TIMING_PATH.parent.mkdir(parents=True, exist_ok=True)
    TIMING_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (STAGE1_SUM / "summary.json").write_text(
        json.dumps({"videos": s1_rows, "totals": totals}, indent=2) + "\n",
        encoding="utf-8",
    )
    (STAGE2_SUM / "summary.json").write_text(
        json.dumps({"videos": s2_rows}, indent=2) + "\n", encoding="utf-8"
    )
    (STAGE3_ROOT / "summary.json").write_text(
        json.dumps({"videos": s3_rows}, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "pipeline totals: s1=%.3fs s2=%.3fs s3=%.3fs all=%.3fs → %s",
        totals["stage1_sec"],
        totals["stage2_sec"],
        totals["stage3_sec"],
        totals["all_stages_sec"],
        TIMING_PATH,
    )
    return payload


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    video_ids = list(args.video_id) if args.video_id else load_target_video_ids(args.video_list)
    if not args.skip_pipeline:
        run_pipeline(video_ids, use_gpu=not bool(args.no_gpu))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

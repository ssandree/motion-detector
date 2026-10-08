#!/usr/bin/env python3
"""Run Stage1→2→3 on 489dosa frame-sequence clips."""

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
    PipelineConfig,
    load_target_video_ids,
)
from motion_analyzer.motion_map import compute_video_base_motion  # noqa: E402
from stage3 import process_video  # noqa: E402

logger = logging.getLogger("run_489dosa_clips")

DATASET_ROOT = Path("/data/datasets/489dosa")
CLIP_LIST = REPO_ROOT / "configs" / "489dosa_clips.txt"
TAG = "489dosa_p13_t5"
STAGE1_NPZ = REPO_ROOT / "data" / "stage1" / TAG
STAGE2_NPZ = REPO_ROOT / "data" / "stage2" / TAG
STAGE3_ROOT = REPO_ROOT / "outputs" / "stage3" / TAG / "roi_tube"
STAGE3_3D = REPO_ROOT / "outputs" / "stage3" / TAG / "roi_tube_3dviz"
SUMMARY_PATH = REPO_ROOT / "outputs" / "stage3" / TAG / "summary.json"
# Frames are already sampled at 3 fps in meta.json.
SAMPLING_FPS = 3.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_list", type=Path, default=CLIP_LIST)
    p.add_argument("--video_id", type=str, nargs="*", default=None)
    p.add_argument("--skip-stage1", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--skip-stage3", action="store_true")
    p.add_argument("--no_gpu", action="store_true")
    return p.parse_args()


def _cfg() -> PipelineConfig:
    cfg = PipelineConfig(
        sampling_fps=SAMPLING_FPS,
        video_search_roots=[DATASET_ROOT],
        fusion=DEFAULT_FUSION,
    )
    cfg.validate()
    return cfg


def run_stage1(video_ids: list[str], *, use_gpu: bool) -> list[dict]:
    cfg = _cfg()
    rows: list[dict] = []
    STAGE1_NPZ.mkdir(parents=True, exist_ok=True)
    for i, vid in enumerate(video_ids, start=1):
        logger.info("[stage1 %d/%d] %s", i, len(video_ids), vid)
        try:
            info = compute_video_base_motion(
                vid,
                cfg=cfg,
                data_root=STAGE1_NPZ,
                use_gpu=use_gpu,
            )
            rows.append(
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
            rows.append({"video_id": vid, "error": str(exc)})
    return rows


def run_stage2(video_ids: list[str]) -> list[dict]:
    cfg = _cfg()
    rows: list[dict] = []
    STAGE2_NPZ.mkdir(parents=True, exist_ok=True)
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
            rows.append(
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
            rows.append({"video_id": vid, "error": str(exc)})
    return rows


def run_stage3(video_ids: list[str]) -> list[dict]:
    cfg = _cfg()
    rows: list[dict] = []
    STAGE3_ROOT.mkdir(parents=True, exist_ok=True)
    STAGE3_3D.mkdir(parents=True, exist_ok=True)
    for i, vid in enumerate(video_ids, start=1):
        logger.info("[stage3 %d/%d] %s", i, len(video_ids), vid)
        try:
            info = process_video(
                vid,
                cfg=cfg,
                fusion_root=STAGE2_NPZ,
                output_root=STAGE3_ROOT,
                fusion=DEFAULT_FUSION,
                write_3d=True,
                viz_3d_root=STAGE3_3D,
            )
            rows.append(
                {
                    "video_id": vid,
                    "pipeline_sec": float(info["pipeline_sec"]),
                    "num_tracks": info["num_tracks"],
                    "mp4": info.get("visualization_mp4"),
                    "png_3d": info.get("visualization_3d"),
                }
            )
            logger.info(
                "  pipeline %.3fs tracks=%s mp4=%s",
                info["pipeline_sec"],
                info["num_tracks"],
                info.get("visualization_mp4"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", vid, exc)
            rows.append({"video_id": vid, "error": str(exc)})
    return rows


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    video_ids = list(args.video_id) if args.video_id else load_target_video_ids(args.video_list)
    if not video_ids:
        raise SystemExit("no video ids")

    use_gpu = not bool(args.no_gpu)
    logger.info(
        "489dosa clips=%d sampling_fps=%.1f gpu=%s dataset=%s",
        len(video_ids),
        SAMPLING_FPS,
        use_gpu,
        DATASET_ROOT,
    )

    payload: dict = {"video_ids": video_ids, "sampling_fps": SAMPLING_FPS}
    if not args.skip_stage1:
        payload["stage1"] = run_stage1(video_ids, use_gpu=use_gpu)

    if not args.skip_stage2:
        payload["stage2"] = run_stage2(video_ids)

    if not args.skip_stage3:
        payload["stage3"] = run_stage3(video_ids)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("summary → %s", SUMMARY_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

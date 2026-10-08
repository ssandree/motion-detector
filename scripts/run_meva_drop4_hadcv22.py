#!/usr/bin/env python3
"""Run Stage1→2→3 on all MEVA drop-4-hadcv22 AVI clips (~201 × ~5 min @ 5 fps)."""

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

logger = logging.getLogger("run_meva_drop4_hadcv22")

MEVA_ROOT = Path("/data/datasets/MEVA/drop-4-hadcv22")
VIDEO_LIST = REPO_ROOT / "configs" / "meva_drop4_hadcv22_all.txt"
TAG = "meva_drop4_hadcv22"
STAGE1_NPZ = REPO_ROOT / "data" / "cache" / TAG
STAGE2_NPZ = REPO_ROOT / "data" / "stage2" / "2_gap_fusion" / TAG
STAGE1_SUM = REPO_ROOT / "outputs" / "stage1" / "1_base_motion" / TAG
STAGE2_SUM = REPO_ROOT / "outputs" / "stage2" / "2_gap_fusion" / TAG
STAGE3_ROOT = REPO_ROOT / "outputs" / "stage3" / "3_roi_tube" / TAG
STAGE3_3D = REPO_ROOT / "outputs" / "stage3" / "3_roi_tube_3dviz" / TAG
SUMMARY_PATH = REPO_ROOT / "outputs" / "stage3" / TAG / "run_summary.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", type=Path, default=MEVA_ROOT)
    p.add_argument("--video_list", type=Path, default=VIDEO_LIST)
    p.add_argument(
        "--refresh-list",
        action="store_true",
        help="Rescan dataset_root for *.avi and rewrite --video_list.",
    )
    p.add_argument("--video_id", type=str, nargs="*", default=None)
    p.add_argument("--skip-stage1", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--skip-stage3", action="store_true")
    p.add_argument("--no_gpu", action="store_true", help="Stage1 CPU path.")
    p.add_argument("--no-3d", action="store_true", help="Skip Stage3 3D PNG figures.")
    return p.parse_args()


def discover_video_ids(root: Path) -> list[str]:
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")
    ids = sorted(p.stem for p in root.rglob("*.avi"))
    if not ids:
        raise SystemExit(f"no .avi files under {root}")
    return ids


def ensure_video_list(root: Path, list_path: Path, *, refresh: bool) -> list[str]:
    ids = discover_video_ids(root)
    if refresh or not list_path.is_file():
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        logger.info("wrote %d video ids → %s", len(ids), list_path)
    loaded = load_target_video_ids(list_path)
    if len(loaded) != len(ids):
        logger.warning(
            "video list count %d != scan count %d (use --refresh-list to sync)",
            len(loaded),
            len(ids),
        )
    return loaded


def _cfg(dataset_root: Path) -> PipelineConfig:
    cfg = PipelineConfig(
        sampling_fps=5.0,
        video_search_roots=[dataset_root.resolve()],
        fusion=DEFAULT_FUSION,
    )
    cfg.validate()
    return cfg


def _count_ok(rows: list[dict]) -> int:
    return sum(1 for r in rows if "error" not in r)


def run_stage1(
    video_ids: list[str],
    *,
    dataset_root: Path,
    use_gpu: bool,
) -> list[dict]:
    cfg = _cfg(dataset_root)
    rows: list[dict] = []
    STAGE1_NPZ.mkdir(parents=True, exist_ok=True)
    STAGE1_SUM.mkdir(parents=True, exist_ok=True)
    for i, vid in enumerate(video_ids, start=1):
        logger.info("[stage1 %d/%d] %s", i, len(video_ids), vid)
        try:
            info = compute_video_base_motion(
                vid,
                cfg=cfg,
                data_root=STAGE1_NPZ,
                use_gpu=use_gpu,
            )
            rows.append(info)
            logger.info(
                "  maps=%s pipeline %.3fs wall %.3fs gpu=%s",
                info.get("map_shape"),
                float(info.get("pipeline_sec") or 0.0),
                float(info.get("elapsed_sec") or 0.0),
                info.get("stage1_gpu"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", vid, exc)
            rows.append({"video_id": vid, "error": str(exc)})
    return rows


def run_stage2(video_ids: list[str], *, dataset_root: Path) -> list[dict]:
    cfg = _cfg(dataset_root)
    rows: list[dict] = []
    STAGE2_NPZ.mkdir(parents=True, exist_ok=True)
    STAGE2_SUM.mkdir(parents=True, exist_ok=True)
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
            rows.append(info)
            logger.info(
                "  unit=%s pipeline %.3fs wall %.3fs",
                info.get("map_shape_unit"),
                float(info.get("pipeline_sec") or 0.0),
                float(info.get("elapsed_sec") or 0.0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", vid, exc)
            rows.append({"video_id": vid, "error": str(exc)})
    return rows


def run_stage3(
    video_ids: list[str],
    *,
    dataset_root: Path,
    write_3d: bool,
) -> list[dict]:
    cfg = _cfg(dataset_root)
    rows: list[dict] = []
    STAGE3_ROOT.mkdir(parents=True, exist_ok=True)
    if write_3d:
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
                write_3d=write_3d,
                viz_3d_root=None if not write_3d else STAGE3_3D,
            )
            rows.append(info)
            logger.info(
                "  tubes=%s pipeline %.3fs wall %.3fs → %s",
                info.get("num_tubes", info.get("num_tracks")),
                float(info.get("pipeline_sec") or 0.0),
                float(info.get("elapsed_sec") or 0.0),
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
    dataset_root = args.dataset_root.expanduser().resolve()

    if args.video_id:
        video_ids = list(args.video_id)
    else:
        video_ids = ensure_video_list(
            dataset_root,
            args.video_list.expanduser().resolve(),
            refresh=bool(args.refresh_list),
        )
    if not video_ids:
        raise SystemExit("no video ids")

    use_gpu = not bool(args.no_gpu)
    write_3d = not bool(args.no_3d)
    logger.info(
        "MEVA clips=%d sampling_fps=5.0 gpu=%s write_3d=%s dataset=%s",
        len(video_ids),
        use_gpu,
        write_3d,
        dataset_root,
    )
    logger.info(
        "artifacts: stage1=%s stage2=%s stage3=%s",
        STAGE1_NPZ,
        STAGE2_NPZ,
        STAGE3_ROOT,
    )

    payload: dict = {
        "dataset_root": str(dataset_root),
        "video_list": str(args.video_list),
        "num_videos": len(video_ids),
        "sampling_fps": 5.0,
        "stage1_npz_root": str(STAGE1_NPZ),
        "stage2_npz_root": str(STAGE2_NPZ),
        "stage3_output_root": str(STAGE3_ROOT),
    }

    if not args.skip_stage1:
        payload["stage1"] = run_stage1(
            video_ids, dataset_root=dataset_root, use_gpu=use_gpu
        )
        payload["stage1_ok"] = _count_ok(payload["stage1"])

    if not args.skip_stage2:
        payload["stage2"] = run_stage2(video_ids, dataset_root=dataset_root)
        payload["stage2_ok"] = _count_ok(payload["stage2"])

    if not args.skip_stage3:
        payload["stage3"] = run_stage3(
            video_ids, dataset_root=dataset_root, write_3d=write_3d
        )
        payload["stage3_ok"] = _count_ok(payload["stage3"])

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("run summary → %s", SUMMARY_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

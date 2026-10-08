#!/usr/bin/env python3
"""Stage1→2→3 in memory. Writes ``<video_id>_roi_tracks.json`` only.

No Stage1/Stage2 NPZ, no overlay MP4, no 3D PNG.
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

from motion_analyzer.config import PipelineConfig, load_target_video_ids  # noqa: E402
from motion_analyzer.direct_roi import run_video_roi_tracks  # noqa: E402

logger = logging.getLogger("run_roi_tracks_direct")

MEVA_ROOT = Path("/data/datasets/MEVA/drop-4-hadcv22")
VIDEO_LIST = REPO_ROOT / "configs" / "meva_drop4_hadcv22_all.txt"
DEFAULT_OUT = REPO_ROOT / "outputs" / "stage3" / "3_roi_tube_direct" / "meva_drop4_hadcv22"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", type=Path, default=MEVA_ROOT)
    p.add_argument("--video_list", type=Path, default=VIDEO_LIST)
    p.add_argument("--video_id", type=str, nargs="*", default=None)
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no_gpu", action="store_true", help="Stage1 CPU path.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a clip when its _roi_tracks.json is already present.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.video_id:
        video_ids = list(args.video_id)
    else:
        video_ids = load_target_video_ids(args.video_list.expanduser().resolve())
    if not video_ids:
        raise SystemExit("no video ids")

    cfg = PipelineConfig(
        sampling_fps=5.0,
        video_search_roots=[dataset_root],
    )
    cfg.validate()
    use_gpu = not bool(args.no_gpu)
    logger.info(
        "direct ROI clips=%d sampling_fps=5.0 gpu=%s out=%s",
        len(video_ids),
        use_gpu,
        output_root,
    )

    rows: list[dict] = []
    n_ok = 0
    for i, vid in enumerate(video_ids, start=1):
        dest = output_root / f"{vid}_roi_tracks.json"
        if args.skip_existing and dest.is_file():
            logger.info("[%d/%d] %s skip existing", i, len(video_ids), vid)
            rows.append({"video_id": vid, "tracks_json": str(dest), "skipped": True})
            n_ok += 1
            continue
        logger.info("[%d/%d] %s", i, len(video_ids), vid)
        try:
            info = run_video_roi_tracks(
                vid,
                cfg=cfg,
                output_root=output_root,
                use_gpu=use_gpu,
            )
            rows.append(info)
            n_ok += 1
            logger.info(
                "  rois=%s events=%s stage1 %.3fs stage2 %.3fs stage3 %.3fs gpu=%s",
                info["num_final_rois"],
                info["num_block_events"],
                info["stage1_sec"],
                info["stage2_sec"],
                info["stage3_sec"],
                info["stage1_gpu"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", vid, exc)
            rows.append({"video_id": vid, "error": str(exc)})

    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "num_videos": len(video_ids),
        "num_ok": n_ok,
        "sampling_fps": 5.0,
        "npz": False,
        "visualization": False,
        "videos": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("run summary → %s (%d/%d ok)", summary_path, n_ok, len(video_ids))
    return 0 if n_ok == len(video_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())

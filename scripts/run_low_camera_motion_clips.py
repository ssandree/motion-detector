#!/usr/bin/env python3
"""Stage1→2→3 + VLM frame export for /data/datasets/low_camera_motion_clips.

Resumes existing NPZ / tracks / JPEGs. Re-scans the dataset after a quiet period
so clips that arrive while copying are included.

CUDA Farneback requires the local OpenCV CUDA build (Python 3.14)::

  source ~/.local/opencv-cuda/env.sh
  PYTHONPATH=src python3.14 scripts/run_low_camera_motion_clips.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motion_analyzer.opencv_cuda_bootstrap import (  # noqa: E402
    bootstrap_opencv_cuda,
    reload_cv2_if_needed,
)

bootstrap_opencv_cuda()
reload_cv2_if_needed()

from motion_analyzer.aggregation import fuse_video, fusion_npz_path  # noqa: E402
from motion_analyzer.config import (  # noqa: E402
    DEFAULT_FUSION,
    DEFAULT_SPATIAL_AGG,
    PipelineConfig,
)
from motion_analyzer.motion_map import (  # noqa: E402
    base_motion_npz_path,
    compute_video_base_motion,
    gaps_tag,
)
from motion_analyzer.config import STAGE1_GAPS  # noqa: E402
from stage3 import process_video  # noqa: E402

logger = logging.getLogger("run_low_camera_motion_clips")

DATASET_ROOT = Path("/data/datasets/low_camera_motion_clips")
TAG = "low_camera_motion"
STAGE1_NPZ = REPO_ROOT / "data" / "stage1" / TAG
STAGE2_NPZ = REPO_ROOT / "data" / "stage2" / TAG
STAGE3_ROOT = REPO_ROOT / "outputs" / "harrypotter_low_camera_motion"
SUMMARY_PATH = STAGE3_ROOT / "pipeline_summary.json"
CLIP_LIST = REPO_ROOT / "configs" / "low_camera_motion_clips.txt"
VLM_READY = STAGE3_ROOT / "vlm_ready_videos.txt"

VLM_DATA = Path("/data/datasets/low_camera_motion_vlm")
VIDEO_OUT = VLM_DATA / "videos"
FPS_ROOT = VLM_DATA / "fps_sampling_frames"
ROI_ROOT = VLM_DATA / "proposed_ROI"
COMBO_ROOT = VLM_DATA / "proposed_ROI_w_fps"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    p.add_argument("--video_id", type=str, nargs="*", default=None)
    p.add_argument("--video_list", type=Path, default=None)
    p.add_argument("--wait_stable_sec", type=float, default=45.0)
    p.add_argument("--wait_stable_max_sec", type=float, default=1200.0)
    p.add_argument("--skip-stage1", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--skip-stage3", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--overwrite-export", action="store_true")
    return p.parse_args()


def _cfg(dataset_root: Path) -> PipelineConfig:
    cfg = PipelineConfig(
        sampling_fps=5.0,
        video_search_roots=[Path(dataset_root).resolve()],
        fusion=DEFAULT_FUSION,
    )
    cfg.validate()
    return cfg


def scan_clips(dataset_root: Path) -> list[tuple[str, Path]]:
    root = Path(dataset_root).resolve()
    clips = sorted(root.rglob("*.mp4"))
    rows: list[tuple[str, Path]] = []
    seen: dict[str, Path] = {}
    for path in clips:
        vid = path.stem
        prev = seen.get(vid)
        if prev is not None and prev.resolve() != path.resolve():
            raise SystemExit(f"duplicate video_id {vid}: {prev} vs {path}")
        seen[vid] = path
        rows.append((vid, path))
    return rows


def wait_until_stable(
    dataset_root: Path, *, quiet_sec: float, max_sec: float
) -> list[tuple[str, Path]]:
    if quiet_sec <= 0:
        return scan_clips(dataset_root)
    logger.info(
        "Waiting until %s is unchanged for %.0fs (max %.0fs)",
        dataset_root,
        quiet_sec,
        max_sec,
    )
    started = time.time()
    last = scan_clips(dataset_root)
    last_change = time.time()
    last_n = len(last)
    while True:
        time.sleep(min(10.0, max(quiet_sec, 1.0)))
        now = scan_clips(dataset_root)
        if len(now) != last_n or [a[0] for a in now] != [a[0] for a in last]:
            logger.info("dataset grew/changed: %d -> %d clips", last_n, len(now))
            last = now
            last_n = len(now)
            last_change = time.time()
        elapsed_quiet = time.time() - last_change
        elapsed_all = time.time() - started
        if elapsed_quiet >= quiet_sec:
            logger.info("dataset stable at %d clips", last_n)
            return last
        if elapsed_all >= max_sec:
            logger.warning(
                "wait_stable timed out after %.0fs with %d clips; continuing",
                elapsed_all,
                last_n,
            )
            return last


def write_clip_list(path: Path, video_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{vid}\n" for vid in video_ids), encoding="utf-8")


def _ok_npz(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def run_stage1(video_ids: list[str], *, cfg: PipelineConfig, use_gpu: bool) -> list[dict]:
    STAGE1_NPZ.mkdir(parents=True, exist_ok=True)
    tag = gaps_tag(STAGE1_GAPS)
    rows: list[dict] = []
    for i, vid in enumerate(video_ids, start=1):
        npz = base_motion_npz_path(STAGE1_NPZ, vid, tag=tag)
        if _ok_npz(npz):
            logger.info("[stage1 %d/%d] %s skip existing %s", i, len(video_ids), vid, npz.name)
            rows.append({"video_id": vid, "npz": str(npz), "skipped": True})
            continue
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
                    "stage1_gpu": bool(info.get("stage1_gpu")),
                    "npz": info["base_motion_npz"],
                }
            )
            logger.info(
                "  pipeline %.3fs gpu=%s",
                info["pipeline_sec"],
                info.get("stage1_gpu"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", vid, exc)
            rows.append({"video_id": vid, "error": str(exc)})
    return rows


def run_stage2(video_ids: list[str], *, cfg: PipelineConfig) -> list[dict]:
    STAGE2_NPZ.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, vid in enumerate(video_ids, start=1):
        npz = fusion_npz_path(STAGE2_NPZ, vid, fusion=DEFAULT_FUSION)
        if _ok_npz(npz):
            logger.info("[stage2 %d/%d] %s skip existing %s", i, len(video_ids), vid, npz.name)
            rows.append({"video_id": vid, "npz": str(npz), "skipped": True})
            continue
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
                    "npz": info["fusion_npz"],
                }
            )
            logger.info("  pipeline %.3fs", info["pipeline_sec"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", vid, exc)
            rows.append({"video_id": vid, "error": str(exc)})
    return rows


def run_stage3(video_ids: list[str], *, cfg: PipelineConfig) -> list[dict]:
    STAGE3_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, vid in enumerate(video_ids, start=1):
        tracks = STAGE3_ROOT / f"{vid}_roi_tracks.json"
        if tracks.is_file() and tracks.stat().st_size > 0:
            logger.info("[stage3 %d/%d] %s skip existing tracks", i, len(video_ids), vid)
            try:
                payload = json.loads(tracks.read_text(encoding="utf-8"))
                n_tubes = int(
                    payload.get("num_final_rois")
                    or len(payload.get("tracks") or payload.get("tubes") or [])
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                n_tubes = None
            rows.append(
                {
                    "video_id": vid,
                    "tracks_json": str(tracks),
                    "num_tubes": n_tubes,
                    "skipped": True,
                }
            )
            continue
        logger.info("[stage3 %d/%d] %s", i, len(video_ids), vid)
        try:
            info = process_video(
                vid,
                cfg=cfg,
                fusion_root=STAGE2_NPZ,
                output_root=STAGE3_ROOT,
                fusion=DEFAULT_FUSION,
                write_3d=False,
                write_overlay=False,
            )
            rows.append(
                {
                    "video_id": vid,
                    "pipeline_sec": float(info["pipeline_sec"]),
                    "num_tracks": info["num_tracks"],
                    "num_tubes": info.get("num_tubes", info["num_tracks"]),
                    "tracks_json": info.get("tracks_json"),
                }
            )
            logger.info(
                "  pipeline %.3fs tubes=%s",
                info["pipeline_sec"],
                info.get("num_tubes", info["num_tracks"]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("  FAILED %s: %s", vid, exc)
            rows.append({"video_id": vid, "error": str(exc)})
    return rows


def run_export(video_ids: list[str], *, dataset_root: Path, overwrite: bool) -> int:
    py = Path(sys.executable)
    cmd = [
        str(py),
        str(REPO_ROOT / "scripts" / "export_vlm_sampling.py"),
        "--video_list",
        str(CLIP_LIST),
        "--fusion_root",
        str(STAGE2_NPZ),
        "--tracks_root",
        str(STAGE3_ROOT),
        "--video_out",
        str(VIDEO_OUT),
        "--fps_root",
        str(FPS_ROOT),
        "--roi_root",
        str(ROI_ROOT),
        "--combo_root",
        str(COMBO_ROOT),
        "--video_search_root",
        str(Path(dataset_root).resolve()),
    ]
    if overwrite:
        cmd.append("--overwrite")
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}:{env['PYTHONPATH']}"
    logger.info("export: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO_ROOT), env=env)


def _failed_ids(rows: list[dict]) -> set[str]:
    return {str(r["video_id"]) for r in rows if "error" in r}


def write_vlm_ready(export_summary: Path) -> list[str]:
    if not export_summary.is_file():
        logger.warning("export summary missing: %s", export_summary)
        return []
    payload = json.loads(export_summary.read_text(encoding="utf-8"))
    ready: list[str] = []
    for row in payload.get("videos") or []:
        vid = str(row.get("video_id") or "")
        n_tubes = int(row.get("num_roi_tubes") or 0)
        if vid and n_tubes > 0:
            ready.append(vid)
    VLM_READY.parent.mkdir(parents=True, exist_ok=True)
    write_clip_list(VLM_READY, ready)
    logger.info("VLM-ready videos (ROI tubes > 0): %d -> %s", len(ready), VLM_READY)
    return ready


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    if args.video_id:
        clips = [(vid, Path()) for vid in args.video_id]
    elif args.video_list:
        ids = [
            line.strip()
            for line in args.video_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        clips = [(vid, Path()) for vid in ids]
    else:
        clips = wait_until_stable(
            dataset_root,
            quiet_sec=float(args.wait_stable_sec),
            max_sec=float(args.wait_stable_max_sec),
        )

    video_ids = [vid for vid, _ in clips]
    if not video_ids:
        raise SystemExit(f"no mp4 clips under {dataset_root}")
    write_clip_list(CLIP_LIST, video_ids)
    logger.info("processing %d clips", len(video_ids))

    cfg = _cfg(dataset_root)
    payload: dict = {
        "dataset_root": str(dataset_root),
        "num_videos": len(video_ids),
        "video_ids": video_ids,
        "stage1_npz": str(STAGE1_NPZ),
        "stage2_npz": str(STAGE2_NPZ),
        "stage3_root": str(STAGE3_ROOT),
        "vlm_data": str(VLM_DATA),
    }

    s1: list[dict] = []
    s2: list[dict] = []
    s3: list[dict] = []
    if not args.skip_stage1:
        s1 = run_stage1(video_ids, cfg=cfg, use_gpu=not bool(args.no_gpu))
        payload["stage1"] = s1
    fail = _failed_ids(s1) if s1 else set()
    s2_ids = [v for v in video_ids if v not in fail]
    if not args.skip_stage2:
        s2 = run_stage2(s2_ids, cfg=cfg)
        payload["stage2"] = s2
        fail |= _failed_ids(s2)
    s3_ids = [v for v in video_ids if v not in fail]
    if not args.skip_stage3:
        s3 = run_stage3(s3_ids, cfg=cfg)
        payload["stage3"] = s3
        fail |= _failed_ids(s3)

    export_ids = [v for v in video_ids if v not in fail]
    write_clip_list(CLIP_LIST, export_ids)
    export_rc = 0
    if not args.skip_export:
        export_rc = run_export(
            export_ids, dataset_root=dataset_root, overwrite=bool(args.overwrite_export)
        )
        payload["export_returncode"] = export_rc
        ready = write_vlm_ready(STAGE3_ROOT / "vlm_sampling_summary.json")
        payload["vlm_ready_videos"] = ready
        payload["num_vlm_ready"] = len(ready)

    payload["num_failed"] = len(fail)
    payload["failed_video_ids"] = sorted(fail)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("pipeline summary -> %s", SUMMARY_PATH)
    if fail:
        logger.warning("failed clips: %d", len(fail))
    if export_rc != 0:
        return export_rc
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())

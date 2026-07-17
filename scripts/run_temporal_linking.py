#!/usr/bin/env python3
"""Temporal linking over existing frame-level Motion Blobs (192×192 / factor=12).

Does NOT recompute Farneback, 16×16 features, 12×12 RMS aggregation, MAD
threshold, or 8-CC. Loads components.csv (or equivalent) and links blobs.

Outputs (per video folder):
  temporal_tracks.csv
  temporal_tracks.json
  temporal_linking_summary.json
  temporal_linking_overlay.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from motion_analyzer.temporal.temporal_linking import (  # noqa: E402
    TemporalLinkingConfig,
    assign_timeline_lanes,
    build_overlay_index,
    compose_frame_with_timeline,
    infer_sample_fps_from_blobs,
    link_blobs,
    load_blobs_from_components_csv,
    render_temporal_overlay_frame,
    save_temporal_linking_outputs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_temporal_linking")

DEFAULT_COMPONENTS_ROOT = Path(
    "/data1/vailab02_dir/vlm_motion/motion-detector/Result/"
    "residual_motion_120s_block16_rms_aggregation_3x2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data1/vailab02_dir/vlm_motion/motion-detector/Result/"
    "residual_motion_120s_block16_4video"
)
DEFAULT_FACTOR = 12
DEFAULT_PIXEL_SIZE = 192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--components_root", type=Path, default=DEFAULT_COMPONENTS_ROOT)
    p.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Per-video results folder root (default: residual_motion_120s_block16_4video).",
    )
    p.add_argument("--video_names", nargs="+", default=None)
    p.add_argument("--factor", type=int, default=DEFAULT_FACTOR)
    p.add_argument("--pixel_size", type=int, default=DEFAULT_PIXEL_SIZE)
    p.add_argument("--max_gap", type=int, default=1)
    p.add_argument("--min_track_length", type=int, default=2)
    p.add_argument("--center_distance_threshold", type=float, default=1.5)
    p.add_argument("--motion_difference_threshold", type=float, default=0.85)
    p.add_argument("--expand_margin_px", type=float, default=float(DEFAULT_PIXEL_SIZE))
    p.add_argument("--expand_prop", type=float, default=0.5)
    p.add_argument("--trail_length", type=int, default=12)
    p.add_argument("--skip_overlay", action="store_true")
    return p.parse_args()


def discover_videos(components_root: Path, names: list[str] | None) -> list[str]:
    if names:
        return list(names)
    found: list[str] = []
    for d in sorted(components_root.iterdir()):
        if d.is_dir() and (d / "components.csv").is_file():
            found.append(d.name)
    return found


def load_video_meta(video_dir: Path) -> dict[str, Any]:
    cfg_path = video_dir / "config.json"
    if not cfg_path.is_file():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def frame_size_from_meta(meta: dict[str, Any], video_path: Path | None) -> tuple[int, int]:
    insp = meta.get("cache_inspection") or {}
    w = insp.get("frame_width")
    h = insp.get("frame_height")
    if w and h:
        return int(w), int(h)
    if video_path is not None and video_path.is_file():
        cap = cv2.VideoCapture(str(video_path))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if fw > 0 and fh > 0:
            return fw, fh
    return 1280, 720


def load_sample_frame_pairs(video_dir: Path, blobs: list[Any]) -> list[tuple[int, int]]:
    """(frame_index, frame_idx) pairs covering the sampled timeline."""
    summary_path = video_dir / "summary_12x12.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        frames = summary.get("frames") or []
        pairs = [
            (int(fr["frame_index"]), int(fr["frame_idx"])) for fr in frames
        ]
        if pairs:
            return pairs

    by_frame_index: dict[int, int] = {}
    for b in blobs:
        by_frame_index.setdefault(int(b.frame_index), int(b.frame_idx))
    return sorted(by_frame_index.items(), key=lambda x: x[0])


def write_overlay(
    *,
    video_path: Path,
    tracks: list[Any],
    blobs: list[Any],
    frame_pairs: list[tuple[int, int]],
    output_mp4: Path,
    trail_length: int,
) -> dict[str, Any]:
    if not tracks:
        logger.warning("No kept tracks for overlay: %s", output_mp4.name)

    if not frame_pairs:
        pairs: dict[int, int] = {}
        for tr in tracks:
            for obs in tr.observations:
                pairs[obs.blob.frame_index] = obs.blob.frame_idx
        frame_pairs = sorted(pairs.items(), key=lambda x: x[0])

    sample_fps = infer_sample_fps_from_blobs(blobs, fallback=5.0)
    overlay_index = build_overlay_index(tracks)

    if frame_pairs:
        t_min = int(frame_pairs[0][0])
        t_max = int(frame_pairs[-1][0])
    else:
        t_min, t_max = 0, 1
    if tracks:
        t_min = min(t_min, min(tr.start_frame_index for tr in tracks))
        t_max = max(t_max, max(tr.end_frame_index for tr in tracks))

    n_lanes = max((lane for _, lane in assign_timeline_lanes(tracks)), default=0) + 1
    timeline_h = 22 + 10 * 2 + max(n_lanes, 1) * 22 + 4

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ok0, probe = cap.read()
    if not ok0 or probe is None:
        cap.release()
        raise RuntimeError(f"Failed to read first frame: {video_path}")
    frame_h, frame_w = probe.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_w, out_h = frame_w, frame_h + timeline_h
    writer = cv2.VideoWriter(
        str(output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(sample_fps), 0.1),
        (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {output_mp4}")

    n_written = 0
    for frame_index, frame_idx in frame_pairs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        elif frame.shape[0] != frame_h or frame.shape[1] != frame_w:
            frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

        entries = overlay_index.get(int(frame_index), [])
        vis = render_temporal_overlay_frame(
            frame, entries, trail_length=trail_length
        )
        n_box = len(entries)
        hud = f"fi={frame_index} frame_idx={frame_idx} tracks={n_box}"
        cv2.putText(
            vis,
            hud,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        composed = compose_frame_with_timeline(
            vis,
            tracks,
            current_frame_index=int(frame_index),
            t_min=t_min,
            t_max=t_max,
            timeline_height=timeline_h,
        )
        writer.write(composed)
        n_written += 1

    cap.release()
    writer.release()
    return {
        "overlay_mp4": str(output_mp4),
        "overlay_frames": n_written,
        "overlay_fps": float(sample_fps),
        "frame_width": out_w,
        "frame_height": out_h,
        "timeline_height": timeline_h,
        "timeline_lanes": n_lanes,
    }


def process_video(
    video_name: str,
    components_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    video_dir = components_root / video_name
    components_csv = video_dir / "components.csv"
    if not components_csv.is_file():
        raise FileNotFoundError(f"Missing components.csv: {components_csv}")

    meta = load_video_meta(video_dir)
    video_path_str = meta.get("source_video_path")
    video_path = Path(video_path_str) if video_path_str else None
    if video_path is None or not video_path.is_file():
        raise FileNotFoundError(
            f"source_video_path missing/invalid in {video_dir / 'config.json'}"
        )

    frame_w, frame_h = frame_size_from_meta(meta, video_path)
    blobs = load_blobs_from_components_csv(
        components_csv,
        factor=int(args.factor),
        pixel_size=int(args.pixel_size),
    )
    if not blobs:
        raise RuntimeError(
            f"No blobs for factor={args.factor} pixel_size={args.pixel_size} in {components_csv}"
        )

    cfg = TemporalLinkingConfig(
        max_gap=int(args.max_gap),
        min_track_length=int(args.min_track_length),
        center_distance_threshold=float(args.center_distance_threshold),
        motion_difference_threshold=float(args.motion_difference_threshold),
        expand_margin_px=float(args.expand_margin_px),
        expand_prop=float(args.expand_prop),
        block_px=int(args.pixel_size),
        trail_length=int(args.trail_length),
    )

    t0 = time.time()
    tracks, stats = link_blobs(blobs, cfg, frame_w=frame_w, frame_h=frame_h)

    if output_root is None:
        raise ValueError("output_root is required")
    out_dir = output_root / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "video_name": video_name,
        "components_csv": str(components_csv),
        "source_video_path": str(video_path),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "factor": int(args.factor),
        "pixel_size": int(args.pixel_size),
        "frame_index_basis": (
            "components.csv frame_index (sequential sampled detection index); "
            "frame_gap uses frame_index deltas. frame_idx is original video frame."
        ),
        "bbox_format": "[x1,y1,x2,y2] pixel coords; x2/y2 exclusive upper bounds from cell edges",
        "original_video_fps_note": "read at runtime via OpenCV CAP_PROP_FPS",
        **stats,
    }

    paths = save_temporal_linking_outputs(out_dir, tracks, summary)

    overlay_info: dict[str, Any] = {}
    if not args.skip_overlay:
        overlay_path = out_dir / "temporal_linking_overlay.mp4"
        frame_pairs = load_sample_frame_pairs(video_dir, blobs)
        overlay_info = write_overlay(
            video_path=video_path,
            tracks=tracks,
            blobs=blobs,
            frame_pairs=frame_pairs,
            output_mp4=overlay_path,
            trail_length=int(args.trail_length),
        )
        summary.update(overlay_info)
        # rewrite summary with overlay fields
        paths["temporal_linking_summary"].write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        paths["temporal_linking_overlay"] = overlay_path

    elapsed = time.time() - t0
    logger.info(
        "Wrote %s tracks=%d blobs=%d (kept_len_mean=%.2f) in %.1fs → %s",
        video_name,
        summary["final_trajectory_count"],
        summary["input_blob_count"],
        summary["track_length_mean"],
        elapsed,
        out_dir,
    )
    return {
        "video_name": video_name,
        "output_dir": str(out_dir),
        **{k: str(v) for k, v in paths.items()},
        "summary": {
            "input_blob_count": summary["input_blob_count"],
            "trajectories_before_filter": summary["trajectories_before_filter"],
            "one_frame_trajectory_count": summary["one_frame_trajectory_count"],
            "final_trajectory_count": summary["final_trajectory_count"],
            "gap1_link_count": summary["gap1_link_count"],
            "track_length_mean": summary["track_length_mean"],
            "track_length_median": summary["track_length_median"],
            "track_length_max": summary["track_length_max"],
        },
        "elapsed_sec": elapsed,
    }


def main() -> None:
    args = parse_args()
    components_root = args.components_root.resolve()
    output_root = args.output_root.resolve()
    videos = discover_videos(components_root, args.video_names)
    if not videos:
        raise SystemExit(f"No videos with components.csv under {components_root}")

    logger.info("Components root: %s", components_root)
    logger.info("Output root: %s", output_root)
    logger.info(
        "Videos (%d): %s | factor=%d pixel=%d max_gap=%d min_len=%d center_thr=%.3f",
        len(videos),
        videos,
        args.factor,
        args.pixel_size,
        args.max_gap,
        args.min_track_length,
        args.center_distance_threshold,
    )

    batch: list[dict[str, Any]] = []
    for name in videos:
        batch.append(process_video(name, components_root, output_root, args))

    batch_path = output_root / "batch_summary_temporal_linking.json"
    batch_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    logger.info("Batch summary → %s", batch_path)


if __name__ == "__main__":
    main()

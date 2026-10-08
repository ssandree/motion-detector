#!/usr/bin/env python3
"""Export VLM-inference frame folders for a fixed 9-video VIRAT subset.

Does not change official Stage3 ROI clustering. Tubes are loaded from
existing ``*_roi_tracks.json`` or generated once with ``stage3.process_video``.

Outputs (default roots under /data/datasets):

  virat_9vid/{VIDEO_ID}.mp4
  fps_sampling_frames/fps0_25/{VIDEO_ID}/global_frame_{seq:06d}_t{ts}.jpg
  fps_sampling_frames/fps_2/{VIDEO_ID}/global_frame_{seq:06d}_t{ts}.jpg
  proposed_ROI/{VIDEO_ID}/roi_{id:02d}/tube{id:02d}_frame_….jpg
  proposed_ROI_w_fps/{VIDEO_ID}/
    global/…
    roi_01/…
    metadata.json

Global sampling picks the original-video frame nearest each target timestamp
(0, 1/fps, 2/fps, …). ROI crops use the existing lifetime AABB on a 5/3 fps
grid over each tube's [t0, t1] (Stage3 5fps timeline, every 3rd frame — same
spacing as previous proposed_frames_v2). Existing JPEGs are skipped unless
``--overwrite``.

Example::

  cd /path/to/motion-detector
  export PYTHONPATH=src
  python scripts/export_vlm_sampling.py
  python scripts/export_vlm_sampling.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motion_analyzer.config import (  # noqa: E402
    DEFAULT_FUSION,
    PipelineConfig,
    load_target_video_ids,
)
from motion_analyzer.sampling_metadata import (  # noqa: E402
    RoiMetaInput,
    SavedFrameRef,
    build_sampling_metadata,
    normalize_bbox,
    source_region_from_normalized_bbox,
    write_sampling_metadata,
)
from motion_analyzer.video_io import resolve_video_path  # noqa: E402
from motion_analyzer.visualization import grid_bbox_to_pixels  # noqa: E402
from stage3.roi_tube import resolve_stage2_npz  # noqa: E402
from stage3 import process_video  # noqa: E402

logger = logging.getLogger("export_vlm_sampling")

JPEG_QUALITY = 95
JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]

DEFAULT_VIDEO_IDS = REPO_ROOT / "configs" / "virat_9vid.txt"
DEFAULT_FUSION_ROOT = (
    REPO_ROOT / "data" / "stage2" / "p13_t5_sample30_6x6_tanstrip_floor05_mean_max"
)
DEFAULT_TRACKS_ROOT = REPO_ROOT / "outputs" / "stage3" / "3_roi_tube" / "vlm_9vid_tracks"

DEFAULT_VIDEO_OUT = Path("/data/datasets/virat_9vid")
DEFAULT_FPS_ROOT = Path("/data/datasets/fps_sampling_frames")
DEFAULT_ROI_ROOT = Path("/data/datasets/proposed_ROI")
DEFAULT_COMBO_ROOT = Path("/data/datasets/proposed_ROI_w_fps")

# Stage3 motion timeline is 5 fps; previous VLM proposed_frames used 0.6 s.
DEFAULT_ROI_FPS = 5.0 / 3.0


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    native_fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        if self.frame_count <= 0 or self.native_fps <= 0:
            return 0.0
        return float(self.frame_count - 1) / float(self.native_fps)

    def timestamp_of(self, frame_idx: int) -> float:
        fps = float(self.native_fps) if self.native_fps > 0 else 30.0
        return float(frame_idx) / fps


@dataclass(frozen=True)
class PlannedJpeg:
    dest: Path
    frame_idx: int
    timestamp_sec: float
    seq: int = 0


@dataclass(frozen=True)
class RoiTubePlan:
    """One ROI tube: lifetime bbox + planned crop JPEGs (may be empty)."""

    roi_id: int
    tube_id: int
    time_range: tuple[float, float]
    pixel_bbox: tuple[int, int, int, int]
    normalized_bbox: tuple[float, float, float, float]
    source_region: str
    frames: list[PlannedJpeg]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video_list", type=Path, default=DEFAULT_VIDEO_IDS)
    p.add_argument("--video_id", type=str, nargs="*", default=None)
    p.add_argument("--fusion_root", type=Path, default=DEFAULT_FUSION_ROOT)
    p.add_argument("--fusion", type=str, default=DEFAULT_FUSION)
    p.add_argument(
        "--tracks_root",
        type=Path,
        default=DEFAULT_TRACKS_ROOT,
        help="Stage3 roi_tracks JSON root (created if missing).",
    )
    p.add_argument("--video_out", type=Path, default=DEFAULT_VIDEO_OUT)
    p.add_argument("--fps_root", type=Path, default=DEFAULT_FPS_ROOT)
    p.add_argument("--roi_root", type=Path, default=DEFAULT_ROI_ROOT)
    p.add_argument("--combo_root", type=Path, default=DEFAULT_COMBO_ROOT)
    p.add_argument(
        "--video_search_root",
        type=Path,
        nargs="*",
        default=None,
        help="Override PipelineConfig.video_search_roots (default: VIRAT roots).",
    )
    p.add_argument("--global_fps_025", type=float, default=0.25)
    p.add_argument("--global_fps_1", type=float, default=1.0)
    p.add_argument("--global_fps_2", type=float, default=2.0)
    p.add_argument(
        "--roi_fps",
        type=float,
        default=DEFAULT_ROI_FPS,
        help="ROI crop sampling rate along each tube lifetime (default 5/3).",
    )
    p.add_argument(
        "--roi_context_margin",
        type=float,
        default=0.0,
        help=(
            "Expand each tight ROI by this fraction of its width/height on every "
            "side (pixel AABB, not grid blocks). E.g. 0.1 → +10%% each side. "
            "Clipped to the frame."
        ),
    )
    p.add_argument("--jpeg_quality", type=int, default=JPEG_QUALITY)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate JPEGs / video copies even if the destination exists.",
    )
    p.add_argument(
        "--rerun_stage3",
        action="store_true",
        help="Recompute official Stage3 tracks even if JSON already exists.",
    )
    return p.parse_args()


def format_timestamp(seconds: float) -> str:
    """``00.0``, ``00.5``, ``04.0``, ``11.6`` — 2-digit integer part, 1 decimal."""
    value = round(float(seconds), 1)
    if abs(value) < 1e-9:
        value = 0.0
    sign = "-" if value < 0 else ""
    body = f"{abs(value):.1f}"
    integer, frac = body.split(".")
    return f"{sign}{int(integer):02d}.{frac}"


def global_name(seq: int, timestamp_sec: float) -> str:
    return f"global_frame_{seq:06d}_t{format_timestamp(timestamp_sec)}.jpg"


def roi_name(tube_id: int, seq: int, timestamp_sec: float) -> str:
    return f"tube{int(tube_id):02d}_frame_{seq:06d}_t{format_timestamp(timestamp_sec)}.jpg"


def target_timestamps(duration_sec: float, fps: float) -> list[float]:
    if duration_sec < 0 or fps <= 0:
        return []
    n = int(np.floor(duration_sec * fps + 1e-9)) + 1
    out: list[float] = []
    for i in range(n):
        t = float(i) / float(fps)
        if t > duration_sec + 1e-9:
            break
        out.append(t)
    return out


def nearest_frame_index(timestamp_sec: float, native_fps: float, frame_count: int) -> int:
    if frame_count <= 0:
        return 0
    fps = float(native_fps) if native_fps > 0 else 30.0
    idx = int(round(float(timestamp_sec) * fps))
    return min(max(idx, 0), int(frame_count) - 1)


def inspect_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if native_fps <= 1e-6:
        native_fps = 30.0
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return VideoInfo(
        path=path,
        native_fps=native_fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def ensure_link_or_copy(src: Path, dst: Path, *, overwrite: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return "skip"
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        try:
            os.symlink(src, dst)
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def save_jpeg(path: Path, bgr: np.ndarray, *, quality: int, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        return "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to write JPEG: {path}")
    return "write"


def extract_frames(video_path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    """Sequential decode; more reliable than random seek on some codecs."""
    wanted = sorted(int(i) for i in indices if int(i) >= 0)
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    got: dict[int, np.ndarray] = {}
    target_i = 0
    frame_idx = 0
    last = wanted[-1]
    try:
        while frame_idx <= last and target_i < len(wanted):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx == wanted[target_i]:
                got[frame_idx] = frame
                target_i += 1
            frame_idx += 1
    finally:
        cap.release()
    missing = [i for i in wanted if i not in got]
    if missing:
        raise RuntimeError(
            f"{video_path.name}: failed to decode frames {missing[:8]}"
            f"{'…' if len(missing) > 8 else ''}"
        )
    return got


def load_npz_timeline(fusion_npz: Path) -> dict:
    with np.load(fusion_npz) as data:
        if "timestamp_sec_curr" not in data.files or "frame_idx_curr" not in data.files:
            raise KeyError(f"{fusion_npz}: missing timestamp_sec_curr / frame_idx_curr")
        return {
            "timestamp_sec_curr": np.asarray(data["timestamp_sec_curr"], dtype=np.float64),
            "frame_idx_curr": np.asarray(data["frame_idx_curr"], dtype=np.int32),
            "video_width": int(np.asarray(data.get("video_width", 0)).item())
            if "video_width" in data.files
            else 0,
            "video_height": int(np.asarray(data.get("video_height", 0)).item())
            if "video_height" in data.files
            else 0,
            "unit_cell_px": int(np.asarray(data.get("unit_cell_px", 64)).item()),
        }


def tracks_path(tracks_root: Path, video_id: str) -> Path:
    return tracks_root / f"{video_id}_roi_tracks.json"


def load_graph_cheby_tracks(path: Path) -> tuple[list[dict], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    variant = str(data.get("variant") or "")
    allowed = {
        "graph_then_constrained_chebyshev",
        "coarse_gc_compose_v2c",
        "stage3_v2_coarse_gc_compose",
        "stage3_light_uf_compose",
        "stage3_v2_light_uf_compose",
    }
    if variant and variant not in allowed:
        raise ValueError(
            f"{path}: unexpected Stage3 variant {variant!r} (allowed {sorted(allowed)})"
        )
    tracks = data.get("tracks") or data.get("tubes") or []
    cell_px = int(data.get("cell_px") or 64)
    return list(tracks), cell_px


def ensure_tracks(
    video_id: str,
    *,
    cfg: PipelineConfig,
    fusion_root: Path,
    tracks_root: Path,
    fusion: str,
    rerun: bool,
) -> Path:
    path = tracks_path(tracks_root, video_id)
    if path.is_file() and not rerun:
        try:
            load_graph_cheby_tracks(path)
            logger.info("  tracks reuse %s", path)
            return path
        except ValueError as exc:
            logger.warning("  %s — recomputing Stage3", exc)
    tracks_root.mkdir(parents=True, exist_ok=True)
    logger.info("  running official Stage3 (tracks only)")
    process_video(
        video_id,
        cfg=cfg,
        fusion_root=fusion_root,
        output_root=tracks_root,
        fusion=fusion,
        write_3d=False,
        write_overlay=False,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Stage3 did not write {path}")
    return path


def plan_global_frames(
    dest_dir: Path,
    info: VideoInfo,
    fps: float,
) -> list[PlannedJpeg]:
    planned: list[PlannedJpeg] = []
    seen: set[int] = set()
    for target in target_timestamps(info.duration_sec, fps):
        idx = nearest_frame_index(target, info.native_fps, info.frame_count)
        if idx in seen:
            continue
        seen.add(idx)
        ts = info.timestamp_of(idx)
        seq = len(planned) + 1
        planned.append(
            PlannedJpeg(
                dest=dest_dir / global_name(seq, ts),
                frame_idx=idx,
                timestamp_sec=ts,
                seq=seq,
            )
        )
    return planned


def expand_pixel_bbox(
    bbox: tuple[int, int, int, int],
    *,
    frame_width: int,
    frame_height: int,
    margin_frac: float,
) -> tuple[int, int, int, int]:
    """Grow AABB by ``margin_frac`` of width/height on each side; clip to frame."""
    x0, y0, x1, y1 = (int(v) for v in bbox)
    m = float(margin_frac)
    if m <= 0.0:
        return x0, y0, x1, y1
    w = max(x1 - x0, 1)
    h = max(y1 - y0, 1)
    dx = w * m
    dy = h * m
    nx0 = int(np.floor(x0 - dx))
    ny0 = int(np.floor(y0 - dy))
    nx1 = int(np.ceil(x1 + dx))
    ny1 = int(np.ceil(y1 + dy))
    nx0 = max(0, min(nx0, int(frame_width)))
    nx1 = max(0, min(nx1, int(frame_width)))
    ny0 = max(0, min(ny0, int(frame_height)))
    ny1 = max(0, min(ny1, int(frame_height)))
    if nx1 <= nx0 or ny1 <= ny0:
        return x0, y0, x1, y1
    return nx0, ny0, nx1, ny1


def plan_roi_tubes(
    roi_root_for_video: Path,
    info: VideoInfo,
    tracks: list[dict],
    timeline: dict,
    *,
    roi_fps: float,
    cell_px: int,
    context_margin: float = 0.0,
) -> list[RoiTubePlan]:
    """Plan per-tube ROI crops under ``roi_root_for_video/roi_{id:02d}/``."""
    ts_curr = np.asarray(timeline["timestamp_sec_curr"], dtype=np.float64)
    n_mag = int(ts_curr.shape[0])
    fw = int(timeline["video_width"] or info.width)
    fh = int(timeline["video_height"] or info.height)
    plans: list[RoiTubePlan] = []

    ordered = sorted(
        tracks,
        key=lambda t: (
            int(t.get("t0", 0)),
            -int(t.get("num_cells") or 0),
            int(t.get("tube_id") or 0),
        ),
    )
    for tube in ordered:
        members = tube.get("members") or []
        t0 = int(tube.get("t0", 0))
        t1 = int(tube.get("t1", t0))
        if members and (tube.get("t0") is None or tube.get("t1") is None):
            t0 = min(int(m["t0"]) for m in members)
            t1 = max(int(m["t1"]) for m in members)
        t0 = max(0, min(t0, n_mag - 1)) if n_mag else 0
        t1 = max(0, min(t1, n_mag - 1)) if n_mag else 0
        if t1 < t0:
            continue
        bbox_grid = tube.get("spatial_bbox_grid")
        if not bbox_grid:
            continue
        px = grid_bbox_to_pixels(
            tuple(int(v) for v in bbox_grid),
            unit_pixel_size=int(cell_px),
            frame_width=fw,
            frame_height=fh,
        )
        x0, y0, x1, y1 = expand_pixel_bbox(
            tuple(int(v) for v in px),
            frame_width=fw,
            frame_height=fh,
            margin_frac=float(context_margin),
        )
        if x1 <= x0 or y1 <= y0:
            continue

        tube_id = int(tube.get("tube_id") or (len(plans) + 1))
        roi_id = len(plans) + 1
        t0s = float(ts_curr[t0])
        t1s = float(ts_curr[t1])
        norm = normalize_bbox((x0, y0, x1, y1), frame_width=fw, frame_height=fh)
        region = None
        if tube.get("source_region"):
            region = str(tube["source_region"])
        elif tube.get("place_class"):
            region = str(tube["place_class"])
        else:
            region = source_region_from_normalized_bbox(norm)

        dest_dir = roi_root_for_video / f"roi_{roi_id:02d}"
        frames: list[PlannedJpeg] = []
        seen: set[int] = set()
        seq = 0
        k = 0
        while True:
            target = t0s + (float(k) / float(roi_fps))
            if target > t1s + 1e-6:
                break
            k += 1
            idx = nearest_frame_index(target, info.native_fps, info.frame_count)
            if idx in seen:
                continue
            seen.add(idx)
            ts = info.timestamp_of(idx)
            seq += 1
            frames.append(
                PlannedJpeg(
                    dest=dest_dir / roi_name(tube_id, seq, ts),
                    frame_idx=idx,
                    timestamp_sec=ts,
                    seq=seq,
                )
            )

        plans.append(
            RoiTubePlan(
                roi_id=roi_id,
                tube_id=tube_id,
                time_range=(t0s, t1s),
                pixel_bbox=(x0, y0, x1, y1),
                normalized_bbox=(norm[0], norm[1], norm[2], norm[3]),
                source_region=region,
                frames=frames,
            )
        )
    return plans


def crop_bgr(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    h, w = frame.shape[:2]
    x0 = min(max(x0, 0), w)
    x1 = min(max(x1, 0), w)
    y0 = min(max(y0, 0), h)
    y1 = min(max(y1, 0), h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty crop bbox={(x0, y0, x1, y1)} frame={(w, h)}")
    return frame[y0:y1, x0:x1]


def write_planned(
    video_path: Path,
    items: list[PlannedJpeg],
    *,
    quality: int,
    overwrite: bool,
    crops: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int]:
    if not items:
        return 0, 0
    need_idx = set()
    for i, item in enumerate(items):
        if overwrite or not item.dest.exists():
            need_idx.add(item.frame_idx)
    frames = extract_frames(video_path, need_idx) if need_idx else {}
    n_write = 0
    n_skip = 0
    for i, item in enumerate(items):
        if item.dest.exists() and not overwrite:
            n_skip += 1
            continue
        frame = frames[item.frame_idx]
        image = crop_bgr(frame, crops[i]) if crops is not None else frame
        save_jpeg(item.dest, image, quality=quality, overwrite=True)
        n_write += 1
    return n_write, n_skip


def sync_jpgs(src_dir: Path, dst_dir: Path, *, overwrite: bool) -> int:
    """Hardlink/copy every ``*.jpg`` under ``src_dir`` (recursive) into ``dst_dir``
    preserving relative subpaths (``global/…``, ``roi_01/…``).
    """
    if not src_dir.is_dir():
        return 0
    n = 0
    for src in sorted(src_dir.rglob("*.jpg")):
        rel = src.relative_to(src_dir)
        ensure_link_or_copy(src, dst_dir / rel, overwrite=overwrite)
        n += 1
    return n


def flatten_roi_plans(
    plans: list[RoiTubePlan],
) -> tuple[list[PlannedJpeg], list[tuple[int, int, int, int]]]:
    items: list[PlannedJpeg] = []
    bboxes: list[tuple[int, int, int, int]] = []
    for plan in plans:
        for fr in plan.frames:
            items.append(fr)
            bboxes.append(plan.pixel_bbox)
    return items, bboxes


def build_metadata_for_video(
    *,
    video_id: str,
    global_items: list[PlannedJpeg],
    roi_plans: list[RoiTubePlan],
) -> dict:
    """Map planned paths onto combo-relative ``file`` strings and build JSON."""
    g_refs: list[SavedFrameRef] = [
        SavedFrameRef(
            timestamp=float(item.timestamp_sec),
            file=f"global/{item.dest.name}",
            frame_index=int(item.seq or 0),
        )
        for item in global_items
    ]

    roi_inputs: list[RoiMetaInput] = []
    for plan in roi_plans:
        frames = [
            SavedFrameRef(
                timestamp=float(fr.timestamp_sec),
                file=f"roi_{plan.roi_id:02d}/{fr.dest.name}",
                frame_index=int(fr.seq or 0),
            )
            for fr in plan.frames
        ]
        if not frames:
            continue
        roi_inputs.append(
            RoiMetaInput(
                roi_id=int(plan.roi_id),
                normalized_bbox=plan.normalized_bbox,
                frames=frames,
                # Let metadata module derive overlap-based source_region.
                source_region=None,
            )
        )

    return build_sampling_metadata(
        video_id=video_id,
        global_frames=g_refs,
        rois=roi_inputs,
    )


def print_video_stats(row: dict) -> None:
    print(
        f"{row['video_id']}\n"
        f"num_0.25fps_global {row['num_0.25fps_global']}\n"
        f"num_1fps_global {row.get('num_1fps_global', 0)}\n"
        f"num_2fps_global {row['num_2fps_global']}\n"
        f"num_roi_frames {row['num_roi_frames']}\n"
        f"num_roi_tubes {row['num_roi_tubes']}\n"
        f"num_proposed_roi_w_fps {row['num_proposed_roi_w_fps']}"
    )


def process_one(
    video_id: str,
    *,
    cfg: PipelineConfig,
    args: argparse.Namespace,
) -> dict:
    src_video = resolve_video_path(video_id, cfg.video_search_roots)
    dst_video = args.video_out / f"{video_id}.mp4"
    action = ensure_link_or_copy(src_video, dst_video, overwrite=args.overwrite)
    logger.info("%s source=%s virat_9vid=%s", video_id, src_video, action)

    info = inspect_video(src_video)
    fusion_npz = resolve_stage2_npz(args.fusion_root, video_id, fusion=args.fusion)
    timeline = load_npz_timeline(fusion_npz)
    tracks_json = ensure_tracks(
        video_id,
        cfg=cfg,
        fusion_root=args.fusion_root,
        tracks_root=args.tracks_root,
        fusion=args.fusion,
        rerun=bool(args.rerun_stage3),
    )
    tracks, cell_px = load_graph_cheby_tracks(tracks_json)
    cell_px = int(timeline["unit_cell_px"] or cell_px)

    fps025_dir = args.fps_root / "fps0_25" / video_id
    fps1_dir = args.fps_root / "fps_1" / video_id
    fps2_dir = args.fps_root / "fps_2" / video_id
    roi_dir = args.roi_root / video_id
    combo_dir = args.combo_root / video_id
    combo_global = combo_dir / "global"

    g025 = plan_global_frames(fps025_dir, info, float(args.global_fps_025))
    g1 = plan_global_frames(fps1_dir, info, float(args.global_fps_1))
    g2 = plan_global_frames(fps2_dir, info, float(args.global_fps_2))
    # Combo global uses the same 0.25fps plan, destinated under combo/global/.
    g_combo = [
        PlannedJpeg(
            dest=combo_global / item.dest.name,
            frame_idx=item.frame_idx,
            timestamp_sec=item.timestamp_sec,
            seq=item.seq,
        )
        for item in g025
    ]
    roi_plans = plan_roi_tubes(
        roi_dir,
        info,
        tracks,
        timeline,
        roi_fps=float(args.roi_fps),
        cell_px=cell_px,
        context_margin=float(args.roi_context_margin),
    )
    # Mirror ROI layout into combo/{roi_XX}/ with identical filenames.
    combo_roi_plans: list[RoiTubePlan] = []
    for plan in roi_plans:
        combo_frames = [
            PlannedJpeg(
                dest=combo_dir / f"roi_{plan.roi_id:02d}" / fr.dest.name,
                frame_idx=fr.frame_idx,
                timestamp_sec=fr.timestamp_sec,
                seq=fr.seq,
            )
            for fr in plan.frames
        ]
        combo_roi_plans.append(
            RoiTubePlan(
                roi_id=plan.roi_id,
                tube_id=plan.tube_id,
                time_range=plan.time_range,
                pixel_bbox=plan.pixel_bbox,
                normalized_bbox=plan.normalized_bbox,
                source_region=plan.source_region,
                frames=combo_frames,
            )
        )

    roi_items, roi_bboxes = flatten_roi_plans(roi_plans)
    combo_roi_items, _combo_bboxes = flatten_roi_plans(combo_roi_plans)

    w025, s025 = write_planned(
        src_video, g025, quality=args.jpeg_quality, overwrite=args.overwrite
    )
    w1, s1 = write_planned(
        src_video, g1, quality=args.jpeg_quality, overwrite=args.overwrite
    )
    w2, s2 = write_planned(
        src_video, g2, quality=args.jpeg_quality, overwrite=args.overwrite
    )
    wroi, sroi = write_planned(
        src_video,
        roi_items,
        quality=args.jpeg_quality,
        overwrite=args.overwrite,
        crops=roi_bboxes,
    )
    # Prefer hardlinks from already-written fps0_25 / proposed_ROI into combo.
    n_g_link = 0
    for src_item, dst_item in zip(g025, g_combo):
        ensure_link_or_copy(src_item.dest, dst_item.dest, overwrite=args.overwrite)
        n_g_link += 1
    n_r_link = 0
    for src_item, dst_item in zip(roi_items, combo_roi_items):
        ensure_link_or_copy(src_item.dest, dst_item.dest, overwrite=args.overwrite)
        n_r_link += 1

    meta = build_metadata_for_video(
        video_id=video_id,
        global_items=g_combo,
        roi_plans=combo_roi_plans,
    )
    meta_path = write_sampling_metadata(combo_dir / "metadata.json", meta)
    roi_only_meta = build_sampling_metadata(
        video_id=video_id,
        global_frames=None,
        rois=[
            RoiMetaInput(
                roi_id=int(plan.roi_id),
                normalized_bbox=plan.normalized_bbox,
                frames=[
                    SavedFrameRef(
                        timestamp=float(fr.timestamp_sec),
                        file=f"roi_{plan.roi_id:02d}/{fr.dest.name}",
                        frame_index=int(fr.seq or 0),
                    )
                    for fr in plan.frames
                ],
            )
            for plan in roi_plans
            if plan.frames
        ],
    )
    write_sampling_metadata(roi_dir / "metadata.json", roi_only_meta)

    n_combo = len(list(combo_dir.rglob("*.jpg")))
    logger.info(
        "  jpeg 0.25fps write=%d skip=%d | 1fps write=%d skip=%d | 2fps write=%d skip=%d "
        "| roi write=%d skip=%d tubes=%d | combo link g=%d roi=%d | metadata=%s",
        w025,
        s025,
        w1,
        s1,
        w2,
        s2,
        wroi,
        sroi,
        len(roi_plans),
        n_g_link,
        n_r_link,
        meta_path,
    )

    row = {
        "video_id": video_id,
        "video_path": str(src_video),
        "native_fps": info.native_fps,
        "frame_count": info.frame_count,
        "duration_sec": round(info.duration_sec, 3),
        "num_0.25fps_global": len(g025),
        "num_1fps_global": len(g1),
        "num_2fps_global": len(g2),
        "num_roi_frames": len(roi_items),
        "num_roi_tubes": len(roi_plans),
        "num_proposed_roi_w_fps": n_combo,
        "metadata_json": str(meta_path),
        "tracks_json": str(tracks_json),
    }
    print_video_stats(row)
    print()
    return row


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    cfg = PipelineConfig(fusion=str(args.fusion))
    if args.video_search_root:
        cfg.video_search_roots = [Path(p).expanduser().resolve() for p in args.video_search_root]
    cfg.validate()

    args.fusion_root = args.fusion_root.resolve()
    args.tracks_root = args.tracks_root.resolve()
    args.video_out = args.video_out.resolve()
    args.fps_root = args.fps_root.resolve()
    args.roi_root = args.roi_root.resolve()
    args.combo_root = args.combo_root.resolve()
    if not args.fusion_root.is_dir():
        raise SystemExit(f"fusion_root not found: {args.fusion_root}")

    if args.video_id:
        video_ids = list(args.video_id)
    else:
        video_ids = load_target_video_ids(args.video_list)

    args.video_out.mkdir(parents=True, exist_ok=True)
    (args.fps_root / "fps0_25").mkdir(parents=True, exist_ok=True)
    (args.fps_root / "fps_1").mkdir(parents=True, exist_ok=True)
    (args.fps_root / "fps_2").mkdir(parents=True, exist_ok=True)
    args.roi_root.mkdir(parents=True, exist_ok=True)
    args.combo_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failed: list[dict] = []
    for i, video_id in enumerate(video_ids, start=1):
        logger.info("[%d/%d] %s", i, len(video_ids), video_id)
        try:
            rows.append(process_one(video_id, cfg=cfg, args=args))
        except Exception as exc:  # noqa: BLE001
            logger.exception("FAILED %s: %s", video_id, exc)
            failed.append({"video_id": video_id, "error": str(exc)})

    print("===== summary =====")
    print(f"num_videos {len(video_ids)}")
    print(f"num_ok {len(rows)}")
    print(f"num_failed {len(failed)}")
    if rows:
        print(f"sum_0.25fps_global {sum(r['num_0.25fps_global'] for r in rows)}")
        print(f"sum_1fps_global {sum(r.get('num_1fps_global', 0) for r in rows)}")
        print(f"sum_2fps_global {sum(r['num_2fps_global'] for r in rows)}")
        print(f"sum_roi_frames {sum(r['num_roi_frames'] for r in rows)}")
        print(f"sum_roi_tubes {sum(r['num_roi_tubes'] for r in rows)}")
        print(f"sum_proposed_roi_w_fps {sum(r['num_proposed_roi_w_fps'] for r in rows)}")
    print()
    print("video_id\tnum_0.25fps_global\tnum_2fps_global\tnum_roi_frames\tnum_roi_tubes\tnum_proposed_roi_w_fps")
    for r in rows:
        print(
            f"{r['video_id']}\t{r['num_0.25fps_global']}\t{r['num_2fps_global']}\t"
            f"{r['num_roi_frames']}\t{r['num_roi_tubes']}\t{r['num_proposed_roi_w_fps']}"
        )
    for f in failed:
        print(f"{f['video_id']}\tERROR\t{f['error']}")

    summary_path = args.tracks_root / "vlm_sampling_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "jpeg_quality": int(args.jpeg_quality),
                "roi_fps": float(args.roi_fps),
                "roi_context_margin": float(args.roi_context_margin),
                "global_fps": [
                    float(args.global_fps_025),
                    float(args.global_fps_1),
                    float(args.global_fps_2),
                ],
                "fusion_root": str(args.fusion_root),
                "tracks_root": str(args.tracks_root),
                "video_out": str(args.video_out),
                "fps_root": str(args.fps_root),
                "roi_root": str(args.roi_root),
                "combo_root": str(args.combo_root),
                "num_ok": len(rows),
                "num_failed": len(failed),
                "videos": rows,
                "failed": failed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

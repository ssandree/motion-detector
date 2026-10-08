"""Video loading and 5fps sampling helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from motion_analyzer.optical_flow import iter_sampled_frames, sample_frame_indices

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")


@dataclass
class SampledFrame:
    sampled_index: int
    frame_idx: int
    timestamp_sec: float
    bgr: np.ndarray


def _sorted_frame_paths(directory: Path) -> list[Path]:
    paths = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.name)


def _clip_dir_from_path(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir() and path.name == "frames":
        return path.parent
    return path


def _frames_dir_for_clip(clip_dir: Path) -> Path:
    clip_dir = _clip_dir_from_path(clip_dir)
    nested = clip_dir / "frames"
    return nested if nested.is_dir() else clip_dir


def _native_fps_for_clip(clip_dir: Path, *, fallback: float) -> float:
    meta_path = _clip_dir_from_path(clip_dir) / "meta.json"
    if not meta_path.is_file():
        return float(fallback)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return float(fallback)
    for key in ("sampled_fps", "native_fps", "fps"):
        value = meta.get(key)
        if value is not None:
            try:
                fps = float(value)
            except (TypeError, ValueError):
                continue
            if fps > 0:
                return fps
    return float(fallback)


def is_frame_sequence(path: str | Path) -> bool:
    clip_dir = _clip_dir_from_path(Path(path))
    if not clip_dir.is_dir():
        return False
    return bool(_sorted_frame_paths(_frames_dir_for_clip(clip_dir)))


def resolve_frame_sequence_dir(video_id: str, search_roots: Iterable[Path]) -> Path | None:
    for root in search_roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            continue
        candidate = root / video_id
        if candidate.is_dir() and is_frame_sequence(candidate):
            return _clip_dir_from_path(candidate)
    return None


def read_first_frame(video_path: str | Path) -> np.ndarray:
    """Return the first decoded BGR frame without sampling the whole video."""
    path = Path(video_path)
    if is_frame_sequence(path):
        frames_dir = _frames_dir_for_clip(path)
        paths = _sorted_frame_paths(frames_dir)
        if not paths:
            raise RuntimeError(f"No frames in {frames_dir}")
        frame = cv2.imread(str(paths[0]))
        if frame is None:
            raise RuntimeError(f"Failed to read first frame: {paths[0]}")
        return frame

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame: {path}")
    return frame


def resolve_video_path(video_id: str, search_roots: Iterable[Path]) -> Path:
    roots = [Path(r).expanduser().resolve() for r in search_roots]
    for root in roots:
        if not root.exists():
            continue
        for ext in VIDEO_EXTENSIONS:
            direct = root / f"{video_id}{ext}"
            if direct.is_file():
                return direct
            matches = sorted(root.rglob(f"{video_id}{ext}"))
            if matches:
                return matches[0]
    seq = resolve_frame_sequence_dir(video_id, roots)
    if seq is not None:
        return seq
    exts = ", ".join(VIDEO_EXTENSIONS)
    raise FileNotFoundError(
        f"Could not resolve video or frame sequence for {video_id} "
        f"(looked for {video_id}{{{exts}}} or {video_id}/frames/)"
    )


def sample_frame_sequence_frames(
    clip_dir: str | Path,
    target_fps: float,
    max_seconds: float | None = None,
) -> list[SampledFrame]:
    clip_dir = _clip_dir_from_path(Path(clip_dir))
    paths = _sorted_frame_paths(_frames_dir_for_clip(clip_dir))
    if len(paths) < 8:
        raise ValueError(f"Not enough frames in {clip_dir}: {len(paths)}")

    native_fps = _native_fps_for_clip(clip_dir, fallback=float(target_fps))
    indices = sample_frame_indices(native_fps, target_fps, len(paths))
    sampled: list[SampledFrame] = []
    for i, frame_idx in enumerate(indices):
        ts = float(frame_idx) / native_fps
        if max_seconds is not None and ts > float(max_seconds):
            break
        bgr = cv2.imread(str(paths[int(frame_idx)]))
        if bgr is None:
            raise RuntimeError(f"Failed to read frame: {paths[int(frame_idx)]}")
        sampled.append(
            SampledFrame(
                sampled_index=i,
                frame_idx=int(frame_idx),
                timestamp_sec=ts,
                bgr=bgr,
            )
        )
    if len(sampled) < 8:
        raise ValueError(f"Not enough sampled frames in sequence: {clip_dir}")
    return sampled


def sample_video_frames(
    video_path: str | Path,
    target_fps: float,
    max_seconds: float | None = None,
) -> list[SampledFrame]:
    path = Path(video_path)
    if is_frame_sequence(path):
        return sample_frame_sequence_frames(path, target_fps, max_seconds=max_seconds)

    sampled: list[SampledFrame] = []
    for i, (frame_idx, ts, bgr) in enumerate(iter_sampled_frames(str(path), target_fps)):
        if max_seconds is not None and float(ts) > float(max_seconds):
            break
        sampled.append(
            SampledFrame(
                sampled_index=i,
                frame_idx=int(frame_idx),
                timestamp_sec=float(ts),
                bgr=bgr,
            )
        )
    if len(sampled) < 8:
        raise ValueError(f"Not enough sampled frames in video: {video_path}")
    return sampled


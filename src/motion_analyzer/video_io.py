"""Video loading and 5fps sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from motion_analyzer.optical_flow import iter_sampled_frames


@dataclass
class SampledFrame:
    sampled_index: int
    frame_idx: int
    timestamp_sec: float
    bgr: np.ndarray


def resolve_video_path(video_id: str, search_roots: Iterable[Path]) -> Path:
    roots = [Path(r).expanduser().resolve() for r in search_roots]
    for root in roots:
        if not root.exists():
            continue
        direct = root / f"{video_id}.mp4"
        if direct.is_file():
            return direct
        matches = sorted(root.rglob(f"{video_id}.mp4"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not resolve video: {video_id}.mp4")


def sample_video_frames(
    video_path: str | Path,
    target_fps: float,
    max_seconds: float | None = None,
) -> list[SampledFrame]:
    sampled: list[SampledFrame] = []
    for i, (frame_idx, ts, bgr) in enumerate(iter_sampled_frames(str(video_path), target_fps)):
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


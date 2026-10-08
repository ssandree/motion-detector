"""A seeking video reader for qwen-vl-utils, registered for this task only.

``vlm_inference.video_io.read_video_av`` decodes every frame into a list before
subsampling. A 688 s 1080p VIRAT clip is about 20k frames, which is roughly 128 GB of
RGB, so the process gets OOM-killed. These annotation passes only ever need 160 frames,
so this reader seeks to each wanted frame and keeps just those.

This registers an extra backend under a private name and points qwen-vl-utils at it.
``vlm_inference`` is left untouched, so other experiments keep the reader they had.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import av
import numpy as np
import torch

logger = logging.getLogger(__name__)

BACKEND_NAME = "vc_seek"

# Decoding forward from a keyframe is cheap; seeking is not free. Below this many
# frames a single sequential pass beats one seek per target frame.
SEQUENTIAL_DECODE_MAX_FRAMES = 600


def _to_local_path(video_path: str) -> str:
    if video_path.startswith("file://"):
        return unquote(urlparse(video_path).path)
    return video_path


def _probe(container: av.container.InputContainer, stream: Any) -> tuple[int, float]:
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    if fps <= 1e-6:
        fps = 30.0
    total = int(stream.frames) if stream.frames and stream.frames > 0 else 0
    if total <= 0 and stream.duration and stream.time_base:
        total = int(round(float(stream.duration * stream.time_base) * fps))
    if total <= 0 and container.duration:
        total = int(round((container.duration / av.time_base) * fps))
    return total, fps


def _sequential(stream_frames, wanted: set[int]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    highest = max(wanted)
    for index, frame in enumerate(stream_frames):
        if index in wanted:
            out[index] = frame.to_ndarray(format="rgb24")
        if index >= highest:
            break
    return out


def _by_seeking(
    container: av.container.InputContainer,
    stream: Any,
    indices: list[int],
    fps: float,
) -> dict[int, np.ndarray]:
    """Seek to the keyframe before each wanted frame, then decode forward to it."""
    out: dict[int, np.ndarray] = {}
    time_base = stream.time_base
    start_time = stream.start_time or 0
    for index in indices:
        target_pts = int(index / fps / time_base) + start_time
        try:
            container.seek(target_pts, stream=stream)
        except av.AVError:
            continue
        picked = None
        for frame in container.decode(stream):
            if frame.pts is None:
                picked = frame
                break
            if frame.pts >= target_pts:
                picked = frame
                break
            picked = frame
            # Guard against a stream whose pts never reaches the target.
            if frame.pts > target_pts + int(1.0 / time_base):
                break
        if picked is not None:
            out[index] = picked.to_ndarray(format="rgb24")
    return out


def read_video_seek(ele: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any], float]:
    """qwen-vl-utils backend: decode only the frames the sampler asks for."""
    from qwen_vl_utils.vision_process import calculate_video_frame_range, smart_nframes

    video_path = _to_local_path(ele["video"])
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    started = time.time()
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total_frames, video_fps = _probe(container, stream)
        if total_frames <= 0:
            raise RuntimeError(f"Could not determine frame count for {video_path}")

        start_frame, end_frame, _ = calculate_video_frame_range(
            ele, total_frames=total_frames, video_fps=video_fps
        )
        clipped_total = end_frame - start_frame + 1
        if clipped_total <= 0:
            raise RuntimeError(f"Empty frame range for {video_path}")

        nframes = smart_nframes(ele, total_frames=clipped_total, video_fps=video_fps)
        offsets = torch.linspace(0, clipped_total - 1, nframes).round().long().tolist()
        abs_indices = [start_frame + int(offset) for offset in offsets]

        if total_frames <= SEQUENTIAL_DECODE_MAX_FRAMES:
            frames = _sequential(container.decode(stream), set(abs_indices))
        else:
            frames = _by_seeking(container, stream, sorted(set(abs_indices)), video_fps)
            if not frames:
                container.seek(0)
                frames = _sequential(container.decode(stream), set(abs_indices))

        if not frames:
            raise RuntimeError(f"No frames decoded from {video_path}")

        # A seek can miss; fall back to the nearest frame actually decoded so the
        # tensor keeps the requested length and stays time-ordered.
        available = sorted(frames)
        selected: list[np.ndarray] = []
        resolved: list[int] = []
        for index in abs_indices:
            if index in frames:
                nearest = index
            else:
                nearest = min(available, key=lambda got: abs(got - index))
            selected.append(frames[nearest])
            resolved.append(nearest)

        video = torch.from_numpy(np.stack(selected, axis=0)).permute(0, 3, 1, 2).contiguous()
        sample_fps = nframes / max(clipped_total, 1e-6) * video_fps
        metadata = dict(
            fps=video_fps,
            frames_indices=resolved,
            total_num_frames=total_frames,
            video_backend=BACKEND_NAME,
        )
        logger.info(
            "%s: path=%s total=%s fps=%.3f nframes=%s decoded=%s time=%.2fs",
            BACKEND_NAME,
            video_path,
            total_frames,
            video_fps,
            nframes,
            len(frames),
            time.time() - started,
        )
        return video, metadata, sample_fps
    finally:
        container.close()


def install() -> None:
    """Point qwen-vl-utils at this reader for the current process."""
    import os

    import qwen_vl_utils.vision_process as vp

    vp.VIDEO_READER_BACKENDS[BACKEND_NAME] = read_video_seek
    vp.FORCE_QWENVL_VIDEO_READER = BACKEND_NAME
    os.environ["FORCE_QWENVL_VIDEO_READER"] = BACKEND_NAME
    if hasattr(vp.get_video_reader_backend, "cache_clear"):
        vp.get_video_reader_backend.cache_clear()

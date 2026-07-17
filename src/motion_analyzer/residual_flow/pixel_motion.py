"""Pixel-level motion map computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class MotionConfig:
    target_fps: float = 5.0
    use_flow: bool = False
    use_global_compensation: bool = True
    flow_scale: float = 0.5
    alpha: float = 1.0
    beta: float = 0.5
    blur_ksize: int = 7
    blur_sigma: float = 1.5


def _normalize_map(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def compute_frame_diff(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    blur_ksize: int = 5,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """Grayscale absolute difference with Gaussian blur and normalization."""
    diff = cv2.absdiff(prev_gray, curr_gray).astype(np.float32)
    if blur_ksize > 1:
        diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), blur_sigma)
    return _normalize_map(diff)


def compute_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Farneback dense optical flow field (HxWx2, float32).

    *scale* < 1 runs flow on a downscaled image for speed (vectors scaled back).
    """
    h, w = prev_gray.shape[:2]
    if scale < 1.0:
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        prev_s = cv2.resize(prev_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        curr_s = cv2.resize(curr_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        flow_s = cv2.calcOpticalFlowFarneback(
            prev_s,
            curr_s,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flow = cv2.resize(flow_s, (w, h), interpolation=cv2.INTER_LINEAR)
        inv = 1.0 / scale
        flow[..., 0] *= inv
        flow[..., 1] *= inv
        return flow.astype(np.float32)

    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    ).astype(np.float32)


def compute_flow_magnitude(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    blur_ksize: int = 5,
    blur_sigma: float = 1.0,
    *,
    flow_scale: float = 1.0,
) -> np.ndarray:
    """Farneback optical flow magnitude, blurred and normalized."""
    flow = compute_optical_flow(prev_gray, curr_gray, scale=flow_scale)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
    if blur_ksize > 1:
        magnitude = cv2.GaussianBlur(magnitude, (blur_ksize, blur_ksize), blur_sigma)
    return _normalize_map(magnitude)


def combine_motion_map(
    frame_diff: np.ndarray,
    flow_mag: np.ndarray | None,
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    use_flow: bool = False,
) -> np.ndarray:
    """Combine frame diff and optional flow magnitude."""
    if use_flow and flow_mag is not None:
        combined = alpha * frame_diff + beta * flow_mag
    else:
        combined = alpha * frame_diff
    return _normalize_map(combined)


def compute_motion_map_pair(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    config: MotionConfig,
) -> tuple[np.ndarray, float, dict]:
    """
    Compute motion map and metadata for a consecutive frame pair.

    Returns (motion_map, frame_diff_mean, global_motion_dict).
    """
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    global_meta: dict = {
        "global_motion_dx": 0.0,
        "global_motion_dy": 0.0,
        "global_motion_scale": 1.0,
        "global_motion_rotation": 0.0,
        "global_motion_score": 0.0,
        "residual_motion_mean": 0.0,
        "residual_motion_area_ratio": 0.0,
        "camera_motion_flag": False,
    }

    if config.use_global_compensation:
        from motion_analyzer.residual_flow.global_motion import compute_motion_compensation_layers

        layers = compute_motion_compensation_layers(
            prev_gray,
            curr_gray,
            flow_scale=config.flow_scale,
            blur_ksize=config.blur_ksize,
            blur_sigma=config.blur_sigma,
            alpha=0.6,
            beta=0.4,
        )
        gm = layers.metadata
        global_meta = {
            "global_motion_dx": gm.dx,
            "global_motion_dy": gm.dy,
            "global_motion_scale": gm.scale,
            "global_motion_rotation": gm.rotation,
            "global_motion_score": gm.global_motion_score,
            "residual_motion_mean": gm.residual_motion_mean,
            "residual_motion_area_ratio": gm.residual_motion_area_ratio,
            "camera_motion_flag": gm.camera_motion_flag,
            "global_flow_method": gm.global_flow_method,
            "original_motion_mean": gm.original_motion_mean,
            "global_motion_mean": gm.global_motion_mean,
            "flow_direction_coherence": gm.flow_direction_coherence,
        }
        frame_diff_mean = float(layers.residual_map.mean())
        motion_map = layers.residual_map
        global_meta["_layers"] = layers
        return motion_map, frame_diff_mean, global_meta

    frame_diff = compute_frame_diff(
        prev_gray, curr_gray, config.blur_ksize, config.blur_sigma
    )
    frame_diff_mean = float(frame_diff.mean())
    global_meta["residual_motion_mean"] = frame_diff_mean
    global_meta["residual_motion_area_ratio"] = float(
        (frame_diff > 0.05).sum() / max(frame_diff.size, 1)
    )

    flow_mag = None
    if config.use_flow:
        flow_mag = compute_flow_magnitude(
            prev_gray, curr_gray, config.blur_ksize, config.blur_sigma,
            flow_scale=config.flow_scale,
        )

    motion_map = combine_motion_map(
        frame_diff,
        flow_mag,
        alpha=config.alpha,
        beta=config.beta,
        use_flow=config.use_flow,
    )
    return motion_map, frame_diff_mean, global_meta


def sample_frame_indices(native_fps: float, target_fps: float, frame_count: int) -> list[int]:
    """Return frame indices to sample at approximately *target_fps*."""
    if native_fps <= 0 or frame_count <= 0:
        return []
    step = max(1, int(round(native_fps / target_fps)))
    return list(range(0, frame_count, step))


def iter_sampled_frames(
    video_path: str,
    target_fps: float,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """
    Yield (frame_idx, timestamp_sec, bgr_frame) sampled at *target_fps*.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(sample_frame_indices(native_fps, target_fps, frame_count))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            timestamp = frame_idx / native_fps
            yield frame_idx, timestamp, frame
        frame_idx += 1

    cap.release()


def process_video_motion_maps(
    video_path: str,
    config: MotionConfig,
) -> tuple[
    list[np.ndarray],
    list[float],
    list[int],
    list[float],
    list[dict],
    dict[int, np.ndarray],
    dict[str, list[np.ndarray]],
]:
    """
    Process entire video and return motion maps with metadata.

    Returns:
        motion_maps (suppressed), frame_diff_means, frame_indices, timestamps_sec,
        global_motion_rows, sampled_gray, layer_maps
    """
    motion_maps: list[np.ndarray] = []
    layer_maps: dict[str, list[np.ndarray]] = {
        "original": [],
        "global": [],
        "residual": [],
    }
    frame_diff_means: list[float] = []
    pair_frame_indices: list[int] = []
    pair_timestamps: list[float] = []
    global_motion_rows: list[dict] = []
    sampled_gray: dict[int, np.ndarray] = {}

    prev_frame: np.ndarray | None = None

    for frame_idx, timestamp, frame in iter_sampled_frames(video_path, config.target_fps):
        sampled_gray[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            motion_map, fd_mean, gm_meta = compute_motion_map_pair(
                prev_frame, frame, config
            )
            layers = gm_meta.pop("_layers", None)
            if layers is not None:
                layer_maps["original"].append(layers.original_map)
                layer_maps["global"].append(layers.global_motion_map)
                layer_maps["residual"].append(layers.residual_map)
            motion_maps.append(motion_map)
            frame_diff_means.append(fd_mean)
            pair_frame_indices.append(frame_idx)
            pair_timestamps.append(timestamp)
            global_motion_rows.append(gm_meta)

        prev_frame = frame

    return (
        motion_maps,
        frame_diff_means,
        pair_frame_indices,
        pair_timestamps,
        global_motion_rows,
        sampled_gray,
        layer_maps,
    )

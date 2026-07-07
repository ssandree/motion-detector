"""I/O utilities for video discovery and unified result persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

DEFAULT_RESULT_ROOT_V1 = "Result/motion_analyzer_v1"
DEFAULT_VIDEO_DIR_V1 = r"C:\Datasets\VIRAT\video"

# Unified layout per video:
#   {video_name}/motion_blobs_overlay.mp4
#   {video_name}/_cache/motion_maps.npz
#   {video_name}/_data/*.csv, *.json, *.md


def normalize_video_stem(name: str) -> str:
    """Strip .mp4 suffix so ``--video_name foo.mp4`` and ``foo`` both work."""
    stem = name.strip()
    if stem.lower().endswith(".mp4"):
        return stem[:-4]
    return stem


def video_result_dir(result_root: str | Path, video_stem: str) -> Path:
    """Return (and create) ``{result_root}/{video_stem}/``."""
    out = Path(result_root) / video_stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def cache_dir(video_dir: Path) -> Path:
    d = video_dir / "_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir(video_dir: Path) -> Path:
    d = video_dir / "_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def overlay_video_path(video_dir: Path) -> Path:
    return video_dir / "motion_blobs_overlay.mp4"


def list_videos(video_root: str | Path, max_videos: int | None = None) -> list[Path]:
    """
    Return sorted mp4 paths under *video_root* (searches subfolders recursively).

    Expected layout::

        {video_root}/videos-00/*.mp4
        {video_root}/videos-01/*.mp4
        ...
    """
    root = Path(video_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory not found: {root}")

    videos = sorted(root.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No mp4 files found under {root}")

    if max_videos is not None:
        videos = videos[:max_videos]
    return videos


def resolve_video_path(stem: str, video_root: str | Path) -> Path:
    """Find ``{stem}.mp4`` under *video_root* (direct child or any subfolder)."""
    root = Path(video_root)
    name = f"{normalize_video_stem(stem)}.mp4"

    direct = root / name
    if direct.is_file():
        return direct

    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Video not found: {name} under {root}")
    if len(matches) > 1:
        return matches[0]
    return matches[0]


def list_result_videos(
    result_root: str | Path,
    max_videos: int | None = None,
    *,
    require_cache: bool = False,
    require_blobs: bool = False,
) -> list[Path]:
    """
    List per-video result directories under *result_root*.

    Filters:
        require_cache: must have ``_cache/motion_maps.npz``
        require_blobs: must have ``_data/motion_blobs.csv``
    """
    root = Path(result_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Result root not found: {root}")

    dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if require_cache and not (child / "_cache" / "motion_maps.npz").exists():
            if not (child / "_cache" / "motion_maps.npy").exists():
                continue
        if require_blobs and not (child / "_data" / "motion_blobs.csv").exists():
            continue
        dirs.append(child)

    if max_videos is not None:
        dirs = dirs[:max_videos]
    if not dirs:
        raise FileNotFoundError(f"No matching video outputs in {root}")
    return dirs


def get_video_metadata(video_path: str | Path) -> dict[str, Any]:
    """Read basic metadata from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / native_fps if native_fps > 0 else 0.0
    cap.release()

    return {
        "native_fps": float(native_fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }


def save_sampled_gray(
    gray_by_idx: dict[int, np.ndarray],
    video_dir: Path,
) -> Path:
    """Save sampled grayscale frames to ``_cache/sampled_gray.npz``."""
    cdir = cache_dir(video_dir)
    path = cdir / "sampled_gray.npz"
    if not gray_by_idx:
        return path
    indices = np.array(sorted(gray_by_idx.keys()), dtype=np.int32)
    frames = np.stack([gray_by_idx[i] for i in indices], axis=0).astype(np.uint8)
    np.savez_compressed(path, frame_indices=indices, gray_frames=frames)
    return path


def load_sampled_gray(video_dir: Path) -> dict[int, np.ndarray]:
    """Load ``_cache/sampled_gray.npz`` if present."""
    path = video_dir / "_cache" / "sampled_gray.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    indices = data["frame_indices"]
    frames = data["gray_frames"]
    return {int(idx): frames[i] for i, idx in enumerate(indices)}


def save_motion_maps(
    motion_maps: np.ndarray,
    video_dir: Path,
    *,
    compressed: bool = True,
) -> Path:
    """Save motion maps to ``_cache/motion_maps.npz``."""
    cdir = cache_dir(video_dir)
    if compressed:
        path = cdir / "motion_maps.npz"
        np.savez_compressed(path, motion_maps=motion_maps.astype(np.float32))
    else:
        path = cdir / "motion_maps.npy"
        np.save(path, motion_maps.astype(np.float32))
    return path


def save_motion_map_layers(
    layer_maps: dict[str, list[np.ndarray]],
    video_dir: Path,
    *,
    compressed: bool = True,
) -> Path | None:
    """Cache original/global/residual/suppressed map stacks for debug overlays."""
    if not layer_maps or not layer_maps.get("suppressed"):
        return None
    arrays = {
        key: np.stack(maps, axis=0).astype(np.float32)
        for key, maps in layer_maps.items()
        if maps
    }
    cdir = cache_dir(video_dir)
    path = cdir / "motion_map_layers.npz"
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)
    return path


def save_frame_features(features: list[dict[str, Any]], video_dir: Path) -> Path:
    """Write frame-level features to ``_data/frame_motion_features.csv``."""
    path = data_dir(video_dir) / "frame_motion_features.csv"
    pd.DataFrame(features).to_csv(path, index=False)
    return path


def save_json(path: Path, obj: dict[str, Any]) -> Path:
    """Write a JSON file with numpy/path-safe serialization."""

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)
    return path


def load_motion_maps(video_dir: Path) -> np.ndarray:
    """Load motion maps from ``_cache/``."""
    cdir = video_dir / "_cache"
    npz_path = cdir / "motion_maps.npz"
    npy_path = cdir / "motion_maps.npy"
    if npz_path.exists():
        return np.load(npz_path)["motion_maps"]
    if npy_path.exists():
        return np.load(npy_path)
    raise FileNotFoundError(f"No motion maps in {cdir}")


def load_frame_features(video_dir: Path) -> pd.DataFrame:
    """Load ``_data/frame_motion_features.csv``."""
    path = data_dir(video_dir) / "frame_motion_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Frame features not found: {path}")
    return pd.read_csv(path)


def load_pixel_summary(video_dir: Path) -> dict[str, Any]:
    """Load pixel-stage summary from ``_data/summary_pixel_v1.json``."""
    path = data_dir(video_dir) / "summary_pixel_v1.json"
    if not path.exists():
        raise FileNotFoundError(f"Pixel summary not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_motion_blobs(video_dir: Path) -> pd.DataFrame:
    """Load ``_data/motion_blobs.csv``."""
    path = data_dir(video_dir) / "motion_blobs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Motion blobs not found: {path}")
    return pd.read_csv(path)

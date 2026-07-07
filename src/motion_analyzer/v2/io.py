"""I/O helpers for Motion Analyzer v2 with optional v1 artifact reuse."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from motion_analyzer.v1.io import (
    data_dir,
    load_frame_features,
    load_motion_blobs,
    load_motion_maps,
    normalize_video_stem,
    video_result_dir,
)

logger = logging.getLogger(__name__)

DEFAULT_RESULT_ROOT_V2 = "Result/motion_analyzer_v2"
DEFAULT_VIDEO_DIR_V2 = r"C:/Datasets/VIRAT/video"

# Canonical v1-stage artifact names (scripts/*_v1.py output).
SUMMARY_PIXEL_V1 = "_data/summary_pixel_v1.json"
SUMMARY_BOX_V1 = "_data/summary_box_v1.json"
SUMMARY_VIDEO_V1 = "_data/summary_v1.json"

# Pre-rename legacy filenames still found in older Result/ trees.
LEGACY_ALIASES: dict[str, list[str]] = {
    SUMMARY_PIXEL_V1: ["_data/summary_pixel.json"],
    SUMMARY_BOX_V1: ["_data/summary_box.json"],
    SUMMARY_VIDEO_V1: ["_data/summary_video.json"],
}

V1_ARTIFACTS = [
    ("_cache/motion_maps.npz", "_cache/motion_maps.npz"),
    ("_cache/sampled_gray.npz", "_cache/sampled_gray.npz"),
    ("_data/frame_motion_features.csv", "_data/frame_motion_features.csv"),
    (SUMMARY_PIXEL_V1, SUMMARY_PIXEL_V1),
    ("_data/motion_blobs.csv", "_data/motion_blobs.csv"),
    (SUMMARY_BOX_V1, SUMMARY_BOX_V1),
    ("_data/motion_tubes.csv", "_data/motion_tubes.csv"),
    (SUMMARY_VIDEO_V1, SUMMARY_VIDEO_V1),
    ("motion_blobs_overlay.mp4", "motion_blobs_overlay.mp4"),
]


def resolve_artifact_path(video_dir: Path, rel_path: str) -> Path | None:
    """Return existing path for *rel_path* (canonical or legacy alias)."""
    direct = video_dir / rel_path
    if direct.is_file():
        return direct
    for legacy in LEGACY_ALIASES.get(rel_path, []):
        candidate = video_dir / legacy
        if candidate.is_file():
            return candidate
    return None


def object_overlay_path(video_dir: Path) -> Path:
    return video_dir / "object_motion_overlay.mp4"


def has_artifact(video_dir: Path, rel_path: str) -> bool:
    return resolve_artifact_path(video_dir, rel_path) is not None


def _source_candidates(v1_dir: Path, rel_path: str) -> list[Path]:
    candidates = [v1_dir / rel_path]
    candidates.extend(v1_dir / legacy for legacy in LEGACY_ALIASES.get(rel_path, []))
    return candidates


def try_reuse_v1_artifact(
    v2_dir: Path,
    v1_dir: Path,
    rel_path: str,
) -> bool:
    """Copy a single artifact from v1 to v2 if v2 is missing it (canonical name)."""
    if has_artifact(v2_dir, rel_path):
        return True

    dst = v2_dir / rel_path
    for src in _source_candidates(v1_dir, rel_path):
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info("Reused v1 artifact: %s -> %s", src, dst)
        return True
    return False


def ensure_v1_artifacts(
    v2_dir: Path,
    v1_result_root: Path | None,
    required: list[str],
) -> dict[str, bool]:
    """Try to populate missing v2 artifacts from v1 result dir."""
    status = {rel: has_artifact(v2_dir, rel) for rel in required}
    if v1_result_root is None:
        return status
    v1_dir = v1_result_root / v2_dir.name
    if not v1_dir.is_dir():
        return status
    for rel in required:
        if not status[rel]:
            status[rel] = try_reuse_v1_artifact(v2_dir, v1_dir, rel)
    return status


def load_pixel_summary(video_dir: Path) -> dict[str, Any]:
    """Load pixel summary (``summary_pixel_v1.json`` or legacy ``summary_pixel.json``)."""
    path = resolve_artifact_path(video_dir, SUMMARY_PIXEL_V1)
    if path is None:
        raise FileNotFoundError(
            f"Pixel summary not found under {video_dir} "
            f"(tried {SUMMARY_PIXEL_V1} and legacy aliases)"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_motion_tubes(video_dir: Path):
    import pandas as pd

    path = data_dir(video_dir) / "motion_tubes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Motion tubes not found: {path}")
    return pd.read_csv(path)


def save_dataframe(df, video_dir: Path, filename: str) -> Path:
    import pandas as pd

    path = data_dir(video_dir) / filename
    if isinstance(df, pd.DataFrame):
        out = df
    else:
        out = pd.DataFrame(df)
    if out.empty and len(out.columns) == 0:
        out = pd.DataFrame(columns=[])
    out.to_csv(path, index=False)
    return path


def get_video_context(video_dir: Path) -> dict[str, Any]:
    """Load common video metadata from pixel summary + frame features."""
    pixel_summary = load_pixel_summary(video_dir)
    frame_df = load_frame_features(video_dir)
    shape = pixel_summary.get("motion_map_shape", [0, 720, 1280])
    frame_h = int(shape[1]) if len(shape) > 1 else 720
    frame_w = int(shape[2]) if len(shape) > 2 else 1280
    sampled_fps = float(pixel_summary.get("target_fps", 5.0))
    video_path = pixel_summary.get("video_path", "")
    duration_sec = float(frame_df["timestamp_sec"].max()) if len(frame_df) else 0.0
    return {
        "pixel_summary": pixel_summary,
        "frame_df": frame_df,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "sampled_fps": sampled_fps,
        "video_path": video_path,
        "duration_sec": duration_sec,
        "video_name": video_dir.name,
    }


# Re-export v1 loaders used by v2 pipeline.
__all__ = [
    "DEFAULT_RESULT_ROOT_V2",
    "DEFAULT_VIDEO_DIR_V2",
    "SUMMARY_PIXEL_V1",
    "SUMMARY_BOX_V1",
    "SUMMARY_VIDEO_V1",
    "data_dir",
    "ensure_v1_artifacts",
    "get_video_context",
    "has_artifact",
    "load_frame_features",
    "load_motion_blobs",
    "load_motion_maps",
    "load_motion_tubes",
    "load_pixel_summary",
    "normalize_video_stem",
    "resolve_artifact_path",
    "save_dataframe",
    "video_result_dir",
]

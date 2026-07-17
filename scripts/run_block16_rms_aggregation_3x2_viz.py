#!/usr/bin/env python3
"""Visualize spatial RMS aggregation sizes from cached 16×16 block motion.

Reuses existing Farneback → 16×16 sub-block RMS cache only.
Does NOT recompute optical flow / residual flow.

Layout (3×2):
  Original (1×1 raw heatmap) | 2×2 (32px) | 4×4 (64px)
  8×8 (128px) | 12×12 (192px) | 16×16 (256px)

- Original: base 16×16 RMS heatmap only (no CC / bbox).
- Aggregated panels: pooled RMS heatmap + MAD + 8-CC bboxes.
- Edge windows that are not a full k×k of sub-blocks are still pooled
  (ceil grid); remainder motion is never discarded.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motion_analyzer.block_motion.representation import (  # noqa: E402
    MAD_CONSISTENCY,
    median_mad_threshold,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT_ROOT = Path(
    "/data1/vailab02_dir/vlm_motion/motion-detector/Result/"
    "residual_motion_120s_block16_4video"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data1/vailab02_dir/vlm_motion/motion-detector/Result/"
    "residual_motion_120s_block16_rms_aggregation_3x2"
)
DEFAULT_VIDEO_DIRS = [
    Path("/data1/vailab02_dir/Classification_DB/VIRAT/videos-00"),
    Path("/data1/vailab02_dir/Classification_DB/VIRAT/videos-01"),
    Path("/data1/vailab02_dir/Classification_DB/VIRAT/videos-04"),
    Path("/data1/vailab02_dir/Classification_DB/VIRAT/videos-05"),
    Path("/data1/vailab02_dir/Classification_DB/VIRAT"),
]

# (sub_block_factor, title) — Original(1x1) is handled separately.
AGG_LEVELS: list[tuple[int, str]] = [
    (2, "2x2 blocks (32x32 px)"),
    (4, "4x4 blocks (64x64 px)"),
    (8, "8x8 blocks (128x128 px)"),
    (12, "12x12 blocks (192x192 px)"),
    (16, "16x16 blocks (256x256 px)"),
]

BASE_BLOCK_PX = 16
DEFAULT_MAD_SCALE = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3x2 RMS aggregation-size comparison from 16x16 block cache."
    )
    p.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument(
        "--video_names",
        nargs="+",
        default=None,
        help="Video folder names under input_root (default: all subdirs with cache).",
    )
    p.add_argument("--video_dir", type=Path, default=None, help="Extra VIRAT search root.")
    p.add_argument("--panel_width", type=int, default=640)
    p.add_argument("--panel_height", type=int, default=360)
    p.add_argument("--title_height", type=int, default=40)
    p.add_argument("--overlay_alpha", type=float, default=0.55)
    p.add_argument("--mad_scale", type=float, default=DEFAULT_MAD_SCALE)
    p.add_argument(
        "--norm_percentile",
        type=float,
        default=99.0,
        help="Video-level vmax percentile over base 16x16 RMS (vmin=0).",
    )
    return p.parse_args()


def resolve_video_path(stem: str, extra_dir: Path | None = None) -> Path | None:
    search_roots: list[Path] = []
    if extra_dir is not None:
        search_roots.append(extra_dir)
    search_roots.extend(DEFAULT_VIDEO_DIRS)
    seen: set[Path] = set()
    for root in search_roots:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        direct = root / f"{stem}.mp4"
        if direct.is_file():
            return direct
        matches = sorted(root.rglob(f"{stem}.mp4"))
        if matches:
            return matches[0]
    return None


def load_rms_cache(cache_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load TxRowsxCols RMS maps from representation cache (no flow recompute)."""
    meta_path = cache_dir / "representation_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    feat_path = cache_dir / "block_features.npz"
    score_path = cache_dir / "block_score_map.npy"
    heat_path = cache_dir / "block_heatmap.npy"

    rms: np.ndarray | None = None
    cache_file = ""
    if feat_path.is_file():
        with np.load(feat_path) as z:
            if "rms_mag" not in z.files:
                raise KeyError(f"rms_mag missing in {feat_path}; keys={z.files}")
            rms = z["rms_mag"].astype(np.float32)
            cache_file = str(feat_path)
    elif score_path.is_file():
        if str(meta.get("block_score", meta.get("heatmap_feature", ""))) not in (
            "rms_mag",
            "rms",
            "",
        ):
            raise ValueError(
                f"{score_path} present but metadata score is not rms_mag: "
                f"{meta.get('block_score')}"
            )
        rms = np.load(score_path).astype(np.float32)
        cache_file = str(score_path)
    elif heat_path.is_file():
        rms = np.load(heat_path).astype(np.float32)
        cache_file = str(heat_path)
    else:
        raise FileNotFoundError(
            f"No block_features.npz / block_score_map.npy under {cache_dir}"
        )

    if rms.ndim != 3:
        raise ValueError(f"RMS map expected TxRowsxCols, got {rms.shape}")

    bs = int(meta.get("block_size", BASE_BLOCK_PX))
    if bs != BASE_BLOCK_PX:
        raise ValueError(f"Expected base block_size={BASE_BLOCK_PX}, got {bs} in {cache_dir}")

    meta = dict(meta)
    meta["_rms_cache_file"] = cache_file
    return rms, meta


def pool_rms_map(rms_hw: np.ndarray, factor: int) -> tuple[np.ndarray, dict[str, int]]:
    """Pooled RMS over non-overlapping windows: sqrt(mean(rms^2)).

    Uses a ceil grid so remainder edge strips that are smaller than a full
    ``factor×factor`` window are still aggregated (never discarded).
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    rows, cols = int(rms_hw.shape[0]), int(rms_hw.shape[1])
    out_r = int(math.ceil(rows / float(factor)))
    out_c = int(math.ceil(cols / float(factor)))
    if out_r < 1 or out_c < 1:
        raise ValueError(f"Grid {rows}x{cols} too small for {factor}x{factor} aggregation")

    pooled = np.zeros((out_r, out_c), dtype=np.float32)
    window_h = np.zeros((out_r, out_c), dtype=np.int32)
    window_w = np.zeros((out_r, out_c), dtype=np.int32)
    for i in range(out_r):
        r0 = i * factor
        r1 = min((i + 1) * factor, rows)
        for j in range(out_c):
            c0 = j * factor
            c1 = min((j + 1) * factor, cols)
            patch = rms_hw[r0:r1, c0:c1].astype(np.float32, copy=False)
            pooled[i, j] = float(np.sqrt(np.mean(np.square(patch))))
            window_h[i, j] = int(r1 - r0)
            window_w[i, j] = int(c1 - c0)

    full_windows = int(np.sum((window_h == factor) & (window_w == factor)))
    info = {
        "sub_factor": int(factor),
        "pixel_block_nominal": int(factor * BASE_BLOCK_PX),
        "grid_rows": int(out_r),
        "grid_cols": int(out_c),
        "base_sub_rows": int(rows),
        "base_sub_cols": int(cols),
        "full_windows": full_windows,
        "partial_windows": int(out_r * out_c - full_windows),
        "covers_all_base_cells": True,
        "valid_pixel_h": int(rows * BASE_BLOCK_PX),
        "valid_pixel_w": int(cols * BASE_BLOCK_PX),
    }
    return pooled, info


def expand_pooled_to_base(
    pooled: np.ndarray,
    factor: int,
    base_rows: int,
    base_cols: int,
) -> np.ndarray:
    """Paint each pooled cell onto its (possibly partial) base sub-block region."""
    out = np.zeros((base_rows, base_cols), dtype=np.float32)
    for i in range(pooled.shape[0]):
        r0 = i * factor
        r1 = min((i + 1) * factor, base_rows)
        for j in range(pooled.shape[1]):
            c0 = j * factor
            c1 = min((j + 1) * factor, base_cols)
            out[r0:r1, c0:c1] = pooled[i, j]
    return out


def cell_pixel_bounds(
    r: int,
    c: int,
    *,
    factor: int,
    base_rows: int,
    base_cols: int,
    frame_h: int,
    frame_w: int,
    base_px: int = BASE_BLOCK_PX,
) -> tuple[int, int, int, int]:
    r0 = r * factor
    r1 = min((r + 1) * factor, base_rows)
    c0 = c * factor
    c1 = min((c + 1) * factor, base_cols)
    x1 = c0 * base_px
    y1 = r0 * base_px
    x2 = min(c1 * base_px, frame_w)
    y2 = min(r1 * base_px, frame_h)
    return int(x1), int(y1), int(x2), int(y2)


def extract_cc_bboxes(
    pooled: np.ndarray,
    *,
    factor: int,
    base_rows: int,
    base_cols: int,
    frame_h: int,
    frame_w: int,
    mad_scale: float,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """MAD active mask + 8-connectivity CC → pixel bboxes on (possibly partial) cells."""
    threshold, _med, _mad, active = median_mad_threshold(
        pooled,
        mad_scale=float(mad_scale),
        mad_consistency=MAD_CONSISTENCY,
    )
    active_u8 = active.astype(np.uint8)
    if not np.any(active_u8):
        return active, float(threshold), []

    num_labels, labels, _stats, _ = cv2.connectedComponentsWithStats(
        active_u8, connectivity=8
    )
    blobs: list[dict[str, Any]] = []
    for label_id in range(1, num_labels):
        ys, xs = np.where(labels == label_id)
        if ys.size == 0:
            continue
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        x1, y1, _, _ = cell_pixel_bounds(
            r0,
            c0,
            factor=factor,
            base_rows=base_rows,
            base_cols=base_cols,
            frame_h=frame_h,
            frame_w=frame_w,
        )
        _, _, x2, y2 = cell_pixel_bounds(
            r1,
            c1,
            factor=factor,
            base_rows=base_rows,
            base_cols=base_cols,
            frame_h=frame_h,
            frame_w=frame_w,
        )
        component = labels == label_id
        mean_score = float(pooled[component].mean())
        peak_score = float(pooled[component].max())
        blobs.append(
            {
                "blob_id": int(label_id),
                "bbox": [x1, y1, x2, y2],
                "active_block_count": int(component.sum()),
                "mean_block_score": mean_score,
                "peak_block_score": peak_score,
            }
        )
    return active, float(threshold), blobs


def video_level_vmax(rms_stack: np.ndarray, percentile: float) -> float:
    flat = rms_stack.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 1.0
    vmax = float(np.percentile(flat, percentile)) if percentile < 100.0 else float(np.max(flat))
    if vmax <= 1e-8:
        vmax = float(np.max(flat)) if flat.size else 1.0
    if vmax <= 1e-8:
        vmax = 1.0
    return vmax


def infer_output_fps(meta: dict[str, Any], fallback: float = 5.0) -> float:
    frames = list(meta.get("frames", []) or [])
    if len(frames) >= 2:
        dts: list[float] = []
        for a, b in zip(frames[:-1], frames[1:]):
            try:
                dt = float(b["timestamp_sec"]) - float(a["timestamp_sec"])
            except Exception:
                continue
            if dt > 1e-6:
                dts.append(dt)
        if dts:
            med = float(np.median(dts))
            if med > 1e-6:
                return float(1.0 / med)
    return float(fallback)


def letterbox_to_panel(
    image_bgr: np.ndarray,
    panel_w: int,
    panel_h: int,
    *,
    pad_color: tuple[int, int, int] = (20, 20, 20),
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((panel_h, panel_w, 3), pad_color, dtype=np.uint8)
    scale = min(panel_w / float(w), panel_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((panel_h, panel_w, 3), pad_color, dtype=np.uint8)
    x0 = (panel_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def draw_title_bar(title: str, width: int, height: int) -> np.ndarray:
    bar = np.full((height, width, 3), (28, 28, 28), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    x = max(8, (width - tw) // 2)
    y = (height + th) // 2
    cv2.putText(bar, title, (x, y), font, scale, (235, 235, 235), thickness, cv2.LINE_AA)
    return bar


def make_cell(
    content_bgr: np.ndarray,
    title: str,
    panel_w: int,
    panel_h: int,
    title_h: int,
) -> np.ndarray:
    body = letterbox_to_panel(content_bgr, panel_w, panel_h)
    title_bar = draw_title_bar(title, panel_w, title_h)
    return np.vstack([title_bar, body])


def normalize_shared(arr: np.ndarray, vmax: float) -> np.ndarray:
    if vmax <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr.astype(np.float32) / float(vmax), 0.0, 1.0)


def paint_base_heatmap(
    frame_bgr: np.ndarray,
    base_map: np.ndarray,
    *,
    vmax: float,
    alpha: float,
    pure_heatmap: bool = False,
) -> np.ndarray:
    """Upsample base 16×16-grid map to full frame coverage (no edge discard)."""
    h, w = frame_bgr.shape[:2]
    rows, cols = int(base_map.shape[0]), int(base_map.shape[1])
    norm = normalize_shared(base_map, vmax)
    heat_small = cv2.applyColorMap(
        np.clip(norm * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    up_h = min(h, rows * BASE_BLOCK_PX)
    up_w = min(w, cols * BASE_BLOCK_PX)
    heat_up = cv2.resize(heat_small, (up_w, up_h), interpolation=cv2.INTER_NEAREST)
    heat_full = np.zeros((h, w, 3), dtype=np.uint8)
    heat_full[:up_h, :up_w] = heat_up
    # Any true pixel remainder beyond complete 16px tiles (rare) stays black/dim.
    if pure_heatmap:
        out = heat_full.copy()
        if up_h < h:
            out[up_h:h, :] = 0
        if up_w < w:
            out[:up_h, up_w:w] = 0
        return out
    out = frame_bgr.copy()
    region = out[:up_h, :up_w]
    out[:up_h, :up_w] = cv2.addWeighted(region, 1.0 - alpha, heat_up, alpha, 0.0)
    return out


def draw_bboxes(
    frame_bgr: np.ndarray,
    blobs: list[dict[str, Any]],
    *,
    color: tuple[int, int, int] = (0, 165, 255),
) -> np.ndarray:
    out = frame_bgr.copy()
    for blob in blobs:
        x1, y1, x2, y2 = [int(v) for v in blob["bbox"][:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = (
            f"id={blob.get('blob_id', '?')} "
            f"m={float(blob.get('mean_block_score', 0.0)):.2f} "
            f"n={int(blob.get('active_block_count', 0))}"
        )
        cv2.putText(
            out,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def build_grid_canvas(cells: list[np.ndarray], cols: int = 3) -> np.ndarray:
    if len(cells) != 6:
        raise ValueError(f"Expected 6 panels, got {len(cells)}")
    rows = []
    for r in range(2):
        row_cells = cells[r * cols : (r + 1) * cols]
        rows.append(np.hstack(row_cells))
    return np.vstack(rows)


def process_video(
    cache_dir: Path,
    output_dir: Path,
    *,
    video_dirs_extra: Path | None,
    panel_w: int,
    panel_h: int,
    title_h: int,
    overlay_alpha: float,
    norm_percentile: float,
    mad_scale: float,
) -> dict[str, Any]:
    video_name = cache_dir.name
    t0 = time.time()
    rms_stack, meta = load_rms_cache(cache_dir)
    frame_meta = list(meta.get("frames", []) or [])
    t_frames = int(rms_stack.shape[0])
    if len(frame_meta) != t_frames:
        logger.warning(
            "%s: frames meta %d != rms T %d; using available indices",
            video_name,
            len(frame_meta),
            t_frames,
        )

    base_rows, base_cols = int(rms_stack.shape[1]), int(rms_stack.shape[2])
    if frame_meta:
        frame_h = int(frame_meta[0].get("frame_height", base_rows * BASE_BLOCK_PX))
        frame_w = int(frame_meta[0].get("frame_width", base_cols * BASE_BLOCK_PX))
    else:
        frame_h = base_rows * BASE_BLOCK_PX
        frame_w = base_cols * BASE_BLOCK_PX

    video_path = resolve_video_path(video_name, video_dirs_extra)
    if video_path is None:
        raise FileNotFoundError(f"Could not resolve source mp4 for {video_name}")

    pooled_by_factor: dict[int, np.ndarray] = {}
    grid_info: dict[str, Any] = {
        "1x1_base_16px": {
            "sub_factor": 1,
            "pixel_block_nominal": BASE_BLOCK_PX,
            "grid_rows": base_rows,
            "grid_cols": base_cols,
            "partial_windows": 0,
            "covers_all_base_cells": True,
            "valid_pixel_h": base_rows * BASE_BLOCK_PX,
            "valid_pixel_w": base_cols * BASE_BLOCK_PX,
        }
    }
    for factor, _title in AGG_LEVELS:
        maps = []
        info = None
        for i in range(t_frames):
            pooled, info = pool_rms_map(rms_stack[i], factor)
            maps.append(pooled)
        pooled_by_factor[factor] = np.stack(maps, axis=0)
        assert info is not None
        grid_info[f"{factor}x{factor}"] = info

    vmax = video_level_vmax(rms_stack, norm_percentile)
    for arr in pooled_by_factor.values():
        vmax = max(vmax, video_level_vmax(arr, norm_percentile))

    output_fps = infer_output_fps(meta, fallback=5.0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None
    src_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame_w)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame_h)

    cell_h = panel_h + title_h
    out_w = panel_w * 3
    out_h = cell_h * 2

    output_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = output_dir / f"{video_name}_block16_rms_aggregation_3x2.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, max(output_fps, 0.1), (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter: {out_mp4}")

    written = 0
    blob_counts: dict[str, list[int]] = {f"{f}x{f}": [] for f, _ in AGG_LEVELS}

    for i in range(t_frames):
        if i < len(frame_meta):
            fm = frame_meta[i]
            frame_idx = int(fm.get("frame_idx", i))
            ts = float(fm.get("timestamp_sec", 0.0))
        else:
            frame_idx = i
            ts = float(i) / max(output_fps, 1e-6)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.warning("%s: failed to read frame_idx=%d (i=%d)", video_name, frame_idx, i)
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        if frame.shape[0] != frame_h or frame.shape[1] != frame_w:
            frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

        hud = (
            f"f={frame_idx}  t={ts:.3f}s  mad_a={mad_scale:g}  "
            f"norm=p{norm_percentile:g} vmax={vmax:.3f}"
        )

        cells: list[np.ndarray] = []

        # Panel 0: original = 1×1 base RMS raw heatmap (no CC/bbox).
        original = paint_base_heatmap(
            frame,
            rms_stack[i],
            vmax=vmax,
            alpha=overlay_alpha,
            pure_heatmap=True,
        )
        cv2.putText(
            original, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA
        )
        cv2.putText(
            original, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA
        )
        cells.append(
            make_cell(original, "Original 1x1 (16px RMS heatmap)", panel_w, panel_h, title_h)
        )

        for factor, title in AGG_LEVELS:
            pooled = pooled_by_factor[factor][i]
            expanded = expand_pooled_to_base(pooled, factor, base_rows, base_cols)
            overlay = paint_base_heatmap(
                frame,
                expanded,
                vmax=vmax,
                alpha=overlay_alpha,
                pure_heatmap=False,
            )
            _active, thr, blobs = extract_cc_bboxes(
                pooled,
                factor=factor,
                base_rows=base_rows,
                base_cols=base_cols,
                frame_h=frame_h,
                frame_w=frame_w,
                mad_scale=mad_scale,
            )
            overlay = draw_bboxes(overlay, blobs)
            blob_counts[f"{factor}x{factor}"].append(len(blobs))
            panel_title = f"{title}  thr={thr:.3f}  blobs={len(blobs)}"
            cells.append(make_cell(overlay, panel_title, panel_w, panel_h, title_h))

        canvas = build_grid_canvas(cells, cols=3)
        writer.write(canvas)
        written += 1

    cap.release()
    writer.release()

    summary: dict[str, Any] = {
        "video_name": video_name,
        "status": "ok",
        "input_cache_dir": str(cache_dir),
        "rms_cache_file": meta.get("_rms_cache_file"),
        "representation_metadata": str(cache_dir / "representation_metadata.json"),
        "source_video_path": str(video_path),
        "base_block_size_px": BASE_BLOCK_PX,
        "score": "rms_only",
        "pooled_rms_formula": "sqrt(mean(sub_block_rms**2)) over ceil windows (partial edges kept)",
        "mad_scale": mad_scale,
        "connectivity": 8,
        "normalization": {
            "scope": "video_level_shared_across_panels",
            "vmin": 0.0,
            "vmax": vmax,
            "percentile": norm_percentile,
            "source": "base_16x16_rms_and_all_pooled_rms",
        },
        "num_frames": t_frames,
        "frames_written": written,
        "output_fps": output_fps,
        "source_video_fps": src_fps,
        "source_video_frame_count": src_frame_count,
        "original_resolution": {"width": src_w, "height": src_h},
        "cache_frame_resolution": {"width": frame_w, "height": frame_h},
        "output_resolution": {"width": out_w, "height": out_h},
        "panel_display_size": {
            "width": panel_w,
            "height": panel_h,
            "title_height": title_h,
        },
        "layout": {
            "rows": [
                [
                    "Original 1x1 (16px RMS heatmap)",
                    "2x2 blocks (32x32 px)",
                    "4x4 blocks (64x64 px)",
                ],
                [
                    "8x8 blocks (128x128 px)",
                    "12x12 blocks (192x192 px)",
                    "16x16 blocks (256x256 px)",
                ],
            ]
        },
        "aggregation_grids": grid_info,
        "mean_blobs_per_frame": {
            k: (float(np.mean(v)) if v else 0.0) for k, v in blob_counts.items()
        },
        "output_mp4": str(out_mp4),
        "elapsed_sec": round(time.time() - t0, 3),
        "notes": [
            "Farneback / residual flow NOT recomputed; only reused 16x16 RMS cache.",
            "Original panel is pure 1x1 base RMS heatmap (no 8-CC bbox).",
            "Aggregated panels keep partial edge windows via ceil pooling.",
            "Aggregated panels draw MAD(alpha) + 8-connectivity CC bboxes.",
        ],
    }
    summary_path = output_dir / f"{video_name}_block16_rms_aggregation_3x2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    logger.info(
        "Wrote %s (%d frames, %dx%d, fps=%.3f, vmax=%.3f) in %.1fs",
        out_mp4.name,
        written,
        out_w,
        out_h,
        output_fps,
        vmax,
        time.time() - t0,
    )
    return summary


def discover_videos(input_root: Path, names: list[str] | None) -> list[Path]:
    if names:
        dirs = [input_root / n for n in names]
    else:
        dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    out: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Missing video cache dir: {d}")
        if not (d / "representation_metadata.json").is_file():
            raise FileNotFoundError(f"No representation_metadata.json in {d}")
        out.append(d)
    return out


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.is_dir():
        logger.error("input_root not found: %s", input_root)
        return 1

    videos = discover_videos(input_root, args.video_names)
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Input root: %s (%d videos)", input_root, len(videos))
    logger.info("Output root: %s", output_root)
    logger.info(
        "Layout: Original1x1 | 2x2 | 4x4 / 8x8 | 12x12 | 16x16  (mad_scale=%.1f, ceil edges)",
        float(args.mad_scale),
    )

    results: list[dict[str, Any]] = []
    for cache_dir in videos:
        out_dir = output_root / cache_dir.name
        try:
            summary = process_video(
                cache_dir,
                out_dir,
                video_dirs_extra=args.video_dir,
                panel_w=int(args.panel_width),
                panel_h=int(args.panel_height),
                title_h=int(args.title_height),
                overlay_alpha=float(args.overlay_alpha),
                norm_percentile=float(args.norm_percentile),
                mad_scale=float(args.mad_scale),
            )
            results.append(summary)
        except Exception as exc:
            logger.exception("Failed on %s: %s", cache_dir.name, exc)
            results.append(
                {
                    "video_name": cache_dir.name,
                    "status": "error",
                    "error": str(exc),
                    "input_cache_dir": str(cache_dir),
                }
            )

    batch = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_videos": len(results),
        "num_ok": sum(1 for r in results if r.get("status") == "ok"),
        "num_error": sum(1 for r in results if r.get("status") != "ok"),
        "layout": [
            ["Original 1x1", "2x2", "4x4"],
            ["8x8", "12x12", "16x16"],
        ],
        "videos": results,
    }
    batch_path = output_root / "batch_summary_block16_rms_aggregation_3x2.json"
    batch_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Batch summary: %s (ok=%d error=%d)", batch_path, batch["num_ok"], batch["num_error"])
    return 0 if batch["num_error"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

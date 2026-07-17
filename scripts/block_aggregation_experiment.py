#!/usr/bin/env python3
"""Block aggregation scale comparison (native MAD threshold).

Implements the comparison protocol matching the 8×8 ``block_aggregation_experiment``
baseline, adapted to a 16×16-pixel base block cache:

  feature: rms_mag
  threshold_mode: native
  mad_k: 1.5
  mad_scale_constant: 1.4826
  active = score >= median + mad_k * mad_scale_constant * MAD
  if MAD < 1e-12: threshold = median
  8-connectivity CC on active mask (no min-size / morphology / weak-grow)
  heatmap: per-panel, per-frame p99 normalization
  pooled_rms = sqrt(sum(base_rms^2) / valid_base_block_count)  (padded zeros excluded)

Layout (3×2):
  1×1 (16px) | 2×2 (32px) | 4×4 (64px)
  8×8 (128px) | 12×12 (192px) | 16×16 (256px)

Does NOT recompute Farneback / residual flow.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("block_aggregation_experiment")

DEFAULT_CACHE_ROOT = Path(
    "/data1/vailab02_dir/vlm_motion/motion-detector/Result/"
    "residual_motion_blob_batch_120s_block16"
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

FEATURE = "rms_mag"
THRESHOLD_MODE = "native"
MAD_K = 1.5
MAD_SCALE_CONSTANT = 1.4826
MAD_EPS = 1e-12
BASE_BLOCK_PX = 16
AGG_FACTORS = (1, 2, 4, 8, 12, 16)
CONNECTIVITY = 8
HEATMAP_PERCENTILE = 99.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument(
        "--video_names",
        nargs="+",
        default=None,
        help="Defaults to video folders already under --output_root.",
    )
    p.add_argument("--panel_width", type=int, default=640)
    p.add_argument("--panel_height", type=int, default=360)
    p.add_argument("--title_height", type=int, default=40)
    p.add_argument("--overlay_alpha", type=float, default=0.35)
    return p.parse_args()


def resolve_video_path(stem: str) -> Path | None:
    seen: set[Path] = set()
    for root in DEFAULT_VIDEO_DIRS:
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


def inspect_cache(cache_dir: Path) -> dict[str, Any]:
    """Load and validate 16×16 rms_mag cache; return stack + meta."""
    meta_path = cache_dir / "representation_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    feat_path = cache_dir / "block_features.npz"
    score_path = cache_dir / "block_score_map.npy"
    if feat_path.is_file():
        with np.load(feat_path) as z:
            keys = list(z.files)
            if FEATURE not in z.files:
                raise KeyError(f"{FEATURE} missing in {feat_path}; keys={keys}")
            rms = z[FEATURE].astype(np.float32)
            cache_file = str(feat_path)
            feature_keys = keys
    elif score_path.is_file():
        rms = np.load(score_path).astype(np.float32)
        cache_file = str(score_path)
        feature_keys = ["block_score_map"]
    else:
        raise FileNotFoundError(f"No block_features.npz / block_score_map.npy in {cache_dir}")

    if rms.ndim != 3:
        raise ValueError(f"Expected TxRowsxCols, got {rms.shape}")

    bs = int(meta.get("block_size", BASE_BLOCK_PX))
    if bs != BASE_BLOCK_PX:
        raise ValueError(f"Expected base block_size={BASE_BLOCK_PX}, got {bs}")

    heat_feat = str(meta.get("heatmap_feature", meta.get("block_score", FEATURE)))
    if heat_feat not in (FEATURE, "rms", ""):
        logger.warning("%s: metadata heatmap_feature=%s (expected %s)", cache_dir.name, heat_feat, FEATURE)

    frames = list(meta.get("frames", []) or [])
    if frames and abs(len(frames) - rms.shape[0]) > 0:
        logger.warning(
            "%s: frames meta %d != rms T %d", cache_dir.name, len(frames), rms.shape[0]
        )

    frame0 = frames[0] if frames else {}
    info = {
        "cache_dir": str(cache_dir),
        "cache_file": cache_file,
        "feature_keys": feature_keys,
        "feature": FEATURE,
        "rms_shape": list(rms.shape),
        "dtype": str(rms.dtype),
        "num_frames": int(rms.shape[0]),
        "grid_rows": int(rms.shape[1]),
        "grid_cols": int(rms.shape[2]),
        "block_size": bs,
        "frame_height": int(frame0.get("frame_height", rms.shape[1] * BASE_BLOCK_PX)),
        "frame_width": int(frame0.get("frame_width", rms.shape[2] * BASE_BLOCK_PX)),
        "heatmap_feature": heat_feat,
        "zero_frac": float((rms == 0).mean()),
    }
    return {"rms": rms, "meta": meta, "info": info, "frames": frames}


def pool_rms_map(rms_hw: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Ceil-window pooled RMS; padded zeros are excluded from the valid count.

    pooled = sqrt(sum(base_rms^2) / valid_base_block_count)
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    rows, cols = int(rms_hw.shape[0]), int(rms_hw.shape[1])
    out_r = int(math.ceil(rows / float(factor)))
    out_c = int(math.ceil(cols / float(factor)))
    pad_r = out_r * factor - rows
    pad_c = out_c * factor - cols

    padded = np.pad(rms_hw.astype(np.float64), ((0, pad_r), (0, pad_c)), mode="constant", constant_values=0.0)
    valid = np.ones((rows, cols), dtype=np.float64)
    valid = np.pad(valid, ((0, pad_r), (0, pad_c)), mode="constant", constant_values=0.0)

    blocked = padded.reshape(out_r, factor, out_c, factor)
    valid_b = valid.reshape(out_r, factor, out_c, factor)
    sum_sq = (blocked * blocked * valid_b).sum(axis=(1, 3))
    count = valid_b.sum(axis=(1, 3))
    count_safe = np.maximum(count, 1.0)
    pooled = np.sqrt(sum_sq / count_safe).astype(np.float32)
    pooled[count <= 0] = 0.0
    return pooled, count.astype(np.float32)


def native_mad_threshold(scores: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """threshold_mode=native: median + mad_k * 1.4826 * MAD (MAD<1e-12 → median)."""
    flat = scores.astype(np.float64).ravel()
    if flat.size == 0:
        return 0.0, 0.0, 0.0, np.zeros_like(scores, dtype=bool)
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    if mad < MAD_EPS:
        thr = med
    else:
        thr = med + float(MAD_K) * float(MAD_SCALE_CONSTANT) * mad
    active = scores >= thr
    return float(thr), med, mad, active.astype(bool)


def extract_8cc_bboxes(
    pooled: np.ndarray,
    active: np.ndarray,
    *,
    factor: int,
    base_rows: int,
    base_cols: int,
    frame_h: int,
    frame_w: int,
) -> list[dict[str, Any]]:
    """8-CC on active mask; no min-size / morphology / weak-grow filters."""
    if active.size == 0 or not np.any(active):
        return []
    num_labels, labels, _stats, _ = cv2.connectedComponentsWithStats(
        active.astype(np.uint8), connectivity=CONNECTIVITY
    )
    blobs: list[dict[str, Any]] = []
    for label_id in range(1, num_labels):
        component = labels == label_id
        ys, xs = np.where(component)
        if ys.size == 0:
            continue
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())

        def cell_bounds(r: int, c: int) -> tuple[int, int, int, int]:
            br0 = r * factor
            br1 = min((r + 1) * factor, base_rows)
            bc0 = c * factor
            bc1 = min((c + 1) * factor, base_cols)
            x1 = bc0 * BASE_BLOCK_PX
            y1 = br0 * BASE_BLOCK_PX
            x2 = min(bc1 * BASE_BLOCK_PX, frame_w)
            y2 = min(br1 * BASE_BLOCK_PX, frame_h)
            return int(x1), int(y1), int(x2), int(y2)

        x1, y1, _, _ = cell_bounds(r0, c0)
        _, _, x2, y2 = cell_bounds(r1, c1)
        vals = pooled[component]
        blobs.append(
            {
                "blob_id": int(label_id),
                "bbox": [x1, y1, x2, y2],
                "grid_r0": r0,
                "grid_r1": r1,
                "grid_c0": c0,
                "grid_c1": c1,
                "active_block_count": int(component.sum()),
                "mean_score": float(vals.mean()),
                "peak_score": float(vals.max()),
            }
        )
    return blobs


def expand_pooled_to_base(
    pooled: np.ndarray, factor: int, base_rows: int, base_cols: int
) -> np.ndarray:
    out = np.zeros((base_rows, base_cols), dtype=np.float32)
    for i in range(pooled.shape[0]):
        r0 = i * factor
        r1 = min((i + 1) * factor, base_rows)
        for j in range(pooled.shape[1]):
            c0 = j * factor
            c1 = min((j + 1) * factor, base_cols)
            out[r0:r1, c0:c1] = pooled[i, j]
    return out


def p99_normalize(arr: np.ndarray) -> np.ndarray:
    flat = arr.astype(np.float32).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    vmax = float(np.percentile(flat, HEATMAP_PERCENTILE))
    if vmax <= 1e-12:
        vmax = float(np.max(flat)) if flat.size else 1.0
    if vmax <= 1e-12:
        vmax = 1.0
    return np.clip(arr.astype(np.float32) / vmax, 0.0, 1.0)


def paint_heatmap(
    frame_bgr: np.ndarray,
    base_map: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Upsample base-grid heatmap with per-call (panel/frame) p99 norm."""
    h, w = frame_bgr.shape[:2]
    rows, cols = int(base_map.shape[0]), int(base_map.shape[1])
    norm = p99_normalize(base_map)
    heat_small = cv2.applyColorMap(
        np.clip(norm * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    up_h = min(h, rows * BASE_BLOCK_PX)
    up_w = min(w, cols * BASE_BLOCK_PX)
    heat_up = cv2.resize(heat_small, (up_w, up_h), interpolation=cv2.INTER_NEAREST)
    out = frame_bgr.copy()
    region = out[:up_h, :up_w]
    out[:up_h, :up_w] = cv2.addWeighted(region, 1.0 - alpha, heat_up, alpha, 0.0)
    return out


def draw_bboxes(frame_bgr: np.ndarray, blobs: list[dict[str, Any]]) -> np.ndarray:
    out = frame_bgr.copy()
    color = (0, 165, 255)
    for blob in blobs:
        x1, y1, x2, y2 = [int(v) for v in blob["bbox"][:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = (
            f"id={blob['blob_id']} m={blob['mean_score']:.2f} n={blob['active_block_count']}"
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


def letterbox(image_bgr: np.ndarray, panel_w: int, panel_h: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((panel_h, panel_w, 3), 20, dtype=np.uint8)
    scale = min(panel_w / float(w), panel_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((panel_h, panel_w, 3), 20, dtype=np.uint8)
    x0 = (panel_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def make_cell(
    content_bgr: np.ndarray, title: str, panel_w: int, panel_h: int, title_h: int
) -> np.ndarray:
    body = letterbox(content_bgr, panel_w, panel_h)
    bar = np.full((title_h, panel_w, 3), 28, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    # Shrink title text if needed.
    while tw > panel_w - 16 and scale > 0.35:
        scale -= 0.05
        (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    x = max(8, (panel_w - tw) // 2)
    y = (title_h + th) // 2
    cv2.putText(bar, title, (x, y), font, scale, (235, 235, 235), thickness, cv2.LINE_AA)
    return np.vstack([bar, body])


def build_grid(cells: list[np.ndarray]) -> np.ndarray:
    if len(cells) != 6:
        raise ValueError(f"Expected 6 panels, got {len(cells)}")
    return np.vstack([np.hstack(cells[0:3]), np.hstack(cells[3:6])])


def infer_output_fps(frames: list[dict[str, Any]], fallback: float = 5.0) -> float:
    if len(frames) >= 2:
        dts = []
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


def process_video(
    *,
    video_name: str,
    cache_dir: Path,
    output_dir: Path,
    panel_w: int,
    panel_h: int,
    title_h: int,
    overlay_alpha: float,
) -> dict[str, Any]:
    t0 = time.time()
    bundle = inspect_cache(cache_dir)
    rms: np.ndarray = bundle["rms"]
    frames: list[dict[str, Any]] = bundle["frames"]
    info: dict[str, Any] = bundle["info"]
    meta: dict[str, Any] = bundle["meta"]

    t_frames = int(rms.shape[0])
    base_rows, base_cols = int(rms.shape[1]), int(rms.shape[2])
    frame_h = int(info["frame_height"])
    frame_w = int(info["frame_width"])

    video_path = resolve_video_path(video_name)
    if video_path is None:
        raise FileNotFoundError(f"Source mp4 not found for {video_name}")

    logger.info(
        "%s cache: shape=%s feature=%s keys=%s frames=%d grid=%dx%d zero_frac=%.4g",
        video_name,
        info["rms_shape"],
        FEATURE,
        info["feature_keys"],
        t_frames,
        base_rows,
        base_cols,
        info["zero_frac"],
    )

    # Precompute pooled maps per factor.
    pooled_by_factor: dict[int, np.ndarray] = {}
    grid_meta: dict[str, Any] = {}
    for factor in AGG_FACTORS:
        maps = []
        counts = []
        for i in range(t_frames):
            pooled, count = pool_rms_map(rms[i], factor)
            maps.append(pooled)
            counts.append(count)
        stacked = np.stack(maps, axis=0)
        pooled_by_factor[factor] = stacked
        grid_meta[f"{factor}x{factor}"] = {
            "factor": factor,
            "pixel_size": factor * BASE_BLOCK_PX,
            "grid_rows": int(stacked.shape[1]),
            "grid_cols": int(stacked.shape[2]),
            "base_rows": base_rows,
            "base_cols": base_cols,
            "covers_all_base_cells": True,
            "mean_valid_count": float(np.mean([c.mean() for c in counts])),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale previous-generation summary naming if present.
    for stale in output_dir.glob("*_block16_rms_aggregation_3x2_summary.json"):
        stale.unlink(missing_ok=True)

    out_mp4 = output_dir / f"{video_name}_block16_rms_aggregation_3x2.mp4"
    output_fps = infer_output_fps(frames)
    cell_h = panel_h + title_h
    out_w, out_h = panel_w * 3, cell_h * 2
    writer = cv2.VideoWriter(
        str(out_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(output_fps, 0.1),
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_mp4}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    component_rows: list[dict[str, Any]] = []
    scale_frame_rows: dict[int, list[dict[str, Any]]] = {f: [] for f in AGG_FACTORS}

    for i in range(t_frames):
        if i < len(frames):
            fm = frames[i]
            frame_idx = int(fm.get("frame_idx", i))
            ts = float(fm.get("timestamp_sec", 0.0))
        else:
            frame_idx = i
            ts = float(i) / max(output_fps, 1e-6)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        if frame.shape[0] != frame_h or frame.shape[1] != frame_w:
            frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

        cells: list[np.ndarray] = []
        for factor in AGG_FACTORS:
            pooled = pooled_by_factor[factor][i]
            thr, med, mad, active = native_mad_threshold(pooled)
            blobs = extract_8cc_bboxes(
                pooled,
                active,
                factor=factor,
                base_rows=base_rows,
                base_cols=base_cols,
                frame_h=frame_h,
                frame_w=frame_w,
            )
            expanded = expand_pooled_to_base(pooled, factor, base_rows, base_cols)
            vis = paint_heatmap(frame, expanded, alpha=overlay_alpha)
            vis = draw_bboxes(vis, blobs)

            px = factor * BASE_BLOCK_PX
            title = f"{factor}x{factor} blocks ({px}x{px} px) thr={thr:.3f} blobs={len(blobs)}"
            cells.append(make_cell(vis, title, panel_w, panel_h, title_h))

            scale_frame_rows[factor].append(
                {
                    "frame_index": i,
                    "frame_idx": frame_idx,
                    "timestamp_sec": ts,
                    "threshold": thr,
                    "median": med,
                    "mad": mad,
                    "n_blobs": len(blobs),
                    "n_active_cells": int(active.sum()),
                    "grid_rows": int(pooled.shape[0]),
                    "grid_cols": int(pooled.shape[1]),
                }
            )
            for blob in blobs:
                x1, y1, x2, y2 = blob["bbox"]
                component_rows.append(
                    {
                        "video_name": video_name,
                        "frame_index": i,
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(ts, 6),
                        "factor": factor,
                        "pixel_size": px,
                        "blob_id": blob["blob_id"],
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "active_block_count": blob["active_block_count"],
                        "mean_score": round(blob["mean_score"], 6),
                        "peak_score": round(blob["peak_score"], 6),
                        "threshold": round(thr, 6),
                        "median": round(med, 6),
                        "mad": round(mad, 6),
                    }
                )

        writer.write(build_grid(cells))

    cap.release()
    writer.release()

    # components.csv
    components_path = output_dir / "components.csv"
    fieldnames = [
        "video_name",
        "frame_index",
        "frame_idx",
        "timestamp_sec",
        "factor",
        "pixel_size",
        "blob_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "active_block_count",
        "mean_score",
        "peak_score",
        "threshold",
        "median",
        "mad",
    ]
    with components_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in component_rows:
            w.writerow(row)

    # per-scale summary JSON
    scale_summaries: dict[str, Any] = {}
    for factor in AGG_FACTORS:
        rows = scale_frame_rows[factor]
        thr_vals = [r["threshold"] for r in rows]
        n_blobs = [r["n_blobs"] for r in rows]
        summary = {
            "video_name": video_name,
            "factor": factor,
            "pixel_size": factor * BASE_BLOCK_PX,
            "label": f"{factor}x{factor}",
            "feature": FEATURE,
            "threshold_mode": THRESHOLD_MODE,
            "mad_k": MAD_K,
            "mad_scale_constant": MAD_SCALE_CONSTANT,
            "connectivity": CONNECTIVITY,
            "heatmap_norm": "per_panel_per_frame_p99",
            "grid": grid_meta[f"{factor}x{factor}"],
            "num_frames": len(rows),
            "mean_threshold": float(np.mean(thr_vals)) if thr_vals else 0.0,
            "mean_blobs_per_frame": float(np.mean(n_blobs)) if n_blobs else 0.0,
            "total_blobs": int(sum(n_blobs)),
            "frames": rows,
        }
        path = output_dir / f"summary_{factor}x{factor}.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        scale_summaries[f"{factor}x{factor}"] = {
            "path": str(path),
            "mean_threshold": summary["mean_threshold"],
            "mean_blobs_per_frame": summary["mean_blobs_per_frame"],
            "total_blobs": summary["total_blobs"],
            "grid": summary["grid"],
        }

    config = {
        "video_name": video_name,
        "cache_root": str(cache_dir.parent),
        "cache_dir": str(cache_dir),
        "cache_inspection": info,
        "source_video_path": str(video_path),
        "output_dir": str(output_dir),
        "feature": FEATURE,
        "threshold_mode": THRESHOLD_MODE,
        "mad_k": MAD_K,
        "mad_scale_constant": MAD_SCALE_CONSTANT,
        "mad_eps": MAD_EPS,
        "active_rule": "pooled_score >= threshold",
        "threshold_rule": "median + mad_k * mad_scale_constant * MAD; if MAD < mad_eps then median",
        "connectivity": CONNECTIVITY,
        "min_blob_size_filter": False,
        "morphology": False,
        "weak_grow": False,
        "heatmap_norm": "per_panel_per_frame_p99",
        "base_block_px": BASE_BLOCK_PX,
        "aggregation_factors": list(AGG_FACTORS),
        "pixel_sizes": {f"{f}x{f}": f * BASE_BLOCK_PX for f in AGG_FACTORS},
        "pooled_rms_formula": "sqrt(sum(base_rms**2) / valid_base_block_count)",
        "padding_zeros_excluded": True,
        "layout": {
            "rows": [
                ["1x1 (16px)", "2x2 (32px)", "4x4 (64px)"],
                ["8x8 (128px)", "12x12 (192px)", "16x16 (256px)"],
            ]
        },
        "panel_display_size": {
            "width": panel_w,
            "height": panel_h,
            "title_height": title_h,
        },
        "overlay_alpha": overlay_alpha,
        "output_mp4": str(out_mp4),
        "components_csv": str(components_path),
        "scale_summaries": scale_summaries,
        "num_frames": t_frames,
        "output_fps": output_fps,
        "output_resolution": {"width": out_w, "height": out_h},
        "notes": [
            "Farneback / residual flow NOT recomputed; reused 16x16 rms_mag cache only.",
            "Native MAD threshold implemented locally (mad_eps=1e-12, active uses >=).",
            "Does not call residual_block_motion.median_mad_threshold (EPS=1e-8, > vs >= differs).",
            "No min blob size / morphology / weak-grow post-processing.",
        ],
        "representation_metadata": {
            "block_size": meta.get("block_size"),
            "heatmap_feature": meta.get("heatmap_feature"),
            "block_score": meta.get("block_score"),
            "num_frames": meta.get("num_frames"),
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Wrote %s (+ config/components/scale summaries) frames=%d in %.1fs",
        out_mp4.name,
        t_frames,
        time.time() - t0,
    )
    return {
        "video_name": video_name,
        "status": "ok",
        "output_mp4": str(out_mp4),
        "config_json": str(config_path),
        "components_csv": str(components_path),
        "scale_summaries": scale_summaries,
        "num_frames": t_frames,
        "elapsed_sec": config["elapsed_sec"],
    }


def discover_video_names(output_root: Path, names: list[str] | None) -> list[str]:
    if names:
        return list(names)
    return sorted(p.name for p in output_root.iterdir() if p.is_dir())


def main() -> int:
    args = parse_args()
    cache_root = args.cache_root.resolve()
    output_root = args.output_root.resolve()
    if not cache_root.is_dir():
        logger.error("cache_root not found: %s", cache_root)
        return 1
    output_root.mkdir(parents=True, exist_ok=True)

    video_names = discover_video_names(output_root, args.video_names)
    if not video_names:
        logger.error("No target videos under %s", output_root)
        return 1

    logger.info("Cache root: %s", cache_root)
    logger.info("Output root (overwrite): %s", output_root)
    logger.info("Videos (%d): %s", len(video_names), video_names)
    logger.info(
        "Protocol: feature=%s mode=%s mad_k=%s const=%s factors=%s",
        FEATURE,
        THRESHOLD_MODE,
        MAD_K,
        MAD_SCALE_CONSTANT,
        list(AGG_FACTORS),
    )

    # Preflight: inspect every cache before writing.
    for name in video_names:
        cache_dir = cache_root / name
        if not cache_dir.is_dir():
            logger.error("Missing 16x16 cache for %s under %s", name, cache_root)
            return 1
        info = inspect_cache(cache_dir)["info"]
        logger.info(
            "PREFLIGHT %s shape=%s feature_keys=%s frames=%d",
            name,
            info["rms_shape"],
            info["feature_keys"],
            info["num_frames"],
        )

    results: list[dict[str, Any]] = []
    for name in video_names:
        try:
            results.append(
                process_video(
                    video_name=name,
                    cache_dir=cache_root / name,
                    output_dir=output_root / name,
                    panel_w=int(args.panel_width),
                    panel_h=int(args.panel_height),
                    title_h=int(args.title_height),
                    overlay_alpha=float(args.overlay_alpha),
                )
            )
        except Exception as exc:
            logger.exception("Failed %s: %s", name, exc)
            results.append({"video_name": name, "status": "error", "error": str(exc)})

    batch = {
        "cache_root": str(cache_root),
        "output_root": str(output_root),
        "feature": FEATURE,
        "threshold_mode": THRESHOLD_MODE,
        "mad_k": MAD_K,
        "mad_scale_constant": MAD_SCALE_CONSTANT,
        "aggregation_factors": list(AGG_FACTORS),
        "num_videos": len(results),
        "num_ok": sum(1 for r in results if r.get("status") == "ok"),
        "num_error": sum(1 for r in results if r.get("status") != "ok"),
        "videos": results,
    }
    batch_path = output_root / "batch_summary_block_aggregation_experiment.json"
    batch_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
    # Replace old batch summary name if present.
    old_batch = output_root / "batch_summary_block16_rms_aggregation_3x2.json"
    if old_batch.is_file():
        old_batch.unlink()
    logger.info("Batch summary: %s (ok=%d error=%d)", batch_path, batch["num_ok"], batch["num_error"])
    return 0 if batch["num_error"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

"""Aggregate fixed-size sub-block RMS maps into larger non-overlapping windows.

Adopted scale (this project):
  base sub-block = 16×16 px
  aggregation factor = 12  →  effective block = 192×192 px

pooled_rms = sqrt(sum(base_rms^2) / valid_base_block_count)
Padded edge cells (value 0 from padding) are excluded from the valid count.

After aggregation, native MAD threshold + 8-connectivity CC are applied and
saved alongside the RMS map (no min-size / morphology / weak-grow).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12
DEFAULT_BASE_BLOCK_PX = 16
DEFAULT_AGG_FACTOR = 12
DEFAULT_FEATURE = "rms_mag"
DEFAULT_MAD_K = 1.5
DEFAULT_MAD_SCALE_CONSTANT = 1.4826
DEFAULT_CONNECTIVITY = 8

DEFAULT_INPUT_ROOT = Path(
    "/data1/vailab02_dir/vlm_motion/motion-detector/Result/"
    "residual_motion_120s_block16_4video"
)


@dataclass
class SubBlockAggregationConfig:
    """Spatial aggregation of a precomputed base-block score map."""

    base_block_px: int = DEFAULT_BASE_BLOCK_PX
    agg_factor: int = DEFAULT_AGG_FACTOR
    feature: str = DEFAULT_FEATURE
    mad_k: float = DEFAULT_MAD_K
    mad_scale_constant: float = DEFAULT_MAD_SCALE_CONSTANT
    mad_eps: float = EPS
    connectivity: int = DEFAULT_CONNECTIVITY
    apply_cc: bool = True

    @property
    def aggregated_pixel_size(self) -> int:
        return int(self.base_block_px) * int(self.agg_factor)

    @property
    def artifact_dirname(self) -> str:
        return f"sub_block_agg_{self.agg_factor}x{self.agg_factor}"


@dataclass
class AggregatedFrame:
    """One frame of aggregated sub-block scores."""

    frame_idx: int
    timestamp_sec: float
    score_map: np.ndarray
    valid_count: np.ndarray
    frame_height: int
    frame_width: int
    base_grid_rows: int
    base_grid_cols: int


def pool_rms_map(
    base_rms: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ceil-window pooled RMS; padded zeros are excluded from the valid count."""
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if base_rms.ndim != 2:
        raise ValueError(f"base_rms must be HxW, got {base_rms.shape}")

    rows, cols = int(base_rms.shape[0]), int(base_rms.shape[1])
    out_r = int(math.ceil(rows / float(factor)))
    out_c = int(math.ceil(cols / float(factor)))
    pad_r = out_r * factor - rows
    pad_c = out_c * factor - cols

    padded = np.pad(
        base_rms.astype(np.float64),
        ((0, pad_r), (0, pad_c)),
        mode="constant",
        constant_values=0.0,
    )
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


def pool_rms_stack(
    base_rms_stack: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply :func:`pool_rms_map` over a TxHxW stack."""
    if base_rms_stack.ndim != 3:
        raise ValueError(f"base_rms_stack must be TxHxW, got {base_rms_stack.shape}")
    pooled_list: list[np.ndarray] = []
    count_list: list[np.ndarray] = []
    for i in range(base_rms_stack.shape[0]):
        pooled, count = pool_rms_map(base_rms_stack[i], factor)
        pooled_list.append(pooled)
        count_list.append(count)
    return (
        np.stack(pooled_list, axis=0).astype(np.float32),
        np.stack(count_list, axis=0).astype(np.float32),
    )


def native_mad_threshold(
    scores: np.ndarray,
    *,
    mad_k: float = DEFAULT_MAD_K,
    mad_scale_constant: float = DEFAULT_MAD_SCALE_CONSTANT,
    mad_eps: float = EPS,
) -> tuple[float, float, float, np.ndarray]:
    """Native MAD: median + mad_k * const * MAD; if MAD < eps → threshold = median."""
    flat = scores.astype(np.float64).ravel()
    if flat.size == 0:
        return 0.0, 0.0, 0.0, np.zeros_like(scores, dtype=bool)
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    if mad < float(mad_eps):
        thr = med
    else:
        thr = med + float(mad_k) * float(mad_scale_constant) * mad
    active = scores >= thr
    return float(thr), med, mad, active.astype(bool)


def cell_pixel_bounds(
    r: int,
    c: int,
    *,
    factor: int,
    base_rows: int,
    base_cols: int,
    frame_h: int,
    frame_w: int,
    base_px: int,
) -> tuple[int, int, int, int]:
    br0 = r * factor
    br1 = min((r + 1) * factor, base_rows)
    bc0 = c * factor
    bc1 = min((c + 1) * factor, base_cols)
    x1 = bc0 * base_px
    y1 = br0 * base_px
    x2 = min(bc1 * base_px, frame_w)
    y2 = min(br1 * base_px, frame_h)
    return int(x1), int(y1), int(x2), int(y2)


def extract_8cc_blobs(
    pooled: np.ndarray,
    active: np.ndarray,
    *,
    factor: int,
    base_rows: int,
    base_cols: int,
    frame_h: int,
    frame_w: int,
    base_px: int,
    connectivity: int = DEFAULT_CONNECTIVITY,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """8-CC on active mask; no min-size / morphology / weak-grow."""
    labels = np.zeros(pooled.shape, dtype=np.int32)
    if active.size == 0 or not np.any(active):
        return labels, []

    conn = 8 if int(connectivity) >= 8 else 4
    num_labels, label_map, _stats, _ = cv2.connectedComponentsWithStats(
        active.astype(np.uint8), connectivity=conn
    )
    labels = label_map.astype(np.int32)
    blobs: list[dict[str, Any]] = []
    for label_id in range(1, num_labels):
        component = labels == label_id
        ys, xs = np.where(component)
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
            base_px=base_px,
        )
        _, _, x2, y2 = cell_pixel_bounds(
            r1,
            c1,
            factor=factor,
            base_rows=base_rows,
            base_cols=base_cols,
            frame_h=frame_h,
            frame_w=frame_w,
            base_px=base_px,
        )
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
    return labels, blobs


def load_base_rms_cache(
    representation_dir: str | Path,
    *,
    feature: str = DEFAULT_FEATURE,
    expected_block_size: int = DEFAULT_BASE_BLOCK_PX,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Load base sub-block RMS stack from a block-representation cache dir."""
    root = Path(representation_dir)
    meta_path = root / "representation_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    bs = int(meta.get("block_size", expected_block_size))
    if bs != int(expected_block_size):
        raise ValueError(
            f"Expected base block_size={expected_block_size}, got {bs} in {root}"
        )

    feat_path = root / "block_features.npz"
    score_path = root / "block_score_map.npy"
    heat_path = root / "block_heatmap.npy"

    rms: np.ndarray | None = None
    source_file = ""
    if feat_path.is_file():
        with np.load(feat_path) as z:
            if feature not in z.files:
                raise KeyError(f"{feature!r} missing in {feat_path}; keys={z.files}")
            rms = z[feature].astype(np.float32)
            source_file = str(feat_path)
    elif score_path.is_file():
        rms = np.load(score_path).astype(np.float32)
        source_file = str(score_path)
    elif heat_path.is_file():
        rms = np.load(heat_path).astype(np.float32)
        source_file = str(heat_path)
    else:
        raise FileNotFoundError(
            f"No block_features.npz / block_score_map.npy under {root}"
        )

    if rms.ndim != 3:
        raise ValueError(f"Expected TxRowsxCols, got {rms.shape}")

    frames = list(meta.get("frames", []) or [])
    meta = dict(meta)
    meta["_rms_source_file"] = source_file
    meta["_feature"] = feature
    return rms, meta, frames


def aggregate_from_representation(
    representation_dir: str | Path,
    *,
    config: SubBlockAggregationConfig | None = None,
) -> tuple[list[AggregatedFrame], dict[str, Any]]:
    """Load base cache and aggregate to the configured factor."""
    cfg = config or SubBlockAggregationConfig()
    rms, meta, frames = load_base_rms_cache(
        representation_dir,
        feature=cfg.feature,
        expected_block_size=cfg.base_block_px,
    )
    pooled, counts = pool_rms_stack(rms, cfg.agg_factor)
    t, base_r, base_c = int(rms.shape[0]), int(rms.shape[1]), int(rms.shape[2])

    results: list[AggregatedFrame] = []
    for i in range(t):
        if i < len(frames):
            fm = frames[i]
            frame_idx = int(fm.get("frame_idx", i))
            ts = float(fm.get("timestamp_sec", 0.0))
            frame_h = int(fm.get("frame_height", base_r * cfg.base_block_px))
            frame_w = int(fm.get("frame_width", base_c * cfg.base_block_px))
        else:
            frame_idx = i
            ts = float(i)
            frame_h = base_r * cfg.base_block_px
            frame_w = base_c * cfg.base_block_px

        results.append(
            AggregatedFrame(
                frame_idx=frame_idx,
                timestamp_sec=ts,
                score_map=pooled[i],
                valid_count=counts[i],
                frame_height=frame_h,
                frame_width=frame_w,
                base_grid_rows=base_r,
                base_grid_cols=base_c,
            )
        )

    info = {
        "source_representation_dir": str(Path(representation_dir).resolve()),
        "source_rms_file": meta.get("_rms_source_file"),
        "feature": cfg.feature,
        "base_block_px": int(cfg.base_block_px),
        "agg_factor": int(cfg.agg_factor),
        "aggregated_pixel_size": cfg.aggregated_pixel_size,
        "num_frames": t,
        "base_grid": [base_r, base_c],
        "agg_grid": [int(pooled.shape[1]), int(pooled.shape[2])],
        "pooled_rms_formula": "sqrt(sum(base_rms**2) / valid_base_block_count)",
        "padding_zeros_excluded": True,
        "representation_metadata": {
            k: meta[k]
            for k in (
                "stage",
                "block_size",
                "heatmap_feature",
                "block_score",
                "num_frames",
                "source",
            )
            if k in meta
        },
    }
    return results, info


def save_sub_block_aggregation(
    frames: list[AggregatedFrame],
    out_dir: str | Path,
    *,
    config: SubBlockAggregationConfig,
    extra_meta: dict[str, Any] | None = None,
    video_name: str | None = None,
) -> dict[str, str]:
    """Persist aggregated RMS + (optional) native-MAD / 8-CC artifacts."""
    if not frames:
        raise ValueError("No aggregated frames to save")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config
    stem = video_name or out.parent.name

    score_stack = np.stack([f.score_map for f in frames], axis=0).astype(np.float32)
    count_stack = np.stack([f.valid_count for f in frames], axis=0).astype(np.float32)

    rms_path = out / "aggregated_rms_mag.npy"
    count_path = out / "valid_base_block_count.npy"
    np.save(rms_path, score_stack)
    np.save(count_path, count_stack)

    paths: dict[str, str] = {
        "aggregated_rms_mag_npy": str(rms_path),
        "valid_base_block_count_npy": str(count_path),
    }

    frame_meta: list[dict[str, Any]] = []
    active_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    component_rows: list[dict[str, Any]] = []
    total_blobs = 0

    for i, fr in enumerate(frames):
        entry: dict[str, Any] = {
            "frame_index": i,
            "frame_idx": fr.frame_idx,
            "timestamp_sec": round(float(fr.timestamp_sec), 4),
            "frame_height": fr.frame_height,
            "frame_width": fr.frame_width,
            "base_grid_rows": fr.base_grid_rows,
            "base_grid_cols": fr.base_grid_cols,
            "agg_grid_rows": int(fr.score_map.shape[0]),
            "agg_grid_cols": int(fr.score_map.shape[1]),
        }

        if cfg.apply_cc:
            thr, med, mad, active = native_mad_threshold(
                fr.score_map,
                mad_k=cfg.mad_k,
                mad_scale_constant=cfg.mad_scale_constant,
                mad_eps=cfg.mad_eps,
            )
            labels, blobs = extract_8cc_blobs(
                fr.score_map,
                active,
                factor=cfg.agg_factor,
                base_rows=fr.base_grid_rows,
                base_cols=fr.base_grid_cols,
                frame_h=fr.frame_height,
                frame_w=fr.frame_width,
                base_px=cfg.base_block_px,
                connectivity=cfg.connectivity,
            )
            active_list.append(active.astype(np.uint8))
            label_list.append(labels.astype(np.int32))
            entry.update(
                {
                    "threshold": thr,
                    "median": med,
                    "mad": mad,
                    "n_active_cells": int(active.sum()),
                    "n_blobs": len(blobs),
                }
            )
            total_blobs += len(blobs)
            px = cfg.aggregated_pixel_size
            for blob in blobs:
                x1, y1, x2, y2 = blob["bbox"]
                component_rows.append(
                    {
                        "video_name": stem,
                        "frame_index": i,
                        "frame_idx": fr.frame_idx,
                        "timestamp_sec": round(float(fr.timestamp_sec), 6),
                        "factor": int(cfg.agg_factor),
                        "pixel_size": int(px),
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
                        "grid_r0": blob["grid_r0"],
                        "grid_r1": blob["grid_r1"],
                        "grid_c0": blob["grid_c0"],
                        "grid_c1": blob["grid_c1"],
                    }
                )

        frame_meta.append(entry)

    if cfg.apply_cc and active_list:
        active_path = out / "active_mask.npy"
        labels_path = out / "cc_labels.npy"
        components_path = out / "components.csv"
        np.save(active_path, np.stack(active_list, axis=0).astype(np.uint8))
        np.save(labels_path, np.stack(label_list, axis=0).astype(np.int32))

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
            "grid_r0",
            "grid_r1",
            "grid_c0",
            "grid_c1",
        ]
        with components_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in component_rows:
                w.writerow(row)

        paths.update(
            {
                "active_mask_npy": str(active_path),
                "cc_labels_npy": str(labels_path),
                "components_csv": str(components_path),
            }
        )

    meta: dict[str, Any] = {
        "stage": "sub_block_aggregation",
        "feature": cfg.feature,
        "base_block_px": int(cfg.base_block_px),
        "agg_factor": int(cfg.agg_factor),
        "agg_label": f"{cfg.agg_factor}x{cfg.agg_factor}",
        "aggregated_pixel_size": cfg.aggregated_pixel_size,
        "pooled_rms_formula": "sqrt(sum(base_rms**2) / valid_base_block_count)",
        "padding_zeros_excluded": True,
        "num_frames": len(frames),
        "agg_grid_rows": int(score_stack.shape[1]),
        "agg_grid_cols": int(score_stack.shape[2]),
        "apply_cc": bool(cfg.apply_cc),
        "frames": frame_meta,
    }
    if cfg.apply_cc:
        meta.update(
            {
                "threshold_mode": "native",
                "mad_k": float(cfg.mad_k),
                "mad_scale_constant": float(cfg.mad_scale_constant),
                "mad_eps": float(cfg.mad_eps),
                "connectivity": int(cfg.connectivity),
                "min_blob_size_filter": False,
                "morphology": False,
                "weak_grow": False,
                "total_blobs": int(total_blobs),
                "mean_blobs_per_frame": float(total_blobs) / max(len(frames), 1),
            }
        )
    if extra_meta:
        meta.update(extra_meta)

    meta_path = out / "aggregation_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["aggregation_metadata_json"] = str(meta_path)

    logger.info(
        "Saved sub-block aggregation → %s (%d frames, factor=%dx%d, px=%d, grid=%dx%d, cc=%s, blobs=%d)",
        out,
        len(frames),
        cfg.agg_factor,
        cfg.agg_factor,
        cfg.aggregated_pixel_size,
        score_stack.shape[1],
        score_stack.shape[2],
        cfg.apply_cc,
        total_blobs,
    )
    return paths


def load_sub_block_aggregation(
    aggregation_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load saved aggregation artifacts."""
    root = Path(aggregation_dir)
    meta_path = root / "aggregation_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    score_path = root / "aggregated_rms_mag.npy"
    if not score_path.is_file():
        raise FileNotFoundError(f"Missing aggregated_rms_mag.npy under {root}")
    count_path = root / "valid_base_block_count.npy"
    if not count_path.is_file():
        raise FileNotFoundError(f"Missing {count_path}")

    score = np.load(score_path).astype(np.float32)
    counts = np.load(count_path).astype(np.float32)
    if score.shape != counts.shape:
        raise ValueError(f"Shape mismatch score {score.shape} vs count {counts.shape}")
    return score, counts, meta


def load_sub_block_cc(
    aggregation_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, Path, dict[str, Any]]:
    """Load 8-CC artifacts saved next to the aggregation cache."""
    root = Path(aggregation_dir)
    _score, _counts, meta = load_sub_block_aggregation(root)
    active_path = root / "active_mask.npy"
    labels_path = root / "cc_labels.npy"
    components_path = root / "components.csv"
    for p in (active_path, labels_path, components_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing 8-CC artifact: {p}")
    active = np.load(active_path).astype(np.uint8)
    labels = np.load(labels_path).astype(np.int32)
    return active, labels, components_path, meta


def process_and_save_video(
    representation_dir: str | Path,
    *,
    config: SubBlockAggregationConfig | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate one video's base cache and save under ``sub_block_agg_{k}x{k}/``."""
    cfg = config or SubBlockAggregationConfig()
    rep = Path(representation_dir)
    out = Path(output_dir) if output_dir is not None else rep / cfg.artifact_dirname

    frames, info = aggregate_from_representation(rep, config=cfg)
    paths = save_sub_block_aggregation(
        frames,
        out,
        config=cfg,
        video_name=rep.name,
        extra_meta={
            "source_representation_dir": info["source_representation_dir"],
            "source_rms_file": info["source_rms_file"],
            "base_grid": info["base_grid"],
            "representation_metadata": info["representation_metadata"],
        },
    )
    return {
        "video_name": rep.name,
        "status": "ok",
        "representation_dir": str(rep),
        "output_dir": str(out),
        "num_frames": info["num_frames"],
        "base_grid": info["base_grid"],
        "agg_grid": info["agg_grid"],
        "aggregated_pixel_size": cfg.aggregated_pixel_size,
        "apply_cc": cfg.apply_cc,
        "artifacts": paths,
    }


def discover_representation_dirs(
    input_root: Path,
    video_names: list[str] | None = None,
) -> list[Path]:
    if video_names:
        dirs = [input_root / n for n in video_names]
    else:
        dirs = sorted(p for p in input_root.iterdir() if p.is_dir())
    out: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Missing video cache dir: {d}")
        if not (d / "representation_metadata.json").is_file():
            raise FileNotFoundError(f"No representation_metadata.json in {d}")
        out.append(d)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate 16×16 sub-block RMS to 12×12 (192×192 px) + 8-CC and save."
    )
    p.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--video_names", nargs="+", default=None)
    p.add_argument("--base_block_px", type=int, default=DEFAULT_BASE_BLOCK_PX)
    p.add_argument("--agg_factor", type=int, default=DEFAULT_AGG_FACTOR)
    p.add_argument("--feature", type=str, default=DEFAULT_FEATURE)
    p.add_argument("--mad_k", type=float, default=DEFAULT_MAD_K)
    p.add_argument("--no_cc", action="store_true", help="Skip native MAD + 8-CC artifacts.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        logger.error("input_root not found: %s", input_root)
        return 1

    cfg = SubBlockAggregationConfig(
        base_block_px=int(args.base_block_px),
        agg_factor=int(args.agg_factor),
        feature=str(args.feature),
        mad_k=float(args.mad_k),
        apply_cc=not bool(args.no_cc),
    )
    videos = discover_representation_dirs(input_root, args.video_names)
    logger.info(
        "Aggregating %d video(s): base=%dpx factor=%dx%d → %dpx  feature=%s  cc=%s mad_k=%s",
        len(videos),
        cfg.base_block_px,
        cfg.agg_factor,
        cfg.agg_factor,
        cfg.aggregated_pixel_size,
        cfg.feature,
        cfg.apply_cc,
        cfg.mad_k,
    )

    results: list[dict[str, Any]] = []
    for rep_dir in videos:
        try:
            results.append(process_and_save_video(rep_dir, config=cfg))
        except Exception as exc:
            logger.exception("Failed %s: %s", rep_dir.name, exc)
            results.append(
                {
                    "video_name": rep_dir.name,
                    "status": "error",
                    "error": str(exc),
                    "representation_dir": str(rep_dir),
                }
            )

    batch = {
        "input_root": str(input_root),
        "base_block_px": cfg.base_block_px,
        "agg_factor": cfg.agg_factor,
        "aggregated_pixel_size": cfg.aggregated_pixel_size,
        "feature": cfg.feature,
        "apply_cc": cfg.apply_cc,
        "mad_k": cfg.mad_k,
        "mad_scale_constant": cfg.mad_scale_constant,
        "num_videos": len(results),
        "num_ok": sum(1 for r in results if r.get("status") == "ok"),
        "num_error": sum(1 for r in results if r.get("status") != "ok"),
        "videos": results,
    }
    batch_path = input_root / f"batch_summary_sub_block_agg_{cfg.agg_factor}x{cfg.agg_factor}.json"
    batch_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Batch summary: %s (ok=%d error=%d)", batch_path, batch["num_ok"], batch["num_error"])
    return 0 if batch["num_error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

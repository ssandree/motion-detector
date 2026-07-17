"""Residual-flow block representation (fixed pixel block grids, no blob stage)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-8
MAD_CONSISTENCY = 1.4826
SUPPORTED_BLOCK_SIZES = (2, 4, 8, 16)


@dataclass
class BlockRepresentationConfig:
    """Block motion representation only (features + score map, no threshold)."""

    block_size: int = 16
    heatmap_feature: str = "rms_mag"
    topk_ratio: float = 0.1
    active_mag_threshold: float = 0.5
    overlay_alpha: float = 0.55


@dataclass
class BlockFeatureMaps:
    """Per-frame block feature grids (rows × cols)."""

    mean_mag: np.ndarray
    rms_mag: np.ndarray
    topk_mag: np.ndarray
    std_mag: np.ndarray
    active_pixel_ratio: np.ndarray
    direction_coherence: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "mean_mag": self.mean_mag,
            "rms_mag": self.rms_mag,
            "topk_mag": self.topk_mag,
            "std_mag": self.std_mag,
            "active_pixel_ratio": self.active_pixel_ratio,
            "direction_coherence": self.direction_coherence,
        }


@dataclass
class FrameBlockRepresentation:
    """Block representation output: features + score map only."""

    frame_idx: int
    timestamp_sec: float
    features: BlockFeatureMaps
    score_map: np.ndarray
    frame_height: int
    frame_width: int

    @property
    def heatmap(self) -> np.ndarray:
        return self.score_map


def _block_grid_shape(h: int, w: int, block_size: int) -> tuple[int, int]:
    bs = max(1, int(block_size))
    return h // bs, w // bs


def _reshape_blocks(arr2d: np.ndarray, n_rows: int, n_cols: int, bs: int) -> np.ndarray:
    """(H,W) → (n_rows, n_cols, bs*bs) for integer-tiled frames."""
    return (
        arr2d.reshape(n_rows, bs, n_cols, bs)
        .transpose(0, 2, 1, 3)
        .reshape(n_rows, n_cols, bs * bs)
    )


def compute_block_features(
    residual_flow: np.ndarray,
    *,
    block_size: int = 16,
    topk_ratio: float = 0.1,
    active_mag_threshold: float = 0.5,
) -> BlockFeatureMaps:
    """Aggregate residual flow into fixed pixel-size blocks (vectorized)."""
    if residual_flow.ndim != 3 or residual_flow.shape[2] != 2:
        raise ValueError(f"residual_flow must be HxWx2, got {residual_flow.shape}")

    h, w = residual_flow.shape[:2]
    bs = max(1, int(block_size))
    n_rows, n_cols = _block_grid_shape(h, w, bs)
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f"Frame {h}x{w} too small for block_size={bs}")

    hc, wc = n_rows * bs, n_cols * bs
    flow = residual_flow[:hc, :wc]
    fx = flow[..., 0].astype(np.float64)
    fy = flow[..., 1].astype(np.float64)
    mag = np.sqrt(fx * fx + fy * fy)

    mag_b = _reshape_blocks(mag, n_rows, n_cols, bs)
    fx_b = _reshape_blocks(fx, n_rows, n_cols, bs)
    fy_b = _reshape_blocks(fy, n_rows, n_cols, bs)

    mean_mag = mag_b.mean(axis=-1).astype(np.float32)
    rms_mag = np.sqrt(np.mean(mag_b * mag_b, axis=-1)).astype(np.float32)
    std_mag = mag_b.std(axis=-1).astype(np.float32)
    active_pixel_ratio = (mag_b > float(active_mag_threshold)).mean(axis=-1).astype(np.float32)

    ratio = float(np.clip(topk_ratio, 1e-3, 1.0))
    k = max(1, int(np.ceil(bs * bs * ratio)))
    if k >= mag_b.shape[-1]:
        topk_mag = mean_mag
    else:
        part = np.partition(mag_b, mag_b.shape[-1] - k, axis=-1)
        topk_mag = part[..., -k:].mean(axis=-1).astype(np.float32)

    active = mag_b > EPS
    angles = np.arctan2(fy_b, fx_b)
    cos_a = np.where(active, np.cos(angles), 0.0)
    sin_a = np.where(active, np.sin(angles), 0.0)
    n_active = active.sum(axis=-1).astype(np.float64)
    mean_cos = cos_a.sum(axis=-1) / np.maximum(n_active, 1.0)
    mean_sin = sin_a.sum(axis=-1) / np.maximum(n_active, 1.0)
    coherence = np.sqrt(mean_cos * mean_cos + mean_sin * mean_sin)
    min_active = 4 if bs * bs >= 4 else 1
    coherence = np.where(n_active >= min_active, coherence, 0.0).astype(np.float32)

    return BlockFeatureMaps(
        mean_mag=mean_mag,
        rms_mag=rms_mag,
        topk_mag=topk_mag,
        std_mag=std_mag,
        active_pixel_ratio=active_pixel_ratio,
        direction_coherence=coherence,
    )


def median_mad_threshold(
    scores: np.ndarray,
    *,
    mad_scale: float = 4.0,
    mad_consistency: float = MAD_CONSISTENCY,
) -> tuple[float, float, float, np.ndarray]:
    """Return (threshold, median, mad, active_mask)."""
    flat = scores.astype(np.float64).ravel()
    if flat.size == 0:
        return 0.0, 0.0, 0.0, np.zeros_like(scores, dtype=bool)

    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med)))
    if mad < EPS:
        threshold = med
        active = scores > med
    else:
        threshold = med + float(mad_scale) * float(mad_consistency) * mad
        active = scores >= threshold
    return float(threshold), med, mad, active.astype(bool)


def process_block_representation_frame(
    residual_flow: np.ndarray,
    *,
    frame_idx: int,
    timestamp_sec: float,
    config: BlockRepresentationConfig | None = None,
) -> FrameBlockRepresentation:
    """Block representation only: features + score_map."""
    cfg = config or BlockRepresentationConfig()
    features = compute_block_features(
        residual_flow,
        block_size=cfg.block_size,
        topk_ratio=cfg.topk_ratio,
        active_mag_threshold=cfg.active_mag_threshold,
    )
    feat_map = features.as_dict().get(cfg.heatmap_feature)
    if feat_map is None:
        raise ValueError(
            f"Unknown heatmap_feature={cfg.heatmap_feature!r}; "
            f"expected one of {list(features.as_dict())}"
        )
    h, w = residual_flow.shape[:2]
    return FrameBlockRepresentation(
        frame_idx=int(frame_idx),
        timestamp_sec=float(timestamp_sec),
        features=features,
        score_map=feat_map.astype(np.float32),
        frame_height=int(h),
        frame_width=int(w),
    )


def process_block_representation(
    residual_flows: np.ndarray,
    frame_indices: list[int],
    timestamps_sec: list[float],
    config: BlockRepresentationConfig | None = None,
) -> list[FrameBlockRepresentation]:
    """Run block representation over all residual-flow frames."""
    results: list[FrameBlockRepresentation] = []
    for i in range(residual_flows.shape[0]):
        results.append(
            process_block_representation_frame(
                residual_flows[i],
                frame_idx=frame_indices[i],
                timestamp_sec=timestamps_sec[i],
                config=config,
            )
        )
    return results


def load_residual_flow_bundle(
    residual_flow_dir: str | Path,
) -> tuple[np.ndarray, list[int], list[float], dict[str, Any]]:
    """Load residual_flow.npy + frame indices/timestamps from global_motion.json."""
    root = Path(residual_flow_dir)
    flow_path = root / "residual_flow.npy"
    meta_path = root / "global_motion.json"
    if not flow_path.is_file():
        raise FileNotFoundError(f"Missing residual_flow.npy under {root}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing global_motion.json under {root}")

    flows = np.load(flow_path)
    if flows.ndim != 4 or flows.shape[-1] != 2:
        raise ValueError(f"residual_flow.npy expected TxHxWx2, got {flows.shape}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = meta.get("frames", [])
    if len(frames) != flows.shape[0]:
        raise ValueError(
            f"Frame count mismatch: residual_flow T={flows.shape[0]} "
            f"vs global_motion frames={len(frames)}"
        )
    frame_indices = [int(f["frame_idx_curr"]) for f in frames]
    timestamps = [float(f["timestamp_sec_curr"]) for f in frames]
    return flows.astype(np.float32), frame_indices, timestamps, meta


def save_block_representation_artifacts(
    representations: list[FrameBlockRepresentation],
    out_dir: str | Path,
    *,
    config: BlockRepresentationConfig,
    source_meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save block representation: features + score map."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config

    if not representations:
        raise ValueError("No block representations to save")

    feature_stacks: dict[str, list[np.ndarray]] = {
        "mean_mag": [],
        "rms_mag": [],
        "topk_mag": [],
        "std_mag": [],
        "active_pixel_ratio": [],
        "direction_coherence": [],
    }
    score_maps: list[np.ndarray] = []
    frame_meta: list[dict[str, Any]] = []

    for rep in representations:
        for key, arr in rep.features.as_dict().items():
            feature_stacks[key].append(arr)
        score_maps.append(rep.score_map.astype(np.float32))
        frame_meta.append(
            {
                "frame_idx": rep.frame_idx,
                "timestamp_sec": round(rep.timestamp_sec, 4),
                "frame_height": rep.frame_height,
                "frame_width": rep.frame_width,
                "grid_rows": int(rep.score_map.shape[0]),
                "grid_cols": int(rep.score_map.shape[1]),
            }
        )

    feat_path = out / "block_features.npz"
    if int(cfg.block_size) <= 2:
        np.savez_compressed(
            feat_path,
            rms_mag=np.stack(feature_stacks["rms_mag"], axis=0).astype(np.float32),
        )
    else:
        np.savez_compressed(
            feat_path,
            **{k: np.stack(v, axis=0).astype(np.float32) for k, v in feature_stacks.items()},
        )

    score_stack = np.stack(score_maps, axis=0).astype(np.float32)
    score_path = out / "block_score_map.npy"
    heat_path = out / "block_heatmap.npy"
    np.save(score_path, score_stack)
    np.save(heat_path, score_stack)

    meta = {
        "stage": "block_representation",
        "block_size": int(cfg.block_size),
        "heatmap_feature": cfg.heatmap_feature,
        "block_score": cfg.heatmap_feature,
        "num_frames": len(representations),
        "topk_ratio": cfg.topk_ratio,
        "active_mag_threshold": cfg.active_mag_threshold,
        "source": source_meta or {},
        "frames": frame_meta,
    }
    meta_path = out / "representation_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Saved block representation: %s + %s (%d frames, block=%dx%d, score=%s)",
        feat_path.name,
        score_path.name,
        len(representations),
        cfg.block_size,
        cfg.block_size,
        cfg.heatmap_feature,
    )
    return {
        "block_features_npz": str(feat_path),
        "block_score_map_npy": str(score_path),
        "block_heatmap_npy": str(heat_path),
        "representation_metadata_json": str(meta_path),
    }


def load_block_representation_bundle(
    representation_dir: str | Path,
    *,
    residual_flow_dir: str | Path | None = None,
    block_size: int = 16,
    score_key: str = "rms_mag",
) -> tuple[list[FrameBlockRepresentation], dict[str, Any]]:
    """Load block representation artifacts from disk."""
    root = Path(representation_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Representation dir not found: {root}")

    meta: dict[str, Any] = {}
    meta_path = root / "representation_metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    score_path = root / "block_score_map.npy"
    heat_path = root / "block_heatmap.npy"
    feat_path = root / "block_features.npz"

    score_stack: np.ndarray | None = None
    if score_path.is_file():
        score_stack = np.load(score_path).astype(np.float32)
    elif heat_path.is_file():
        score_stack = np.load(heat_path).astype(np.float32)
    elif feat_path.is_file():
        with np.load(feat_path) as z:
            if score_key not in z.files:
                raise FileNotFoundError(
                    f"Missing score key {score_key!r} in {feat_path}; keys={z.files}"
                )
            score_stack = z[score_key].astype(np.float32)
    else:
        raise FileNotFoundError(
            f"No block_score_map.npy / block_heatmap.npy / block_features.npz under {root}"
        )

    if score_stack.ndim != 3:
        raise ValueError(f"score map expected TxRowsxCols, got {score_stack.shape}")

    feature_maps: dict[str, np.ndarray] = {}
    if feat_path.is_file():
        with np.load(feat_path) as z:
            feature_maps = {k: z[k].astype(np.float32) for k in z.files}

    frame_meta = list(meta.get("frames", []) or [])
    residual_meta: dict[str, Any] = {}
    if residual_flow_dir is not None:
        residual_meta_path = Path(residual_flow_dir) / "global_motion.json"
        if residual_meta_path.is_file():
            residual_meta = json.loads(residual_meta_path.read_text(encoding="utf-8"))

    residual_frames = list(residual_meta.get("frames", []) or [])
    t = int(score_stack.shape[0])
    bs = int(meta.get("block_size", block_size))

    representations: list[FrameBlockRepresentation] = []
    for i in range(t):
        if i < len(frame_meta):
            fm = frame_meta[i]
            frame_idx = int(fm.get("frame_idx", i))
            timestamp_sec = float(fm.get("timestamp_sec", 0.0))
            frame_h = int(fm.get("frame_height", score_stack.shape[1] * bs))
            frame_w = int(fm.get("frame_width", score_stack.shape[2] * bs))
        elif i < len(residual_frames):
            rf = residual_frames[i]
            frame_idx = int(rf["frame_idx_curr"])
            timestamp_sec = float(rf["timestamp_sec_curr"])
            frame_h = int(score_stack.shape[1] * bs)
            frame_w = int(score_stack.shape[2] * bs)
        else:
            frame_idx = i
            timestamp_sec = float(i)
            frame_h = int(score_stack.shape[1] * bs)
            frame_w = int(score_stack.shape[2] * bs)

        if feature_maps:
            score_i = score_stack[i]
            zeros = np.zeros_like(score_i)

            def _feat(key: str, default: np.ndarray) -> np.ndarray:
                arr = feature_maps.get(key)
                if arr is None:
                    return default
                return arr[i]

            features = BlockFeatureMaps(
                mean_mag=_feat("mean_mag", score_i),
                rms_mag=_feat("rms_mag", score_i),
                topk_mag=_feat("topk_mag", score_i),
                std_mag=_feat("std_mag", zeros),
                active_pixel_ratio=_feat("active_pixel_ratio", zeros),
                direction_coherence=_feat("direction_coherence", zeros),
            )
        else:
            score_i = score_stack[i]
            zeros = np.zeros_like(score_i)
            features = BlockFeatureMaps(
                mean_mag=score_i,
                rms_mag=score_i,
                topk_mag=score_i,
                std_mag=zeros,
                active_pixel_ratio=zeros,
                direction_coherence=zeros,
            )

        representations.append(
            FrameBlockRepresentation(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                features=features,
                score_map=score_stack[i].astype(np.float32),
                frame_height=frame_h,
                frame_width=frame_w,
            )
        )

    bundle_meta = {
        "representation_dir": str(root),
        "residual_flow_dir": str(residual_flow_dir) if residual_flow_dir else None,
        "block_size": bs,
        "score_key": score_key,
        "num_frames": t,
        "grid_rows": int(score_stack.shape[1]),
        "grid_cols": int(score_stack.shape[2]),
        "representation_metadata": meta,
    }
    return representations, bundle_meta

"""Stage1 GPU path: keep gray on device, Farneback + R_ap + 8×8 spatial mean.

Pipeline:
  CPU decode/gray
  → upload once + ¼ resize (GPU, kept)
  → CUDA Farneback (flow stays on GPU)
  → CUDA Sobel R_ap on the same ¼ gray (no extra upload)
  → v = Σ(R_ap v)/Σ(R_ap) over 8×8 via boxFilter (temporal_radius=0 for Stage1)
  → download only the 16px grid
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np
import logging

from motion_analyzer.config import (
    APERTURE_ALPHA,
    APERTURE_GAMMA,
    APERTURE_TAU_EDGE,
    INPUT_SCALE,
    STRUCTURE_TENSOR_EPS,
    STRUCTURE_TENSOR_SIGMA,
)
from motion_analyzer.farneback import (
    _get_cuda_farneback_algo,
    downscale_gray_cuda,
    quarter_scale_size,
    require_cuda_farneback,
    upload_gray_gpu,
)
from motion_analyzer.structure_tensor import aperture_reliability_gpu

logger = logging.getLogger("stage1_gpu")


def block_grid_shape(height: int, width: int, *, stride: int = 4) -> tuple[int, int]:
    return (
        int(math.ceil(height / float(stride))),
        int(math.ceil(width / float(stride))),
    )


def _scale_and_threshold_flow_gpu(
    flow_gpu: cv2.cuda.GpuMat,
    *,
    scale_x: float,
    scale_y: float,
    mag_threshold: float,
) -> tuple[cv2.cuda.GpuMat, cv2.cuda.GpuMat]:
    """Return (u, v) GpuMats in original-px units, optionally zeroed by ‖v‖ thr."""
    u, v = cv2.cuda.split(flow_gpu)
    if float(scale_x) != 1.0:
        u = cv2.cuda.multiplyWithScalar(u, float(scale_x))
    if float(scale_y) != 1.0:
        v = cv2.cuda.multiplyWithScalar(v, float(scale_y))
    if float(mag_threshold) > 0.0:
        # Avoid cuda.merge (Python binding often returns ndarray); ‖v‖ via sqrt(u²+v²).
        mag = cv2.cuda.sqrt(
            cv2.cuda.add(cv2.cuda.multiply(u, u), cv2.cuda.multiply(v, v))
        )
        # THRESH_BINARY: 1 if mag > thr else 0 (float32 mask).
        _ret, keep_f = cv2.cuda.threshold(
            mag, float(mag_threshold), 1.0, cv2.THRESH_BINARY
        )
        u = cv2.cuda.multiply(u, keep_f)
        v = cv2.cuda.multiply(v, keep_f)
    return u, v


@lru_cache(maxsize=8)
def _box_filter_32f(window: int):
    require_cuda_farneback()
    w = int(window)
    return cv2.cuda.createBoxFilter(
        cv2.CV_32FC1,
        cv2.CV_32FC1,
        (w, w),
        (-1, -1),
    )


@lru_cache(maxsize=16)
def _center_remap_maps(
    height: int, width: int, *, stride: int, window: int
) -> tuple[cv2.cuda.GpuMat, cv2.cuda.GpuMat, int, int]:
    """GpuMat mapx/mapy sampling boxFilter centers on the stride grid."""
    del window  # centers depend on stride only; window kept for cache clarity
    require_cuda_farneback()
    hb, wb = block_grid_shape(height, width, stride=int(stride))
    ys = np.empty((hb, wb), dtype=np.float32)
    xs = np.empty((hb, wb), dtype=np.float32)
    half = int(stride) // 2
    for row in range(hb):
        cy = min(row * int(stride) + half, int(height) - 1)
        for col in range(wb):
            cx = min(col * int(stride) + half, int(width) - 1)
            ys[row, col] = float(cy)
            xs[row, col] = float(cx)
    mapx = cv2.cuda.GpuMat()
    mapy = cv2.cuda.GpuMat()
    mapx.upload(xs)
    mapy.upload(ys)
    return mapx, mapy, hb, wb


def _spatial_mean_grid_gpu(
    u_gpu: cv2.cuda.GpuMat,
    v_gpu: cv2.cuda.GpuMat,
    *,
    height: int,
    width: int,
    stride: int,
    window: int,
    weight_gpu: cv2.cuda.GpuMat | None = None,
) -> np.ndarray:
    """8×8 (window) mean @ stride → host (Hb,Wb,2). Dense flow never leaves GPU.

    If ``weight_gpu`` is set, returns Σ(w v)/Σ(w) via boxFilter on (w·u, w·v, w).
    """
    filt = _box_filter_32f(int(window))
    mapx, mapy, _hb, _wb = _center_remap_maps(
        int(height), int(width), stride=int(stride), window=int(window)
    )
    if weight_gpu is None:
        u_f = filt.apply(u_gpu)
        v_f = filt.apply(v_gpu)
        u_s = cv2.cuda.remap(u_f, mapx, mapy, interpolation=cv2.INTER_NEAREST)
        v_s = cv2.cuda.remap(v_f, mapx, mapy, interpolation=cv2.INTER_NEAREST)
        out = np.empty((u_s.size()[1], u_s.size()[0], 2), dtype=np.float32)
        out[..., 0] = np.asarray(u_s.download(), dtype=np.float32)
        out[..., 1] = np.asarray(v_s.download(), dtype=np.float32)
        return out

    uw = cv2.cuda.multiply(u_gpu, weight_gpu)
    vw = cv2.cuda.multiply(v_gpu, weight_gpu)
    u_s = cv2.cuda.remap(
        filt.apply(uw), mapx, mapy, interpolation=cv2.INTER_NEAREST
    )
    v_s = cv2.cuda.remap(
        filt.apply(vw), mapx, mapy, interpolation=cv2.INTER_NEAREST
    )
    w_s = cv2.cuda.remap(
        filt.apply(weight_gpu), mapx, mapy, interpolation=cv2.INTER_NEAREST
    )
    su = np.asarray(u_s.download(), dtype=np.float64)
    sv = np.asarray(v_s.download(), dtype=np.float64)
    sw = np.asarray(w_s.download(), dtype=np.float64)
    # Keep Σ(w v) and Σ(w) in channels 0/1/2 via packing: store (su, sv) and
    # encode sw in a side array by returning 3-channel host block.
    out = np.empty(su.shape + (3,), dtype=np.float32)
    out[..., 0] = su.astype(np.float32)
    out[..., 1] = sv.astype(np.float32)
    out[..., 2] = sw.astype(np.float32)
    return out


def _temporal_mean_blocks(
    blocks: np.ndarray, *, temporal_radius: int
) -> np.ndarray:
    """Mean over [t−R, t+R] on (T,Hb,Wb,2) host blocks (already spatially reduced)."""
    stack = np.asarray(blocks, dtype=np.float32)
    if stack.ndim != 4 or stack.shape[-1] != 2:
        raise ValueError(f"expected (T,Hb,Wb,2), got {stack.shape}")
    t_len = int(stack.shape[0])
    r = int(temporal_radius)
    if r <= 0:
        return stack.copy()
    prefix = np.empty((t_len + 1,) + stack.shape[1:], dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(stack.astype(np.float64, copy=False), axis=0, out=prefix[1:])
    out = np.empty_like(stack)
    for t in range(t_len):
        t0 = max(0, t - r)
        t1 = min(t_len, t + r + 1)
        out[t] = ((prefix[t1] - prefix[t0]) / float(t1 - t0)).astype(np.float32)
    return out


def _temporal_weighted_mean_blocks(
    blocks: np.ndarray, *, temporal_radius: int
) -> np.ndarray:
    """3D weighted mean from per-frame spatial (mean(w u), mean(w v), mean(w))."""
    stack = np.asarray(blocks, dtype=np.float32)
    if stack.ndim != 4 or stack.shape[-1] != 3:
        raise ValueError(f"expected (T,Hb,Wb,3), got {stack.shape}")
    t_len = int(stack.shape[0])
    r = int(temporal_radius)
    prefix = np.empty((t_len + 1,) + stack.shape[1:], dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(stack.astype(np.float64, copy=False), axis=0, out=prefix[1:])
    out = np.empty(stack.shape[:3] + (2,), dtype=np.float32)
    for t in range(t_len):
        t0 = max(0, t - r)
        t1 = min(t_len, t + r + 1)
        acc = prefix[t1] - prefix[t0]
        w = np.maximum(acc[..., 2], 1e-12)
        out[t, ..., 0] = (acc[..., 0] / w).astype(np.float32)
        out[t, ..., 1] = (acc[..., 1] / w).astype(np.float32)
        out[t] = np.where(acc[..., 2:3] > 0, out[t], 0.0)
    return out


def upload_quarter_gray_sequence(
    gray: list[np.ndarray],
    *,
    input_scale: float = INPUT_SCALE,
) -> tuple[list[cv2.cuda.GpuMat], int, int, int, int]:
    """Upload each full-res gray once, keep ¼ GpuMats for Farneback reuse."""
    require_cuda_farneback()
    if not gray:
        raise ValueError("gray sequence is empty")
    full_h, full_w = gray[0].shape[:2]
    scaled_h, scaled_w = quarter_scale_size(full_h, full_w, input_scale=input_scale)
    out: list[cv2.cuda.GpuMat] = []
    for frame in gray:
        if frame.shape[:2] != (full_h, full_w):
            raise ValueError(f"gray shape mismatch: {frame.shape} vs {(full_h, full_w)}")
        full_gpu = upload_gray_gpu(frame)
        out.append(
            downscale_gray_cuda(
                full_gpu, scaled_w=scaled_w, scaled_h=scaled_h, blur_sigma=None
            )
        )
    return out, full_h, full_w, scaled_h, scaled_w


def _compute_gap_mean_flow_from_quarter(
    gray_q: list,
    *,
    gap: int,
    stride: int,
    window: int,
    temporal_radius: int,
    mag_threshold: float,
    full_h: int,
    full_w: int,
    scaled_h: int,
    scaled_w: int,
    use_aperture_reliability: bool = True,
    tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
    tau_edge: float = APERTURE_TAU_EDGE,
    gamma: float = APERTURE_GAMMA,
    alpha: float = APERTURE_ALPHA,
    eps: float = STRUCTURE_TENSOR_EPS,
) -> np.ndarray:
    """Farneback + R_ap 8×8×±R mean on already-uploaded ¼ gray."""
    gap = int(gap)
    if gap < 1:
        raise ValueError("gap must be >= 1")
    if len(gray_q) <= gap:
        raise ValueError(f"need >{gap} frames, got {len(gray_q)}")
    if int(window) < int(stride):
        raise ValueError(f"window ({window}) must be >= stride ({stride})")

    scale_x = float(full_w) / float(scaled_w)
    scale_y = float(full_h) / float(scaled_h)
    algo = _get_cuda_farneback_algo()
    use_r = bool(use_aperture_reliability)

    spatial_blocks: list[np.ndarray] = []
    for curr in range(gap, len(gray_q)):
        flow_gpu = algo.calc(gray_q[curr - gap], gray_q[curr], None)
        u_gpu, v_gpu = _scale_and_threshold_flow_gpu(
            flow_gpu,
            scale_x=scale_x,
            scale_y=scale_y,
            mag_threshold=float(mag_threshold),
        )
        weight = None
        if use_r:
            weight = aperture_reliability_gpu(
                gray_q[curr],
                u_gpu,
                v_gpu,
                tensor_sigma=float(tensor_sigma),
                tau_edge=float(tau_edge),
                gamma=float(gamma),
                alpha=float(alpha),
                eps=float(eps),
            )
        spatial_blocks.append(
            _spatial_mean_grid_gpu(
                u_gpu,
                v_gpu,
                height=scaled_h,
                width=scaled_w,
                stride=int(stride),
                window=int(window),
                weight_gpu=weight,
            )
        )

    blocks = np.stack(spatial_blocks, axis=0)
    if use_r:
        return _temporal_weighted_mean_blocks(
            blocks, temporal_radius=int(temporal_radius)
        )
    return _temporal_mean_blocks(blocks, temporal_radius=int(temporal_radius))


def compute_gap1_mean_flow_gpu(
    gray: list[np.ndarray],
    *,
    gap: int = 1,
    stride: int = 4,
    window: int = 8,
    temporal_radius: int = 2,
    mag_threshold: float = 0.6,
    input_scale: float = INPUT_SCALE,
    use_aperture_reliability: bool = True,
    tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
    tau_edge: float = APERTURE_TAU_EDGE,
    gamma: float = APERTURE_GAMMA,
    alpha: float = APERTURE_ALPHA,
    eps: float = STRUCTURE_TENSOR_EPS,
) -> np.ndarray:
    """Gap Farneback + R_ap + 8×8×±R weighted mean → (T_flow, Hb, Wb, 2).

    Default: C1 aggregation ``v = Σ(R_ap v) / Σ(R_ap)`` over 8×8×(2R+1).
    """
    require_cuda_farneback()
    gray_q, full_h, full_w, scaled_h, scaled_w = upload_quarter_gray_sequence(
        gray, input_scale=float(input_scale)
    )
    return _compute_gap_mean_flow_from_quarter(
        gray_q,
        gap=int(gap),
        stride=int(stride),
        window=int(window),
        temporal_radius=int(temporal_radius),
        mag_threshold=float(mag_threshold),
        full_h=full_h,
        full_w=full_w,
        scaled_h=scaled_h,
        scaled_w=scaled_w,
        use_aperture_reliability=bool(use_aperture_reliability),
        tensor_sigma=float(tensor_sigma),
        tau_edge=float(tau_edge),
        gamma=float(gamma),
        alpha=float(alpha),
        eps=float(eps),
    )


def compute_aligned_gap1_gpu(
    gray: list[np.ndarray],
    sampled,
    *,
    gap: int = 1,
    block_size: int = 4,
    spatial_win: int = 8,
    temporal_radius: int = 2,
    mag_threshold: float = 0.6,
    align_start: int | None = None,
    use_aperture_reliability: bool = True,
    tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
    tau_edge: float = APERTURE_TAU_EDGE,
    gamma: float = APERTURE_GAMMA,
    alpha: float = APERTURE_ALPHA,
) -> tuple[np.ndarray, np.ndarray, list[dict], int]:
    """Stage1 Gap1 GPU path → U, M, meta_rows, align_start."""
    gap = int(gap)
    start = int(gap if align_start is None else align_start)
    if start < gap:
        start = gap
    if len(sampled) <= start:
        raise ValueError(f"need >{start} sampled frames, got {len(sampled)}")

    u_full = compute_gap1_mean_flow_gpu(
        gray,
        gap=gap,
        stride=int(block_size),
        window=int(spatial_win),
        temporal_radius=int(temporal_radius),
        mag_threshold=float(mag_threshold),
        use_aperture_reliability=bool(use_aperture_reliability),
        tensor_sigma=float(tensor_sigma),
        tau_edge=float(tau_edge),
        gamma=float(gamma),
        alpha=float(alpha),
    )
    # u_full[0] ↔ sampled index ``gap``; slice to align_start.
    idx0 = int(start) - int(gap)
    u = np.asarray(u_full[idx0:], dtype=np.float32)
    expected = len(sampled) - int(start)
    if int(u.shape[0]) != expected:
        raise RuntimeError(f"aligned length mismatch {u.shape[0]} vs {expected}")

    meta_rows: list[dict] = []
    for offset in range(expected):
        current_pos = int(start) + offset
        current = sampled[current_pos]
        previous = sampled[current_pos - gap]
        meta_rows.append(
            {
                "sampled_index_curr": current.sampled_index,
                "frame_idx_curr": current.frame_idx,
                "timestamp_sec_curr": current.timestamp_sec,
                f"sampled_index_prev_gap{gap}": previous.sampled_index,
                f"frame_idx_prev_gap{gap}": previous.frame_idx,
                f"timestamp_sec_prev_gap{gap}": previous.timestamp_sec,
            }
        )
    m = np.linalg.norm(u, axis=-1).astype(np.float32)
    return u, m, meta_rows, start


def compute_aligned_gap_stacks_gpu(
    gray: list[np.ndarray],
    sampled,
    *,
    gaps: tuple[int, ...],
    block_size: int = 4,
    spatial_win: int = 8,
    temporal_radius: int = 2,
    mag_threshold: float = 0.6,
    align_start: int | None = None,
    use_aperture_reliability: bool = True,
    tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
    tau_edge: float = APERTURE_TAU_EDGE,
    gamma: float = APERTURE_GAMMA,
    alpha: float = APERTURE_ALPHA,
    input_scale: float = INPUT_SCALE,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[dict], int]:
    """Multi-gap GPU C1: upload ¼ gray once, then R_ap-weighted 8×8×±R per gap."""
    require_cuda_farneback()
    gaps = tuple(int(g) for g in gaps)
    if not gaps:
        raise ValueError("gaps must be non-empty")
    if len(sampled) <= max(gaps):
        raise ValueError(
            f"need at least {max(gaps) + 1} sampled frames, got {len(sampled)}"
        )
    start = int(max(gaps) if align_start is None else align_start)
    if start < max(gaps):
        start = int(max(gaps))
    if int(spatial_win) < int(block_size):
        raise ValueError(
            f"spatial_win ({spatial_win}) must be >= block_size ({block_size})"
        )

    gray_q, full_h, full_w, scaled_h, scaled_w = upload_quarter_gray_sequence(
        gray, input_scale=float(input_scale)
    )

    u_stacks: dict[int, np.ndarray] = {}
    for gap in gaps:
        logger.info("  GPU R-weighted 8×8 mean gap=%d", int(gap))
        u_full = _compute_gap_mean_flow_from_quarter(
            gray_q,
            gap=int(gap),
            stride=int(block_size),
            window=int(spatial_win),
            temporal_radius=int(temporal_radius),
            mag_threshold=float(mag_threshold),
            full_h=full_h,
            full_w=full_w,
            scaled_h=scaled_h,
            scaled_w=scaled_w,
            use_aperture_reliability=bool(use_aperture_reliability),
            tensor_sigma=float(tensor_sigma),
            tau_edge=float(tau_edge),
            gamma=float(gamma),
            alpha=float(alpha),
        )
        idx0 = int(start) - int(gap)
        u_stacks[int(gap)] = np.asarray(u_full[idx0:], dtype=np.float32)

    n_aligned = int(u_stacks[gaps[0]].shape[0])
    expected = len(sampled) - int(start)
    if n_aligned != expected:
        raise RuntimeError(f"aligned length mismatch {n_aligned} vs {expected}")
    for gap in gaps:
        if int(u_stacks[gap].shape[0]) != n_aligned:
            raise RuntimeError(
                f"gap {gap} length {u_stacks[gap].shape[0]} != {n_aligned}"
            )

    meta_rows: list[dict] = []
    for offset in range(n_aligned):
        current_pos = int(start) + offset
        current = sampled[current_pos]
        meta = {
            "sampled_index_curr": current.sampled_index,
            "frame_idx_curr": current.frame_idx,
            "timestamp_sec_curr": current.timestamp_sec,
        }
        for gap in gaps:
            previous = sampled[current_pos - gap]
            meta[f"sampled_index_prev_gap{gap}"] = previous.sampled_index
            meta[f"frame_idx_prev_gap{gap}"] = previous.frame_idx
            meta[f"timestamp_sec_prev_gap{gap}"] = previous.timestamp_sec
        meta_rows.append(meta)

    m_stacks = {
        gap: np.linalg.norm(u_stacks[gap], axis=-1).astype(np.float32) for gap in gaps
    }
    return u_stacks, m_stacks, meta_rows, start


"""Structure-tensor aperture / texture reliability for dense optical flow.

Computes per-pixel reliability in [0, 1] from the local 2×2 structure tensor
on the same grayscale resolution used as Farneback input. Flow itself is never
modified here — reliability is metadata for optional weighted aggregation.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Mapping

import cv2
import numpy as np

from motion_analyzer.config import (
    APERTURE_ALPHA,
    APERTURE_GAMMA,
    APERTURE_TAU_EDGE,
    INPUT_SCALE,
    STRUCTURE_TENSOR_EPS,
    STRUCTURE_TENSOR_SIGMA,
    STRUCTURE_TENSOR_STRENGTH_TAU,
)
from motion_analyzer.farneback import require_cuda_farneback

logger = logging.getLogger(__name__)

DEFAULT_LAMBDA2_PERCENTILES = (50, 75, 90, 95, 99, 99.9)


@lru_cache(maxsize=2)
def _cuda_sobel_filters() -> tuple[object, object]:
    """Cached 3×3 CUDA Sobel (dx=1, dy=1) on float32 gray."""
    require_cuda_farneback()
    sx = cv2.cuda.createSobelFilter(cv2.CV_32FC1, cv2.CV_32FC1, 1, 0, ksize=3)
    sy = cv2.cuda.createSobelFilter(cv2.CV_32FC1, cv2.CV_32FC1, 0, 1, ksize=3)
    return sx, sy


@lru_cache(maxsize=8)
def _cuda_gaussian_32f(sigma: float):
    require_cuda_farneback()
    return cv2.cuda.createGaussianFilter(
        cv2.CV_32FC1, cv2.CV_32FC1, (0, 0), float(sigma)
    )


def gray_gpu_as_float32(gray_gpu: cv2.cuda.GpuMat) -> cv2.cuda.GpuMat:
    """uint8/float GpuMat → CV_32FC1 (no host round-trip)."""
    if int(gray_gpu.type()) == int(cv2.CV_32FC1):
        return gray_gpu
    return gray_gpu.convertTo(cv2.CV_32FC1, cv2.cuda.GpuMat())


def sobel_gradients_gpu(
    gray_gpu: cv2.cuda.GpuMat,
) -> tuple[cv2.cuda.GpuMat, cv2.cuda.GpuMat]:
    """3×3 Sobel Ix, Iy on GPU. ``gray_gpu`` is Farneback ¼ gray (uint8 or float32)."""
    require_cuda_farneback()
    img = gray_gpu_as_float32(gray_gpu)
    sx, sy = _cuda_sobel_filters()
    return sx.apply(img), sy.apply(img)


def compute_structure_tensor_fields_gpu(
    gray_gpu: cv2.cuda.GpuMat,
    *,
    tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
) -> dict[str, cv2.cuda.GpuMat]:
    """GPU Sobel → Gaussian-smoothed structure tensor → λ1≥λ2≥0 (GpuMats)."""
    require_cuda_farneback()
    ix, iy = sobel_gradients_gpu(gray_gpu)
    gauss = _cuda_gaussian_32f(float(tensor_sigma))
    jxx = gauss.apply(cv2.cuda.multiply(ix, ix))
    jyy = gauss.apply(cv2.cuda.multiply(iy, iy))
    jxy = gauss.apply(cv2.cuda.multiply(ix, iy))
    trace = cv2.cuda.add(jxx, jyy)
    diff = cv2.cuda.subtract(jxx, jyy)
    four_jxy2 = cv2.cuda.multiplyWithScalar(cv2.cuda.multiply(jxy, jxy), 4.0)
    disc = cv2.cuda.maxWithScalar(
        cv2.cuda.add(cv2.cuda.multiply(diff, diff), four_jxy2), 0.0
    )
    delta = cv2.cuda.sqrt(disc)
    lambda1 = cv2.cuda.maxWithScalar(
        cv2.cuda.multiplyWithScalar(cv2.cuda.add(trace, delta), 0.5), 0.0
    )
    lambda2 = cv2.cuda.maxWithScalar(
        cv2.cuda.multiplyWithScalar(cv2.cuda.subtract(trace, delta), 0.5), 0.0
    )
    return {
        "Ix": ix,
        "Iy": iy,
        "Jxx": jxx,
        "Jyy": jyy,
        "Jxy": jxy,
        "lambda1": lambda1,
        "lambda2": lambda2,
    }


def structure_tensor_reliability_gpu(
    gray_gpu: cv2.cuda.GpuMat,
    *,
    tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
    strength_tau: float = STRUCTURE_TENSOR_STRENGTH_TAU,
    eps: float = STRUCTURE_TENSOR_EPS,
) -> cv2.cuda.GpuMat:
    """Legacy R_st = clip(shape × strength) on GPU. Aggregation weight only."""
    fields = compute_structure_tensor_fields_gpu(
        gray_gpu, tensor_sigma=float(tensor_sigma)
    )
    tau = max(float(strength_tau), float(eps))
    shape = cv2.cuda.divide(
        fields["lambda2"], cv2.cuda.addWithScalar(fields["lambda1"], float(eps))
    )
    neg = cv2.cuda.multiplyWithScalar(fields["lambda2"], -1.0 / tau)
    strength = cv2.cuda.addWithScalar(cv2.cuda.multiplyWithScalar(cv2.cuda.exp(neg), -1.0), 1.0)
    rel = cv2.cuda.multiply(shape, strength)
    rel = cv2.cuda.maxWithScalar(rel, 0.0)
    return cv2.cuda.minWithScalar(rel, 1.0)


def _gpu_one_minus(src: cv2.cuda.GpuMat) -> cv2.cuda.GpuMat:
    return cv2.cuda.addWithScalar(cv2.cuda.multiplyWithScalar(src, -1.0), 1.0)


def _gpu_where_mask(
    cond_gt: cv2.cuda.GpuMat,
    if_true: cv2.cuda.GpuMat,
    if_false: cv2.cuda.GpuMat,
) -> cv2.cuda.GpuMat:
    """cond_gt is float {0,1}: out = cond*if_true + (1-cond)*if_false."""
    return cv2.cuda.add(
        cv2.cuda.multiply(cond_gt, if_true),
        cv2.cuda.multiply(_gpu_one_minus(cond_gt), if_false),
    )


def aperture_reliability_gpu(
    gray_gpu: cv2.cuda.GpuMat,
    u_gpu: cv2.cuda.GpuMat,
    v_gpu: cv2.cuda.GpuMat,
    *,
    tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
    tau_edge: float = APERTURE_TAU_EDGE,
    gamma: float = APERTURE_GAMMA,
    alpha: float = APERTURE_ALPHA,
    eps: float = STRUCTURE_TENSOR_EPS,
    min_flow_mag: float = 1e-6,
) -> cv2.cuda.GpuMat:
    """Direction-aware R_ap ∈ [0,1] on GPU (aggregation weight, flow unchanged).

    P_ap = strong_edge * edgeness * alignment^γ
    R_ap = clip(1 - α * P_ap, 0, 1)
    """
    fields = compute_structure_tensor_fields_gpu(
        gray_gpu, tensor_sigma=float(tensor_sigma)
    )
    lambda1 = fields["lambda1"]
    lambda2 = fields["lambda2"]
    eps_f = float(eps)
    ratio = cv2.cuda.divide(lambda2, cv2.cuda.addWithScalar(lambda1, eps_f))
    edgeness = _gpu_one_minus(ratio)
    _ret, strong_edge = cv2.cuda.threshold(
        lambda1, float(tau_edge), 1.0, cv2.THRESH_BINARY
    )

    # n ∝ (Jxy, λ1 - Jxx); fallback (λ1 - Jyy, Jxy)
    nx = fields["Jxy"]
    ny = cv2.cuda.subtract(lambda1, fields["Jxx"])
    n_norm = cv2.cuda.sqrt(
        cv2.cuda.add(cv2.cuda.multiply(nx, nx), cv2.cuda.multiply(ny, ny))
    )
    alt_nx = cv2.cuda.subtract(lambda1, fields["Jyy"])
    alt_ny = fields["Jxy"]
    alt_norm = cv2.cuda.sqrt(
        cv2.cuda.add(
            cv2.cuda.multiply(alt_nx, alt_nx), cv2.cuda.multiply(alt_ny, alt_ny)
        )
    )
    _ret, use_primary = cv2.cuda.threshold(
        n_norm, eps_f, 1.0, cv2.THRESH_BINARY
    )
    nx = _gpu_where_mask(use_primary, nx, alt_nx)
    ny = _gpu_where_mask(use_primary, ny, alt_ny)
    n_norm = _gpu_where_mask(use_primary, n_norm, alt_norm)
    n_safe = cv2.cuda.maxWithScalar(n_norm, eps_f)
    nx = cv2.cuda.divide(nx, n_safe)
    ny = cv2.cuda.divide(ny, n_safe)
    ones = cv2.cuda.GpuMat(nx.size()[1], nx.size()[0], cv2.CV_32FC1)
    zeros = cv2.cuda.GpuMat(nx.size()[1], nx.size()[0], cv2.CV_32FC1)
    ones.setTo(1.0)
    zeros.setTo(0.0)
    nx = _gpu_where_mask(use_primary, nx, ones)
    ny = _gpu_where_mask(use_primary, ny, zeros)
    tx = cv2.cuda.multiplyWithScalar(ny, -1.0)
    ty = nx

    mag = cv2.cuda.sqrt(
        cv2.cuda.add(cv2.cuda.multiply(u_gpu, u_gpu), cv2.cuda.multiply(v_gpu, v_gpu))
    )
    mag_safe = cv2.cuda.maxWithScalar(mag, float(min_flow_mag))
    alignment = cv2.cuda.abs(
        cv2.cuda.add(cv2.cuda.multiply(u_gpu, tx), cv2.cuda.multiply(v_gpu, ty))
    )
    alignment = cv2.cuda.divide(alignment, mag_safe)
    alignment = cv2.cuda.minWithScalar(cv2.cuda.maxWithScalar(alignment, 0.0), 1.0)
    _ret, mag_ok = cv2.cuda.threshold(
        mag, float(min_flow_mag), 1.0, cv2.THRESH_BINARY
    )
    alignment = cv2.cuda.multiply(alignment, mag_ok)

    g = max(float(gamma), 0.0)
    if abs(g - 2.0) < 1e-6:
        align_g = cv2.cuda.multiply(alignment, alignment)
    else:
        align_g = cv2.cuda.exp(
            cv2.cuda.multiplyWithScalar(cv2.cuda.log(cv2.cuda.maxWithScalar(alignment, 1e-12)), g)
        )
    p_ap = cv2.cuda.multiply(
        cv2.cuda.multiply(strong_edge, edgeness), align_g
    )
    r_ap = _gpu_one_minus(cv2.cuda.multiplyWithScalar(p_ap, float(alpha)))
    r_ap = cv2.cuda.minWithScalar(cv2.cuda.maxWithScalar(r_ap, 0.0), 1.0)
    return r_ap


def downscale_gray_for_flow(
    gray: np.ndarray,
    *,
    input_scale: float = INPUT_SCALE,
) -> np.ndarray:
    """Match Farneback input resolution (INTER_AREA @ INPUT_SCALE, CPU)."""
    arr = np.asarray(gray)
    if arr.ndim != 2:
        raise ValueError(f"expected gray HxW, got {arr.shape}")
    height, width = arr.shape[:2]
    scaled_h = max(1, int(height * float(input_scale)))
    scaled_w = max(1, int(width * float(input_scale)))
    if (scaled_h, scaled_w) == (height, width):
        return np.ascontiguousarray(arr)
    return cv2.resize(arr, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)


def compute_structure_tensor_fields(
    gray: np.ndarray,
    *,
    tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
) -> dict[str, np.ndarray]:
    """Sobel gradients → Gaussian-smoothed structure tensor → λ1≥λ2≥0."""
    img = np.asarray(gray)
    if img.ndim != 2:
        raise ValueError(f"expected gray HxW, got {img.shape}")
    if img.dtype != np.float32:
        img_f = img.astype(np.float32)
    else:
        img_f = img

    ix = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)

    # ksize=0 → derive from sigma (OpenCV convention).
    jxx = cv2.GaussianBlur(ix * ix, (0, 0), float(tensor_sigma))
    jyy = cv2.GaussianBlur(iy * iy, (0, 0), float(tensor_sigma))
    jxy = cv2.GaussianBlur(ix * iy, (0, 0), float(tensor_sigma))

    trace = jxx + jyy
    diff = jxx - jyy
    delta = np.sqrt(np.maximum(diff * diff + 4.0 * jxy * jxy, 0.0))
    lambda1 = 0.5 * (trace + delta)
    lambda2 = 0.5 * (trace - delta)
    lambda1 = np.maximum(lambda1, 0.0).astype(np.float32)
    lambda2 = np.maximum(lambda2, 0.0).astype(np.float32)

    return {
        "Ix": ix.astype(np.float32),
        "Iy": iy.astype(np.float32),
        "Jxx": jxx.astype(np.float32),
        "Jyy": jyy.astype(np.float32),
        "Jxy": jxy.astype(np.float32),
        "lambda1": lambda1,
        "lambda2": lambda2,
    }


def edge_tangent_from_structure_tensor(
    *,
    jxx: np.ndarray,
    jyy: np.ndarray,
    jxy: np.ndarray,
    lambda1: np.ndarray,
    eps: float = STRUCTURE_TENSOR_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unit edge-normal n (λ1 eigenvector) and tangent t = (-ny, nx).

    Returns ``(nx, ny, tx, ty)``.
    """
    # λ1 eigenvector of [[Jxx,Jxy],[Jxy,Jyy]]: n ∝ (Jxy, λ1 - Jxx)
    nx = np.asarray(jxy, dtype=np.float32)
    ny = (np.asarray(lambda1, dtype=np.float32) - np.asarray(jxx, dtype=np.float32)).astype(
        np.float32
    )
    n_norm = np.sqrt(nx * nx + ny * ny)
    # Fallback when first formula is near-degenerate.
    alt_nx = (np.asarray(lambda1, dtype=np.float32) - np.asarray(jyy, dtype=np.float32)).astype(
        np.float32
    )
    alt_ny = np.asarray(jxy, dtype=np.float32)
    alt_norm = np.sqrt(alt_nx * alt_nx + alt_ny * alt_ny)
    use_alt = n_norm < float(eps)
    nx = np.where(use_alt, alt_nx, nx)
    ny = np.where(use_alt, alt_ny, ny)
    n_norm = np.where(use_alt, alt_norm, n_norm)
    n_safe = np.maximum(n_norm, float(eps))
    nx = (nx / n_safe).astype(np.float32)
    ny = (ny / n_safe).astype(np.float32)
    # Flat / undefined: leave a harmless default; strong_edge gate zeros penalty.
    undefined = n_norm < float(eps)
    nx = np.where(undefined, 1.0, nx).astype(np.float32)
    ny = np.where(undefined, 0.0, ny).astype(np.float32)
    tx = (-ny).astype(np.float32)
    ty = nx.astype(np.float32)
    return nx, ny, tx, ty


def compute_aperture_reliability(
    gray: np.ndarray,
    flow: np.ndarray,
    *,
    tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
    tau_edge: float = APERTURE_TAU_EDGE,
    gamma: float = APERTURE_GAMMA,
    alpha: float = APERTURE_ALPHA,
    eps: float = STRUCTURE_TENSOR_EPS,
    min_flow_mag: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Direction-aware aperture reliability R_ap ∈ [0,1] (aggregation weight only).

    P_ap = strong_edge * edgeness * alignment^γ
    R_ap = clip(1 - α * P_ap, 0, 1)

    Does not modify ``flow``. Does not include legacy R_st (shape×strength).
    """
    fields = compute_structure_tensor_fields(gray, tensor_sigma=float(tensor_sigma))
    lambda1 = fields["lambda1"]
    lambda2 = fields["lambda2"]
    flow_arr = np.asarray(flow, dtype=np.float32)
    if flow_arr.shape[:2] != lambda1.shape:
        raise ValueError(
            f"flow shape {flow_arr.shape[:2]} != structure tensor {lambda1.shape}"
        )

    ratio = (lambda2 / (lambda1 + float(eps))).astype(np.float32)
    edgeness = (1.0 - ratio).astype(np.float32)
    strong_edge = (lambda1 > float(tau_edge)).astype(np.float32)

    _nx, _ny, tx, ty = edge_tangent_from_structure_tensor(
        jxx=fields["Jxx"],
        jyy=fields["Jyy"],
        jxy=fields["Jxy"],
        lambda1=lambda1,
        eps=float(eps),
    )
    u = flow_arr[..., 0]
    v = flow_arr[..., 1]
    mag = np.sqrt(u * u + v * v).astype(np.float32)
    mag_safe = np.maximum(mag, float(min_flow_mag))
    # alignment = |flow · t| / ||flow||  (t is unit)
    alignment = (np.abs(u * tx + v * ty) / mag_safe).astype(np.float32)
    alignment = np.clip(alignment, 0.0, 1.0)
    # Near-zero flow: alignment unused (strong_edge*edgeness still defined, but
    # unstable orientation — force alignment 0 so P_ap does not spuriously fire).
    alignment = np.where(mag < float(min_flow_mag), 0.0, alignment).astype(np.float32)

    g = max(float(gamma), 0.0)
    p_ap = (strong_edge * edgeness * np.power(alignment, g)).astype(np.float32)
    r_ap = np.clip(1.0 - float(alpha) * p_ap, 0.0, 1.0).astype(np.float32)

    return {
        "lambda1": lambda1,
        "lambda2": lambda2,
        "edgeness": edgeness,
        "strong_edge": strong_edge,
        "tangent_x": tx,
        "tangent_y": ty,
        "alignment": alignment,
        "P_ap": p_ap,
        "R_ap": r_ap,
        "flow_mag": mag,
        "Jxx": fields["Jxx"],
        "Jyy": fields["Jyy"],
        "Jxy": fields["Jxy"],
    }


def aperture_reliability_stack_for_flow(
    gray: list[np.ndarray],
    flow_stack: np.ndarray,
    *,
    gap: int,
    tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
    tau_edge: float = APERTURE_TAU_EDGE,
    gamma: float = APERTURE_GAMMA,
    alpha: float = APERTURE_ALPHA,
    eps: float = STRUCTURE_TENSOR_EPS,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """R_ap stack aligned with dense flow stack indices [gap, T).

    Also returns stacked diagnostics (edgeness, alignment, strong_edge, …).
    Uses current-frame gray at Farneback resolution for each flow's curr index.
    """
    stack = np.asarray(flow_stack, dtype=np.float32)
    if stack.ndim != 4 or stack.shape[-1] != 2:
        raise ValueError(f"expected (T,H,W,2) flow stack, got {stack.shape}")
    t_len = int(stack.shape[0])
    r_maps: list[np.ndarray] = []
    diag_keys = (
        "edgeness",
        "alignment",
        "strong_edge",
        "P_ap",
        "lambda1",
        "lambda2",
    )
    diag_lists: dict[str, list[np.ndarray]] = {k: [] for k in diag_keys}
    for i in range(t_len):
        curr = int(gap) + i
        gray_q = downscale_gray_for_flow(gray[curr])
        ap = compute_aperture_reliability(
            gray_q,
            stack[i],
            tensor_sigma=float(tensor_sigma),
            tau_edge=float(tau_edge),
            gamma=float(gamma),
            alpha=float(alpha),
            eps=float(eps),
        )
        r_maps.append(ap["R_ap"])
        for k in diag_keys:
            diag_lists[k].append(ap[k])
    r_stack = np.stack(r_maps, axis=0).astype(np.float32)
    diags = {k: np.stack(v, axis=0).astype(np.float32) for k, v in diag_lists.items()}
    return r_stack, diags


def compute_structure_tensor_reliability(
    gray: np.ndarray,
    *,
    tensor_sigma: float = STRUCTURE_TENSOR_SIGMA,
    strength_tau: float = STRUCTURE_TENSOR_STRENGTH_TAU,
    eps: float = STRUCTURE_TENSOR_EPS,
) -> dict[str, np.ndarray]:
    """Legacy ST reliability (shape×strength). Kept for ablation; not used in R_ap.

    shape_reliability = λ2 / (λ1 + eps)
    strength_reliability = 1 - exp(-λ2/τ)
    reliability = clip(shape * strength, 0, 1)

    Does not modify flow.
    """
    fields = compute_structure_tensor_fields(gray, tensor_sigma=float(tensor_sigma))
    lambda1 = fields["lambda1"]
    lambda2 = fields["lambda2"]
    tau = max(float(strength_tau), float(eps))
    shape = (lambda2 / (lambda1 + float(eps))).astype(np.float32)
    strength = (1.0 - np.exp(-lambda2 / tau)).astype(np.float32)
    reliability = np.clip(shape * strength, 0.0, 1.0).astype(np.float32)
    return {
        "lambda1": lambda1,
        "lambda2": lambda2,
        "shape_reliability": shape,
        "strength_reliability": strength,
        "reliability": reliability,
        "Ix": fields["Ix"],
        "Iy": fields["Iy"],
        "Jxx": fields["Jxx"],
        "Jyy": fields["Jyy"],
        "Jxy": fields["Jxy"],
    }


def lambda2_percentile_stats(
    lambda2: np.ndarray,
    *,
    percentiles: tuple[float, ...] = DEFAULT_LAMBDA2_PERCENTILES,
) -> dict[str, float]:
    """Finite λ2 percentiles for choosing strength_tau."""
    arr = np.asarray(lambda2, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"P{p:g}": float("nan") for p in percentiles}
    values = np.percentile(arr, list(percentiles))
    return {f"P{p:g}": float(v) for p, v in zip(percentiles, values)}


def log_lambda2_percentiles(
    lambda2: np.ndarray,
    *,
    percentiles: tuple[float, ...] = DEFAULT_LAMBDA2_PERCENTILES,
    label: str = "lambda2",
) -> dict[str, float]:
    stats = lambda2_percentile_stats(lambda2, percentiles=percentiles)
    logger.info("%s percentiles:", label)
    for key, value in stats.items():
        logger.info("  %-6s: %.6g", key, value)
    return stats


def roi_reliability_stats(
    *,
    flow: np.ndarray | None,
    st: Mapping[str, np.ndarray],
    rois: Mapping[str, tuple[int, int, int, int]],
    eps: float = STRUCTURE_TENSOR_EPS,
) -> dict[str, dict[str, float]]:
    """Per-ROI means for flow mag / λ / reliability / retention.

    ROI boxes are ``(x1, y1, x2, y2)`` in the same pixel grid as ``st`` maps
    (typically Farneback 1/4 resolution).
    """
    lambda1 = np.asarray(st["lambda1"], dtype=np.float32)
    lambda2 = np.asarray(st["lambda2"], dtype=np.float32)
    shape = np.asarray(st["shape_reliability"], dtype=np.float32)
    strength = np.asarray(st["strength_reliability"], dtype=np.float32)
    reliability = np.asarray(st["reliability"], dtype=np.float32)
    height, width = reliability.shape[:2]

    mag: np.ndarray | None
    if flow is None:
        mag = None
    else:
        flow_arr = np.asarray(flow, dtype=np.float32)
        if flow_arr.shape[:2] != (height, width):
            raise ValueError(
                f"flow shape {flow_arr.shape[:2]} != reliability {(height, width)}"
            )
        mag = np.linalg.norm(flow_arr, axis=-1).astype(np.float32)

    out: dict[str, dict[str, float]] = {}
    for name, box in rois.items():
        x1, y1, x2, y2 = (int(v) for v in box)
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            out[name] = {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "n_pixels": 0.0,
            }
            continue
        sl = (slice(y1, y2), slice(x1, x2))
        row: dict[str, float] = {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "n_pixels": float((y2 - y1) * (x2 - x1)),
            "mean_lambda1": float(np.mean(lambda1[sl])),
            "mean_lambda2": float(np.mean(lambda2[sl])),
            "mean_shape_reliability": float(np.mean(shape[sl])),
            "mean_strength_reliability": float(np.mean(strength[sl])),
            "mean_reliability": float(np.mean(reliability[sl])),
        }
        if mag is not None:
            m = mag[sl]
            r = reliability[sl]
            mean_mag = float(np.mean(m))
            mean_weighted = float(np.mean(m * r))
            row["mean_flow_magnitude"] = mean_mag
            row["mean_weighted_magnitude"] = mean_weighted
            row["retention"] = mean_weighted / (mean_mag + float(eps))
        out[name] = row
    return out


def normalize_for_display(
    arr: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    percentile_hi: float = 99.0,
) -> np.ndarray:
    """Map array to uint8 for debug PNGs; does not alter source floats."""
    a = np.asarray(arr, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    lo = float(np.min(finite) if vmin is None else vmin)
    if vmax is None:
        hi = float(np.percentile(finite, percentile_hi))
    else:
        hi = float(vmax)
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return np.uint8(np.round(scaled * 255.0))


def save_reliability_debug_images(
    out_dir,
    *,
    gray: np.ndarray,
    flow: np.ndarray | None,
    st: Mapping[str, np.ndarray],
    prefix: str = "debug",
) -> dict[str, str]:
    """Write A–H debug PNGs. Returns map of label → path."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def _write(name: str, img: np.ndarray) -> None:
        path = out / f"{prefix}_{name}.png"
        if img.ndim == 2:
            ok = cv2.imwrite(str(path), img)
        else:
            ok = cv2.imwrite(str(path), img)
        if not ok:
            raise RuntimeError(f"failed to write {path}")
        paths[name] = str(path)

    g = np.asarray(gray)
    if g.dtype != np.uint8:
        g8 = normalize_for_display(g, vmin=0.0, vmax=255.0)
    else:
        g8 = g
    _write("gray", g8)

    if flow is not None:
        mag = np.linalg.norm(np.asarray(flow, dtype=np.float32), axis=-1)
        mag_u8 = normalize_for_display(mag, vmin=0.0)
        _write("flow_mag", cv2.applyColorMap(mag_u8, cv2.COLORMAP_TURBO))
        weighted = mag * np.asarray(st["reliability"], dtype=np.float32)
        w_u8 = normalize_for_display(weighted, vmin=0.0)
        _write("weighted_mag", cv2.applyColorMap(w_u8, cv2.COLORMAP_TURBO))

    for key, disp_name, use_turbo in (
        ("lambda1", "lambda1", True),
        ("lambda2", "lambda2", True),
        ("shape_reliability", "shape_reliability", False),
        ("strength_reliability", "strength_reliability", False),
        ("reliability", "flow_reliability", False),
    ):
        u8 = normalize_for_display(
            st[key],
            vmin=0.0,
            vmax=1.0 if "reliability" in key else None,
        )
        if use_turbo:
            _write(disp_name, cv2.applyColorMap(u8, cv2.COLORMAP_TURBO))
        else:
            _write(disp_name, cv2.applyColorMap(u8, cv2.COLORMAP_TURBO))

    return paths

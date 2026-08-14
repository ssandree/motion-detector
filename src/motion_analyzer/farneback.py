"""Farneback dense optical flow (CUDA via OpenCV cudaoptflow)."""

from __future__ import annotations

import threading
from typing import Mapping

from motion_analyzer.opencv_cuda_bootstrap import bootstrap_opencv_cuda, reload_cv2_if_needed

bootstrap_opencv_cuda()
reload_cv2_if_needed()

import cv2
import numpy as np

from motion_analyzer.config import (
    FARNEBACK_DEVICE,
    FARNEBACK_ITERATIONS,
    FARNEBACK_LEVELS,
    FARNEBACK_POLY_N,
    FARNEBACK_POLY_SIGMA,
    FARNEBACK_PYR_SCALE,
    FARNEBACK_USE_GAUSSIAN,
    FARNEBACK_WINSIZE,
    INPUT_SCALE,
)

_LOCK = threading.Lock()
_CUDA_ALGO = None
_CUDA_CHECKED = False
_GAUSS_FILTERS: dict[tuple[int, float], object] = {}


def farneback_device() -> str:
    """Resolved device string used for Farneback (`cuda`)."""
    return str(FARNEBACK_DEVICE).strip().lower()


def require_cuda_farneback() -> None:
    """Raise if OpenCV CUDA Farneback is unavailable."""
    global _CUDA_CHECKED
    if _CUDA_CHECKED:
        return
    if not hasattr(cv2, "cuda"):
        raise RuntimeError(
            "OpenCV CUDA module missing. Build with scripts/setup/build_opencv_cuda.sh "
            "and `source ~/.local/opencv-cuda/env.sh`."
        )
    try:
        n_dev = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"OpenCV CUDA device probe failed: {exc}") from exc
    if n_dev < 1:
        raise RuntimeError(
            "OpenCV reports 0 CUDA devices. Ensure a CUDA-enabled OpenCV build is active "
            "(source ~/.local/opencv-cuda/env.sh) and the GPU is visible."
        )
    create = getattr(cv2.cuda, "FarnebackOpticalFlow", None)
    if create is None or not hasattr(create, "create"):
        raise RuntimeError(
            "cv2.cuda.FarnebackOpticalFlow unavailable. Rebuild OpenCV with "
            "opencv_contrib cudaoptflow (scripts/setup/build_opencv_cuda.sh)."
        )
    _CUDA_CHECKED = True


def _farneback_flags() -> int:
    return cv2.OPTFLOW_FARNEBACK_GAUSSIAN if FARNEBACK_USE_GAUSSIAN else 0


def _get_cuda_farneback_algo():
    global _CUDA_ALGO
    require_cuda_farneback()
    with _LOCK:
        if _CUDA_ALGO is None:
            _CUDA_ALGO = cv2.cuda.FarnebackOpticalFlow.create(
                numLevels=int(FARNEBACK_LEVELS),
                pyrScale=float(FARNEBACK_PYR_SCALE),
                fastPyramids=False,
                winSize=int(FARNEBACK_WINSIZE),
                numIters=int(FARNEBACK_ITERATIONS),
                polyN=int(FARNEBACK_POLY_N),
                polySigma=float(FARNEBACK_POLY_SIGMA),
                flags=int(_farneback_flags()),
            )
        return _CUDA_ALGO


def _get_cuda_gaussian_filter(sigma: float, *, src_type: int = cv2.CV_8UC1):
    """Cached CUDA Gaussian filter; ksize auto from σ when (0,0)."""
    key = (int(src_type), float(sigma))
    with _LOCK:
        filt = _GAUSS_FILTERS.get(key)
        if filt is None:
            require_cuda_farneback()
            filt = cv2.cuda.createGaussianFilter(
                int(src_type),
                int(src_type),
                (0, 0),
                float(sigma),
            )
            _GAUSS_FILTERS[key] = filt
        return filt


def upload_gray_gpu(gray: np.ndarray) -> cv2.cuda.GpuMat:
    """Upload contiguous uint8 HxW gray to a new GpuMat."""
    require_cuda_farneback()
    arr = np.ascontiguousarray(gray)
    if arr.ndim != 2:
        raise ValueError(f"expected gray HxW, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    gpu = cv2.cuda.GpuMat()
    gpu.upload(arr)
    return gpu


def downscale_gray_cuda(
    src_gpu: cv2.cuda.GpuMat,
    *,
    scaled_w: int,
    scaled_h: int,
    blur_sigma: float | None = None,
) -> cv2.cuda.GpuMat:
    """Optional Gaussian blur then INTER_AREA 1/N resize — all on GPU."""
    require_cuda_farneback()
    src = src_gpu
    if blur_sigma is not None and float(blur_sigma) > 0:
        filt = _get_cuda_gaussian_filter(float(blur_sigma), src_type=src.type())
        src = filt.apply(src)
    return cv2.cuda.resize(
        src,
        (int(scaled_w), int(scaled_h)),
        interpolation=cv2.INTER_AREA,
    )


def calc_farneback_gpu(
    prev_gpu: cv2.cuda.GpuMat, curr_gpu: cv2.cuda.GpuMat
) -> np.ndarray:
    """Dense Farneback on two same-size CUDA gray GpuMats → (H, W, 2) float32."""
    algo = _get_cuda_farneback_algo()
    gpu_flow = algo.calc(prev_gpu, curr_gpu, None)
    return np.asarray(gpu_flow.download(), dtype=np.float32)


def calc_farneback(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Dense Farneback flow between two same-size gray images → (H, W, 2) float32.

    Uses CUDA when FARNEBACK_DEVICE == \"cuda\" (required by default).
    """
    prev = np.asarray(prev_gray)
    curr = np.asarray(curr_gray)
    if prev.ndim != 2 or curr.ndim != 2:
        raise ValueError(f"expected gray HxW arrays, got {prev.shape} / {curr.shape}")
    if prev.shape != curr.shape:
        raise ValueError(f"shape mismatch: {prev.shape} vs {curr.shape}")

    device = farneback_device()
    if device != "cuda":
        raise RuntimeError(
            f"FARNEBACK_DEVICE={device!r} is not supported; set FARNEBACK_DEVICE='cuda'."
        )

    return calc_farneback_gpu(upload_gray_gpu(prev), upload_gray_gpu(curr))


def quarter_scale_size(
    height: int, width: int, *, input_scale: float = INPUT_SCALE
) -> tuple[int, int]:
    scaled_h = max(1, int(height * float(input_scale)))
    scaled_w = max(1, int(width * float(input_scale)))
    return scaled_h, scaled_w


def compute_dense_flow_from_gpu(
    prev_gpu: cv2.cuda.GpuMat,
    curr_gpu: cv2.cuda.GpuMat,
    *,
    full_h: int,
    full_w: int,
    blur_sigma: float | None = None,
    input_scale: float = INPUT_SCALE,
) -> np.ndarray:
    """GPU blur(optional) → INTER_AREA 1/4 → Farneback; vectors in original px."""
    scaled_h, scaled_w = quarter_scale_size(full_h, full_w, input_scale=input_scale)
    prev_s = downscale_gray_cuda(
        prev_gpu, scaled_w=scaled_w, scaled_h=scaled_h, blur_sigma=blur_sigma
    )
    curr_s = downscale_gray_cuda(
        curr_gpu, scaled_w=scaled_w, scaled_h=scaled_h, blur_sigma=blur_sigma
    )
    flow = calc_farneback_gpu(prev_s, curr_s)
    flow[..., 0] *= float(full_w) / float(scaled_w)
    flow[..., 1] *= float(full_h) / float(scaled_h)
    return flow


def compute_dense_flow_original_px(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    blur_sigma: float | None = None,
) -> np.ndarray:
    """1/4-scale Farneback; vectors restored to original-pixel units (h/4, w/4, 2).

    Downscale runs on GPU: optional Gaussian(σ) then INTER_AREA, then CUDA Farneback.
    """
    height, width = prev_gray.shape[:2]
    if curr_gray.shape[:2] != (height, width):
        raise ValueError(f"shape mismatch: {prev_gray.shape} vs {curr_gray.shape}")
    device = farneback_device()
    if device != "cuda":
        raise RuntimeError(
            f"FARNEBACK_DEVICE={device!r} is not supported; set FARNEBACK_DEVICE='cuda'."
        )
    return compute_dense_flow_from_gpu(
        upload_gray_gpu(prev_gray),
        upload_gray_gpu(curr_gray),
        full_h=height,
        full_w=width,
        blur_sigma=blur_sigma,
    )


def compute_dense_flows_resize_methods(
    prev_gpu: cv2.cuda.GpuMat,
    curr_gpu: cv2.cuda.GpuMat,
    *,
    full_h: int,
    full_w: int,
    blur_sigmas: Mapping[str, float | None],
    input_scale: float = INPUT_SCALE,
) -> dict[str, np.ndarray]:
    """Run Farneback once per named blur_sigma (None = INTER_AREA only)."""
    return {
        name: compute_dense_flow_from_gpu(
            prev_gpu,
            curr_gpu,
            full_h=full_h,
            full_w=full_w,
            blur_sigma=sigma,
            input_scale=input_scale,
        )
        for name, sigma in blur_sigmas.items()
    }

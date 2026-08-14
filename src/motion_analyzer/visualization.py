"""Shared drawing helpers for pipeline visualizations."""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / float(w), height / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def make_cell(
    title: str,
    panel: np.ndarray,
    panel_w: int = 640,
    panel_h: int = 360,
    title_h: int = 40,
    font_scale: float = 0.55,
    thickness: int = 2,
) -> np.ndarray:
    body = letterbox(panel, panel_w, panel_h)
    bar = np.full((title_h, panel_w, 3), 28, dtype=np.uint8)
    # Keep text vertically centered for smaller title bars / font sizes.
    baseline_y = max(12, int(round(title_h * 0.68)))
    cv2.putText(
        bar,
        title,
        (8, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(font_scale),
        (235, 235, 235),
        int(thickness),
        cv2.LINE_AA,
    )
    return np.vstack([bar, body])


def expand_cells(
    values: np.ndarray,
    *,
    cell_px: int,
    frame_height: int,
    frame_width: int,
) -> np.ndarray:
    expanded = np.repeat(np.repeat(values, cell_px, axis=0), cell_px, axis=1)
    return expanded[:frame_height, :frame_width]


def heat_overlay(
    frame: np.ndarray,
    values: np.ndarray,
    *,
    cell_px: int,
    max_alpha: float,
) -> np.ndarray:
    """Overlay a [0,1] grid heatmap onto a BGR frame."""
    frame_h, frame_w = frame.shape[:2]
    level = np.clip(values, 0.0, 1.0).astype(np.float32)
    heat_small = cv2.applyColorMap(
        np.uint8(np.round(level * 255.0)), cv2.COLORMAP_TURBO
    )
    heat = expand_cells(
        heat_small, cell_px=cell_px, frame_height=frame_h, frame_width=frame_w
    )
    alpha = expand_cells(
        level * max_alpha,
        cell_px=cell_px,
        frame_height=frame_h,
        frame_width=frame_w,
    )[..., None]
    blended = frame.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha
    return np.uint8(np.clip(blended, 0, 255))


def grid_bbox_to_pixels(
    bbox: tuple[int, int, int, int],
    *,
    unit_pixel_size: int,
    frame_width: int,
    frame_height: int,
) -> list[int]:
    x0, y0, x1, y1 = bbox
    return [
        int(x0 * unit_pixel_size),
        int(y0 * unit_pixel_size),
        int(min(frame_width, x1 * unit_pixel_size)),
        int(min(frame_height, y1 * unit_pixel_size)),
    ]

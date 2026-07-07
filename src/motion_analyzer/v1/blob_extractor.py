"""Grid-based motion blob extraction with flow filtering and region merging."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from motion_analyzer.v1.pixel_motion import compute_optical_flow

EPS = 1e-8


@dataclass
class BlobExtractConfig:
    """Divide each frame into a fixed grid; merge 4-connected active cells."""

    grid_rows: int = 20
    grid_cols: int = 20
    cell_motion_threshold: float = 0.05
    use_flow_filter: bool = True
    min_flow_magnitude: float = 0.5
    min_direction_coherence: float = 0.45
    flow_scale: float = 0.5


@dataclass
class RawBlob:
    """One merged region of 4-connected active grid cells."""

    blob_id: int
    grid_row: int
    grid_col: int
    grid_cells: list[tuple[int, int]]
    num_grid_cells: int
    x1: int
    y1: int
    x2: int
    y2: int
    component_area: int
    label_mask: np.ndarray
    mean_flow_magnitude: float = 0.0
    flow_direction_coherence: float = 0.0
    source_blob_ids: list[int] = field(default_factory=list)


def _cell_bounds(
    row: int,
    col: int,
    frame_h: int,
    frame_w: int,
    grid_rows: int,
    grid_cols: int,
) -> tuple[int, int, int, int]:
    y1 = row * frame_h // grid_rows
    y2 = (row + 1) * frame_h // grid_rows if row < grid_rows - 1 else frame_h
    x1 = col * frame_w // grid_cols
    x2 = (col + 1) * frame_w // grid_cols if col < grid_cols - 1 else frame_w
    return x1, y1, x2, y2


def _cell_flow_stats(
    flow: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[float, float]:
    """
    Return (mean_magnitude, direction_coherence) for a grid cell.

    direction_coherence ∈ [0, 1]: 1 = all vectors same direction, 0 = random.
    """
    patch = flow[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0, 0.0

    fx = patch[..., 0].astype(np.float64)
    fy = patch[..., 1].astype(np.float64)
    mag = np.sqrt(fx * fx + fy * fy)
    mean_mag = float(mag.mean())

    active = mag > EPS
    if active.sum() < 4:
        return mean_mag, 0.0

    angles = np.arctan2(fy[active], fx[active])
    mean_cos = float(np.cos(angles).mean())
    mean_sin = float(np.sin(angles).mean())
    coherence = float(np.sqrt(mean_cos * mean_cos + mean_sin * mean_sin))
    return mean_mag, coherence


def _passes_flow_filter(
    mean_mag: float,
    coherence: float,
    cfg: BlobExtractConfig,
) -> bool:
    if not cfg.use_flow_filter:
        return True
    # Require meaningful displacement; reject flicker with near-zero flow.
    if mean_mag < cfg.min_flow_magnitude:
        return False
    # Reject oscillating / wind-like motion with inconsistent flow direction.
    return coherence >= cfg.min_direction_coherence


def _find_connected_regions(
    active_cells: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    """4-connected component labeling on grid coordinates."""
    remaining = set(active_cells)
    regions: list[list[tuple[int, int]]] = []

    while remaining:
        start = remaining.pop()
        region: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([start])

        while queue:
            r, c = queue.popleft()
            region.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in remaining:
                    remaining.remove(nb)
                    queue.append(nb)

        regions.append(region)

    return regions


def _region_to_blob(
    blob_id: int,
    cells: list[tuple[int, int]],
    motion_map: np.ndarray,
    flow: np.ndarray | None,
    cfg: BlobExtractConfig,
) -> RawBlob:
    h, w = motion_map.shape
    xs1, ys1, xs2, ys2 = [], [], [], []
    mag_sum = 0.0
    coh_sum = 0.0

    for row, col in cells:
        x1, y1, x2, y2 = _cell_bounds(row, col, h, w, cfg.grid_rows, cfg.grid_cols)
        xs1.append(x1)
        ys1.append(y1)
        xs2.append(x2)
        ys2.append(y2)
        if flow is not None:
            m, c = _cell_flow_stats(flow, x1, y1, x2, y2)
            mag_sum += m
            coh_sum += c

    n = len(cells)
    rx1, ry1, rx2, ry2 = min(xs1), min(ys1), max(xs2), max(ys2)
    label_mask = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)

    centroid_row = int(round(sum(r for r, _ in cells) / n))
    centroid_col = int(round(sum(c for _, c in cells) / n))

    return RawBlob(
        blob_id=blob_id,
        grid_row=centroid_row,
        grid_col=centroid_col,
        grid_cells=cells,
        num_grid_cells=n,
        x1=rx1,
        y1=ry1,
        x2=rx2,
        y2=ry2,
        component_area=(ry2 - ry1) * (rx2 - rx1),
        label_mask=label_mask,
        mean_flow_magnitude=round(mag_sum / n, 4) if flow is not None else 0.0,
        flow_direction_coherence=round(coh_sum / n, 4) if flow is not None else 0.0,
        source_blob_ids=[blob_id],
    )


def extract_blobs_from_motion_map(
    motion_map: np.ndarray,
    config: BlobExtractConfig | None = None,
    *,
    prev_gray: np.ndarray | None = None,
    curr_gray: np.ndarray | None = None,
    flow: np.ndarray | None = None,
    cell_motion_threshold_override: float | None = None,
) -> list[RawBlob]:
    """
    Detect motion on a 20×20 grid, filter with optical flow coherence, merge
    4-connected active cells into one bbox per region.
    """
    cfg = config or BlobExtractConfig()
    h, w = motion_map.shape
    cell_threshold = (
        cell_motion_threshold_override
        if cell_motion_threshold_override is not None
        else cfg.cell_motion_threshold
    )

    if flow is None and prev_gray is not None and curr_gray is not None:
        flow = compute_optical_flow(prev_gray, curr_gray, scale=cfg.flow_scale)

    active_cells: set[tuple[int, int]] = set()

    for row in range(cfg.grid_rows):
        for col in range(cfg.grid_cols):
            x1, y1, x2, y2 = _cell_bounds(row, col, h, w, cfg.grid_rows, cfg.grid_cols)
            cell = motion_map[y1:y2, x1:x2]
            if cell.size == 0 or float(cell.mean()) < cell_threshold:
                continue

            if flow is not None:
                mean_mag, coherence = _cell_flow_stats(flow, x1, y1, x2, y2)
                if not _passes_flow_filter(mean_mag, coherence, cfg):
                    continue

            active_cells.add((row, col))

    regions = _find_connected_regions(active_cells)
    blobs: list[RawBlob] = []
    for blob_id, cells in enumerate(regions):
        blobs.append(_region_to_blob(blob_id, cells, motion_map, flow, cfg))

    return blobs


def read_gray_frames_cached(
    video_path: str,
    frame_indices: set[int],
) -> dict[int, np.ndarray]:
    """Read grayscale frames for *frame_indices* (seeks per index)."""
    if not frame_indices:
        return {}

    cache: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return cache

    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            cache[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()
    return cache


def load_gray_from_dict(
    gray_by_idx: dict[int, np.ndarray],
    frame_indices: set[int],
) -> dict[int, np.ndarray]:
    """Subset a gray-frame dict to the requested indices."""
    return {idx: gray_by_idx[idx] for idx in frame_indices if idx in gray_by_idx}


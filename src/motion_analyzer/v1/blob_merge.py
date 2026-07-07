"""Spatial merging of nearby motion blobs within a single frame."""

from __future__ import annotations

import numpy as np

from motion_analyzer.v1.blob_extractor import RawBlob, BlobExtractConfig


def _bbox_center(blob: RawBlob) -> tuple[float, float]:
    return (blob.x1 + blob.x2) / 2.0, (blob.y1 + blob.y2) / 2.0


def _bbox_gap(a: RawBlob, b: RawBlob) -> float:
    """Minimum edge-to-edge distance; negative if overlapping."""
    dx = max(0, max(a.x1, b.x1) - min(a.x2, b.x2))
    dy = max(0, max(a.y1, b.y1) - min(a.y2, b.y2))
    if dx == 0 and dy == 0:
        return -1.0
    return float(np.hypot(dx, dy))


def _x_overlap_ratio(a: RawBlob, b: RawBlob) -> float:
    overlap = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    min_w = max(1, min(a.x2 - a.x1, b.x2 - b.x1))
    return overlap / min_w


def _should_merge(
    a: RawBlob,
    b: RawBlob,
    *,
    base_dist: float,
    frame_h: int,
) -> bool:
    gap = _bbox_gap(a, b)
    ca = _bbox_center(a)
    cb = _bbox_center(b)
    center_dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))

    small_a = (a.y2 - a.y1) < 0.08 * frame_h
    small_b = (b.y2 - b.y1) < 0.08 * frame_h
    vertical = abs(ca[1] - cb[1]) > abs(ca[0] - cb[0])
    x_ov = _x_overlap_ratio(a, b)

    threshold = base_dist
    if small_a and small_b and vertical and x_ov > 0.3:
        threshold = base_dist * 1.35
    elif small_a or small_b:
        threshold = base_dist * 1.15

    return gap <= threshold or center_dist <= threshold


def _merge_pair(a: RawBlob, b: RawBlob, new_id: int) -> RawBlob:
    cells = list({c for c in a.grid_cells + b.grid_cells})
    source_ids = sorted(set(a.source_blob_ids + b.source_blob_ids))

    rx1 = min(a.x1, b.x1)
    ry1 = min(a.y1, b.y1)
    rx2 = max(a.x2, b.x2)
    ry2 = max(a.y2, b.y2)
    n = len(cells)
    centroid_row = int(round(sum(r for r, _ in cells) / n))
    centroid_col = int(round(sum(c for _, c in cells) / n))

    return RawBlob(
        blob_id=new_id,
        grid_row=centroid_row,
        grid_col=centroid_col,
        grid_cells=cells,
        num_grid_cells=n,
        x1=rx1,
        y1=ry1,
        x2=rx2,
        y2=ry2,
        component_area=(rx2 - rx1) * (ry2 - ry1),
        label_mask=np.ones((ry2 - ry1, rx2 - rx1), dtype=bool),
        mean_flow_magnitude=round((a.mean_flow_magnitude + b.mean_flow_magnitude) / 2, 4),
        flow_direction_coherence=round(
            (a.flow_direction_coherence + b.flow_direction_coherence) / 2, 4
        ),
        source_blob_ids=source_ids,
    )


def merge_nearby_raw_blobs(
    blobs: list[RawBlob],
    motion_map: np.ndarray,
    flow: np.ndarray | None,
    cfg: BlobExtractConfig,
    *,
    distance_ratio: float = 0.06,
) -> list[RawBlob]:
    """
    Merge spatially close raw blobs in the same frame (after 4-connected merge).

    Vertical, small blobs (e.g. limbs/head) are merged more aggressively.
    """
    if len(blobs) <= 1:
        return blobs

    h, _ = motion_map.shape
    base_dist = distance_ratio * h
    active = list(blobs)

    changed = True
    while changed:
        changed = False
        used = [False] * len(active)
        merged_list: list[RawBlob] = []
        new_id = 0

        for i in range(len(active)):
            if used[i]:
                continue
            current = active[i]
            used[i] = True
            for j in range(i + 1, len(active)):
                if used[j]:
                    continue
                if _should_merge(current, active[j], base_dist=base_dist, frame_h=h):
                    current = _merge_pair(current, active[j], new_id)
                    used[j] = True
                    changed = True
            current.blob_id = new_id
            merged_list.append(current)
            new_id += 1
        active = merged_list

    return active

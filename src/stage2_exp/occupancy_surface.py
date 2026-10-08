"""Stage2-only (x, y, t) occupancy surface. No SAM, no object tracking.

Default occupancy is 16px fused magnitude. Unit occupancy is 64px MU
(after 4×4 spatial aggregation), then painted back as 4×4 of 16px cells
so the axes stay on the 16px grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label

from motion_analyzer.config import AGGREGATION_BLOCK, ORIGINAL_CELL_PX, ROI_THRESHOLD

_T_ASPECT = 2.8
_LIGHT = np.array([0.35, 0.80, 0.45], dtype=np.float64)
_BASE_RGB = np.array([0.42, 0.44, 0.48], dtype=np.float64)


def occupancy_from_magnitude(
    mag: np.ndarray,
    *,
    threshold: float = ROI_THRESHOLD,
    sigma: float = 0.6,
    min_voxels: int = 40,
) -> np.ndarray:
    """(T, Y, X) bool occupancy from Stage2 magnitude."""
    vol = np.nan_to_num(np.asarray(mag, dtype=np.float32), nan=0.0)
    if float(sigma) > 0:
        vol = gaussian_filter(vol, sigma=float(sigma))
    occ = vol >= float(threshold)
    if int(min_voxels) <= 1:
        return occ
    labels, n_cc = label(occ, structure=np.ones((3, 3, 3), dtype=np.int8))
    if n_cc <= 0:
        return occ
    counts = np.bincount(labels.ravel())
    keep = np.zeros(counts.shape, dtype=bool)
    keep[1:] = counts[1:] >= int(min_voxels)
    return keep[labels]


def expand_unit_occupancy(
    occ: np.ndarray, *, block: int = AGGREGATION_BLOCK, fine_hw: tuple[int, int] | None = None
) -> np.ndarray:
    """Paint each 64px occupancy cell as block×block 16px cells. Time is unchanged.

    If ``fine_hw`` is (H, W) of the 16px map, the padded 4×4 expand is cropped
    back to that grid so axes match the original 16px occupancy figure.
    """
    arr = np.asarray(occ, dtype=bool)
    if arr.ndim != 3:
        raise ValueError(f"expected (T,Y,X), got {arr.shape}")
    scale = int(block)
    painted = arr if scale <= 1 else np.repeat(np.repeat(arr, scale, axis=1), scale, axis=2)
    if fine_hw is not None:
        fh, fw = int(fine_hw[0]), int(fine_hw[1])
        painted = painted[:, :fh, :fw]
    return painted


def fine_grid_hw(meta: dict) -> tuple[int, int] | None:
    cell = int(meta.get("original_cell_px") or ORIGINAL_CELL_PX)
    width = int(meta.get("video_width") or 0)
    height = int(meta.get("video_height") or 0)
    if cell <= 0 or width <= 0 or height <= 0:
        return None
    return height // cell, width // cell


def voxel_surface_quads(occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exposed voxel faces as quads in plot coords (x, t, y)."""
    binary = np.asarray(occ, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"expected (T,Y,X), got {binary.shape}")
    p = np.pad(binary, 1, mode="constant", constant_values=False)
    center = p[1:-1, 1:-1, 1:-1]
    chunks: list[np.ndarray] = []
    normals: list[np.ndarray] = []

    def emit(corners: np.ndarray, normal: tuple[float, float, float]) -> None:
        if corners.shape[0] == 0:
            return
        chunks.append(corners)
        normals.append(
            np.broadcast_to(np.asarray(normal, dtype=np.float64), (corners.shape[0], 3)).copy()
        )

    def corners_of(t, y, x, pts) -> np.ndarray:
        tt = t.astype(np.float64)
        yy = y.astype(np.float64)
        xx = x.astype(np.float64)
        stacked = [np.stack(pt(xx, tt, yy), axis=1) for pt in pts]
        return np.stack(stacked, axis=1)

    # +x (neighbor x+1 empty) — plane x+1, plot-normal +X
    t, y, x = np.nonzero(center & ~p[1:-1, 1:-1, 2:])
    emit(
        corners_of(
            t,
            y,
            x,
            (
                lambda xx, tt, yy: (xx + 1.0, tt, yy),
                lambda xx, tt, yy: (xx + 1.0, tt + 1.0, yy),
                lambda xx, tt, yy: (xx + 1.0, tt + 1.0, yy + 1.0),
                lambda xx, tt, yy: (xx + 1.0, tt, yy + 1.0),
            ),
        ),
        (1.0, 0.0, 0.0),
    )
    # -x
    t, y, x = np.nonzero(center & ~p[1:-1, 1:-1, :-2])
    emit(
        corners_of(
            t,
            y,
            x,
            (
                lambda xx, tt, yy: (xx, tt, yy),
                lambda xx, tt, yy: (xx, tt, yy + 1.0),
                lambda xx, tt, yy: (xx, tt + 1.0, yy + 1.0),
                lambda xx, tt, yy: (xx, tt + 1.0, yy),
            ),
        ),
        (-1.0, 0.0, 0.0),
    )
    # +y
    t, y, x = np.nonzero(center & ~p[1:-1, 2:, 1:-1])
    emit(
        corners_of(
            t,
            y,
            x,
            (
                lambda xx, tt, yy: (xx, tt, yy + 1.0),
                lambda xx, tt, yy: (xx + 1.0, tt, yy + 1.0),
                lambda xx, tt, yy: (xx + 1.0, tt + 1.0, yy + 1.0),
                lambda xx, tt, yy: (xx, tt + 1.0, yy + 1.0),
            ),
        ),
        (0.0, 0.0, 1.0),
    )
    # -y
    t, y, x = np.nonzero(center & ~p[1:-1, :-2, 1:-1])
    emit(
        corners_of(
            t,
            y,
            x,
            (
                lambda xx, tt, yy: (xx, tt, yy),
                lambda xx, tt, yy: (xx, tt + 1.0, yy),
                lambda xx, tt, yy: (xx + 1.0, tt + 1.0, yy),
                lambda xx, tt, yy: (xx + 1.0, tt, yy),
            ),
        ),
        (0.0, 0.0, -1.0),
    )
    # +t
    t, y, x = np.nonzero(center & ~p[2:, 1:-1, 1:-1])
    emit(
        corners_of(
            t,
            y,
            x,
            (
                lambda xx, tt, yy: (xx, tt + 1.0, yy),
                lambda xx, tt, yy: (xx + 1.0, tt + 1.0, yy),
                lambda xx, tt, yy: (xx + 1.0, tt + 1.0, yy + 1.0),
                lambda xx, tt, yy: (xx, tt + 1.0, yy + 1.0),
            ),
        ),
        (0.0, 1.0, 0.0),
    )
    # -t
    t, y, x = np.nonzero(center & ~p[:-2, 1:-1, 1:-1])
    emit(
        corners_of(
            t,
            y,
            x,
            (
                lambda xx, tt, yy: (xx, tt, yy),
                lambda xx, tt, yy: (xx, tt, yy + 1.0),
                lambda xx, tt, yy: (xx + 1.0, tt, yy + 1.0),
                lambda xx, tt, yy: (xx + 1.0, tt, yy),
            ),
        ),
        (0.0, -1.0, 0.0),
    )

    if not chunks:
        return np.zeros((0, 4, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0), np.concatenate(normals, axis=0)


def _face_colors(normals: np.ndarray) -> np.ndarray:
    light = _LIGHT / np.linalg.norm(_LIGHT)
    lambert = np.clip(np.asarray(normals, dtype=np.float64) @ light, 0.0, 1.0)
    shade = 0.32 + 0.68 * lambert
    rgb = np.clip(_BASE_RGB[None, :] * shade[:, None], 0.0, 1.0)
    return np.concatenate([rgb, np.full((rgb.shape[0], 1), 0.92)], axis=1)


def _style_axes(ax, *, grid_w: int, grid_h: int, num_frames: int, t_aspect: float) -> None:
    tw = float(max(grid_w, 1))
    th = float(max(grid_h, 1))
    tt = float(max(num_frames, 1))
    ax.set_xlim(0, tw)
    ax.set_ylim(0, tt)
    ax.set_zlim(0, th)
    ax.set_xlabel("x (16px cell)")
    ax.set_ylabel("t (frame)")
    ax.set_zlabel("y (16px cell)")
    ax.invert_zaxis()
    spatial = max(tw, th)
    ax.set_box_aspect((tw / spatial, float(t_aspect), th / spatial))
    ax.set_facecolor((0.96, 0.97, 0.99))
    ax.xaxis.pane.set_facecolor((0.93, 0.94, 0.97, 0.6))
    ax.yaxis.pane.set_facecolor((0.93, 0.94, 0.97, 0.6))
    ax.zaxis.pane.set_facecolor((0.93, 0.94, 0.97, 0.6))


def render_occupancy_surface(
    occ: np.ndarray,
    *,
    out_path: Path,
    title: str = "",
    dpi: int = 160,
    t_aspect: float = _T_ASPECT,
    views: tuple[tuple[float, float], ...] | None = None,
) -> Path:
    """Occupancy wrap from one or more viewpoints (default: elev=18, azim=-55)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_len, grid_h, grid_w = (int(v) for v in occ.shape)
    quads, normals = voxel_surface_quads(occ)
    colors = _face_colors(normals) if len(normals) else np.zeros((0, 4))

    view_list = ((18.0, -55.0),) if views is None else tuple(views)
    if not view_list:
        raise ValueError("views must not be empty")
    n_views = len(view_list)
    fig_w = 8.0 if n_views == 1 else 7.2 * n_views
    fig = plt.figure(figsize=(fig_w, 6.4), facecolor="white")
    for i, (elev, azim) in enumerate(view_list, start=1):
        ax = fig.add_subplot(1, n_views, i, projection="3d", computed_zorder=False)
        if len(quads):
            mesh = Poly3DCollection(
                quads,
                facecolors=colors,
                edgecolors="none",
                linewidths=0.0,
                shade=False,
            )
            ax.add_collection3d(mesh)
        _style_axes(ax, grid_w=grid_w, grid_h=grid_h, num_frames=t_len, t_aspect=t_aspect)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev} azim={azim}", fontsize=10, pad=6)
    if title:
        fig.suptitle(title, fontsize=11, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_occupancy_footprint(
    occ: np.ndarray,
    *,
    out_path: Path,
    cell_px: int = ORIGINAL_CELL_PX,
    title: str = "",
) -> Path:
    """Mean occupancy over t — where motion lingered."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    footprint = np.asarray(occ, dtype=bool).any(axis=0)
    dwell = np.asarray(occ, dtype=np.float32).mean(axis=0)
    fig, ax = plt.subplots(figsize=(8.0, 4.6), facecolor="white")
    ax.imshow(dwell, cmap="magma", vmin=0.0, vmax=max(float(dwell.max()), 1e-6), origin="upper")
    ax.contour(footprint.astype(np.float32), levels=[0.5], colors="white", linewidths=0.6)
    ax.set_title(title or f"occupancy dwell | cell={int(cell_px)}px")
    ax.set_xlabel("x (cell)")
    ax.set_ylabel("y (cell)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_stage2_fine_magnitude(fusion_npz: Path, *, fusion: str = "rms") -> tuple[np.ndarray, dict]:
    """16px fused magnitude (M_fused), not 64px MU."""
    from stage3.roi_tube import load_stage2_unit_map

    return load_stage2_unit_map(Path(fusion_npz), prefer_unit=False, fusion=fusion)


def load_stage2_unit_magnitude(fusion_npz: Path, *, fusion: str = "rms") -> tuple[np.ndarray, dict]:
    """64px fused magnitude (MU), after temporal fusion and 4×4 spatial aggregation."""
    from stage3.roi_tube import load_stage2_unit_map

    return load_stage2_unit_map(Path(fusion_npz), prefer_unit=True, fusion=fusion)

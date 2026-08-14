"""3D spatio-temporal figures for ROI tubes.

Plot axes (display):
  X = spatial x (block)
  Y = time t (frame)   ← elongated
  Z = spatial y (block)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stage3.hysteresis_tube import RoiTube

# Distinct tube colors (RGBA 0-1).
_TUBE_COLORS = (
    (0.25, 0.45, 0.85, 0.28),
    (0.15, 0.65, 0.55, 0.28),
    (0.75, 0.35, 0.20, 0.28),
    (0.55, 0.30, 0.75, 0.28),
    (0.85, 0.55, 0.15, 0.28),
    (0.20, 0.55, 0.80, 0.28),
    (0.60, 0.20, 0.45, 0.28),
    (0.30, 0.70, 0.30, 0.28),
)

# Visual length of t-axis relative to the larger spatial axis.
_T_ASPECT = 2.8


def _cuboid_vertices(
    x0: float, x1: float, y0: float, y1: float, z0: float, z1: float
) -> np.ndarray:
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )


_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (1, 2, 6, 5),
    (0, 3, 7, 4),
)


def _add_cuboid(ax, x0, x1, y0, y1, z0, z1, *, facecolor, edgecolor, linewidth=0.6):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = _cuboid_vertices(x0, x1, y0, y1, z0, z1)
    faces = [[verts[i] for i in face] for face in _FACES]
    poly = Poly3DCollection(
        faces,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        shade=False,
    )
    ax.add_collection3d(poly)


def _xy_t_to_plot(
    x0: float, x1: float, y0: float, y1: float, t0: float, t1: float
) -> tuple[float, float, float, float, float, float]:
    """Map data (x,y,t) → plot (X=x, Y=t, Z=y)."""
    return x0, x1, t0, t1, y0, y1


def render_tubes_3d(
    tubes: list[RoiTube],
    *,
    grid_h: int,
    grid_w: int,
    num_frames: int,
    out_path: Path,
    title: str = "",
    dpi: int = 160,
    t_aspect: float = _T_ASPECT,
) -> Path:
    """Write PNG: block-event prisms + tube AABB (X=x, Y=t elongated, Z=y)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.0, 7.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor((0.96, 0.97, 0.99))

    tw = float(max(grid_w, 1))
    th = float(max(grid_h, 1))
    tt = float(max(num_frames, 1))

    # Outer volume.
    _add_cuboid(
        ax,
        *_xy_t_to_plot(0.0, tw, 0.0, th, 0.0, tt),
        facecolor=(0.85, 0.88, 0.95, 0.04),
        edgecolor=(0.35, 0.40, 0.55, 0.55),
        linewidth=1.0,
    )

    for tube in tubes:
        rgba = _TUBE_COLORS[(tube.tube_id - 1) % len(_TUBE_COLORS)]
        edge = (rgba[0] * 0.55, rgba[1] * 0.55, rgba[2] * 0.55, 0.85)
        for member in tube.members:
            _add_cuboid(
                ax,
                *_xy_t_to_plot(
                    float(member.x),
                    float(member.x + 1),
                    float(member.y),
                    float(member.y + 1),
                    float(member.t0),
                    float(member.t1 + 1),
                ),
                facecolor=rgba,
                edgecolor=edge,
                linewidth=0.45,
            )
        sx0, sy0, sx1, sy1 = tube.spatial_bbox()
        _add_cuboid(
            ax,
            *_xy_t_to_plot(
                float(sx0),
                float(sx1),
                float(sy0),
                float(sy1),
                float(tube.t0),
                float(tube.t1 + 1),
            ),
            facecolor=(rgba[0], rgba[1], rgba[2], 0.06),
            edgecolor=(rgba[0], rgba[1], rgba[2], 0.95),
            linewidth=1.4,
        )

    ax.set_xlim(0, tw)
    ax.set_ylim(0, tt)
    ax.set_zlim(0, th)
    ax.set_xlabel("x (block)")
    ax.set_ylabel("t (frame)")
    ax.set_zlabel("y (block)")
    ax.invert_zaxis()  # image-like spatial y down

    # Stretch t (plot-Y) so it reads longer than spatial axes.
    spatial_ref = max(tw, th)
    ax.set_box_aspect(
        (
            tw / spatial_ref,
            float(t_aspect),
            th / spatial_ref,
        )
    )
    ax.view_init(elev=18, azim=-55)
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    ax.xaxis.pane.set_facecolor((0.93, 0.94, 0.97, 0.6))
    ax.yaxis.pane.set_facecolor((0.93, 0.94, 0.97, 0.6))
    ax.zaxis.pane.set_facecolor((0.93, 0.94, 0.97, 0.6))
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

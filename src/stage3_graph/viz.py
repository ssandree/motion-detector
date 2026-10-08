"""Stage3_graph visualizations: ROI overlay video + 3D cluster segments."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from motion_analyzer.config import HEAT_VMAX, HEAT_VMIN
from motion_analyzer.visualization import grid_bbox_to_pixels, heat_overlay
from stage3.roi_tube import HEAT_MAX_ALPHA, _linear_heat_level
from stage3_graph.nodes import GraphNode
from stage3_graph.tubes import GraphRoiTube

# Distinct BGR colors for overlay (OpenCV).
CLUSTER_COLORS_BGR = (
    (0, 220, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 200, 0),
    (0, 255, 255),
    (180, 105, 255),
    (50, 200, 50),
    (255, 100, 100),
    (255, 255, 0),
    (0, 128, 255),
    (40, 40, 255),
    (200, 180, 0),
    (0, 200, 180),
    (220, 80, 160),
    (100, 220, 255),
    (160, 255, 80),
)

# Match Stage3 tube_3d_viz: plot Y = time (elongated).
_T_ASPECT = 2.8


def cluster_color_bgr(cluster_id: int) -> tuple[int, int, int]:
    return CLUSTER_COLORS_BGR[(int(cluster_id) - 1) % len(CLUSTER_COLORS_BGR)]


def cluster_color_rgba(cluster_id: int, *, alpha: float = 0.35) -> tuple[float, float, float, float]:
    b, g, r = cluster_color_bgr(cluster_id)
    return (r / 255.0, g / 255.0, b / 255.0, float(alpha))


def draw_graph_roi_frame(
    frame: np.ndarray,
    tubes: list[GraphRoiTube],
    *,
    t: int,
    cell_px: int,
    mag_frame: np.ndarray | None = None,
    heat_vmin: float = HEAT_VMIN,
    heat_vmax: float = HEAT_VMAX,
    heat_alpha: float = HEAT_MAX_ALPHA,
    thickness: int = 2,
) -> np.ndarray:
    """Frame + Stage2 turbo MU heatmap + fixed lifetime AABB per cluster."""
    if mag_frame is not None and np.isfinite(mag_frame).any():
        out = heat_overlay(
            frame,
            _linear_heat_level(
                mag_frame, vmin=float(heat_vmin), vmax=float(heat_vmax)
            ),
            cell_px=int(cell_px),
            max_alpha=float(heat_alpha),
        )
    else:
        out = frame.copy()
    fh, fw = out.shape[:2]
    for tube in tubes:
        bbox = tube.bbox_at(t)
        if bbox is None:
            continue
        color = cluster_color_bgr(tube.cluster_id)
        pxb = grid_bbox_to_pixels(
            bbox,
            unit_pixel_size=int(cell_px),
            frame_width=fw,
            frame_height=fh,
        )
        cv2.rectangle(
            out,
            (pxb[0], pxb[1]),
            (pxb[2] - 1, pxb[3] - 1),
            color,
            int(thickness),
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"C{tube.cluster_id}",
            (pxb[0] + 4, max(16, pxb[1] + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def write_graph_roi_overlay(
    *,
    frames_bgr: list[np.ndarray],
    frame_indices: list[int],
    tubes: list[GraphRoiTube],
    cell_px: int,
    out_path: Path,
    fps: float,
    mag: np.ndarray | None = None,
) -> int:
    """Write ``graph_roi_overlay.mp4`` (or video-prefixed name)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_bgr:
        return 0
    fh, fw = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (fw, fh),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {out_path}")
    mag_arr = None if mag is None else np.asarray(mag)
    written = 0
    try:
        for t, samp_idx in enumerate(frame_indices):
            if samp_idx < 0 or samp_idx >= len(frames_bgr):
                continue
            mag_frame = None
            if mag_arr is not None and 0 <= t < mag_arr.shape[0]:
                mag_frame = mag_arr[t]
            writer.write(
                draw_graph_roi_frame(
                    frames_bgr[samp_idx],
                    tubes,
                    t=t,
                    cell_px=int(cell_px),
                    mag_frame=mag_frame,
                )
            )
            written += 1
    finally:
        writer.release()
    return written


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
    """Map data (x,y,t) → plot (X=x, Y=t, Z=y). Same as Stage3 tube_3d_viz."""
    return x0, x1, t0, t1, y0, y1


def render_graph_clusters_3d(
    nodes: list[GraphNode],
    labels: list[int],
    *,
    tubes: list[GraphRoiTube] | None = None,
    grid_h: int,
    grid_w: int,
    num_frames: int,
    out_path: Path,
    title: str = "",
    dpi: int = 160,
    t_aspect: float = _T_ASPECT,
) -> Path:
    """Stage3-matched layout: X=x, Y=t (elongated), Z=y.

    Member events as prisms + optional lifetime AABB outline per cluster.
    """
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

    _add_cuboid(
        ax,
        *_xy_t_to_plot(0.0, tw, 0.0, th, 0.0, tt),
        facecolor=(0.85, 0.88, 0.95, 0.04),
        edgecolor=(0.35, 0.40, 0.55, 0.55),
        linewidth=1.0,
    )

    for node, lab in zip(nodes, labels):
        rgba = cluster_color_rgba(int(lab), alpha=0.32)
        edge = (rgba[0] * 0.55, rgba[1] * 0.55, rgba[2] * 0.55, 0.9)
        _add_cuboid(
            ax,
            *_xy_t_to_plot(
                float(node.x),
                float(node.x + 1),
                float(node.y),
                float(node.y + 1),
                float(node.t0),
                float(node.t1 + 1),
            ),
            facecolor=rgba,
            edgecolor=edge,
            linewidth=0.45,
        )

    if tubes:
        for tube in tubes:
            rgba = cluster_color_rgba(int(tube.cluster_id), alpha=0.06)
            x0, y0, x1, y1 = tube.spatial_extent()
            _add_cuboid(
                ax,
                *_xy_t_to_plot(
                    float(x0),
                    float(x1),
                    float(y0),
                    float(y1),
                    float(tube.t0),
                    float(tube.t1 + 1),
                ),
                facecolor=rgba,
                edgecolor=(rgba[0], rgba[1], rgba[2], 0.95),
                linewidth=1.4,
            )

    ax.set_xlim(0, tw)
    ax.set_ylim(0, tt)
    ax.set_zlim(0, th)
    ax.set_xlabel("x (block)")
    ax.set_ylabel("t (frame)")
    ax.set_zlabel("y (block)")
    ax.invert_zaxis()

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

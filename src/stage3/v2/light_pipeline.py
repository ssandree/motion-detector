"""Official Stage3: events → attractive Union-Find split → ROI composition.

Drops coarse grouping and GAEC+KL Graph-cut. Edge costs and AABB composition
match exp_v2c.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from motion_analyzer.config import (
    DEFAULT_FUSION,
    HEAT_VMAX,
    HEAT_VMIN,
    PipelineConfig,
    ROI_MAX_GAP,
    ROI_MIN_BLOCK_EVENT,
    ROI_MIN_BLOCK_PX,
    ROI_MIN_TUBE_CELLS,
    ROI_MIN_TUBE_DURATION,
    ROI_SUPPRESS_CONTAINED,
    ROI_TAU_HIGH,
    ROI_TAU_LOW,
)
from motion_analyzer.video_io import resolve_video_path, sample_video_frames
from stage3.hysteresis_tube import (
    apply_stage3_roi_filters,
    filter_partial_block_events,
    frame_indices_from_meta,
    tube_to_dict,
)
from stage3_graph.filter_nodes import filter_isolated_single_block_nodes
from stage3.roi_tube import (
    HEAT_MAX_ALPHA,
    load_stage2_unit_map,
    resolve_stage2_npz,
    write_roi_overlay_mp4,
)
from stage3.tube_3d_viz import render_tubes_3d
from stage3_graph.config import SIGMA_T, SIGMA_XY, W_T, W_XY
from stage3_graph.nodes import build_or_load_events, nodes_from_block_events
from stage3_graph.pipeline import load_unit_flow
from stage3.v2.light_split import (
    CONSERVATIVE_CHEBYSHEV,
    CONSERVATIVE_TAU_GRAPH,
    CONSERVATIVE_TEMPORAL_GAP,
    CONSERVATIVE_W_DIR,
    split_attractive_unionfind,
)
from stage3.v2.roi_composition import (
    COMP_COMPACTNESS_FACTOR,
    COMP_MAX_ROI_PER_FRAME,
    COMP_MAX_TEMPORAL_GAP,
    COMP_TARGET_ROI_PER_FRAME,
    compose_rois,
)

logger = logging.getLogger("stage3.v2.light")


def build_light_roi_tracks(
    video_id: str,
    mag: np.ndarray,
    flow: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    cell_px: int,
    score_key: str,
    fusion_npz: str | None = None,
    events_root: Path | None = None,
    tau_high: float = ROI_TAU_HIGH,
    tau_low: float = ROI_TAU_LOW,
    max_gap: int = ROI_MAX_GAP,
    min_block_event: int = ROI_MIN_BLOCK_EVENT,
    min_block_px: bool = ROI_MIN_BLOCK_PX,
    chebyshev: int = CONSERVATIVE_CHEBYSHEV,
    temporal_gap: int = CONSERVATIVE_TEMPORAL_GAP,
    sigma_xy: float = SIGMA_XY,
    sigma_t: float = SIGMA_T,
    w_xy: float = W_XY,
    w_t: float = W_T,
    w_dir: float = CONSERVATIVE_W_DIR,
    tau_graph: float = CONSERVATIVE_TAU_GRAPH,
    comp_max_temporal_gap: int = COMP_MAX_TEMPORAL_GAP,
    comp_compactness_factor: float = COMP_COMPACTNESS_FACTOR,
    comp_target_rois: int = COMP_TARGET_ROI_PER_FRAME,
    comp_max_rois: int = COMP_MAX_ROI_PER_FRAME,
    min_tube_cells: int = ROI_MIN_TUBE_CELLS,
    min_tube_duration: int = ROI_MIN_TUBE_DURATION,
    suppress_contained: bool = ROI_SUPPRESS_CONTAINED,
    heat_vmin: float = HEAT_VMIN,
    heat_vmax: float = HEAT_VMAX,
    heat_alpha: float = HEAT_MAX_ALPHA,
) -> dict[str, Any]:
    """Events → Union-Find → ROI composition. No video decode and no files."""
    if flow.shape[:3] != mag.shape:
        raise ValueError(f"{video_id}: flow {flow.shape[:3]} != mag {mag.shape}")

    events, _events_meta = build_or_load_events(
        video_id=video_id,
        mag=mag,
        events_root=events_root,
        tau_high=float(tau_high),
        tau_low=float(tau_low),
        max_gap=int(max_gap),
        min_block_event=int(min_block_event),
    )
    dropped_partial: list[int] = []
    if bool(min_block_px):
        events, dropped_partial = filter_partial_block_events(
            events,
            cell_px=int(cell_px),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
        )

    nodes = nodes_from_block_events(events, flow, mag)
    nodes, dropped_isolated = filter_isolated_single_block_nodes(
        nodes, neigh_chebyshev=int(chebyshev)
    )

    t_cluster0 = time.perf_counter()
    comps, split_log = split_attractive_unionfind(
        nodes,
        chebyshev=int(chebyshev),
        temporal_gap=int(temporal_gap),
        sigma_xy=float(sigma_xy),
        sigma_t=float(sigma_t),
        w_xy=float(w_xy),
        w_t=float(w_t),
        w_dir=float(w_dir),
        tau_graph=float(tau_graph),
        use_temporal_bridge=True,
    )
    tubes_composed, comp_log = compose_rois(
        comps,
        max_temporal_gap=int(comp_max_temporal_gap),
        compactness_factor=float(comp_compactness_factor),
        target_rois=int(comp_target_rois),
        max_rois=int(comp_max_rois),
    )
    cluster_sec = round(float(time.perf_counter() - t_cluster0), 6)

    params = {
        "tau_high": float(tau_high),
        "tau_low": float(tau_low),
        "max_gap": int(max_gap),
        "min_tube_cells": int(min_tube_cells),
        "min_tube_duration": int(min_tube_duration),
        "chebyshev": int(chebyshev),
        "temporal_gap": int(temporal_gap),
        "tau_graph": float(tau_graph),
        "w_dir": float(w_dir),
        "comp_compactness_factor": float(comp_compactness_factor),
        "comp_max_rois": int(comp_max_rois),
        "heat_vmin": float(heat_vmin),
        "heat_vmax": float(heat_vmax),
        "heat_alpha": float(heat_alpha),
        "suppress_contained": bool(suppress_contained),
        "min_block_px": bool(min_block_px),
    }
    final_tubes, final_filt = apply_stage3_roi_filters(
        tubes_composed,
        params=params,
        cell_px=int(cell_px),
        frame_width=int(frame_width),
        frame_height=int(frame_height),
        iou_nms_tau=0.5,
    )
    tracks_json: dict[str, Any] = {
        "video_id": video_id,
        "variant": "stage3_light_uf_compose",
        "note": (
            "Step1: events | Step2: attractive Union-Find (hashed local edges) | "
            "Step3: ROI composition (AABB compact)"
        ),
        "params": params,
        "fusion_npz": fusion_npz,
        "score_key": score_key,
        "cell_px": int(cell_px),
        "num_frames": int(mag.shape[0]),
        "num_block_events": len(events),
        "num_dropped_partial": len(dropped_partial),
        "num_dropped_isolated": len(dropped_isolated),
        "num_split_components": len(comps),
        "num_composed_rois": len(tubes_composed),
        "num_final_rois": len(final_tubes),
        "split_log": split_log,
        "composition_log": comp_log,
        "roi_filter": final_filt,
        "cluster_sec": cluster_sec,
        "tracks": [tube_to_dict(t) for t in final_tubes],
        "tubes": [tube_to_dict(t) for t in final_tubes],
    }
    return {"tracks_json": tracks_json, "final_tubes": final_tubes}


def process_video_light(
    video_id: str,
    *,
    cfg: PipelineConfig,
    fusion_root: Path,
    output_root: Path,
    fusion: str = DEFAULT_FUSION,
    prefer_unit: bool = True,
    events_root: Path | None = None,
    tau_high: float = ROI_TAU_HIGH,
    tau_low: float = ROI_TAU_LOW,
    max_gap: int = ROI_MAX_GAP,
    min_block_event: int = ROI_MIN_BLOCK_EVENT,
    min_block_px: bool = ROI_MIN_BLOCK_PX,
    chebyshev: int = CONSERVATIVE_CHEBYSHEV,
    temporal_gap: int = CONSERVATIVE_TEMPORAL_GAP,
    sigma_xy: float = SIGMA_XY,
    sigma_t: float = SIGMA_T,
    w_xy: float = W_XY,
    w_t: float = W_T,
    w_dir: float = CONSERVATIVE_W_DIR,
    tau_graph: float = CONSERVATIVE_TAU_GRAPH,
    comp_max_temporal_gap: int = COMP_MAX_TEMPORAL_GAP,
    comp_compactness_factor: float = COMP_COMPACTNESS_FACTOR,
    comp_target_rois: int = COMP_TARGET_ROI_PER_FRAME,
    comp_max_rois: int = COMP_MAX_ROI_PER_FRAME,
    min_tube_cells: int = ROI_MIN_TUBE_CELLS,
    min_tube_duration: int = ROI_MIN_TUBE_DURATION,
    suppress_contained: bool = ROI_SUPPRESS_CONTAINED,
    write_3d: bool = True,
    write_overlay: bool = True,
    viz_3d_root: Path | None = None,
    video_root: Path | None = None,
    json_root: Path | None = None,
    heat_vmin: float = HEAT_VMIN,
    heat_vmax: float = HEAT_VMAX,
    heat_alpha: float = HEAT_MAX_ALPHA,
    **_ignored,
) -> dict[str, Any]:
    del _ignored
    t_wall0 = time.perf_counter()

    fusion_npz = resolve_stage2_npz(Path(fusion_root), video_id, fusion=fusion)
    mag, meta = load_stage2_unit_map(fusion_npz, prefer_unit=prefer_unit, fusion=fusion)
    flow = load_unit_flow(fusion_npz, fusion=fusion)
    cell_px = int(meta.get("unit_cell_px") or 64)

    video_path = resolve_video_path(video_id, cfg.video_search_roots)
    sampled = sample_video_frames(video_path, cfg.sampling_fps)
    fh, fw = sampled[0].bgr.shape[:2]
    curr_indices = frame_indices_from_meta(meta, mag.shape[0])
    built = build_light_roi_tracks(
        video_id,
        mag,
        flow,
        frame_width=int(fw),
        frame_height=int(fh),
        cell_px=cell_px,
        score_key=str(meta["key"]),
        fusion_npz=str(fusion_npz),
        events_root=events_root,
        tau_high=float(tau_high),
        tau_low=float(tau_low),
        max_gap=int(max_gap),
        min_block_event=int(min_block_event),
        min_block_px=bool(min_block_px),
        chebyshev=int(chebyshev),
        temporal_gap=int(temporal_gap),
        sigma_xy=float(sigma_xy),
        sigma_t=float(sigma_t),
        w_xy=float(w_xy),
        w_t=float(w_t),
        w_dir=float(w_dir),
        tau_graph=float(tau_graph),
        comp_max_temporal_gap=int(comp_max_temporal_gap),
        comp_compactness_factor=float(comp_compactness_factor),
        comp_target_rois=int(comp_target_rois),
        comp_max_rois=int(comp_max_rois),
        min_tube_cells=int(min_tube_cells),
        min_tube_duration=int(min_tube_duration),
        suppress_contained=bool(suppress_contained),
        heat_vmin=float(heat_vmin),
        heat_vmax=float(heat_vmax),
        heat_alpha=float(heat_alpha),
    )
    tracks_json = built["tracks_json"]
    final_tubes = built["final_tubes"]

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tag = (
        f"light_h{tau_high:g}"
        f"_g{chebyshev}t{tau_graph:g}"
        f"_cmp{comp_compactness_factor:g}r{comp_max_rois}"
    )
    vid_dir = Path(video_root) if video_root is not None else output_root
    json_dir = Path(json_root) if json_root is not None else output_root
    vid_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = vid_dir / f"{video_id}_roi_tube_{tag}.mp4"
    n_written = 0
    if write_overlay:
        frames_bgr = [s.bgr for s in sampled]
        n_written = write_roi_overlay_mp4(
            frames_bgr=frames_bgr,
            frame_indices=curr_indices,
            tubes=final_tubes,
            mag=mag,
            cell_px=cell_px,
            out_path=mp4_path,
            fps=float(cfg.sampling_fps),
            heat_vmin=float(heat_vmin),
            heat_vmax=float(heat_vmax),
            heat_alpha=float(heat_alpha),
        )

    fig_3d_path = None
    if write_3d:
        fig_dir = Path(viz_3d_root) if viz_3d_root is not None else output_root
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_3d_path = fig_dir / f"{video_id}_roi_tubes_3d_{tag}.png"
        render_tubes_3d(
            final_tubes,
            grid_h=int(mag.shape[1]),
            grid_w=int(mag.shape[2]),
            num_frames=int(mag.shape[0]),
            out_path=fig_3d_path,
            title=(
                f"{video_id} | light: UF({tracks_json['num_split_components']})"
                f"→ROI({tracks_json['num_final_rois']})"
            ),
        )

    tracks_path = json_dir / f"{video_id}_roi_tracks.json"
    tracks_path.write_text(json.dumps(tracks_json, indent=2) + "\n", encoding="utf-8")

    pipeline_sec = round(float(time.perf_counter() - t_wall0), 6)
    logger.info(
        "%s events=%d → uf_comps=%d → composed=%d → final_roi=%d (cluster=%.3fs wall=%.2fs)",
        video_id,
        tracks_json["num_block_events"],
        tracks_json["num_split_components"],
        tracks_json["num_composed_rois"],
        tracks_json["num_final_rois"],
        tracks_json["cluster_sec"],
        pipeline_sec,
    )
    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "fusion_npz": str(fusion_npz),
        "tracks_json": str(tracks_path),
        "visualization_mp4": str(mp4_path) if write_overlay else None,
        "visualization_3d": str(fig_3d_path) if fig_3d_path else None,
        "visualization_frames": int(n_written),
        "num_events": tracks_json["num_block_events"],
        "num_split_components": tracks_json["num_split_components"],
        "num_gc_components": tracks_json["num_split_components"],
        "num_graph_partitions": tracks_json["num_split_components"],
        "num_composed_rois": tracks_json["num_composed_rois"],
        "num_tubes": tracks_json["num_final_rois"],
        "num_tracks": tracks_json["num_final_rois"],
        "score_key": tracks_json["score_key"],
        "cell_px": int(cell_px),
        "map_shape": list(mag.shape),
        "params": tracks_json["params"],
        "cluster_sec": tracks_json["cluster_sec"],
        "pipeline_sec": pipeline_sec,
    }


_LEGACY_KW = {
    "gc_tau_graph": "tau_graph",
    "gc_w_dir": "w_dir",
    "gc_chebyshev": "chebyshev",
    "gc_temporal_gap": "temporal_gap",
    "gc_sigma_xy": "sigma_xy",
    "gc_sigma_t": "sigma_t",
    "gc_w_xy": "w_xy",
    "gc_w_t": "w_t",
}


def process_video(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Official Stage3 entry (light Union-Find → composition)."""
    remapped = dict(kwargs)
    for old, new in _LEGACY_KW.items():
        if old in remapped:
            val = remapped.pop(old)
            remapped.setdefault(new, val)
    remapped.pop("gc_kl_passes", None)
    remapped.pop("coarse_chebyshev", None)
    remapped.pop("coarse_temporal_gap", None)
    return process_video_light(*args, **remapped)


__all__ = ["build_light_roi_tracks", "process_video", "process_video_light"]

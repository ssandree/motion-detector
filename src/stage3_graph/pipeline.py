"""Stage3_graph pipeline: events → signed graph → multicut → ROI tubes."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from motion_analyzer.config import (
    DEFAULT_FUSION,
    PipelineConfig,
    ROI_MAX_GAP,
    ROI_MIN_BLOCK_EVENT,
    ROI_TAU_HIGH,
    ROI_TAU_LOW,
)
from motion_analyzer.video_io import resolve_video_path, sample_video_frames
from stage3.roi_tube import list_videos_in_fusion_root, load_stage2_unit_map, resolve_stage2_npz
from stage3_graph.config import (
    EDGE_TAU,
    KL_MAX_PASSES,
    SIGMA_T,
    SIGMA_XY,
    SPATIAL_CHEBYSHEV_MAX,
    TAU_SPATIAL,
    TEMPORAL_GAP_MAX,
    TEMPORAL_PRED_RADIUS,
    W_DIR,
    W_T,
    W_XY,
)
from stage3_graph.edges import build_candidate_edges
from stage3_graph.filter_nodes import filter_isolated_single_block_nodes
from stage3_graph.multicut import solve_multicut
from stage3_graph.nodes import build_or_load_events, nodes_from_block_events
from stage3_graph.tubes import build_roi_tubes, drop_sub_block_tubes, tube_summary
from stage3_graph.viz import render_graph_clusters_3d, write_graph_roi_overlay

logger = logging.getLogger("stage3_graph")

OUTPUT_SUFFIX = "noaff_ovmin_tbridge"


def load_unit_flow(fusion_npz: Path, *, fusion: str = DEFAULT_FUSION) -> np.ndarray:
    """Load unit-grid flow UU_* matching MU grid."""
    tag = str(fusion).lower()
    with np.load(fusion_npz) as data:
        for key in (f"UU_{tag}", "UU_fused", "UU_rms", "UU_mean", "UU_max"):
            if key in data.files:
                flow = np.asarray(data[key], dtype=np.float32)
                break
        else:
            # Aggregate U_fused 4×4 mean if UU missing.
            u_key = None
            for cand in (f"U_{tag}", "U_fused", "U_rms", "U_mean", "U_max"):
                if cand in data.files:
                    u_key = cand
                    break
            if u_key is None:
                raise KeyError(f"{fusion_npz}: missing UU_* / U_* flow")
            fine = np.asarray(data[u_key], dtype=np.float32)
            block = int(np.asarray(data.get("unit_block", 4)).item())
            t, h, w, _ = fine.shape
            hh, ww = h // block, w // block
            cropped = fine[:, : hh * block, : ww * block]
            flow = (
                cropped.reshape(t, hh, block, ww, block, 2)
                .mean(axis=(2, 4))
                .astype(np.float32)
            )
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError(f"bad flow shape {flow.shape}")
    return flow


def process_video(
    video_id: str,
    *,
    cfg: PipelineConfig,
    fusion_root: Path,
    output_root: Path,
    events_root: Path | None = None,
    fusion: str = DEFAULT_FUSION,
    prefer_unit: bool = True,
    tau_high: float = ROI_TAU_HIGH,
    tau_low: float = ROI_TAU_LOW,
    max_gap: int = ROI_MAX_GAP,
    min_block_event: int = ROI_MIN_BLOCK_EVENT,
    spatial_chebyshev: int = SPATIAL_CHEBYSHEV_MAX,
    temporal_gap_max: int = TEMPORAL_GAP_MAX,
    pred_radius: float = TEMPORAL_PRED_RADIUS,
    sigma_xy: float = SIGMA_XY,
    sigma_t: float = SIGMA_T,
    edge_tau: float = EDGE_TAU,
    w_xy: float = W_XY,
    w_t: float = W_T,
    w_dir: float = W_DIR,
    tau_spatial: float = TAU_SPATIAL,
    kl_passes: int = KL_MAX_PASSES,
) -> dict:
    t_wall0 = time.perf_counter()
    fusion_npz = resolve_stage2_npz(fusion_root, video_id, fusion=fusion)
    mag, meta = load_stage2_unit_map(
        fusion_npz, prefer_unit=prefer_unit, fusion=fusion
    )
    flow = load_unit_flow(fusion_npz, fusion=fusion)
    if flow.shape[:3] != mag.shape:
        raise ValueError(
            f"{video_id}: flow {flow.shape[:3]} != mag {mag.shape}"
        )
    cell_px = int(meta["unit_cell_px"]) if prefer_unit else int(
        meta.get("original_cell_px") or cell_px_fallback(meta)
    )

    events, events_meta = build_or_load_events(
        video_id=video_id,
        mag=mag,
        events_root=events_root,
        tau_high=float(tau_high),
        tau_low=float(tau_low),
        max_gap=int(max_gap),
        min_block_event=int(min_block_event),
    )
    nodes = nodes_from_block_events(events, flow, mag)
    n_events_raw = len(nodes)
    nodes, dropped_isolated = filter_isolated_single_block_nodes(
        nodes, neigh_chebyshev=int(spatial_chebyshev)
    )
    spatial_edges, temporal_edges, affinity_edges = build_candidate_edges(
        nodes,
        cell_px=float(cell_px),
        spatial_chebyshev=int(spatial_chebyshev),
        temporal_gap_max=int(temporal_gap_max),
        pred_radius=float(pred_radius),
        sigma_xy=float(sigma_xy),
        sigma_t=float(sigma_t),
        tau=float(edge_tau),
        w_xy=float(w_xy),
        w_t=float(w_t),
        w_dir=float(w_dir),
        tau_spatial=float(tau_spatial),
    )
    if affinity_edges:
        raise RuntimeError(
            f"{video_id}: affinity edges must be empty, got {len(affinity_edges)}"
        )
    all_edges = spatial_edges + temporal_edges
    part = solve_multicut(len(nodes), all_edges, kl_passes=int(kl_passes))
    tubes = build_roi_tubes(nodes, part.labels)

    video_path = resolve_video_path(video_id, cfg.video_search_roots)
    sampled = sample_video_frames(video_path, cfg.sampling_fps)
    fh, fw = sampled[0].bgr.shape[:2]
    tubes, dropped_sub = drop_sub_block_tubes(
        tubes,
        cell_px=int(cell_px),
        frame_width=int(fw),
        frame_height=int(fh),
    )
    # Keep only nodes that belong to surviving VLM tubes; rebuild labels for viz.
    keep_event_ids = {m.event_id for t in tubes for m in t.members}
    viz_nodes = [n for n in nodes if n.event_id in keep_event_ids]
    event_to_cluster = {
        m.event_id: int(t.cluster_id) for t in tubes for m in t.members
    }
    viz_labels = [event_to_cluster[n.event_id] for n in viz_nodes]

    n_events = len(nodes)
    n_spatial = len(spatial_edges)
    n_temporal = len(temporal_edges)
    n_affinity = len(affinity_edges)
    n_spatial_attr = sum(1 for e in spatial_edges if float(e.cost) > 0.0)
    n_spatial_rep = sum(1 for e in spatial_edges if float(e.cost) < 0.0)
    n_spatial_zero = n_spatial - n_spatial_attr - n_spatial_rep
    n_clusters = len(tubes)

    logger.info("number of events: %d (raw=%d dropped_isolated=%d)", n_events, n_events_raw, len(dropped_isolated))
    logger.info("number of spatial edges: %d", n_spatial)
    logger.info("number of temporal edges: %d", n_temporal)
    logger.info("number of affinity edges: %d", n_affinity)
    logger.info(
        "spatial edges attractive/repulsive/zero: %d / %d / %d",
        n_spatial_attr,
        n_spatial_rep,
        n_spatial_zero,
    )
    logger.info(
        "number of final clusters: %d  (dropped_sub_block=%d)",
        n_clusters,
        len(dropped_sub),
    )
    cluster_rows = [tube_summary(t) for t in tubes]
    for row in cluster_rows:
        logger.info(
            "cluster_id=%d / events=%d / temporal_range=%s / spatial_extent=%s",
            row["cluster_id"],
            row["n_events"],
            row["temporal_range"],
            row["spatial_extent_grid"],
        )

    align_start = int(meta.get("align_start_sampled_index") or 0)
    if meta["sampled_index_curr"] is not None:
        curr_indices = [int(i) for i in meta["sampled_index_curr"].tolist()]
    else:
        curr_indices = list(range(align_start, align_start + mag.shape[0]))
    if len(curr_indices) != mag.shape[0]:
        raise ValueError(
            f"{video_id}: mag frames={mag.shape[0]} but indices={len(curr_indices)}"
        )

    out_dir = Path(output_root) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"graph_roi_overlay_{OUTPUT_SUFFIX}.mp4"
    fig_3d_path = out_dir / f"graph_clusters_3d_{OUTPUT_SUFFIX}.png"
    result_path = out_dir / f"graph_result_{OUTPUT_SUFFIX}.json"

    frames_bgr = [s.bgr for s in sampled]
    n_written = write_graph_roi_overlay(
        frames_bgr=frames_bgr,
        frame_indices=curr_indices,
        tubes=tubes,
        cell_px=int(cell_px),
        out_path=overlay_path,
        fps=float(cfg.sampling_fps),
        mag=mag,
    )
    render_graph_clusters_3d(
        viz_nodes,
        viz_labels,
        tubes=tubes,
        grid_h=int(mag.shape[1]),
        grid_w=int(mag.shape[2]),
        num_frames=int(mag.shape[0]),
        out_path=fig_3d_path,
        title=(
            f"{video_id} | {OUTPUT_SUFFIX} | events={n_events}/{n_events_raw} "
            f"clusters={n_clusters} | spatial_e={n_spatial} temporal_e={n_temporal} "
            f"affinity_e={n_affinity} | τ_s={tau_spatial:g} | lifetime AABB"
        ),
    )

    params = {
        "tau_high": float(tau_high),
        "tau_low": float(tau_low),
        "max_gap": int(max_gap),
        "min_block_event": int(min_block_event),
        "spatial_chebyshev": int(spatial_chebyshev),
        "temporal_gap_max": int(temporal_gap_max),
        "pred_radius": float(pred_radius),
        "sigma_xy": float(sigma_xy),
        "sigma_t": float(sigma_t),
        "edge_tau": float(edge_tau),
        "w_xy": float(w_xy),
        "w_t": float(w_t),
        "w_dir": float(w_dir),
        "tau_spatial": float(tau_spatial),
        "kl_passes": int(kl_passes),
        "solver": part.solver,
        "cell_px": int(cell_px),
        "affinity_edges": False,
    }
    result = {
        "video_id": video_id,
        "variant": f"stage3_graph_{OUTPUT_SUFFIX}",
        "note": (
            "No affinity edges. Spatial: S_t = overlap/min(dur); "
            "c = w_xy*S_xy + w_t*S_t - w_dir*min(C_i,C_j)*(1-S_v) - tau_spatial. "
            "Temporal: gap∈[1,gap_max] + Chebyshev≤R, no flow prediction gate; "
            "score (S_xy+S_t+S_v)/3 - tau. GAEC+KL; ROI = lifetime AABB."
        ),
        "params": params,
        "fusion_npz": str(fusion_npz),
        "events_json": None
        if events_meta is None
        else events_meta.get("_events_json"),
        "score_key": meta["key"],
        "num_temporal_events_raw": n_events_raw,
        "num_temporal_events": n_events,
        "num_dropped_isolated": len(dropped_isolated),
        "num_dropped_sub_block_tubes": len(dropped_sub),
        "num_spatial_edges": n_spatial,
        "num_temporal_edges": n_temporal,
        "num_affinity_edges": n_affinity,
        "num_spatial_attractive": n_spatial_attr,
        "num_spatial_repulsive": n_spatial_rep,
        "num_spatial_zero_cost": n_spatial_zero,
        "num_final_clusters": n_clusters,
        "multicut_objective": float(part.objective),
        "clusters": cluster_rows,
        "nodes": [
            {
                "node_id": n.node_id,
                "event_id": n.event_id,
                "x": n.x,
                "y": n.y,
                "t0": n.t0,
                "t1": n.t1,
                "u": n.u,
                "v": n.v,
                "mean_mag": n.mean_mag,
                "coherence": n.coherence,
                "cluster_id": event_to_cluster.get(n.event_id),
            }
            for n in nodes
        ],
        "edges": [
            {
                "i": e.i,
                "j": e.j,
                "kind": e.kind,
                "s_xy": e.s_xy,
                "s_t": e.s_t,
                "s_v": e.s_v,
                "r_dir": e.r_dir,
                "s": e.s,
                "cost": e.cost,
            }
            for e in all_edges
        ],
        "graph_roi_overlay": str(overlay_path),
        "graph_clusters_3d": str(fig_3d_path),
        "visualization_frames": int(n_written),
        "pipeline_sec": round(float(time.perf_counter() - t_wall0), 6),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def cell_px_fallback(meta: dict) -> int:
    return int(meta.get("unit_cell_px") or meta.get("original_cell_px") or 64)


__all__ = [
    "list_videos_in_fusion_root",
    "load_unit_flow",
    "process_video",
]

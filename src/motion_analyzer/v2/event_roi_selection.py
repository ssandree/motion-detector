"""Coverage-aware Event ROI selection and revised ranking scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-8
HIGH_IMPORTANCE_THRESHOLD = 0.35


@dataclass
class RoiSelectionConfig:
  short_event_threshold_sec: float = 4.5
  long_event_min_duration_sec: float = 8.0
  min_high_importance_for_long: float = 0.28
  min_active_frames_for_long: int = 6
  replace_duration_ratio: float = 2.0
  replace_min_revised_ratio: float = 0.55
  diversity_temporal_iou_max: float = 0.55
  diversity_score_margin: float = 1.12
  debug_rejected_rank_limit: int = 8
  human_scale_max_area_ratio: float = 0.025
  human_scale_min_duration_sec: float = 6.0


def compute_blob_persistence_stats(members: list[dict[str, Any]]) -> dict[str, float]:
    if not members:
        return {
            "mean_blob_importance": 0.0,
            "max_blob_importance": 0.0,
            "high_importance_frame_ratio": 0.0,
            "num_blob_active_frames": 0,
        }
    importances = [float(m["blob_importance"]) for m in members]
    by_frame: dict[int, float] = {}
    for m in members:
        fi = int(m["frame_idx"])
        imp = float(m["blob_importance"])
        by_frame[fi] = max(by_frame.get(fi, 0.0), imp)
    active_frames = len(by_frame)
    high_frames = sum(1 for v in by_frame.values() if v >= HIGH_IMPORTANCE_THRESHOLD)
    return {
        "mean_blob_importance": round(float(np.mean(importances)), 4),
        "max_blob_importance": round(float(np.max(importances)), 4),
        "high_importance_frame_ratio": round(high_frames / max(active_frames, 1), 4),
        "num_blob_active_frames": active_frames,
    }


def _high_importance_blob_score(cand: dict[str, Any]) -> float:
    mean_imp = float(cand.get("mean_blob_importance", 0.0))
    max_imp = float(cand.get("max_blob_importance", 0.0))
    hi_ratio = float(cand.get("high_importance_frame_ratio", 0.0))
    active = int(cand.get("num_blob_active_frames", 0))
    return round(min(1.0, (
        0.35 * min(1.0, mean_imp * 1.8)
        + 0.30 * min(1.0, max_imp * 1.2)
        + 0.20 * min(1.0, hi_ratio * 2.0)
        + 0.15 * min(1.0, active / 12.0)
    )), 4)


def _human_scale_motion_score(cand: dict[str, Any], cfg: RoiSelectionConfig) -> float:
    """Localized persistent motion at person scale without requiring YOLO person."""
    mean_area = float(cand.get("mean_bbox_area_ratio", 1.0))
    duration = float(cand.get("duration_sec", 0.0))
    hi = float(cand.get("high_importance_blob_score", 0.0))
    net_disp = float(cand.get("net_displacement_ratio", 0.0))
    if mean_area > cfg.human_scale_max_area_ratio:
        return 0.0
    if duration < cfg.human_scale_min_duration_sec:
        return 0.0
    if hi < 0.25:
        return 0.0
    if net_disp > 0.92:
        return 0.0
    motion_signal = min(1.0, duration / 14.0) * hi
    if 0.04 <= net_disp <= 0.90:
        motion_signal = min(1.0, motion_signal * (1.0 + 0.25 * net_disp))
    return round(motion_signal, 4)


def compute_revised_event_score(
    cand: dict[str, Any],
    *,
    cfg: RoiSelectionConfig | None = None,
) -> dict[str, Any]:
    """Compute coverage-aware revised score components for one candidate."""
    sel_cfg = cfg or RoiSelectionConfig()
    base = float(cand.get("event_relevance_score", cand.get("event_score", 0.0)))
    duration_sec = float(cand.get("duration_sec", 0.0))
    bbox_len = max(int(cand.get("bbox_sequence_len", 1)), 1)
    active_frames = int(cand.get("num_blob_active_frames", 0))
    mean_area = float(cand.get("mean_bbox_area_ratio", 0.0))
    n_seq = bbox_len

    person_near = int(cand.get("person_near_frames", 0))
    vehicle_near = int(cand.get("vehicle_near_frames", 0))
    person_ratio = person_near / n_seq
    vehicle_ratio = vehicle_near / n_seq
    interaction_score = float(cand.get("interaction_score", 0.0))

    hi_blob = _high_importance_blob_score(cand)
    cand["high_importance_blob_score"] = hi_blob

    duration_score = round(min(1.0, duration_sec / 14.0), 4)
    active_ratio = active_frames / max(bbox_len, 1)
    persistence_score = round(min(1.0, (
        0.45 * min(1.0, active_frames / 14.0)
        + 0.35 * active_ratio
        + 0.20 * float(cand.get("localized_persistence_score", 0.0))
    )), 4)

    spatial_compactness_score = round(
        min(1.0, max(0.0, 1.0 - mean_area / 0.10)) if mean_area <= 0.10 else max(0.0, 0.35 - mean_area),
        4,
    )

    vehicle_only_penalty = 0.0
    if vehicle_ratio >= 0.25 and person_ratio < 0.06:
        vehicle_only_penalty = round(min(1.0, 0.45 + 0.35 * vehicle_ratio), 4)

    short_event_penalty = 0.0
    if duration_sec < sel_cfg.short_event_threshold_sec:
        short_event_penalty = round(min(1.0, 1.0 - duration_sec / sel_cfg.short_event_threshold_sec), 4)

    interaction_without_person_penalty = 0.0
    pattern_reason = str(cand.get("pattern_reason", ""))
    if (
        interaction_score >= 0.22
        and person_ratio < 0.06
        and float(cand.get("person_vehicle_dwell_ratio", 0.0)) < 0.06
        and "strong_interaction" in pattern_reason
    ):
        interaction_without_person_penalty = round(min(1.0, interaction_score * 0.75), 4)

    weak_blob_penalty = 0.0
    if hi_blob < 0.18 and duration_sec < sel_cfg.short_event_threshold_sec:
        weak_blob_penalty = 0.35

    human_scale_motion_score = _human_scale_motion_score(cand, sel_cfg)

    revised = round(max(0.0, (
        base
        + 0.70 * duration_score
        + 0.55 * persistence_score
        + 0.65 * hi_blob
        + 0.25 * spatial_compactness_score
        + 0.50 * human_scale_motion_score
        - 0.75 * vehicle_only_penalty
        - 0.85 * short_event_penalty
        - 0.65 * interaction_without_person_penalty
        - 0.40 * weak_blob_penalty
    )), 6)

    return {
        "original_score": round(base, 6),
        "base_event_score": round(base, 6),
        "duration_score": duration_score,
        "persistence_score": persistence_score,
        "high_importance_blob_score": hi_blob,
        "spatial_compactness_score": spatial_compactness_score,
        "human_scale_motion_score": human_scale_motion_score,
        "vehicle_only_penalty": vehicle_only_penalty,
        "short_event_penalty": short_event_penalty,
        "interaction_without_person_penalty": interaction_without_person_penalty,
        "weak_blob_penalty": weak_blob_penalty,
        "temporal_diversity_score": 0.0,
        "revised_event_score": revised,
    }


def annotate_revised_scores(
    candidates: list[dict[str, Any]],
    *,
    cfg: RoiSelectionConfig | None = None,
) -> None:
    sel_cfg = cfg or RoiSelectionConfig()
    for cand in candidates:
        cand.update(compute_revised_event_score(cand, cfg=sel_cfg))

    pool = list(candidates)
    by_orig = sorted(pool, key=lambda c: c.get("original_score", 0.0), reverse=True)
    for rank, cand in enumerate(by_orig, start=1):
        cand["selected_rank_before"] = rank

    by_rev = sorted(pool, key=lambda c: c.get("revised_event_score", 0.0), reverse=True)
    for rank, cand in enumerate(by_rev, start=1):
        cand["selected_rank_after"] = rank


def _temporal_iou(seq_a: list[dict], seq_b: list[dict]) -> float:
    fa = {int(e["frame_idx"]) for e in seq_a}
    fb = {int(e["frame_idx"]) for e in seq_b}
    if not fa or not fb:
        return 0.0
    return len(fa & fb) / len(fa | fb)


def _mean_spatial_iou(seq_a: list[dict], seq_b: list[dict]) -> float:
    fb_by_frame = {int(e["frame_idx"]): e["bbox"] for e in seq_b}
    ious: list[float] = []
    for e in seq_a:
        fi = int(e["frame_idx"])
        if fi in fb_by_frame:
            probe_a = {"x1": e["bbox"][0], "y1": e["bbox"][1], "x2": e["bbox"][2], "y2": e["bbox"][3]}
            probe_b = {"x1": fb_by_frame[fi][0], "y1": fb_by_frame[fi][1], "x2": fb_by_frame[fi][2], "y2": fb_by_frame[fi][3]}
            from motion_analyzer.v2.bbox_utils import bbox_iou
            ious.append(float(bbox_iou(probe_a, probe_b)))
    if not ious:
        mid_a = seq_a[len(seq_a) // 2]["bbox"]
        mid_b = seq_b[len(seq_b) // 2]["bbox"]
        probe_a = {"x1": mid_a[0], "y1": mid_a[1], "x2": mid_a[2], "y2": mid_a[3]}
        probe_b = {"x1": mid_b[0], "y1": mid_b[1], "x2": mid_b[2], "y2": mid_b[3]}
        from motion_analyzer.v2.bbox_utils import bbox_iou
        return float(bbox_iou(probe_a, probe_b))
    return float(np.mean(ious))


def _nms_conflict(
    cand: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    nms_temporal_iou: float,
    nms_spatial_iou: float,
) -> dict[str, Any] | None:
    seq = cand.get("bbox_sequence", [])
    for prev in selected:
        prev_seq = prev.get("bbox_sequence", [])
        t_iou = _temporal_iou(seq, prev_seq)
        s_iou = _mean_spatial_iou(seq, prev_seq)
        if t_iou >= nms_temporal_iou and s_iou >= nms_spatial_iou:
            return {
                "reason": "spatial_temporal_nms",
                "conflicts_with_group_id": prev.get("group_id"),
                "temporal_iou": round(t_iou, 4),
                "spatial_iou": round(s_iou, 4),
            }
        prev_aux = set(prev.get("auxiliary_track_ids", []))
        cand_aux = set(cand.get("auxiliary_track_ids", []))
        if prev_aux and cand_aux and prev_aux & cand_aux and t_iou >= 0.2:
            return {
                "reason": "shared_auxiliary_track",
                "conflicts_with_group_id": prev.get("group_id"),
                "shared_track_ids": sorted(prev_aux & cand_aux),
            }
    return None


def _is_long_persistent_event(cand: dict[str, Any], cfg: RoiSelectionConfig) -> bool:
    return (
        float(cand.get("duration_sec", 0.0)) >= cfg.long_event_min_duration_sec
        and float(cand.get("high_importance_blob_score", 0.0)) >= cfg.min_high_importance_for_long
        and int(cand.get("num_blob_active_frames", 0)) >= cfg.min_active_frames_for_long
        and float(cand.get("short_event_penalty", 0.0)) < 0.45
    )


def _spatial_overlap_action(
    cand: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    spatial_iou_min: float = 0.45,
    duration_ratio: float = 1.05,
) -> tuple[str, int | None]:
    """Decide whether to add, skip, or replace a spatially overlapping ROI."""
    for i, prev in enumerate(selected):
        s_iou = _mean_spatial_iou(cand.get("bbox_sequence", []), prev.get("bbox_sequence", []))
        if s_iou < spatial_iou_min:
            continue
        cand_dur = float(cand.get("duration_sec", 0.0))
        prev_dur = float(prev.get("duration_sec", 0.0))
        if cand_dur >= prev_dur * duration_ratio:
            return "replace", i
        return "skip", None
    return "add", None


def _apply_spatial_overlap_action(
    cand: dict[str, Any],
    selected: list[dict[str, Any]],
    selection_rejected: list[dict[str, Any]],
    *,
    selection_reason: str,
) -> bool:
    """Apply spatial overlap policy. Returns True if candidate was added/replaced."""
    action, replace_idx = _spatial_overlap_action(cand, selected)
    if action == "skip":
        return False
    if action == "replace" and replace_idx is not None:
        replaced = selected[replace_idx]
        replaced["selection_status"] = "replaced"
        replaced["rejection_reason"] = "replaced_by_longer_spatial_duplicate"
        replaced["replacement_reason"] = f"replaced_by_group_{cand.get('group_id')}"
        selection_rejected.append(_reject_record(
            replaced,
            "replaced_by_longer_spatial_duplicate",
            replaced.get("motion_pattern", ""),
            replacement_reason=replaced["replacement_reason"],
            replaced_by_group_id=cand.get("group_id"),
        ))
        selected[replace_idx] = cand
        cand["selection_status"] = "selected"
        cand["selection_reason"] = selection_reason
        cand["replacement_reason"] = f"replaced_group_{replaced.get('group_id')}"
        return True
    cand["selection_status"] = "selected"
    cand["selection_reason"] = selection_reason
    cand["replacement_reason"] = ""
    selected.append(cand)
    return True


def select_coverage_aware_rois(
    candidates: list[dict[str, Any]],
    *,
    max_rois: int = 2,
    eligible_patterns: set[str] | None = None,
    selection_cfg: RoiSelectionConfig | None = None,
    nms_temporal_iou: float = 0.40,
    nms_spatial_iou: float = 0.35,
    min_roi_frames: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Coverage-aware ROI selection with revised ranking and long-event priority.

    Returns (selected_candidates, selection_rejected_records, debug_rejected_high_rank).
    """
    sel_cfg = selection_cfg or RoiSelectionConfig()
    patterns = eligible_patterns or {"event_motion"}

    annotate_revised_scores(candidates, cfg=sel_cfg)

    pool: list[dict[str, Any]] = []
    selection_rejected: list[dict[str, Any]] = []

    for cand in candidates:
        pattern = str(cand.get("motion_pattern", "ambiguous_motion"))
        if pattern not in patterns:
            selection_rejected.append(_reject_record(cand, f"pattern_{pattern}", pattern))
            continue
        rej = cand.get("rejection_reason", "")
        if pattern == "event_motion" and rej == "low_event_relevance_score":
            selection_rejected.append(_reject_record(cand, "low_event_relevance_score", pattern))
            continue
        if pattern == "ambiguous_motion" and rej == "ambiguous_motion":
            selection_rejected.append(_reject_record(cand, "pattern_ambiguous_motion", pattern))
            continue
        if len(cand.get("bbox_sequence", [])) < min_roi_frames:
            selection_rejected.append(_reject_record(cand, "bbox_sequence_below_min_frames", pattern))
            continue
        pool.append(cand)

    by_revised = sorted(pool, key=lambda c: c.get("revised_event_score", 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    replaced_ids: set[int] = set()

    long_candidates = [c for c in by_revised if _is_long_persistent_event(c, sel_cfg)]
    for cand in long_candidates:
        if len(selected) >= max_rois:
            break
        conflict = _nms_conflict(
            cand, selected,
            nms_temporal_iou=nms_temporal_iou,
            nms_spatial_iou=nms_spatial_iou,
        )
        if conflict:
            continue
        if not _apply_spatial_overlap_action(
            cand, selected, selection_rejected, selection_reason="long_persistent_priority",
        ):
            continue

    for cand in by_revised:
        if len(selected) >= max_rois:
            break
        if cand in selected:
            continue
        conflict = _nms_conflict(
            cand, selected,
            nms_temporal_iou=nms_temporal_iou,
            nms_spatial_iou=nms_spatial_iou,
        )
        if conflict:
            continue
        if len(selected) == 1 and max_rois >= 2:
            t_iou = _temporal_iou(cand.get("bbox_sequence", []), selected[0].get("bbox_sequence", []))
            if (
                t_iou >= sel_cfg.diversity_temporal_iou_max
                and cand.get("revised_event_score", 0.0)
                < selected[0].get("revised_event_score", 0.0) * sel_cfg.diversity_score_margin
            ):
                cand["temporal_diversity_score"] = round(1.0 - t_iou, 4)
                continue
        if not _apply_spatial_overlap_action(
            cand, selected, selection_rejected, selection_reason="revised_score_rank",
        ):
            continue

    for i, sel in enumerate(list(selected)):
        if float(sel.get("duration_sec", 0.0)) >= sel_cfg.short_event_threshold_sec:
            continue
        if float(sel.get("short_event_penalty", 0.0)) < 0.25:
            continue
        for cand in by_revised:
            if cand in selected:
                continue
            if not _is_long_persistent_event(cand, sel_cfg):
                continue
            if float(cand.get("duration_sec", 0.0)) < float(sel.get("duration_sec", 0.0)) * sel_cfg.replace_duration_ratio:
                continue
            if float(cand.get("revised_event_score", 0.0)) < float(sel.get("revised_event_score", 0.0)) * sel_cfg.replace_min_revised_ratio:
                continue
            others = [s for j, s in enumerate(selected) if j != i]
            if _nms_conflict(
                cand, others,
                nms_temporal_iou=nms_temporal_iou,
                nms_spatial_iou=nms_spatial_iou,
            ):
                continue
            replaced = selected[i]
            replaced["selection_status"] = "replaced"
            replaced["rejection_reason"] = "replaced_by_longer_persistent_event"
            replaced["replacement_reason"] = f"replaced_by_group_{cand.get('group_id')}"
            selection_rejected.append(_reject_record(
                replaced,
                "replaced_by_longer_persistent_event",
                replaced.get("motion_pattern", ""),
                replacement_reason=replaced["replacement_reason"],
                replaced_by_group_id=cand.get("group_id"),
            ))
            replaced_ids.add(int(replaced.get("group_id", -1)))
            cand["selection_status"] = "selected"
            cand["selection_reason"] = "replaced_short_event"
            cand["replacement_reason"] = f"replaced_group_{replaced.get('group_id')}"
            selected[i] = cand
            break

    selected_ids = {int(c.get("group_id", -1)) for c in selected}
    for cand in by_revised:
        gid = int(cand.get("group_id", -1))
        if gid in selected_ids:
            continue
        if gid in replaced_ids:
            continue
        if cand.get("selection_status") == "replaced":
            continue
        cand["selection_status"] = "rejected"
        reason = "max_roi_tracks_reached"
        conflict = _nms_conflict(
            cand, selected,
            nms_temporal_iou=nms_temporal_iou,
            nms_spatial_iou=nms_spatial_iou,
        )
        if conflict and len(selected) >= max_rois:
            reason = conflict["reason"]
        cand["rejection_reason"] = reason
        cand["replacement_reason"] = ""
        selection_rejected.append(_reject_record(cand, reason, cand.get("motion_pattern", "")))

    debug_rejected = [
        c for c in by_revised
        if int(c.get("group_id", -1)) not in selected_ids
        and c.get("selected_rank_after", 999) <= sel_cfg.debug_rejected_rank_limit
    ]

    selected.sort(key=lambda c: c.get("start_sec", 0.0))
    return selected, selection_rejected, debug_rejected


def _reject_record(
    cand: dict[str, Any],
    reason: str,
    pattern: str,
    *,
    replacement_reason: str = "",
    replaced_by_group_id: int | None = None,
) -> dict[str, Any]:
    rec = {
        "group_id": cand.get("group_id"),
        "reason": reason,
        "motion_pattern": pattern,
        "rejection_reason": cand.get("rejection_reason") or reason,
        "replacement_reason": replacement_reason or cand.get("replacement_reason", ""),
        "original_score": cand.get("original_score"),
        "revised_event_score": cand.get("revised_event_score"),
        "duration_sec": cand.get("duration_sec"),
        "selected_rank_before": cand.get("selected_rank_before"),
        "selected_rank_after": cand.get("selected_rank_after"),
    }
    if replaced_by_group_id is not None:
        rec["replaced_by_group_id"] = replaced_by_group_id
    return rec

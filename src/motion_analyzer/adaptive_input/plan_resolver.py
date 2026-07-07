"""Resolve final VLM input selections from adaptive_input_plan.json."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from motion_analyzer.v1.input_plan_builder import MAX_ROI_TRACKS
from motion_analyzer.v1.input_plan_visualize import (
    canonicalize_plan_roi_tracks,
    plan_motion_type,
)

ANCHOR_FRAME_TYPES = frozenset(
    {
        "event_anchor",
        "tube_anchor_start",
        "tube_anchor_end",
        "tube_anchor",
    }
)


@dataclass
class SparseGlobalInput:
    frame_idx: int
    timestamp_sec: float


@dataclass
class RoiFrameInput:
    roi_id: int
    frame_idx: int
    timestamp_sec: float
    bbox: list[int]


@dataclass
class EventAnchorInput:
    frame_idx: int
    timestamp_sec: float
    label: str = "event_anchor"


@dataclass
class ResolvedInputPlan:
    motion_type: str
    sparse_global: list[SparseGlobalInput] = field(default_factory=list)
    roi_frames: list[RoiFrameInput] = field(default_factory=list)
    event_anchors: list[EventAnchorInput] = field(default_factory=list)

    @property
    def roi_track_count(self) -> int:
        return len({r.roi_id for r in self.roi_frames})

    @property
    def sparse_global_frame_count(self) -> int:
        return len(self.sparse_global)

    @property
    def roi_frame_count(self) -> int:
        return len(self.roi_frames)

    @property
    def event_anchor_count(self) -> int:
        return len(self.event_anchors)

    @property
    def total_vlm_input_count(self) -> int:
        return (
            self.sparse_global_frame_count
            + self.roi_frame_count
            + self.event_anchor_count
        )


def _collect_sparse_global(plan: dict[str, Any]) -> list[SparseGlobalInput]:
    seen: set[int] = set()
    entries: list[SparseGlobalInput] = []

    for block in plan.get("global_inputs", []):
        if block.get("type") != "sparse_global":
            continue
        indices = block.get("frame_indices", [])
        timestamps = block.get("timestamps_sec", [])
        for i, frame_idx in enumerate(indices):
            fi = int(frame_idx)
            if fi in seen:
                continue
            seen.add(fi)
            ts = float(timestamps[i]) if i < len(timestamps) else 0.0
            entries.append(SparseGlobalInput(frame_idx=fi, timestamp_sec=ts))

    for frame in plan.get("frames", []):
        ftype = str(frame.get("type", ""))
        source = str(frame.get("source", ""))
        if ftype != "sparse_global" and source != "sparse_global":
            continue
        fi = int(frame["frame_idx"])
        if fi in seen:
            continue
        seen.add(fi)
        entries.append(
            SparseGlobalInput(
                frame_idx=fi,
                timestamp_sec=float(frame["timestamp_sec"]),
            )
        )

    entries.sort(key=lambda e: e.frame_idx)
    return entries


def _collect_event_anchors(plan: dict[str, Any]) -> list[EventAnchorInput]:
    seen: set[int] = set()
    entries: list[EventAnchorInput] = []

    for anchor in plan.get("event_anchors", []):
        fi = int(anchor["frame_idx"])
        if fi in seen:
            continue
        seen.add(fi)
        entries.append(
            EventAnchorInput(
                frame_idx=fi,
                timestamp_sec=float(anchor["timestamp_sec"]),
                label=str(anchor.get("type", "event_anchor")),
            )
        )

    for frame in plan.get("frames", []):
        ftype = str(frame.get("type", ""))
        if ftype not in ANCHOR_FRAME_TYPES and "anchor" not in ftype:
            continue
        fi = int(frame["frame_idx"])
        if fi in seen:
            continue
        seen.add(fi)
        entries.append(
            EventAnchorInput(
                frame_idx=fi,
                timestamp_sec=float(frame["timestamp_sec"]),
                label=ftype,
            )
        )

    entries.sort(key=lambda e: e.frame_idx)
    return entries


def _collect_roi_frames(
    plan: dict[str, Any],
    *,
    max_tracks: int = MAX_ROI_TRACKS,
) -> list[RoiFrameInput]:
    roi_inputs = canonicalize_plan_roi_tracks(plan)[:max_tracks]
    frames: list[RoiFrameInput] = []
    for roi in roi_inputs:
        roi_id = int(roi.get("roi_id", 0))
        for entry in roi.get("bbox_sequence", []):
            bbox = entry.get("bbox")
            if not bbox:
                continue
            frames.append(
                RoiFrameInput(
                    roi_id=roi_id,
                    frame_idx=int(entry["frame_idx"]),
                    timestamp_sec=float(entry["timestamp_sec"]),
                    bbox=[int(v) for v in bbox],
                )
            )
    frames.sort(key=lambda f: (f.timestamp_sec, f.roi_id, f.frame_idx))
    return frames


def resolve_input_plan(plan: dict[str, Any]) -> ResolvedInputPlan:
    """
    Convert adaptive_input_plan.json into final VLM input selections.

    background_motion → sparse global only
    event_motion      → sparse global + up to 2 ROI tracks + event anchors
    """
    motion_type = plan_motion_type(plan)
    sparse_global = _collect_sparse_global(plan)

    roi_frames: list[RoiFrameInput] = []
    event_anchors: list[EventAnchorInput] = []
    if motion_type == "event_motion":
        roi_frames = _collect_roi_frames(plan)
        event_anchors = _collect_event_anchors(plan)

    return ResolvedInputPlan(
        motion_type=motion_type,
        sparse_global=sparse_global,
        roi_frames=roi_frames,
        event_anchors=event_anchors,
    )

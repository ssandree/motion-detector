"""Which Stage3 ROI tubes actually contain a GT event participant.

BM asks whether a video has motion that is irrelevant to the semantic events yet still
drives the ROI allocation. For VIRAT that is checkable rather than a matter of opinion:
the GT names the tracks that take part in each event, and the object annotation gives
those tracks a box in every frame. An ROI tube that no participant track ever enters is
being driven by something the GT does not care about.

Stage3 tube times are indices into the 5 fps sampled sequence, while the GT tracks are
indexed by original video frame, so the two are bridged through seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gt_context import GtContext, track_boxes, track_categories
from .media import native_fps
from .selection import TubeInfo, VideoTarget

# A participant counts as present in a tube when its box centre is inside the tube box.
# Centres are robust to the tube being a coarse 64 px grid AABB.


@dataclass
class RoiGtEvent:
    """A GT event placed inside one ROI tube, in that tube's own time window."""

    event_id: str
    action: str
    subject: str
    overlap_t0_sec: float
    overlap_t1_sec: float

    def describe(self) -> str:
        return (
            f"[{self.event_id}] {self.subject} {self.action.replace('_', ' ')} "
            f"t={self.overlap_t0_sec:.1f}-{self.overlap_t1_sec:.1f}s"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "action": self.action,
            "subject": self.subject,
            "overlap_t0_sec": round(self.overlap_t0_sec, 2),
            "overlap_t1_sec": round(self.overlap_t1_sec, 2),
        }


@dataclass
class RoiParticipantLink:
    roi_id: str
    overlay_label: str
    t0_sec: float
    t1_sec: float
    participant_tracks: list[str] = field(default_factory=list)
    participant_summary: list[str] = field(default_factory=list)
    other_tracks: list[str] = field(default_factory=list)
    n_participant_samples: int = 0
    gt_events: list[RoiGtEvent] = field(default_factory=list)

    @property
    def has_participant(self) -> bool:
        return bool(self.participant_tracks)

    @property
    def action_types(self) -> set[str]:
        return {event.action for event in self.gt_events}

    def as_dict(self) -> dict[str, Any]:
        return {
            "roi_id": self.roi_id,
            "t0_sec": round(self.t0_sec, 2),
            "t1_sec": round(self.t1_sec, 2),
            "has_gt_participant": self.has_participant,
            "participant_tracks": self.participant_tracks,
            "participant_summary": self.participant_summary,
            "non_participant_tracks_present": self.other_tracks,
            "n_participant_samples": self.n_participant_samples,
            "gt_events": [event.as_dict() for event in self.gt_events],
            "gt_action_types": sorted(self.action_types),
        }


@dataclass
class GtRoiLinkage:
    available: bool
    rois: list[RoiParticipantLink] = field(default_factory=list)
    note: str | None = None

    @property
    def rois_without_participant(self) -> list[RoiParticipantLink]:
        return [roi for roi in self.rois if not roi.has_participant]

    @property
    def has_event_placement(self) -> bool:
        """Whether any GT event could be placed in a tube, so IE/DE can use the GT."""
        return any(roi.gt_events for roi in self.rois)

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "note": self.note,
            "n_rois_with_gt_participant": sum(1 for roi in self.rois if roi.has_participant),
            "n_rois_without_gt_participant": len(self.rois_without_participant),
            "n_gt_events_placed": sum(len(roi.gt_events) for roi in self.rois),
            "rois": [roi.as_dict() for roi in self.rois],
        }


def _centre_inside(box: tuple[int, int, int, int], x: int, y: int, w: int, h: int) -> bool:
    cx, cy = x + w / 2.0, y + h / 2.0
    x0, y0, x1, y1 = box
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _events_in_tube(
    tube: TubeInfo,
    *,
    events: list[Any],
    boxes: dict[str, list[tuple[int, int, int, int, int]]],
    fps: float,
) -> list[RoiGtEvent]:
    """GT events that happen inside this tube, in both time and space.

    An event counts only when its own window overlaps the tube's lifetime and one of
    its participant tracks is actually inside the tube box during that overlap, so an
    event taking place elsewhere in the frame is not credited to this ROI.
    """
    placed: list[RoiGtEvent] = []
    for event in events:
        if not event.has_time:
            continue
        lo = max(tube.t0_sec, float(event.start_sec))
        hi = min(tube.t1_sec, float(event.end_sec))
        if lo > hi:
            continue
        track_ids = [event.subject_track_id, *event.object_track_ids]
        start_frame, end_frame = int(lo * fps), int(hi * fps)
        inside = False
        for track_id in track_ids:
            for frame, x, y, w, h in boxes.get(str(track_id), ()):
                if start_frame <= frame <= end_frame and _centre_inside(
                    tube.pixel_bbox, x, y, w, h
                ):
                    inside = True
                    break
            if inside:
                break
        if inside:
            placed.append(
                RoiGtEvent(
                    event_id=event.event_id,
                    action=event.action,
                    subject=event.subject,
                    overlap_t0_sec=lo,
                    overlap_t1_sec=hi,
                )
            )
    return placed


def _link_one(
    tube: TubeInfo,
    *,
    boxes: dict[str, list[tuple[int, int, int, int, int]]],
    categories: dict[str, str],
    participant_ids: set[str],
    roles: dict[str, list[str]],
    fps: float,
    events: list[Any],
) -> RoiParticipantLink:
    start_frame = int(tube.t0_sec * fps)
    end_frame = int(tube.t1_sec * fps)
    inside_participants: dict[str, int] = {}
    inside_others: set[str] = set()

    for track_id, rows in boxes.items():
        hits = 0
        for frame, x, y, w, h in rows:
            if frame < start_frame or frame > end_frame:
                continue
            if _centre_inside(tube.pixel_bbox, x, y, w, h):
                hits += 1
                if track_id not in participant_ids:
                    break
        if not hits:
            continue
        if track_id in participant_ids:
            inside_participants[track_id] = hits
        else:
            inside_others.add(track_id)

    summary = [
        f"track {track_id} ({categories.get(track_id, 'unknown')}, "
        f"{', '.join(sorted(set(roles.get(track_id, []))) ) or 'participant'})"
        for track_id in sorted(inside_participants, key=int)
    ]
    return RoiParticipantLink(
        roi_id=tube.roi_id,
        overlay_label=tube.overlay_label,
        t0_sec=tube.t0_sec,
        t1_sec=tube.t1_sec,
        participant_tracks=sorted(inside_participants, key=int),
        participant_summary=summary,
        other_tracks=sorted(inside_others, key=int),
        n_participant_samples=sum(inside_participants.values()),
        gt_events=_events_in_tube(tube, events=events, boxes=boxes, fps=fps),
    )


def link_rois_to_gt(target: VideoTarget, gt: GtContext) -> GtRoiLinkage:
    """Per-ROI GT participant presence, or an explicit note when it cannot be computed."""
    if target.dataset != "VIRAT" or not gt.available:
        return GtRoiLinkage(
            available=False,
            note="No GT event annotation for this video, so ROI/participant linkage is unavailable.",
        )

    boxes = track_boxes(target.video_id)
    if not boxes:
        return GtRoiLinkage(
            available=False,
            note="GT events exist but viratdata.objects.txt has no usable track boxes.",
        )

    participants = gt.participants
    participant_ids = {p.track_id for p in participants}
    roles = {p.track_id: p.roles for p in participants}
    categories = {**track_categories(target.video_id)}
    for participant in participants:
        categories.setdefault(participant.track_id, participant.category)

    fps = native_fps(str(target.raw_video)) if target.raw_video else 30.0
    rois = [
        _link_one(
            tube,
            boxes=boxes,
            categories=categories,
            participant_ids=participant_ids,
            roles=roles,
            fps=fps,
            events=gt.events,
        )
        for tube in target.tubes
    ]
    return GtRoiLinkage(available=True, rois=rois)

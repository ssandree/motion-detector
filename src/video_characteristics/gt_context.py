"""Existing GT event annotation, read only, turned into prompt context.

VIRAT has caption-event GT for the 191 selected videos (``caption_event_gt_191``),
so the AD/OD prompt can be grounded on real event records. MEVA drop-4-hadcv22 is
not covered by the kitware MEVA annotation release, so MEVA videos get no GT block
and the prompt says so explicitly rather than inventing events.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .datasets import EVAL_ROOT, DatasetSpec

VIRAT_ANNOTATION_DIR = Path("/data/datasets/VIRAT/annotations")
GT_191_ROOT = EVAL_ROOT / "experiments" / "virat" / "caption_event_gt_191"
REVIEWED_GT_POINTER = GT_191_ROOT / "CURRENT_REVIEWED_GT.json"

# Actions whose name only makes sense once you have ruled out a visually similar
# opposite: enter/exit, in/out, open/close, load/unload, pick up/put down. Naming one
# of these correctly already requires the fine distinction AD is about, so a single
# occurrence counts. This is a property of the GT vocabulary, not a judgement.
POLARITY_GROUPS = {
    "entering_facility": "facility_enter_exit",
    "exiting_facility": "facility_enter_exit",
    "facility_transition": "facility_enter_exit",
    "getting_into_vehicle": "vehicle_in_out",
    "getting_out_of_vehicle": "vehicle_in_out",
    "embarking_vehicle": "vehicle_in_out",
    "disembarking_vehicle": "vehicle_in_out",
    "opening_vehicle_trunk": "trunk_open_close",
    "closing_vehicle_trunk": "trunk_open_close",
    "opening_vehicle_door": "door_open_close",
    "closing_vehicle_door": "door_open_close",
    "loading_object_to_vehicle": "object_load_unload",
    "unloading_object_from_vehicle": "object_load_unload",
    "retrieving_object": "object_pick_put",
    "putting_down_object": "object_pick_put",
    "posture_change": "posture_up_down",
    "ascending": "ascend_descend",
    "descending": "ascend_descend",
}

# Actions whose evidence is a small carried item rather than the person.
SMALL_OBJECT_ACTIONS = {
    "carrying_object",
    "loading_object_to_vehicle",
    "unloading_object_from_vehicle",
}

# A subject box below this fraction of the frame is reported as small.
SMALL_BOX_AREA_RATIO = 0.004


@dataclass
class GtParticipant:
    """A track the GT names as taking part in at least one event.

    BM turns on separating these from everything else that moves, so the track ids
    matter: motion produced by a participant track is never background motion.
    """

    track_id: str
    category: str
    roles: list[str]
    event_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "category": self.category,
            "roles": sorted(set(self.roles)),
            "event_ids": sorted(set(self.event_ids)),
        }


@dataclass
class GtEvent:
    event_id: str
    action: str
    # The human-reviewed GT leaves the time open for events it could not localise
    # (``temporal_localization: not_provided``), so these stay None rather than 0.0.
    start_sec: float | None
    end_sec: float | None
    start_frame: int | None
    end_frame: int | None
    event_form: str
    subject: str
    objects: list[str]
    screen_location: str | None
    polarity_group: str | None
    reference_sentence: str | None = None
    subject_box: dict[str, Any] | None = None
    subject_track_id: str | None = None
    # track_id -> annotated category, so a participating car is not reported as "object".
    object_categories: dict[str, str] = field(default_factory=dict)

    @property
    def object_track_ids(self) -> list[str]:
        return sorted(self.object_categories, key=lambda value: int(value))

    @property
    def has_time(self) -> bool:
        return self.start_sec is not None and self.end_sec is not None

    @property
    def track_ids(self) -> list[str]:
        ids = [self.subject_track_id, *self.object_track_ids]
        return [str(i) for i in ids if i]

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "action": self.action,
            "start_sec": None if self.start_sec is None else round(self.start_sec, 2),
            "end_sec": None if self.end_sec is None else round(self.end_sec, 2),
            "event_form": self.event_form,
            "subject": self.subject,
            "objects": self.objects,
            "screen_location": self.screen_location,
            "polarity_group": self.polarity_group,
            "reference_sentence": self.reference_sentence,
            "subject_box": self.subject_box,
        }


@dataclass
class GtContext:
    available: bool
    source: str | None = None
    events: list[GtEvent] = field(default_factory=list)
    note: str | None = None

    @property
    def polarity_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for event in self.events:
            if event.polarity_group:
                groups.setdefault(event.polarity_group, []).append(event.action)
        return groups

    @property
    def small_object_actions(self) -> list[str]:
        return sorted({e.action for e in self.events if e.action in SMALL_OBJECT_ACTIONS})

    @property
    def participants(self) -> list[GtParticipant]:
        """Every track the GT events name, with the roles it plays."""
        merged: dict[str, GtParticipant] = {}
        for event in self.events:
            pairs = [(event.subject_track_id, event.subject)]
            pairs += list(event.object_categories.items())
            for track_id, category in pairs:
                if not track_id:
                    continue
                key = str(track_id)
                if key not in merged:
                    merged[key] = GtParticipant(
                        track_id=key, category=str(category), roles=[], event_ids=[]
                    )
                merged[key].roles.append(event.action)
                merged[key].event_ids.append(event.event_id)
        return [merged[key] for key in sorted(merged, key=lambda value: int(value))]

    @property
    def small_subject_events(self) -> list[GtEvent]:
        out = []
        for event in self.events:
            box = event.subject_box
            if box and box.get("area_ratio") is not None:
                if float(box["area_ratio"]) < SMALL_BOX_AREA_RATIO:
                    out.append(event)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "source": self.source,
            "note": self.note,
            "n_events": len(self.events),
            "polarity_groups": self.polarity_groups,
            "small_object_actions": self.small_object_actions,
            "n_small_subject_events": len(self.small_subject_events),
            "participants": [p.as_dict() for p in self.participants],
            "events": [event.as_dict() for event in self.events],
        }


@lru_cache(maxsize=1)
def _reviewed_gt_root() -> Path | None:
    """Per-video GT root of the human-reviewed version, if the pointer exists."""
    if not REVIEWED_GT_POINTER.is_file():
        return None
    import json

    pointer = json.loads(REVIEWED_GT_POINTER.read_text(encoding="utf-8"))
    root = pointer.get("per_video_gt_root")
    if not root:
        return None
    path = Path(root)
    return path if path.is_dir() else None


# viratdata.objects.txt object_type codes.
VIRAT_OBJECT_TYPES = {
    1: "person",
    2: "car",
    3: "vehicle",
    4: "object",
    5: "bike",
}


@lru_cache(maxsize=512)
def track_boxes(video_id: str) -> dict[str, list[tuple[int, int, int, int, int]]]:
    """``viratdata.objects.txt`` boxes per track as ``(frame, x, y, w, h)``."""
    path = VIRAT_ANNOTATION_DIR / f"{video_id}.viratdata.objects.txt"
    if not path.is_file():
        return {}
    boxes: dict[str, list[tuple[int, int, int, int, int]]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            track_id = str(int(parts[0]))
            frame = int(parts[2])
            x = int(parts[3])
            y = int(parts[4])
            width = int(parts[5])
            height = int(parts[6])
        except ValueError:
            continue
        boxes.setdefault(track_id, []).append((frame, x, y, width, height))
    for rows in boxes.values():
        rows.sort()
    return boxes


@lru_cache(maxsize=512)
def track_categories(video_id: str) -> dict[str, str]:
    """Object category per track, from the ``object_type`` column."""
    path = VIRAT_ANNOTATION_DIR / f"{video_id}.viratdata.objects.txt"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            track_id = str(int(parts[0]))
            type_id = int(parts[7])
        except ValueError:
            continue
        out.setdefault(track_id, VIRAT_OBJECT_TYPES.get(type_id, "unknown"))
    return out


def _subject_box_stats(
    video_id: str,
    track_id: str | None,
    start_frame: int,
    end_frame: int,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any] | None:
    """Median subject box over the event window, as a fraction of the frame.

    Comes straight from the VIRAT object tracks. Returns ``None`` when the track
    has no boxes in the window, rather than guessing a size.
    """
    if not track_id or frame_width <= 0 or frame_height <= 0:
        return None
    rows = track_boxes(video_id).get(str(track_id))
    if not rows:
        return None
    inside = [(w, h) for frame, _x, _y, w, h in rows if start_frame <= frame <= end_frame]
    if not inside:
        inside = [(w, h) for _f, _x, _y, w, h in rows]
    if not inside:
        return None
    med_w = int(statistics.median(w for w, _ in inside))
    med_h = int(statistics.median(h for _, h in inside))
    frame_area = float(frame_width * frame_height)
    area_ratio = (med_w * med_h) / frame_area if frame_area > 0 else None
    return {
        "track_id": str(track_id),
        "median_width_px": med_w,
        "median_height_px": med_h,
        "area_ratio": None if area_ratio is None else round(area_ratio, 6),
        "n_boxes_in_window": len(inside),
    }


def _load_virat(video_id: str, frame_width: int, frame_height: int) -> GtContext:
    import json

    candidates: list[tuple[str, Path]] = []
    reviewed = _reviewed_gt_root()
    if reviewed is not None:
        candidates.append(("human_review_v2", reviewed / video_id / "caption_events.json"))
    candidates.append(("caption_event_gt_191", GT_191_ROOT / "gt" / video_id / "caption_events.json"))

    for source, path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_events = payload.get("caption_events") or []
        events: list[GtEvent] = []
        for raw in raw_events:
            subject = raw.get("subject") or {}
            track_id = subject.get("track_id")
            start_frame = raw.get("start_frame")
            end_frame = raw.get("end_frame")
            start_sec = raw.get("start_time")
            end_sec = raw.get("end_time")
            action = str(raw.get("action") or "")
            subject_label = str(subject.get("category") or "unknown")
            if subject.get("group"):
                subject_label = f"group of {subject_label}"
            box = None
            if start_frame is not None and end_frame is not None:
                box = _subject_box_stats(
                    video_id,
                    track_id,
                    int(start_frame),
                    int(end_frame),
                    frame_width,
                    frame_height,
                )
            events.append(
                GtEvent(
                    event_id=str(raw.get("caption_event_id") or ""),
                    action=action,
                    start_sec=None if start_sec is None else float(start_sec),
                    end_sec=None if end_sec is None else float(end_sec),
                    start_frame=None if start_frame is None else int(start_frame),
                    end_frame=None if end_frame is None else int(end_frame),
                    event_form=str(raw.get("event_form") or ""),
                    subject=subject_label,
                    objects=sorted(
                        {
                            str(obj.get("category") or "object")
                            for obj in (raw.get("object") or [])
                            if isinstance(obj, dict)
                        }
                    ),
                    screen_location=raw.get("semantic_location"),
                    polarity_group=POLARITY_GROUPS.get(action),
                    reference_sentence=(
                        " ".join(str(raw.get("reference_sentence")).split())
                        if raw.get("reference_sentence")
                        else None
                    ),
                    subject_box=box,
                    subject_track_id=None if track_id is None else str(track_id),
                    object_categories={
                        str(obj["track_id"]): str(obj.get("category") or "object")
                        for obj in (raw.get("object") or [])
                        if isinstance(obj, dict) and obj.get("track_id")
                    },
                )
            )
        events.sort(key=lambda e: (e.start_sec is None, e.start_sec or 0.0, e.event_id))
        return GtContext(
            available=bool(events),
            source=f"{source}:{path}",
            events=events,
            note=None if events else "GT file exists but lists zero caption events.",
        )

    return GtContext(
        available=False,
        source=None,
        note="No VIRAT caption-event GT file for this video.",
    )


def load_gt(spec: DatasetSpec, video_id: str, frame_width: int, frame_height: int) -> GtContext:
    if spec.name == "VIRAT":
        return _load_virat(video_id, frame_width, frame_height)
    return GtContext(
        available=False,
        source=None,
        note=(
            "MEVA drop-4-hadcv22 has no released activity annotation, so no GT events "
            "are supplied for this video."
        ),
    )

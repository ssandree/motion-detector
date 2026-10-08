"""Prompts for the two annotation passes, plus the context blocks they embed.

Pass A judges AD and OD from the original video, because those two are properties
of the semantic content. Pass B judges BM, IE and DE from the Stage3 overlay MP4
plus a table of the real ROI tubes, because those three are claims about the ROI
allocation and must not be guessed from the raw video.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .gt_context import GtContext
    from .gt_roi_link import GtRoiLinkage
    from .selection import VideoTarget

PASS_A_LABELS = ("AD", "OD")
PASS_B_LABELS = ("BM", "IE", "DE")

COMMON_RULES = """Rules for every label:
- Fill in the observation fields first, then decide the labels from what you wrote.
- Judge each label independently. Several labels may be true for the same video.
- A label is true only when you can point at concrete visible evidence.
- `evidence` is a list of short English strings describing what you actually saw.
  Write times in seconds from the start of the clip, like `t=182.0-195.5s`. Never use
  mm:ss. Leave the list empty when the label is false and there is nothing to report.
- A true label with an empty evidence list is not acceptable. Either give the evidence
  or set the label false.
- Every label also needs a `certainty`, which is exactly one of these three words,
  describing how well you could see the thing that label asks about:
  - "high": you clearly saw it, or clearly saw that it is absent.
  - "medium": you are fairly sure, but the view is partly small, brief or occluded.
  - "low": you are guessing. Use "low" whenever a label is false only because you could
    not make out enough to tell, rather than because you saw that it is absent, and
    whenever the frames are too coarse or too far apart to judge.
  Labels you could see well and labels you could barely see must not get the same word.
- `confidence` is a number inside the band of the word you chose: 0.85-1.00 for "high",
  0.70-0.85 for "medium", 0.30-0.70 for "low".
- Never invent events, object identities, or times that you cannot see.
- Output one JSON object and nothing else. No prose, no markdown fence."""

PASS_A_PROMPT = """You are annotating a fixed-camera surveillance video for two
difficulty characteristics of its semantic content: AD and OD.

AD (action_detail) is true when understanding at least one event in this video requires
telling apart actions that look visually similar but are opposite in direction or state,
or that differ only by a small motion. Typical pairs: entering versus exiting a facility,
getting into versus getting out of a vehicle, ascending versus descending, picking up
versus putting down, opening versus closing a door or trunk, loading versus unloading an
object.
One such action is enough. If a person gets out of a car, naming that correctly already
means ruling out getting in, so AD is true even though the opposite never happens in the
clip. You do not need to see both halves of a pair.
AD is false when the events only amount to generic presence or travel, such as a person
walking through the scene, standing, talking, or a vehicle driving past, with no direction
or state that could be read backwards. The presence of an action is not by itself a reason
to answer true.

OD (object_detail) is true when resolving a small or distant person or object, or a small
carried item, is necessary to understand what the event is. Ask yourself: at the size this
target appears, could you still say what it is and what it is doing? If you had to strain,
or you could only tell from context, OD is true.
A person who occupies well under 1% of the frame while carrying the event is a typical
positive, and so is an event that turns on a small carried item such as a bag or box.
OD is false when small figures merely appear somewhere in the frame while the event can
still be understood without resolving them, and false when the actors that carry the event
are large and clearly visible. Being far away is not by itself a reason to answer true;
the identification has to matter for the event.

{rules}

Answer in exactly this shape, and keep this key order:
{{
  "observed_events": ["one short sentence per event you actually saw, with its time in seconds"],
  "AD": {{"label": false, "certainty": "high", "confidence": 0.0, "evidence": []}},
  "OD": {{"label": false, "certainty": "high", "confidence": 0.0, "evidence": []}}
}}

If `observed_events` ends up empty because you could not make out any event, say so by
giving AD and OD a certainty of "low".
"""

PASS_B_PROMPT = """You are annotating a motion-guided ROI result for three
characteristics: BM, IE and DE.

The video you see is the original footage with two things drawn on top:
- Smeared colour is a motion-magnitude heatmap. It is not an object or an event.
- Coloured rectangles labelled E1, E2, E3 are the proposed ROI tubes. The table
  below lists the same tubes with their real time ranges and pixel boxes. Refer to
  them by the `roi_id` from the table.

BM (background_motion) is true when the video contains independent or distractor motion
that is not needed to understand the main semantic events, yet could still drive the
motion-based ROI allocation.
What decides BM is semantic relevance, not what kind of thing is moving. Any of these can
be background motion: trees and vegetation, shadows and reflections, flags, rippling
water, rain, camera jitter, vehicle traffic that takes no part in the events, and
pedestrian traffic that takes no part in the events. A moving person or car is background
motion whenever that person or car is irrelevant to the main events. Equally, none of
these is background motion when it is part of a main event, so do not mark a car or a
person just because it moves.
The target events are listed below. Motion produced by a participant in those events, or
motion that is part of one of them, is never BM.
These frames are sampled about one per second, so slow or periodic motion may not look
like motion to you. The heatmap is the reliable cue, because it accumulates motion over
time: heatmap colour or an ROI box sitting on a region that no event participant ever
occupies is irrelevant motion being scored.
BM is false when all the scored motion belongs to the main events or their participants.
Each BM evidence string should state the motion source, a timestamp in seconds, the
`roi_id` it drives when you can tell, and why that motion is unrelated to the main
events.

IE (integrated_event) is true when a single ROI tube contains two or more different
semantic events inside its own box and time range, for example one ROI that holds
both a person walking and a separate vehicle interaction. Two moments of the same
continuous event do not count, and neither do two events that fall in different
ROIs. IE evidence must name the one `roi_id` that holds both events, and give a time
range for each of them.

DE (different_event) is true when two or more different ROI tubes each carry their own
semantic event or their own evidence, for example one ROI on a vehicle interaction
while another holds a distant pedestrian.
An ROI only counts toward DE if you can see an actual event in it involving a person or
a vehicle. An ROI holding nothing but empty ground, pavement, grass, a parked and
motionless object, a shadow, or heatmap colour does not count, even though a box is
drawn there. Two ROIs showing two parts of the same event do not count either.
DE evidence must name at least two different `roi_id` values and say what event each
one carries. If the table lists only one tube, DE is false.

{rules}

Start by filling in `roi_observations`: one entry for every `roi_id` in the table above,
in the table's order. Each row is a different region over a different stretch of time.
Take them one at a time: look only at that part of the frame, only between those
seconds, and write down what is actually there. Never leave `objects` or `actions`
blank. If a box holds nothing of interest, say so in words, for example
`objects: "empty pavement, no person or vehicle"` and `actions: "nothing happens here"`.
A blank string is not an acceptable answer.

Do not describe the video as a whole and repeat it for every ROI. If your entry for one
ROI would read the same as another, you have not looked at them separately; go back and
say what makes each box different, or state plainly that the box is empty. Most ROIs in
a video hold either nothing or one small part of the action, so differing and often
empty descriptions are the expected answer.

`distinct_events` must list only events happening inside that one box during its own
time range, and an event belongs to the ROI whose box actually contains it. Set
`has_semantic_event` to true only when a person or vehicle does something inside that
box. The BM, IE and DE answers must follow from what you wrote in `roi_observations`.

Answer with JSON in exactly this shape and key order, replacing every `<...>` with your
own text. The angle brackets are placeholders, not literal output:
{{
  "roi_observations": [
    {{"roi_id": "ROI_1", "objects": "<what is inside this box>", "actions": "<what it does, over which seconds>", "has_semantic_event": <true|false>, "distinct_events": ["<event 1>", "<event 2>"]}}
  ],
  "BM": {{"label": <true|false>, "certainty": "<high|medium|low>", "confidence": <0.0-1.0>, "evidence": ["<motion source, t=..s, roi_id, why it is unrelated to the main events>"]}},
  "IE": {{"label": <true|false>, "certainty": "<high|medium|low>", "confidence": <0.0-1.0>, "evidence": ["<the one roi_id, and each event in it with its time range>"]}},
  "DE": {{"label": <true|false>, "certainty": "<high|medium|low>", "confidence": <0.0-1.0>, "evidence": ["<roi_id and its event>", "<another roi_id and its different event>"]}}
}}
"""


def _fmt_sec(value: float) -> str:
    return f"{value:.1f}"


def gt_block(gt: GtContext) -> str:
    """Ground-truth event records for the AD/OD pass, or an explicit absence note."""
    if not gt.available:
        note = gt.note or "No ground-truth event annotation is available for this video."
        return (
            "Ground-truth events: none available.\n"
            f"{note}\n"
            "Judge AD and OD from the video alone. Do not assume events that you cannot see."
        )

    lines = [
        f"Ground-truth events ({len(gt.events)} records from the dataset annotation, "
        "these are facts, not guesses):",
    ]
    for event in gt.events:
        when = (
            f"t={_fmt_sec(event.start_sec)}-{_fmt_sec(event.end_sec)}s"
            if event.has_time
            else "time not annotated"
        )
        parts = [
            f"- [{event.event_id}] {when}",
            f"{event.subject} {event.action.replace('_', ' ')}",
        ]
        if event.objects:
            parts.append(f"(objects: {', '.join(event.objects)})")
        box = event.subject_box
        if box and box.get("area_ratio") is not None:
            parts.append(
                f"[subject box ~{box['median_width_px']}x{box['median_height_px']} px, "
                f"{float(box['area_ratio']) * 100:.3f}% of frame]"
            )
        if event.reference_sentence:
            parts.append(f'note: "{event.reference_sentence}"')
        lines.append(" ".join(parts))

    groups = gt.polarity_groups
    if groups:
        rendered = "; ".join(
            f"{name}: {', '.join(sorted(set(actions)))}" for name, actions in sorted(groups.items())
        )
        lines.append(
            "Direction/state-sensitive actions present in the GT: "
            + rendered
            + ". Each of these is named in a way that only holds if the opposite is ruled "
            "out, so one occurrence is already a fine distinction. Both members of a pair "
            "do not have to appear."
        )
    else:
        lines.append(
            "No direction/state-sensitive action appears in the GT vocabulary for this video."
        )

    small = gt.small_subject_events
    if small:
        lines.append(
            "GT events whose subject is smaller than 0.4% of the frame, measured from the "
            "annotated track boxes: "
            + ", ".join(
                f"{e.event_id} ({e.action}, "
                f"{float(e.subject_box['area_ratio']) * 100:.3f}%)"
                for e in small
            )
            + ". Check in the video whether you could tell what such a target is doing."
        )
    if gt.small_object_actions:
        lines.append(
            "GT actions that hinge on a carried item: " + ", ".join(gt.small_object_actions) + "."
        )
    lines.append(
        "The GT tells you which events exist and when. You still decide whether telling those "
        "events apart needs fine action detail (AD) or needs resolving a small/distant target "
        "(OD). Use the GT times in your evidence, in seconds."
    )
    return "\n".join(lines)


def relevance_block(
    target: VideoTarget,
    gt: GtContext,
    linkage: GtRoiLinkage,
    observed_events: list[str],
) -> str:
    """What counts as a main event for this video, so BM can be judged on relevance.

    Prefers the GT events and participant tracks. Falls back to what pass A reported
    when the dataset has no GT, and says which of the two is in force.
    """
    if gt.available:
        lines = [
            "Main semantic events for this video, from the dataset ground truth. "
            "Treat these as the target events:",
        ]
        for event in gt.events:
            when = (
                f"t={_fmt_sec(event.start_sec)}-{_fmt_sec(event.end_sec)}s"
                if event.has_time
                else "time not annotated"
            )
            lines.append(
                f"- [{event.event_id}] {when} {event.subject} "
                f"{event.action.replace('_', ' ')}"
            )
        participants = gt.participants
        if participants:
            lines.append(
                "Tracks taking part in those events: "
                + "; ".join(
                    f"track {p.track_id} ({p.category})" for p in participants
                )
                + ". Motion from these is not BM."
            )
        if linkage.available:
            lines.append("")
            lines.append(
                "Checked against the annotated track boxes, per ROI tube in the table above:"
            )
            for roi in linkage.rois:
                if roi.has_participant:
                    lines.append(
                        f"- {roi.roi_id}: contains event participants "
                        f"({', '.join(roi.participant_summary)})."
                    )
                else:
                    extra = (
                        f" Non-participant tracks do pass through it: "
                        f"{', '.join(roi.other_tracks)}."
                        if roi.other_tracks
                        else " No annotated object passes through it at all."
                    )
                    lines.append(
                        f"- {roi.roi_id}: no event participant is ever inside it."
                        f"{extra} Whatever drives this ROI is a BM candidate; say what you "
                        "can see moving there."
                    )
        return "\n".join(lines)

    lines = [
        "This video has no ground-truth event annotation, so there is no authoritative "
        "list of target events.",
    ]
    if observed_events:
        lines.append(
            "Watching the original video, these were the main semantic events reported:"
        )
        lines.extend(f"- {event}" for event in observed_events)
        lines.append(
            "Judge BM against these: motion that is clearly unrelated to them, yet is "
            "being scored by the heatmap or covered by an ROI box, is BM. This list "
            "describes the whole frame, so do not copy it into `roi_observations`; "
            "write for each ROI only what happens inside that box."
        )
    else:
        lines.append(
            "No main event could be made out in the original video either. Only call BM "
            "true if you can point at specific irrelevant motion; otherwise say so with a "
            'certainty of "low".'
        )
    return "\n".join(lines)


def _where_in_frame(tube, frame_width: int, frame_height: int) -> str:
    """Plain-language position of a tube, to help tell two ROIs apart."""
    x0, y0, x1, y1 = tube.pixel_bbox
    if tube.area_ratio >= 0.85:
        return "almost the whole frame"
    cx = (x0 + x1) / 2 / max(frame_width, 1)
    cy = (y0 + y1) / 2 / max(frame_height, 1)
    vertical = "top" if cy < 0.34 else "bottom" if cy > 0.66 else "middle"
    horizontal = "left" if cx < 0.34 else "right" if cx > 0.66 else "centre"
    return f"{vertical} {horizontal}"


def roi_block(target: VideoTarget) -> str:
    """The real Stage3 tube table that BM/IE/DE must be grounded on."""
    lines = [
        f"Stage3 ROI tubes for this video ({target.num_roi_tubes} tube(s), "
        f"max {target.max_concurrent_rois} alive at once; "
        f"frame {target.frame_width}x{target.frame_height}, "
        f"clip length {_fmt_sec(target.duration_sec)}s):",
        "| roi_id | overlay label | time range | where in frame "
        "| pixel box x0,y0,x1,y1 | share of frame |",
        "|---|---|---|---|---|---|",
    ]
    for tube in target.tubes:
        box = ",".join(str(v) for v in tube.pixel_bbox)
        where = _where_in_frame(tube, target.frame_width, target.frame_height)
        lines.append(
            f"| {tube.roi_id} | {tube.overlay_label} | "
            f"{_fmt_sec(tube.t0_sec)}-{_fmt_sec(tube.t1_sec)}s | {where} | {box} | "
            f"{tube.area_ratio * 100:.1f}% |"
        )
    if target.concurrent_pairs:
        rendered = "; ".join(
            f"{pair['roi_a']}+{pair['roi_b']} overlap "
            f"{pair['overlap_t0_sec']}-{pair['overlap_t1_sec']}s"
            for pair in target.concurrent_pairs
        )
        lines.append(f"Tubes alive at the same time: {rendered}.")
    else:
        lines.append(
            "No two tubes are alive at the same time, so DE must be false for this video."
        )
    return "\n".join(lines)


def build_pass_a_prompt(target: VideoTarget, gt: GtContext) -> str:
    return (
        PASS_A_PROMPT.format(rules=COMMON_RULES)
        + "\n"
        + f"Video: {target.video_id} ({target.dataset}), "
        + f"{_fmt_sec(target.duration_sec)}s, {target.frame_width}x{target.frame_height}.\n\n"
        + gt_block(gt)
        + "\n"
    )


def build_pass_b_prompt(
    target: VideoTarget,
    gt: GtContext,
    linkage: GtRoiLinkage,
    observed_events: list[str] | None = None,
) -> str:
    return (
        PASS_B_PROMPT.format(rules=COMMON_RULES)
        + "\n"
        + f"Video: {target.video_id} ({target.dataset}).\n\n"
        + roi_block(target)
        + "\n\n"
        + relevance_block(target, gt, linkage, observed_events or [])
        + "\n"
    )


def prompt_settings() -> dict[str, Any]:
    return {
        "pass_a_labels": list(PASS_A_LABELS),
        "pass_b_labels": list(PASS_B_LABELS),
        "pass_a_prompt_template": PASS_A_PROMPT.format(rules=COMMON_RULES),
        "pass_b_prompt_template": PASS_B_PROMPT.format(rules=COMMON_RULES),
    }

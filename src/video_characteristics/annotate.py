"""Run the two annotation passes and assemble one record per video.

Writes only under the chosen output directory. Each video's raw model responses are
cached so a rerun resumes instead of re-inferring, and nothing under ``outputs/stage1``,
``outputs/stage2``, ``outputs/stage3`` or the earlier inference output trees is touched.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import LABELS, LABEL_NAMES
from .datasets import DatasetSpec
from .gt_context import GtContext, load_gt
from .gt_roi_link import GtRoiLinkage, link_rois_to_gt
from .prompts import (
    PASS_A_LABELS,
    PASS_B_LABELS,
    build_pass_a_prompt,
    build_pass_b_prompt,
    prompt_settings,
)
from .selection import VideoTarget
from .vlm import (
    MAX_NEW_TOKENS,
    REPETITION_PENALTY,
    extract_json,
    generate,
    model_settings,
    pass_b_max_new_tokens,
)

STATUS_RESOLVED = "resolved"
STATUS_UNRESOLVED = "unresolved"

EMPTY_LABEL = {
    "label": None,
    "confidence": 0.0,
    "evidence": [],
    "certainty": "",
    "status": STATUS_UNRESOLVED,
}
REVIEW_THRESHOLD = 0.7
# An unresolved label must never look confident.
UNRESOLVED_MAX_CONFIDENCE = 0.5
MAX_ATTEMPTS = 2

# Progress-line symbols; `?` keeps an unresolved label from reading as a negative.
_SYMBOL = {True: "T", False: "F", None: "?"}


# Trailing context must not be \b: in "t=2:08-2:12s" the final "s" is a word char,
# so \b would refuse to close the second stamp.
MMSS_PATTERN = re.compile(r"(?<![\d:.])(\d{1,2}):([0-5]\d)(?:\.(\d+))?(?![\d:])")

# The model reports a word; the number it gives is clamped into that word's band.
# Qwen barely varies the raw float, so the word carries most of the signal.
# "low" stays strictly under REVIEW_THRESHOLD so it always reaches the review list.
CERTAINTY_BANDS = {
    "high": (0.85, 1.00, 0.92),
    "medium": (0.70, 0.85, 0.78),
    "low": (0.30, 0.65, 0.55),
}


def _normalise_times(text: str) -> str:
    """Rewrite ``t=2:08`` style stamps as seconds, which is what the prompt asks for."""

    def repl(match: re.Match[str]) -> str:
        minutes, seconds, frac = match.group(1), match.group(2), match.group(3)
        total = int(minutes) * 60 + int(seconds)
        return f"{total}.{frac}" if frac else f"{total}.0"

    return MMSS_PATTERN.sub(repl, text)


def _coerce_label(payload: Any, key: str) -> dict[str, Any]:
    """Normalise one label block. Unparseable means false at zero confidence."""
    block = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(block, dict):
        return dict(EMPTY_LABEL)
    raw_label = block.get("label")
    if isinstance(raw_label, bool):
        label = raw_label
    else:
        label = str(raw_label).strip().lower() in {"true", "yes", "1"}

    certainty = str(block.get("certainty") or "").strip().lower()
    try:
        confidence = float(block.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    if certainty in CERTAINTY_BANDS:
        low, high, default = CERTAINTY_BANDS[certainty]
        confidence = default if confidence <= 0.0 else max(low, min(high, confidence))
    else:
        certainty = ""
        confidence = max(0.0, min(1.0, confidence))

    raw_evidence = block.get("evidence")
    if isinstance(raw_evidence, str):
        raw_evidence = [raw_evidence]
    evidence = [
        _normalise_times(" ".join(str(item).split()))
        for item in (raw_evidence or [])
        if str(item).strip()
    ]
    return {
        "label": label,
        "confidence": round(confidence, 3),
        "evidence": evidence,
        "certainty": certainty,
        "status": STATUS_RESOLVED,
    }


def _unresolved_reasons(
    key: str,
    block: dict[str, Any],
    *,
    parse_errors: dict[str, str],
    template_echo: bool,
) -> list[str]:
    """Why this label cannot be decided. Empty means it can."""
    reasons: list[str] = []
    source = "pass_a" if key in PASS_A_LABELS else "pass_b"
    if source in parse_errors:
        reasons.append(f"{source}_failed: {parse_errors[source]}")
    if template_echo and key in PASS_B_LABELS:
        reasons.append("pass_b_template_echo")
    if block["certainty"] == "low":
        reasons.append("insufficient_evidence_low_certainty")
    if not block["certainty"] and block["confidence"] <= 0.0:
        reasons.append("no_certainty_or_confidence_reported")
    if block["label"] and not block["evidence"]:
        reasons.append("positive_without_evidence")
    return reasons


def _apply_unresolved(
    parsed: dict[str, dict[str, Any]],
    *,
    parse_errors: dict[str, str],
    template_echo: bool,
    review_flags: dict[str, list[str]],
    structural: set[str],
    applied_rules: list[str],
    extra_reasons: dict[str, list[str]] | None = None,
) -> None:
    """Turn undecidable labels into ``label: null`` instead of forcing them false.

    A label that the model could not judge is not evidence of absence, so it must not
    be counted as a negative. ``structural`` names labels settled by the ROI geometry
    rather than by the model, which stay resolved.
    """
    for key, block in parsed.items():
        if key in structural:
            continue
        reasons = _unresolved_reasons(
            key, block, parse_errors=parse_errors, template_echo=template_echo
        )
        reasons += [r for r in (extra_reasons or {}).get(key, []) if r not in reasons]
        if not reasons:
            continue
        block["label"] = None
        block["status"] = STATUS_UNRESOLVED
        block["unresolved_reasons"] = reasons
        block["confidence"] = round(min(block["confidence"], UNRESOLVED_MAX_CONFIDENCE), 3)
        for reason in reasons:
            if reason not in review_flags.setdefault(key, []):
                review_flags[key].append(reason)
        applied_rules.append(f"{key} unresolved: {', '.join(reasons)}")


def _string_list(payload: Any, key: str) -> list[str]:
    value = payload.get(key) if isinstance(payload, dict) else None
    if isinstance(value, str):
        value = [value]
    return [" ".join(str(item).split()) for item in (value or []) if str(item).strip()]


# Things the model writes when a box holds nothing. Treated as "no event" so a DE
# claim cannot lean on an ROI the model itself described as empty.
NO_CONTENT_PHRASES = (
    "nothing identifiable",
    "nothing",
    "none",
    "n/a",
    "no event",
    "no events",
    "no discernible event",
    "no discernible events",
    "static background",
    "static",
    "background",
    "empty",
)


def _is_unfilled(text: str) -> bool:
    """Whether the model left a field blank or echoed the template placeholder.

    Distinct from `_is_empty_description`: saying a box is empty is an observation,
    while saying nothing at all is a failure to answer.
    """
    cleaned = text.strip()
    return not cleaned or (cleaned.startswith("<") and cleaned.endswith(">"))


def _is_empty_description(text: str) -> bool:
    cleaned = text.strip().lower().strip(".")
    return _is_unfilled(text) or cleaned in NO_CONTENT_PHRASES


def _roi_observations(payload: Any) -> list[dict[str, Any]]:
    value = payload.get("roi_observations") if isinstance(payload, dict) else None
    out: list[dict[str, Any]] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        actions = " ".join(str(item.get("actions") or "").split())
        objects = " ".join(str(item.get("objects") or "").split())
        raw_events = item.get("distinct_events")
        if isinstance(raw_events, str):
            raw_events = [raw_events]
        distinct = [
            _normalise_times(" ".join(str(event).split()))
            for event in (raw_events or [])
            if str(event).strip() and not _is_empty_description(str(event))
        ]
        claimed = item.get("has_semantic_event")
        if isinstance(claimed, bool):
            has_event = claimed
        else:
            has_event = str(claimed).strip().lower() in {"true", "yes", "1"}
        # Do not let a true flag stand when the description says the box is empty.
        if has_event and _is_empty_description(actions) and not distinct:
            has_event = False
        out.append(
            {
                "roi_id": " ".join(str(item.get("roi_id") or "").split()),
                "objects": objects,
                "actions": actions,
                "has_semantic_event": has_event,
                "distinct_events": distinct,
            }
        )
    return out


def _review_flags(
    target: VideoTarget,
    parsed: dict[str, dict[str, Any]],
    extras: dict[str, Any],
    parse_errors: dict[str, str],
) -> dict[str, list[str]]:
    """Per-label reasons a human should look again, beyond the confidence threshold.

    These are consistency checks on the output as assembled: a positive label with no
    evidence, or a pass that reported no observation at all, is not usable as it stands
    even when the model reported high confidence.
    """
    flags: dict[str, list[str]] = {key: [] for key in LABELS}

    for name, keys in (("pass_a", PASS_A_LABELS), ("pass_b", PASS_B_LABELS)):
        if name in parse_errors:
            for key in keys:
                flags[key].append(f"{name}_unusable: {parse_errors[name]}")

    for key in LABELS:
        block = parsed[key]
        if block["confidence"] < REVIEW_THRESHOLD:
            flags[key].append(f"confidence_below_{REVIEW_THRESHOLD}")
        if block["label"] and not block["evidence"]:
            flags[key].append("positive_without_evidence")

    if not extras.get("observed_events"):
        for key in PASS_A_LABELS:
            flags[key].append("no_observed_events_reported")

    described = [obs for obs in extras.get("roi_observations", []) if obs.get("actions")]
    if len(described) < target.num_roi_tubes:
        flags["BM"].append(
            f"only_{len(described)}_of_{target.num_roi_tubes}_rois_described"
        )

    return {key: value for key, value in flags.items() if value}


# Phrases describing a state rather than an event. A parked car or a stationary
# building is not a semantic event, so these never count toward IE or DE.
STATIC_MARKERS = (
    "parked",
    "stationary",
    "remains",
    "no movement",
    "motionless",
    "static",
    "unchanged",
    "idle",
)


def _normalise_event_phrase(text: str) -> str:
    cleaned = " ".join(str(text).split()).lower().strip(" .")
    for prefix in ("a ", "an ", "the "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned


def _event_phrases(obs: dict[str, Any]) -> list[str]:
    """Distinct, non-static event phrases the model listed for one ROI."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in obs.get("distinct_events") or []:
        phrase = _normalise_event_phrase(raw)
        if not phrase or phrase in seen:
            continue
        if any(marker in phrase for marker in STATIC_MARKERS):
            continue
        seen.add(phrase)
        out.append(phrase)
    return out


def _rois_with_events(roi_observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        obs
        for obs in roi_observations
        if obs.get("has_semantic_event") and _event_phrases(obs)
    ]


def _roi_time_range(target: VideoTarget, roi_id: str) -> str:
    for tube in target.tubes:
        if tube.roi_id.upper() == roi_id.upper():
            return f"t={tube.t0_sec:.1f}-{tube.t1_sec:.1f}s"
    return "time range unknown"


ROI_REF_PATTERN = re.compile(r"\bROI[_ ]?(\d+)\b", re.IGNORECASE)
SECONDS_PATTERN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?=s\b|-\d)")


def _bm_evidence_grounded(target: VideoTarget, evidence: list[str]) -> list[str]:
    """Evidence strings whose cited seconds fall inside the ROI tube they name.

    A BM claim says a particular ROI is being driven by motion irrelevant to the
    events. If the cited time lies outside that tube's lifetime, the claim does not
    describe anything the pipeline actually did, so it cannot support BM.
    """
    spans = {
        tube.roi_id.upper(): (tube.t0_sec, tube.t1_sec) for tube in target.tubes
    }
    grounded: list[str] = []
    for item in evidence:
        names = [f"ROI_{n}".upper() for n in ROI_REF_PATTERN.findall(item)]
        known = [spans[name] for name in names if name in spans]
        if not known:
            continue
        seconds = [float(value) for value in SECONDS_PATTERN.findall(item)]
        if not seconds:
            # An ROI named without a time is weak but not contradictory.
            grounded.append(item)
            continue
        # Compare as an interval, so a span that brackets the tube still counts; only a
        # time wholly outside the tube's lifetime contradicts the claim.
        lo, hi = min(seconds), max(seconds)
        if any(lo <= t1 and hi >= t0 for t0, t1 in known):
            grounded.append(item)
    return grounded


def derive_ie(
    target: VideoTarget, roi_observations: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    """IE straight from the per-ROI event lists: one tube holding two or more events.

    The model enumerates the events inside each ROI reliably but answers the IE
    question inconsistently with its own enumeration, so the enumeration decides.
    """
    evidence: list[str] = []
    for obs in roi_observations:
        phrases = _event_phrases(obs)
        if len(phrases) < 2:
            continue
        roi_id = obs.get("roi_id") or "ROI_?"
        evidence.append(
            f"{roi_id} ({_roi_time_range(target, roi_id)}) contains "
            f"{len(phrases)} distinct events: {', '.join(phrases)}"
        )
    return bool(evidence), evidence


def derive_de(
    target: VideoTarget, roi_observations: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    """DE straight from the per-ROI event lists: two tubes carrying different events."""
    with_events = _rois_with_events(roi_observations)
    if len(with_events) < 2:
        return False, []
    # One representative ROI per distinct event set, so the evidence states the actual
    # contrast instead of repeating ROIs that carry the same thing.
    representatives: dict[frozenset[str], str] = {}
    for obs in with_events:
        signature = frozenset(_event_phrases(obs))
        representatives.setdefault(signature, obs.get("roi_id") or "ROI_?")
    if len(representatives) < 2:
        return False, []
    evidence = [
        f"{roi_id} ({_roi_time_range(target, roi_id)}) carries: " + ", ".join(sorted(signature))
        for signature, roi_id in representatives.items()
    ]
    return True, evidence


def derive_ie_from_gt(linkage: GtRoiLinkage) -> tuple[bool, list[str]]:
    """IE from the GT events placed inside each tube: one tube, two or more event kinds."""
    evidence: list[str] = []
    for roi in linkage.rois:
        if len(roi.action_types) < 2:
            continue
        evidence.append(
            f"{roi.roi_id} (t={roi.t0_sec:.1f}-{roi.t1_sec:.1f}s) contains "
            f"{len(roi.gt_events)} GT events of {len(roi.action_types)} kinds: "
            + "; ".join(event.describe() for event in roi.gt_events)
        )
    return bool(evidence), evidence


def derive_de_from_gt(linkage: GtRoiLinkage) -> tuple[bool, list[str]]:
    """DE from the GT: two or more tubes carrying different sets of event kinds."""
    with_events = [roi for roi in linkage.rois if roi.gt_events]
    if len(with_events) < 2:
        return False, []
    representatives: dict[frozenset[str], Any] = {}
    for roi in with_events:
        representatives.setdefault(frozenset(roi.action_types), roi)
    if len(representatives) < 2:
        return False, []
    evidence = [
        f"{roi.roi_id} (t={roi.t0_sec:.1f}-{roi.t1_sec:.1f}s) carries "
        + ", ".join(sorted(roi.action_types))
        for roi in representatives.values()
    ]
    return True, evidence


def _is_template_echo(roi_observations: list[dict[str, Any]]) -> bool:
    """True when the model handed the output skeleton back without filling it in.

    Catches both a blanked-out skeleton and one still carrying the `<...>` placeholders.
    """
    if not roi_observations:
        return True
    return all(
        _is_unfilled(str(obs.get("objects") or ""))
        and _is_unfilled(str(obs.get("actions") or ""))
        and not obs.get("distinct_events")
        for obs in roi_observations
    )


def _looks_unusable(pass_name: str, response: str) -> bool:
    """Whether a response is worth one more attempt: unparseable or an empty skeleton."""
    try:
        payload = extract_json(response)
    except Exception:  # noqa: BLE001
        return True
    if pass_name == "pass_b":
        return _is_template_echo(_roi_observations(payload))
    return not _string_list(payload, "observed_events")


def raw_cache_path(output_dir: Path, target: VideoTarget) -> Path:
    return output_dir / "raw" / target.dataset / f"{target.video_id}.json"


def failed_passes(raw: dict[str, Any]) -> list[str]:
    """Passes in a cached raw record that errored, came back empty, or did not parse.

    A parseable answer that simply reports no events is a valid answer, not a failure.
    """
    failed: list[str] = []
    for name in ("pass_a", "pass_b"):
        block = raw.get(name) or {}
        response = block.get("response") or ""
        if block.get("error") or not response:
            failed.append(name)
            continue
        try:
            extract_json(response)
        except Exception:  # noqa: BLE001
            failed.append(name)
    return failed


def _reusable(cached: dict[str, Any], name: str, prompt: str) -> dict[str, Any] | None:
    """A cached pass worth keeping: same prompt, and a usable response.

    Lets a prompt change in one pass be re-run without paying for the other, and keeps
    a failed response from being cached forever, so a later run gets another attempt.
    """
    block = cached.get(name) or {}
    if not block.get("response") or block.get("prompt") != prompt:
        return None
    if _looks_unusable(name, block["response"]):
        return None
    return {**block, "from_cache": True}


@dataclass(frozen=True)
class GenerationBoost:
    """Generation settings for re-running passes whose earlier answer was unusable.

    Every unusable answer in the full run was a JSON cut off at the token limit, either
    because the answer was long or because the model looped on near-identical lines.
    The first attempt only raises the token limit, so the answer differs from the main
    run in length alone; the second also raises the repetition penalty to break a loop.
    """

    token_scale: float = 2.0
    max_new_tokens_cap: int = 4096
    loop_repetition_penalty: float = 1.15

    def attempts(self, base_max_new_tokens: int) -> list[dict[str, Any]]:
        tokens = min(self.max_new_tokens_cap, int(base_max_new_tokens * self.token_scale))
        return [
            {"max_new_tokens": tokens, "repetition_penalty": None},
            {"max_new_tokens": tokens, "repetition_penalty": self.loop_repetition_penalty},
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_scale": self.token_scale,
            "max_new_tokens_cap": self.max_new_tokens_cap,
            "loop_repetition_penalty": self.loop_repetition_penalty,
        }


def _run_one_pass(
    captioner: Any,
    name: str,
    prompt: str,
    video_path: Any,
    kind: str,
    *,
    max_new_tokens: int | None = None,
    boost: GenerationBoost | None = None,
) -> dict[str, Any]:
    """Generate one pass, retrying once when the response comes back unusable."""
    if video_path is None or not Path(video_path).is_file():
        return {
            "prompt": prompt,
            "input_kind": kind,
            "input_path": None if video_path is None else str(video_path),
            "response": "",
            "error": f"missing {kind}",
        }
    if boost is None:
        schedule = [{"max_new_tokens": max_new_tokens, "repetition_penalty": None}] * MAX_ATTEMPTS
    else:
        schedule = boost.attempts(max_new_tokens or MAX_NEW_TOKENS)
    for attempts, settings in enumerate(schedule, start=1):
        try:
            result = generate(
                captioner,
                Path(video_path),
                prompt,
                max_new_tokens=settings["max_new_tokens"],
                repetition_penalty=settings["repetition_penalty"],
            )
            block = {
                "prompt": prompt,
                "input_kind": kind,
                "input_path": str(video_path),
                "error": None,
                "attempts": attempts,
                "max_new_tokens": settings["max_new_tokens"],
                **result.as_dict(),
            }
            if boost is not None:
                block["generation_boost"] = {
                    **boost.as_dict(),
                    "repetition_penalty_used": settings["repetition_penalty"]
                    or REPETITION_PENALTY,
                }
        except Exception as exc:  # noqa: BLE001
            return {
                "prompt": prompt,
                "input_kind": kind,
                "input_path": str(video_path),
                "response": "",
                "error": str(exc),
                "attempts": attempts,
            }
        # Occasionally the model hands the output skeleton straight back. One retry is
        # far cheaper than losing the labels for the whole video.
        if not _looks_unusable(name, block["response"]):
            return block
    return block


def _observed_events_of(block: dict[str, Any]) -> list[str]:
    if block.get("error") or not block.get("response"):
        return []
    try:
        return _string_list(extract_json(block["response"]), "observed_events")
    except Exception:  # noqa: BLE001
        return []


def run_passes(
    captioner: Any,
    target: VideoTarget,
    gt: GtContext,
    linkage: GtRoiLinkage,
    *,
    output_dir: Path,
    overwrite: bool = False,
    boost: GenerationBoost | None = None,
    regenerate: set[str] | None = None,
) -> dict[str, Any]:
    """Return the cached or freshly generated raw responses for both passes.

    ``boost`` applies only to passes that actually get generated. When ``regenerate``
    names the passes to redo, every other cached pass with a matching prompt is kept
    as it is, even one whose answer would otherwise be retried.
    """
    cache = raw_cache_path(output_dir, target)
    cached: dict[str, Any] = {}
    if cache.is_file() and not overwrite:
        cached = json.loads(cache.read_text(encoding="utf-8"))

    def reuse(name: str, prompt: str) -> dict[str, Any] | None:
        if regenerate is None:
            return _reusable(cached, name, prompt)
        block = cached.get(name) or {}
        if name in regenerate or not block.get("response") or block.get("prompt") != prompt:
            return None
        return {**block, "from_cache": True}

    record: dict[str, Any] = {
        "video_id": target.video_id,
        "dataset": target.dataset,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Pass A runs first so that, when the video has no GT, pass B can be told which
    # events pass A actually saw and judge BM relevance against those.
    prompt_a = build_pass_a_prompt(target, gt)
    record["pass_a"] = reuse("pass_a", prompt_a) or _run_one_pass(
        captioner,
        "pass_a",
        prompt_a,
        target.raw_video,
        "raw_video",
        max_new_tokens=MAX_NEW_TOKENS,
        boost=boost,
    )
    prompt_b = build_pass_b_prompt(
        target, gt, linkage, observed_events=_observed_events_of(record["pass_a"])
    )
    record["pass_b"] = reuse("pass_b", prompt_b) or _run_one_pass(
        captioner,
        "pass_b",
        prompt_b,
        target.overlay_video,
        "stage3_overlay_mp4",
        max_new_tokens=pass_b_max_new_tokens(target.num_roi_tubes),
        boost=boost,
    )
    record["from_cache"] = all(
        record[name].get("from_cache") for name in ("pass_a", "pass_b")
    )

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def build_annotation(
    target: VideoTarget,
    gt: GtContext,
    linkage: GtRoiLinkage,
    raw: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the final per-video annotation from the two raw responses."""
    parse_errors: dict[str, str] = {}
    parsed: dict[str, dict[str, Any]] = {}
    extras: dict[str, Any] = {}

    for name, label_keys in (("pass_a", PASS_A_LABELS), ("pass_b", PASS_B_LABELS)):
        block = raw.get(name) or {}
        payload: dict[str, Any] = {}
        if block.get("error"):
            parse_errors[name] = str(block["error"])
        elif not block.get("response"):
            parse_errors[name] = "empty response"
        else:
            try:
                payload = extract_json(block["response"])
            except Exception as exc:  # noqa: BLE001
                parse_errors[name] = f"JSON parse failed: {exc}"
        for key in label_keys:
            parsed[key] = _coerce_label(payload, key)
        if name == "pass_a":
            extras["observed_events"] = _string_list(payload, "observed_events")
        else:
            extras["roi_observations"] = _roi_observations(payload)

    roi_observations = extras.get("roi_observations", [])
    template_echo = "pass_b" not in parse_errors and _is_template_echo(roi_observations)

    applied_rules: list[str] = []
    model_labels = {key: dict(parsed[key]) for key in PASS_B_LABELS}

    # IE and DE are rebuilt from the per-ROI event lists rather than taken from the
    # model's own IE/DE answers, which contradict that enumeration in both directions.
    # Their certainty follows how completely the ROIs were described, since that is what
    # the derivation rests on, not the model's own IE/DE self-report.
    described = sum(1 for obs in roi_observations if obs.get("actions"))
    coverage = described / target.num_roi_tubes if target.num_roi_tubes else 0.0
    derived_certainty = "high" if coverage >= 1.0 else "medium" if coverage >= 0.5 else "low"

    # With GT, which events sit in which tube is a fact about the annotation and the
    # tube geometry, so it decides IE and DE instead of the model's ROI descriptions.
    use_gt = linkage.available and linkage.has_event_placement
    ie_de_basis = "gt_events_in_roi" if use_gt else "vlm_roi_observations"
    derivers = (
        (("IE", derive_ie_from_gt), ("DE", derive_de_from_gt))
        if use_gt
        else (("IE", derive_ie), ("DE", derive_de))
    )
    for key, derive in derivers:
        label, evidence = (
            derive(linkage) if use_gt else derive(target, roi_observations)
        )
        certainty = "high" if use_gt else derived_certainty
        parsed[key] = {
            "label": label,
            "confidence": CERTAINTY_BANDS[certainty][2],
            "evidence": evidence,
            "certainty": certainty,
            "status": STATUS_RESOLVED,
            "basis": ie_de_basis,
        }
        if use_gt:
            applied_rules.append(
                f"{key} derived from the {linkage.as_dict()['n_gt_events_placed']} GT "
                f"event(s) placed in ROI tubes (model answered {model_labels[key]['label']})"
            )
        else:
            applied_rules.append(
                f"{key} derived from per-ROI event lists "
                f"({described}/{target.num_roi_tubes} ROIs described; "
                f"model answered {model_labels[key]['label']})"
            )

    # The task definition fixes DE to false when there is only one ROI tube: a single
    # tube cannot split evidence across different tubes.
    if target.num_roi_tubes < 2:
        parsed["DE"] = {
            "label": False,
            "confidence": 1.0,
            "evidence": [],
            "certainty": "high",
            "status": STATUS_RESOLVED,
            "basis": "single_roi_tube",
        }
        applied_rules.append("DE false by definition: video has a single Stage3 ROI tube")

    if template_echo:
        applied_rules.append("pass_b returned the empty output template; BM/IE/DE unresolved")

    review_flags = _review_flags(target, parsed, extras, parse_errors)
    if template_echo:
        for key in PASS_B_LABELS:
            review_flags.setdefault(key, []).append("pass_b_template_echo")
    for key in PASS_B_LABELS:
        if model_labels[key]["label"] != parsed[key]["label"]:
            review_flags.setdefault(key, []).append(
                f"derived_label_differs_from_model ({model_labels[key]['label']}"
                f" -> {parsed[key]['label']})"
            )

    # BM asks whether motion unrelated to the events is what drives an ROI. With no ROI
    # described, there is nothing to attribute the motion to, whatever the model answered.
    extra_reasons: dict[str, list[str]] = {}
    if not described:
        extra_reasons["BM"] = ["roi_contents_not_described"]
    if parsed["BM"]["label"]:
        grounded = _bm_evidence_grounded(target, parsed["BM"]["evidence"])
        if not grounded:
            extra_reasons.setdefault("BM", []).append("bm_evidence_not_grounded_in_an_roi")
        elif len(grounded) < len(parsed["BM"]["evidence"]):
            dropped = len(parsed["BM"]["evidence"]) - len(grounded)
            parsed["BM"]["evidence"] = grounded
            applied_rules.append(
                f"BM: dropped {dropped} evidence item(s) whose time fell outside the ROI "
                "tube they named"
            )

    # Labels settled by the annotation or the tube geometry do not depend on the model,
    # so a failed pass B cannot leave them unresolved.
    structural: set[str] = {"IE", "DE"} if use_gt else set()
    if target.num_roi_tubes < 2:
        structural.add("DE")
    _apply_unresolved(
        parsed,
        parse_errors=parse_errors,
        template_echo=template_echo,
        review_flags=review_flags,
        structural=structural,
        applied_rules=applied_rules,
        extra_reasons=extra_reasons,
    )

    record: dict[str, Any] = {
        "video_id": target.video_id,
        "dataset": target.dataset,
        "selection": dict(target.selection),
    }
    for key in LABELS:
        record[key] = parsed[key]

    record["label_names"] = dict(LABEL_NAMES)
    record["observed_events"] = extras.get("observed_events", [])
    record["roi_observations"] = extras.get("roi_observations", [])
    record["stage3"] = target.as_dict()["stage3"]
    # BM rests on knowing which motion is relevant. With GT that comes from the event
    # participant tracks; without it, only from what pass A reported seeing.
    record["BM"]["relevance_basis"] = (
        "gt_event_participants" if gt.available else "vlm_observed_events"
    )
    record["BM"]["gt_derived"] = gt.available
    record["ie_de_basis"] = ie_de_basis
    record["gt_roi_linkage"] = linkage.as_dict()
    record["gt"] = {
        "available": gt.available,
        "source": gt.source,
        "note": gt.note,
        "n_events": len(gt.events),
        "participants": [p.as_dict() for p in gt.participants],
        "polarity_groups": gt.polarity_groups,
        "small_object_actions": gt.small_object_actions,
        "n_small_subject_events": len(gt.small_subject_events),
    }
    record["provenance"] = {
        "raw_video": None if target.raw_video is None else str(target.raw_video),
        "overlay_video": None if target.overlay_video is None else str(target.overlay_video),
        "tracks_json": None if target.tracks_json is None else str(target.tracks_json),
        "pass_a_input": (raw.get("pass_a") or {}).get("input_kind"),
        "pass_b_input": (raw.get("pass_b") or {}).get("input_kind"),
        "sampled_frames": {
            "pass_a": (raw.get("pass_a") or {}).get("sampled_frames"),
            "pass_b": (raw.get("pass_b") or {}).get("sampled_frames"),
        },
        "visual_tokens": {
            "pass_a": (raw.get("pass_a") or {}).get("visual_tokens"),
            "pass_b": (raw.get("pass_b") or {}).get("visual_tokens"),
        },
        "applied_rules": applied_rules,
        "parse_errors": parse_errors,
        "from_cache": bool(raw.get("from_cache")),
    }
    # DE on a single-tube video is settled by the ROI count, so the model's answer and
    # any flag it collected no longer describe the value being reported.
    if target.num_roi_tubes < 2:
        review_flags.pop("DE", None)
    record["provenance"]["model_labels"] = {
        key: {"label": value["label"], "confidence": value["confidence"]}
        for key, value in model_labels.items()
    }
    record["review_flags"] = review_flags
    record["needs_manual_review"] = sorted(review_flags)
    record["unresolved_labels"] = sorted(
        key for key in LABELS if record[key]["status"] == STATUS_UNRESOLVED
    )
    return record


def annotate_targets(
    captioner: Any,
    spec: DatasetSpec,
    targets: list[VideoTarget],
    *,
    output_dir: Path,
    overwrite: bool = False,
    progress: bool = True,
    boost: GenerationBoost | None = None,
    regenerate: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Annotate ``targets``; ``regenerate`` maps video_id to the passes to redo."""
    records: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        gt = load_gt(spec, target.video_id, target.frame_width, target.frame_height)
        if progress:
            print(
                f"[{index}/{len(targets)}] {spec.name} {target.video_id} "
                f"(tubes={target.num_roi_tubes}, gt={'yes' if gt.available else 'no'})",
                flush=True,
            )
        linkage = link_rois_to_gt(target, gt)
        raw = run_passes(
            captioner,
            target,
            gt,
            linkage,
            output_dir=output_dir,
            overwrite=overwrite,
            boost=boost,
            regenerate=None if regenerate is None else regenerate.get(target.video_id, set()),
        )
        record = build_annotation(target, gt, linkage, raw)
        records.append(record)
        if progress:
            flags = " ".join(
                f"{key}={_SYMBOL[record[key]['label']]}@{record[key]['confidence']:.2f}"
                for key in LABELS
            )
            cached = " (cached)" if record["provenance"]["from_cache"] else ""
            print(f"    {flags}{cached}", flush=True)
            for name, error in record["provenance"]["parse_errors"].items():
                print(f"    [warn] {name}: {error}", flush=True)
    return records


def run_settings() -> dict[str, Any]:
    return {
        "model": model_settings(),
        "prompts": prompt_settings(),
        "labels": {key: LABEL_NAMES[key] for key in LABELS},
        "pass_a_input": "original dataset video (AD, OD)",
        "pass_b_input": "existing Stage3 heatmap + lifetime ROI overlay MP4 (BM, IE, DE)",
        "manual_review_threshold": 0.7,
    }

"""Counts, distributions, review list and label-combination frequencies."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from . import LABELS, LABEL_NAMES

REVIEW_THRESHOLD = 0.7

CSV_COLUMNS = [
    "video_id",
    "dataset",
    "roi_set",
    "single_roi_area_lt_090",
    "multi_roi",
    "num_roi_tubes",
    "max_concurrent_rois",
    "mean_roi_area_ratio",
    "duration_sec",
    "gt_available",
    "gt_n_events",
    *[f"{key}_label" for key in LABELS],
    *[f"{key}_status" for key in LABELS],
    *[f"{key}_confidence" for key in LABELS],
    *[f"{key}_certainty" for key in LABELS],
    *[f"{key}_evidence" for key in LABELS],
    "label_combination",
    "fully_resolved",
    "unresolved_labels",
    "bm_relevance_basis",
    "ie_de_basis",
    "min_confidence",
    "needs_manual_review",
    "parse_errors",
]

STATUS_UNRESOLVED = "unresolved"


def status_of(record: dict[str, Any], key: str) -> str:
    """``positive``, ``negative`` or ``unresolved`` for one label of one video."""
    block = record[key]
    if block.get("status") == STATUS_UNRESOLVED or block.get("label") is None:
        return "unresolved"
    return "positive" if block["label"] else "negative"


def unresolved_labels(record: dict[str, Any]) -> list[str]:
    return [key for key in LABELS if status_of(record, key) == "unresolved"]


def is_fully_resolved(record: dict[str, Any]) -> bool:
    return not unresolved_labels(record)


def roi_set_of(record: dict[str, Any]) -> str:
    selection = record.get("selection") or {}
    if selection.get("single_roi_area_lt_090") and selection.get("multi_roi"):
        return "both"
    if selection.get("multi_roi"):
        return "multi_roi"
    if selection.get("single_roi_area_lt_090"):
        return "single_roi_area_lt_090"
    return "unknown"


def label_combination(record: dict[str, Any]) -> str:
    """Positive set of a video, or ``unresolved`` when any label is undecided.

    A combination is only meaningful once every label is settled, since an undecided
    label could belong in the set.
    """
    if not is_fully_resolved(record):
        return "unresolved"
    positives = [key for key in LABELS if record[key]["label"]]
    return "+".join(positives) if positives else "none"


def to_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    stage3 = record.get("stage3") or {}
    gt = record.get("gt") or {}
    row: dict[str, Any] = {
        "video_id": record["video_id"],
        "dataset": record["dataset"],
        "roi_set": roi_set_of(record),
        "single_roi_area_lt_090": record["selection"].get("single_roi_area_lt_090"),
        "multi_roi": record["selection"].get("multi_roi"),
        "num_roi_tubes": stage3.get("num_roi_tubes"),
        "max_concurrent_rois": stage3.get("max_concurrent_rois"),
        "mean_roi_area_ratio": stage3.get("mean_roi_area_ratio"),
        "duration_sec": stage3.get("duration_sec"),
        "gt_available": gt.get("available"),
        "gt_n_events": gt.get("n_events"),
        "label_combination": label_combination(record),
        "fully_resolved": is_fully_resolved(record),
        "unresolved_labels": ";".join(unresolved_labels(record)),
        "bm_relevance_basis": record["BM"].get("relevance_basis", ""),
        "ie_de_basis": record.get("ie_de_basis", ""),
        "min_confidence": min(record[key]["confidence"] for key in LABELS),
        "needs_manual_review": ";".join(record.get("needs_manual_review") or []),
        "parse_errors": ";".join(
            f"{k}:{v}" for k, v in (record.get("provenance", {}).get("parse_errors") or {}).items()
        ),
    }
    for key in LABELS:
        row[f"{key}_label"] = "" if record[key]["label"] is None else record[key]["label"]
        row[f"{key}_status"] = status_of(record, key)
        row[f"{key}_confidence"] = record[key]["confidence"]
        row[f"{key}_certainty"] = record[key].get("certainty", "")
        row[f"{key}_evidence"] = " || ".join(record[key]["evidence"])
    return row


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(to_csv_row(record))


def label_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Positive / negative / unresolved per label.

    ``positive_rate`` is taken over the resolved videos only, so that videos whose
    label could not be judged neither inflate nor deflate it. Any benchmark-score
    analysis should use the same ``n_resolved`` denominator.
    """
    out: dict[str, dict[str, Any]] = {}
    for key in LABELS:
        statuses = Counter(status_of(r, key) for r in records)
        resolved = statuses["positive"] + statuses["negative"]
        out[key] = {
            "name": LABEL_NAMES[key],
            "positive": statuses["positive"],
            "negative": statuses["negative"],
            "unresolved": statuses["unresolved"],
            "n_resolved": resolved,
            "total": len(records),
            "positive_rate": round(statuses["positive"] / resolved, 4) if resolved else None,
            "low_confidence": sum(1 for r in records if r[key]["confidence"] < REVIEW_THRESHOLD),
        }
    return out


def counts_by_group(records: list[dict[str, Any]], group_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(group_fn(record), []).append(record)
    return {
        name: {"n_videos": len(rows), "labels": label_counts(rows)}
        for name, rows in sorted(groups.items())
    }


def combination_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Frequency of each exact positive set, over fully resolved videos only."""
    resolved = [r for r in records if is_fully_resolved(r)]
    counter = Counter(label_combination(record) for record in resolved)
    return [
        {"combination": name, "n_videos": count}
        for name, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def pair_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Co-occurrence of every label pair, independent of the other three.

    Counted per pair over the videos where both labels of that pair are resolved, so
    one undecided label does not remove a video from every other pair.
    """
    out: list[dict[str, Any]] = []
    for i, left in enumerate(LABELS):
        for right in LABELS[i + 1 :]:
            usable = [
                r
                for r in records
                if status_of(r, left) != "unresolved" and status_of(r, right) != "unresolved"
            ]
            n = sum(1 for r in usable if r[left]["label"] and r[right]["label"])
            out.append({"pair": f"{left}+{right}", "n_videos": n, "n_resolved": len(usable)})
    return sorted(out, key=lambda item: (-item["n_videos"], item["pair"]))


def manual_review_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (video, label) that a human should re-check.

    Triggered by an unresolved status, by confidence below the threshold, or by any
    consistency flag raised while assembling the record.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        flags = record.get("review_flags") or {}
        for key in LABELS:
            block = record[key]
            status = status_of(record, key)
            reasons = list(flags.get(key) or [])
            if not reasons and status != "unresolved" and block["confidence"] >= REVIEW_THRESHOLD:
                continue
            rows.append(
                {
                    "video_id": record["video_id"],
                    "dataset": record["dataset"],
                    "roi_set": roi_set_of(record),
                    "label": key,
                    "label_name": LABEL_NAMES[key],
                    "status": status,
                    "value": "" if block["label"] is None else block["label"],
                    "confidence": block["confidence"],
                    "evidence": " || ".join(block["evidence"]),
                    "reason": ";".join(reasons),
                }
            )
    return sorted(rows, key=lambda row: (row["confidence"], row["video_id"], row["label"]))


def write_manual_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "video_id",
        "dataset",
        "roi_set",
        "label",
        "label_name",
        "status",
        "value",
        "confidence",
        "evidence",
        "reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    review = manual_review_rows(records)
    return {
        "n_videos": len(records),
        "n_by_dataset": dict(Counter(r["dataset"] for r in records)),
        "n_by_roi_set": dict(Counter(roi_set_of(r) for r in records)),
        "gt_available": {
            "yes": sum(1 for r in records if (r.get("gt") or {}).get("available")),
            "no": sum(1 for r in records if not (r.get("gt") or {}).get("available")),
        },
        "bm_relevance_basis": dict(
            Counter(r["BM"].get("relevance_basis", "unknown") for r in records)
        ),
        "ie_de_basis": dict(Counter(r.get("ie_de_basis", "unknown") for r in records)),
        "resolution": {
            "n_fully_resolved": sum(1 for r in records if is_fully_resolved(r)),
            "n_with_unresolved": sum(1 for r in records if not is_fully_resolved(r)),
            "unresolved_by_label": dict(
                Counter(key for r in records for key in unresolved_labels(r))
            ),
            "unresolved_reasons": dict(
                Counter(
                    reason
                    for r in records
                    for key in LABELS
                    for reason in (r[key].get("unresolved_reasons") or [])
                )
            ),
        },
        "label_counts_overall": label_counts(records),
        "label_counts_by_dataset": counts_by_group(records, lambda r: r["dataset"]),
        "label_counts_by_roi_set": counts_by_group(records, roi_set_of),
        "label_counts_by_dataset_roi_set": counts_by_group(
            records, lambda r: f"{r['dataset']}/{roi_set_of(r)}"
        ),
        "label_combinations": combination_counts(records),
        "label_pairs": pair_counts(records),
        "manual_review": {
            "threshold": REVIEW_THRESHOLD,
            "n_label_entries": len(review),
            "n_videos": len({row["video_id"] for row in review}),
            "by_label": dict(Counter(row["label"] for row in review)),
            "by_reason": dict(
                Counter(
                    reason
                    for row in review
                    for reason in (row["reason"].split(";") if row["reason"] else ["(none)"])
                )
            ),
        },
        "parse_failures": [
            {
                "video_id": r["video_id"],
                "dataset": r["dataset"],
                "errors": r["provenance"]["parse_errors"],
            }
            for r in records
            if r.get("provenance", {}).get("parse_errors")
        ],
    }


def _label_table(counts: dict[str, dict[str, Any]], total: int) -> list[str]:
    lines = [
        "| label | name | positive | negative | unresolved | positive rate (resolved) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in LABELS:
        row = counts[key]
        rate = "n/a" if row["positive_rate"] is None else f"{row['positive_rate'] * 100:.1f}%"
        lines.append(
            f"| {key} | {row['name']} | {row['positive']} | {row['negative']} | "
            f"{row['unresolved']} | {rate} ({row['n_resolved']}) |"
        )
    return lines


def write_report(path: Path, stats: dict[str, Any], records: list[dict[str, Any]]) -> None:
    total = stats["n_videos"]
    lines = [
        "# Video characteristic annotation (AD/OD/BM/IE/DE)",
        "",
        "Built from the existing Stage3 ROI tube results and the existing VIRAT "
        "caption-event GT. Stage1-3 outputs and earlier inference results were not "
        "modified.",
        "",
        f"- videos: **{total}**",
        "- by dataset: "
        + ", ".join(f"{k} {v}" for k, v in sorted(stats["n_by_dataset"].items())),
        "- by selection set: "
        + ", ".join(f"{k} {v}" for k, v in sorted(stats["n_by_roi_set"].items())),
        f"- GT events available: {stats['gt_available']['yes']} / "
        f"unavailable: {stats['gt_available']['no']}",
        "- BM relevance basis: "
        + ", ".join(f"{k} {v}" for k, v in sorted(stats["bm_relevance_basis"].items())),
        "- IE/DE basis: "
        + ", ".join(f"{k} {v}" for k, v in sorted(stats["ie_de_basis"].items())),
        f"- fully resolved videos: **{stats['resolution']['n_fully_resolved']}**, "
        f"videos with at least one unresolved label: "
        f"{stats['resolution']['n_with_unresolved']}",
        "",
        "## 1. Label counts (all videos)",
        "",
        "`unresolved` means the label could not be judged reliably, so it is neither a "
        "positive nor a negative. Exclude those videos from that label's benchmark-score "
        "analysis; the `positive rate` column already uses the resolved count as its "
        "denominator (shown in brackets).",
        "",
        *_label_table(stats["label_counts_overall"], total),
        "",
    ]
    if stats["resolution"]["unresolved_by_label"]:
        lines.extend(
            [
                "### Why labels were left unresolved",
                "",
                "| reason | label entries |",
                "|---|---:|",
                *[
                    f"| {reason} | {count} |"
                    for reason, count in sorted(
                        stats["resolution"]["unresolved_reasons"].items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )
                ],
                "",
            ]
        )

    lines.extend(["## 2. By dataset", ""])
    for name, block in stats["label_counts_by_dataset"].items():
        lines.extend([f"### {name} ({block['n_videos']} videos)", ""])
        lines.extend(_label_table(block["labels"], block["n_videos"]))
        lines.append("")

    lines.extend(["## 3. By Single-ROI / Multi-ROI", ""])
    for name, block in stats["label_counts_by_roi_set"].items():
        lines.extend([f"### {name} ({block['n_videos']} videos)", ""])
        lines.extend(_label_table(block["labels"], block["n_videos"]))
        lines.append("")

    lines.extend(["## 4. By dataset x selection set", ""])
    for name, block in stats["label_counts_by_dataset_roi_set"].items():
        lines.extend([f"### {name} ({block['n_videos']} videos)", ""])
        lines.extend(_label_table(block["labels"], block["n_videos"]))
        lines.append("")

    lines.extend(
        [
            "## 5. Label combinations",
            "",
            "Exact positive set per video, over the "
            f"{stats['resolution']['n_fully_resolved']} videos where all five labels are "
            "resolved. `none` means all five are false.",
            "",
            "| combination | videos | share |",
            "|---|---:|---:|",
        ]
    )
    n_combo = stats["resolution"]["n_fully_resolved"]
    for row in stats["label_combinations"]:
        share = row["n_videos"] / n_combo if n_combo else 0.0
        lines.append(f"| {row['combination']} | {row['n_videos']} | {share * 100:.1f}% |")

    lines.extend(
        [
            "",
            "### Pairwise co-occurrence",
            "",
            "Videos where both labels are true, regardless of the other three. Counted "
            "per pair over the videos where both of its labels are resolved.",
            "",
            "| pair | both true | videos resolved for the pair |",
            "|---|---:|---:|",
        ]
    )
    for row in stats["label_pairs"]:
        lines.append(f"| {row['pair']} | {row['n_videos']} | {row['n_resolved']} |")

    review = stats["manual_review"]
    lines.extend(
        [
            "",
            "## 6. Manual review targets",
            "",
            f"Label entries flagged by confidence < {review['threshold']} or by a "
            f"consistency check: **{review['n_label_entries']}** across "
            f"**{review['n_videos']}** videos. Full list in `manual_review.csv`.",
            "",
            "| reason | entries |",
            "|---|---:|",
            *[
                f"| {reason} | {count} |"
                for reason, count in sorted(
                    review["by_reason"].items(), key=lambda kv: (-kv[1], kv[0])
                )
            ],
            "",
        ]
    )
    if stats["parse_failures"]:
        lines.extend(
            [
                "## 7. Responses that failed to parse",
                "",
                "| video_id | dataset | errors |",
                "|---|---|---|",
            ]
        )
        for row in stats["parse_failures"]:
            errors = "; ".join(f"{k}: {v}" for k, v in row["errors"].items())
            lines.append(f"| `{row['video_id']}` | {row['dataset']} | {errors} |")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

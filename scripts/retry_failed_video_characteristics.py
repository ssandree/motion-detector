#!/usr/bin/env python3
"""Re-run only the passes that failed (error, empty, or unparseable JSON), and merge them in.

Every failed answer in the full run was a JSON cut off at the token limit, so the
failed passes are regenerated with a larger token budget (and, on the second attempt,
a higher repetition penalty to break loops). Every other pass is reused from the cache
unchanged, including parseable answers that simply report no events; prompts are not
modified. If pass A is regenerated, pass B is regenerated too only when its prompt
changes as a result (it quotes pass A's observed events for videos without GT).

The rest of the existing annotations are kept as they are: the re-annotated records
replace their old versions inside ``{dataset}_annotations.json``, and the combined
JSON/CSV, statistics, report and manual-review list are rebuilt from the merged set.
Everything that gets overwritten is first copied to ``retry_backups/<timestamp>/``.

Check what would run:
    scripts/retry_failed_video_characteristics.py --dry-run

Run:
    scripts/retry_failed_video_characteristics.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from video_characteristics import LABELS, aggregate  # noqa: E402
from video_characteristics.annotate import (  # noqa: E402
    GenerationBoost,
    annotate_targets,
    failed_passes,
    raw_cache_path,
)
from video_characteristics.datasets import DATASETS, get_dataset  # noqa: E402
from video_characteristics.selection import VideoTarget, select_targets  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "video_characteristics"
COMBINED_OUTPUTS = (
    "combined_annotations.json",
    "combined_annotations.csv",
    "statistics.json",
    "report.md",
    "manual_review.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASETS), choices=sorted(DATASETS))
    parser.add_argument(
        "--video-ids",
        nargs="+",
        default=None,
        help="restrict the retry to these video ids (they must still have a failed pass)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the failed videos and passes without loading the model or writing",
    )
    return parser.parse_args()


def _labels_line(record: dict) -> str:
    symbol = {True: "T", False: "F", None: "?"}
    return " ".join(f"{key}={symbol[record[key]['label']]}" for key in LABELS)


def find_failed(
    output_dir: Path, dataset: str, records: list[dict], wanted: set[str] | None
) -> tuple[list[VideoTarget], dict[str, list[str]]]:
    spec = get_dataset(dataset)
    by_id = {target.video_id: target for target in select_targets(spec)}
    failed: dict[str, list[str]] = {}
    for record in records:
        video_id = record["video_id"]
        if wanted is not None and video_id not in wanted:
            continue
        target = by_id.get(video_id)
        if target is None:
            continue
        cache = raw_cache_path(output_dir, target)
        if not cache.is_file():
            failed[video_id] = ["pass_a", "pass_b"]
            continue
        passes = failed_passes(json.loads(cache.read_text(encoding="utf-8")))
        if passes:
            failed[video_id] = passes
    targets = [by_id[video_id] for video_id in failed]
    return targets, failed


def backup(output_dir: Path, plan: list[tuple[str, list[VideoTarget]]]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = output_dir / "retry_backups" / stamp
    root.mkdir(parents=True, exist_ok=False)
    for name in [*(f"{d.lower()}_annotations.json" for d in DATASETS), *COMBINED_OUTPUTS]:
        source = output_dir / name
        if source.is_file():
            shutil.copy2(source, root / name)
    for _, targets in plan:
        for target in targets:
            cache = raw_cache_path(output_dir, target)
            if cache.is_file():
                destination = root / "raw" / target.dataset / cache.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cache, destination)
    return root


def rebuild_combined(output_dir: Path) -> list[dict]:
    records: list[dict] = []
    for name in sorted(DATASETS):
        path = output_dir / f"{name.lower()}_annotations.json"
        if path.is_file():
            records.extend(json.loads(path.read_text(encoding="utf-8")))
    aggregate.write_json(output_dir / "combined_annotations.json", records)
    aggregate.write_csv(output_dir / "combined_annotations.csv", records)
    stats = aggregate.build_statistics(records)
    aggregate.write_json(output_dir / "statistics.json", stats)
    aggregate.write_report(output_dir / "report.md", stats, records)
    aggregate.write_manual_review_csv(
        output_dir / "manual_review.csv", aggregate.manual_review_rows(records)
    )
    return records


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    wanted = set(args.video_ids) if args.video_ids else None

    existing: dict[str, list[dict]] = {}
    plan: list[tuple[str, list[VideoTarget]]] = []
    failed_by_video: dict[str, list[str]] = {}
    for dataset in args.datasets:
        path = output_dir / f"{dataset.lower()}_annotations.json"
        if not path.is_file():
            print(f"skip {dataset}: {path} not found")
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        existing[dataset] = records
        targets, failed = find_failed(output_dir, dataset, records, wanted)
        plan.append((dataset, targets))
        failed_by_video.update(failed)
        print(f"{dataset}: {len(targets)} of {len(records)} video(s) have a failed pass")
        old = {record["video_id"]: record for record in records}
        for target in targets:
            print(
                f"  - {target.video_id} tubes={target.num_roi_tubes} "
                f"failed={','.join(failed[target.video_id])} | now: {_labels_line(old[target.video_id])}"
            )

    total = sum(len(targets) for _, targets in plan)
    if total == 0:
        print("nothing to retry")
        return 0
    boost = GenerationBoost()
    print(
        f"\nretry settings: token budget x{boost.token_scale} (cap {boost.max_new_tokens_cap}), "
        f"second attempt repetition_penalty={boost.loop_repetition_penalty}"
    )
    if args.dry_run:
        print("dry run: model not loaded, nothing written")
        return 0

    backup_dir = backup(output_dir, plan)
    print(f"backed up current outputs to {backup_dir}")

    from video_characteristics.vlm import build_captioner

    print(f"\nloading Qwen3-VL for {total} video(s) ...", flush=True)
    captioner = build_captioner()

    changes: list[dict] = []
    for dataset, targets in plan:
        if not targets:
            continue
        spec = get_dataset(dataset)
        fresh = annotate_targets(
            captioner,
            spec,
            targets,
            output_dir=output_dir,
            boost=boost,
            regenerate={t.video_id: set(failed_by_video[t.video_id]) for t in targets},
        )
        fresh_by_id = {record["video_id"]: record for record in fresh}
        merged = []
        for record in existing[dataset]:
            new = fresh_by_id.get(record["video_id"])
            if new is None:
                merged.append(record)
                continue
            merged.append(new)
            changes.append(
                {
                    "video_id": record["video_id"],
                    "dataset": dataset,
                    "failed_passes": failed_by_video[record["video_id"]],
                    "before": {key: record[key]["label"] for key in LABELS},
                    "after": {key: new[key]["label"] for key in LABELS},
                    "still_failed": sorted(new["provenance"]["parse_errors"]),
                }
            )
        aggregate.write_json(output_dir / f"{dataset.lower()}_annotations.json", merged)

    records = rebuild_combined(output_dir)
    aggregate.write_json(
        output_dir / "retry_log.json",
        {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "backup_dir": str(backup_dir),
            "generation_boost": boost.as_dict(),
            "n_retried": len(changes),
            "n_still_failed": sum(1 for change in changes if change["still_failed"]),
            "videos": changes,
        },
    )

    symbol = {True: "T", False: "F", None: "?"}
    print("\nbefore -> after")
    for change in changes:
        diff = " ".join(
            f"{key}:{symbol[change['before'][key]]}->{symbol[change['after'][key]]}"
            for key in LABELS
            if change["before"][key] != change["after"][key]
        )
        tail = f" | still failed: {','.join(change['still_failed'])}" if change["still_failed"] else ""
        print(f"  {change['video_id']}: {diff or 'no label change'}{tail}")
    recovered = sum(1 for change in changes if not change["still_failed"])
    print(f"\nrecovered {recovered} of {len(changes)} video(s); merged into {len(records)} annotation(s)")
    print(f"details: {output_dir / 'retry_log.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

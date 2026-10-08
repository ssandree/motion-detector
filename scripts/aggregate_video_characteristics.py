#!/usr/bin/env python3
"""Rebuild the combined outputs and statistics from existing annotation JSONs.

Useful after a partial run, or to regenerate the report without touching the model.
Reads ``{dataset}_annotations.json`` from the output directory and rewrites the
combined JSON/CSV, statistics, report and manual-review list in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from video_characteristics import aggregate  # noqa: E402
from video_characteristics.datasets import DATASETS  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "video_characteristics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records: list[dict[str, object]] = []
    for name in sorted(DATASETS):
        path = args.output_dir / f"{name.lower()}_annotations.json"
        if not path.is_file():
            print(f"skip {name}: {path} not found")
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        print(f"{name}: {len(rows)} annotation(s)")
        records.extend(rows)

    if not records:
        print("no annotations found")
        return 1

    aggregate.write_json(args.output_dir / "combined_annotations.json", records)
    aggregate.write_csv(args.output_dir / "combined_annotations.csv", records)
    stats = aggregate.build_statistics(records)
    aggregate.write_json(args.output_dir / "statistics.json", stats)
    aggregate.write_report(args.output_dir / "report.md", stats, records)
    aggregate.write_manual_review_csv(
        args.output_dir / "manual_review.csv", aggregate.manual_review_rows(records)
    )
    print(f"rebuilt combined outputs for {len(records)} video(s) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

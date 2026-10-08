#!/usr/bin/env python3
"""Annotate VIRAT/MEVA videos with the AD/OD/BM/IE/DE multi-label characteristics.

Pass A shows Qwen3-VL the original video (plus the existing GT event records when the
dataset has them) and judges AD and OD. Pass B shows the already-written Stage3
heatmap + ROI overlay MP4, together with the real ROI tube table, and judges BM, IE
and DE. Per-video raw responses are cached, so rerunning resumes.

Nothing outside ``--output-dir`` is written.

Smoke test (3 VIRAT + 3 MEVA):
    scripts/annotate_video_characteristics.py --smoke

Full run:
    scripts/annotate_video_characteristics.py --datasets VIRAT MEVA
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from video_characteristics import aggregate  # noqa: E402
from video_characteristics.annotate import annotate_targets, run_settings  # noqa: E402
from video_characteristics.datasets import DATASETS, get_dataset  # noqa: E402
from video_characteristics.selection import VideoTarget, select_targets  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "video_characteristics"
SMOKE_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "video_characteristics_smoke"
SMOKE_N = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASETS), choices=sorted(DATASETS))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="annotate at most this many videos per dataset",
    )
    parser.add_argument(
        "--video-ids",
        nargs="+",
        default=None,
        help="annotate only these video ids (must still be in the selected set)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"run {SMOKE_N} videos per dataset, balanced across the two selection sets, "
            "into the separate smoke output directory"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run inference even when a cached response exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve targets and print what would run, without loading the model",
    )
    return parser.parse_args()


def smoke_subset(targets: list[VideoTarget], n: int) -> list[VideoTarget]:
    """Take videos from both selection sets so every label has a chance to fire."""
    multi = [t for t in targets if t.selection["multi_roi"]]
    single = [t for t in targets if t.selection["single_roi_area_lt_090"]]
    picked: list[VideoTarget] = []
    pools = [multi, single]
    index = 0
    while len(picked) < n and any(pools):
        pool = pools[index % len(pools)]
        if pool:
            picked.append(pool.pop(0))
        elif not any(pools):
            break
        index += 1
    return picked[:n]


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (SMOKE_OUTPUT_DIR if args.smoke else DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, list[VideoTarget]]] = []
    for name in args.datasets:
        spec = get_dataset(name)
        targets = select_targets(spec)
        if args.video_ids:
            wanted = set(args.video_ids)
            targets = [t for t in targets if t.video_id in wanted]
        elif args.smoke:
            targets = smoke_subset(targets, SMOKE_N)
        if args.limit is not None:
            targets = targets[: args.limit]
        plan.append((name, targets))
        print(f"{name}: {len(targets)} video(s) to annotate")
        for target in targets:
            sets = ",".join(k for k, v in target.selection.items() if v)
            print(
                f"  - {target.video_id} [{sets}] tubes={target.num_roi_tubes} "
                f"maxc={target.max_concurrent_rois} "
                f"raw={'ok' if target.raw_video else 'MISSING'} "
                f"overlay={'ok' if target.overlay_video else 'MISSING'}"
            )

    if args.dry_run:
        print("\ndry run: model not loaded, nothing written")
        return 0

    total = sum(len(targets) for _, targets in plan)
    if total == 0:
        print("nothing to annotate")
        return 0

    from video_characteristics.vlm import build_captioner

    print(f"\nloading Qwen3-VL for {total} video(s) x 2 passes ...", flush=True)
    captioner = build_captioner()

    all_records: list[dict[str, object]] = []
    for name, targets in plan:
        if not targets:
            continue
        spec = get_dataset(name)
        records = annotate_targets(
            captioner,
            spec,
            targets,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )
        aggregate.write_json(output_dir / f"{name.lower()}_annotations.json", records)
        all_records.extend(records)

    aggregate.write_json(output_dir / "combined_annotations.json", all_records)
    aggregate.write_csv(output_dir / "combined_annotations.csv", all_records)

    stats = aggregate.build_statistics(all_records)
    aggregate.write_json(output_dir / "statistics.json", stats)
    aggregate.write_report(output_dir / "report.md", stats, all_records)
    aggregate.write_manual_review_csv(
        output_dir / "manual_review.csv", aggregate.manual_review_rows(all_records)
    )
    aggregate.write_json(
        output_dir / "run_settings.json",
        {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datasets": args.datasets,
            "n_videos": len(all_records),
            "smoke": args.smoke,
            **run_settings(),
        },
    )

    print(f"\nwrote {len(all_records)} annotation(s) to {output_dir}")
    for key, row in stats["label_counts_overall"].items():
        print(f"  {key} ({row['name']}): {row['positive']} positive / {row['negative']} negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

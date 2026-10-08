#!/usr/bin/env python3
"""Extract the annotation target videos from the existing Stage3 ROI tube results.

Selects, per dataset, the Single-ROI videos with mean lifetime area < 0.90 and the
Multi-ROI videos with two tubes alive in the same frame, deduplicates the union, and
checks the counts against the expected values. Reads Stage3 output only; writes the
target manifest into the new output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from video_characteristics.aggregate import write_json  # noqa: E402
from video_characteristics.datasets import DATASETS, get_dataset  # noqa: E402
from video_characteristics.selection import select_targets  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "video_characteristics"

EXPECTED = {
    "VIRAT": {"single_roi_area_lt_090": 132, "multi_roi": 59, "union": 191},
    "MEVA": {"single_roi_area_lt_090": 25, "multi_roi": 63, "union": 88},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(DATASETS),
        choices=sorted(DATASETS),
        help="datasets to select from",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when a count does not match the expected value",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest: dict[str, object] = {"datasets": {}}
    mismatches: list[str] = []

    for name in args.datasets:
        spec = get_dataset(name)
        targets = select_targets(spec)
        counts = {
            "single_roi_area_lt_090": sum(
                1 for t in targets if t.selection["single_roi_area_lt_090"]
            ),
            "multi_roi": sum(1 for t in targets if t.selection["multi_roi"]),
            "both": sum(1 for t in targets if all(t.selection.values())),
            "union": len(targets),
        }
        expected = EXPECTED.get(spec.name, {})
        for key, want in expected.items():
            got = counts.get(key)
            status = "ok" if got == want else "MISMATCH"
            print(f"{spec.name} {key}: {got} (expected {want}) {status}")
            if got != want:
                mismatches.append(f"{spec.name}.{key}: got {got}, expected {want}")

        missing_raw = [t.video_id for t in targets if t.raw_video is None]
        missing_overlay = [t.video_id for t in targets if t.overlay_video is None]
        print(
            f"{spec.name} total={len(targets)} "
            f"videos_scanned={len(spec.video_ids())} "
            f"missing_raw_video={len(missing_raw)} missing_overlay={len(missing_overlay)}"
        )

        manifest["datasets"][spec.name] = {
            "video_list": str(spec.video_list),
            "tracks_root": str(spec.tracks_root),
            "overlay_root": str(spec.overlay_root),
            "gt_root": None if spec.gt_root is None else str(spec.gt_root),
            "n_videos_scanned": len(spec.video_ids()),
            "counts": counts,
            "expected": expected,
            "missing_raw_video": missing_raw,
            "missing_overlay_video": missing_overlay,
            "videos": [t.as_dict() for t in targets],
        }

    out_path = args.output_dir / "targets.json"
    write_json(out_path, manifest)
    print(f"\nwrote {out_path}")

    if mismatches:
        print("\ncount mismatches:")
        for line in mismatches:
            print(f"  - {line}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

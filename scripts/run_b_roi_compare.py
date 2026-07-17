#!/usr/bin/env python3
"""Approach B+ comparison: wider ROI + why-labels across variants.

Runs several B spatial/span settings on the 3 GT videos, scores strict HIT,
and writes context-rich overlay videos for side-by-side visual review.

Outputs under each video:
  method_B_compare/<variant>/
    temporal_groups.json
    summary.json
    overlay_context.mp4
  method_B_compare/compare_report.json
  method_B/  (best / default hop1_long copied as primary B+)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from motion_analyzer.temporal.block_peak_roi import (  # noqa: E402
    build_block_peak_rois,
    save_block_peak_events_json,
    write_context_overlay_video,
)
from motion_analyzer.temporal.frame_window_fusion import save_fused_frame_roi_json  # noqa: E402
from run_causal_roi_ab import build_windows  # noqa: E402
from run_sliding_window_regions import (  # noqa: E402
    DEFAULT_COMPONENTS_SUBDIR,
    DEFAULT_INPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_b_roi_compare")

FPS, FIRST, STEP = 30.0, 6, 6

GT_EVENTS: dict[str, list[dict[str, Any]]] = {
    "VIRAT_S_000201_06_001354_001397": [
        dict(name="alight_left_20_32", t0=20.0, t1=32.0, bbox=(0, 150, 450, 550)),
    ],
    "VIRAT_S_000201_02_000590_000623": [
        dict(name="board_15_22", t0=15.0, t1=22.0, bbox=(700, 280, 1280, 620)),
    ],
    "VIRAT_S_050000_08_001235_001295": [
        dict(name="board_top_09_26", t0=9.0, t1=26.0, bbox=(400, 0, 1100, 420)),
        dict(name="depart_37_59", t0=37.0, t1=59.0, bbox=(400, 0, 1100, 500)),
    ],
}

DEFAULT_VIDEOS = list(GT_EVENTS.keys())

# Visual/spatial variants for comparison
VARIANTS: dict[str, dict[str, Any]] = {
    "B1_seed_narrow": dict(
        spatial_mode="seed",
        neighbor_ratio=1.0,
        max_cells=1,
        span_frac=0.22,
        span_pad=0,
        min_persist=2,
    ),
    "B2_hop1": dict(
        spatial_mode="hop1",
        neighbor_ratio=0.28,
        max_cells=6,
        span_frac=0.15,
        span_pad=1,
        min_persist=2,
    ),
    "B3_hop1_long": dict(
        spatial_mode="hop1",
        neighbor_ratio=0.22,
        max_cells=8,
        span_frac=0.12,
        span_pad=2,
        min_persist=2,
    ),
    "B4_core3": dict(
        spatial_mode="core3",
        neighbor_ratio=0.0,
        max_cells=9,
        span_frac=0.15,
        span_pad=2,
        min_persist=2,
    ),
    "B5_hop1_loose": dict(
        spatial_mode="hop1",
        neighbor_ratio=0.18,
        max_cells=9,
        span_frac=0.10,
        span_pad=3,
        min_persist=2,
    ),
}

# Primary recommendation for method_B copy
PRIMARY_VARIANT = "B3_hop1_long"


def samp(t: float) -> int:
    return max(0, int(round((float(t) * FPS - FIRST) / STEP)))


def sample_to_sec(s: int) -> float:
    return (FIRST + STEP * int(s)) / FPS


def bbox_area(b: tuple[float, float, float, float] | list[float]) -> float:
    return max(0.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))


def bbox_intersect(
    a: tuple[float, float, float, float] | list[float],
    b: tuple[float, float, float, float] | list[float],
) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def time_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def eval_strict(events: list[dict[str, Any]], gt_list: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    n_hit = 0
    for gt in gt_list:
        gs0, gs1 = samp(gt["t0"]), samp(gt["t1"])
        gb = tuple(gt["bbox"])
        ga = bbox_area(gb)
        best = None
        for e in events:
            es0 = int(e["valid_start_frame"])
            es1 = int(e["valid_end_frame"])
            if not time_overlap(es0, es1, gs0, gs1):
                continue
            rb = e["rep_bbox"]
            inter = bbox_intersect(rb, gb)
            ra = bbox_area(rb)
            gt_cov = inter / max(ga, 1e-6)
            roi_in = inter / max(ra, 1e-6)
            hit = gt_cov >= 0.08 or roi_in >= 0.5
            cand = dict(
                event_id=e.get("event_id"),
                role=e.get("role"),
                seed=e.get("seed_cell"),
                n_blocks=e.get("n_rep_blocks"),
                bbox=rb,
                t_sec=[sample_to_sec(es0), sample_to_sec(es1)],
                gt_cov=round(gt_cov, 4),
                roi_in=round(roi_in, 4),
                hit=hit,
                reason=e.get("reason"),
            )
            if best is None or (hit, gt_cov + roi_in) > (
                best["hit"],
                best["gt_cov"] + best["roi_in"],
            ):
                best = cand
        if best and best["hit"]:
            n_hit += 1
        rows.append({"gt": gt["name"], "hit": bool(best and best["hit"]), "best": best})
    return {
        "n_gt": len(gt_list),
        "n_hit": n_hit,
        "all_pass": n_hit == len(gt_list),
        "events": rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--components_subdir", type=str, default=DEFAULT_COMPONENTS_SUBDIR)
    p.add_argument("--video_names", nargs="+", default=DEFAULT_VIDEOS)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    p.add_argument("--primary_variant", type=str, default=PRIMARY_VARIANT)
    p.add_argument("--max_rois", type=int, default=2)
    p.add_argument("--window_size", type=int, default=10)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max_window_candidates", type=int, default=12)
    p.add_argument("--skip_overlay", action="store_true")
    return p.parse_args()


def run_variant(
    ctx: dict[str, Any],
    *,
    variant_name: str,
    params: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fused, events, stats = build_block_peak_rois(
        ctx["windows"],
        n_frames=ctx["n_frames"],
        cfg=ctx["cfg"],
        frame_idx_map=ctx["frame_idx_map"],
        max_rois=int(args.max_rois),
        min_peak=0.05,
        top_onset_peak=0.05,
        **params,
    )
    event_dicts = [e.to_dict() for e in events]
    save_block_peak_events_json(
        out_dir / "temporal_groups.json",
        events,
        meta={
            "approach": "B_plus",
            "variant": variant_name,
            "video_name": ctx["video_name"],
            "frame_area": float(ctx["frame_w"] * ctx["frame_h"]),
            "params": params,
            **{k: stats[k] for k in ("n_raw_events", "n_final_rois", "method", "spatial_mode") if k in stats},
        },
    )
    save_fused_frame_roi_json(
        out_dir / "frame_fused_roi.json", fused, meta={"approach": "B_plus", "variant": variant_name}
    )
    gt = GT_EVENTS.get(ctx["video_name"], [])
    gt_eval = eval_strict(event_dicts, gt) if gt else {"n_gt": 0, "n_hit": 0, "all_pass": True, "events": []}
    summary: dict[str, Any] = {
        "variant": variant_name,
        "params": params,
        "n_events": len(events),
        "selected": event_dicts,
        "gt_eval": gt_eval,
        **stats,
    }
    if not args.skip_overlay:
        overlay = out_dir / "overlay_context.mp4"
        info = write_context_overlay_video(
            video_path=ctx["video_path"],
            events=events,
            frame_pairs=ctx["frame_pairs"],
            output_mp4=overlay,
            variant_name=variant_name,
            cfg=ctx["cfg"],
        )
        summary.update(info)
        summary["overlay"] = str(overlay)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def promote_primary(video_out: Path, variant_name: str) -> None:
    """Copy chosen variant into method_B as the default B+ deliverable."""
    src = video_out / "method_B_compare" / variant_name
    dst = video_out / "method_B"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("temporal_groups.json", "frame_fused_roi.json", "summary.json"):
        p = src / name
        if p.is_file():
            (dst / name).write_bytes(p.read_bytes())
    ov = src / "overlay_context.mp4"
    if ov.is_file():
        (dst / "sliding_window_overlay.mp4").write_bytes(ov.read_bytes())
        (video_out / "sliding_window_overlay_B.mp4").write_bytes(ov.read_bytes())
        (video_out / f"sliding_window_overlay_{variant_name}.mp4").write_bytes(ov.read_bytes())


def main() -> None:
    args = parse_args()
    # Reuse window builder CLI fields via a lightweight namespace
    ab_ns = argparse.Namespace(
        input_root=args.input_root,
        output_root=args.output_root,
        components_subdir=args.components_subdir,
        window_size=args.window_size,
        stride=args.stride,
        max_window_candidates=args.max_window_candidates,
        max_rois=args.max_rois,
        skip_overlay=args.skip_overlay,
        video_names=args.video_names,
        methods=["B"],
    )
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    videos = list(args.video_names)
    variant_names = [v for v in args.variants if v in VARIANTS]
    if not variant_names:
        raise SystemExit(f"No valid variants. Choose from: {list(VARIANTS)}")

    batch: list[dict[str, Any]] = []
    for name in videos:
        t0 = time.time()
        logger.info("=== %s ===", name)
        ctx = build_windows(name, input_root, ab_ns)
        video_out = output_root / name
        compare_root = video_out / "method_B_compare"
        compare_root.mkdir(parents=True, exist_ok=True)

        variant_rows: dict[str, Any] = {}
        for vn in variant_names:
            logger.info("  variant %s ...", vn)
            row = run_variant(
                ctx,
                variant_name=vn,
                params=VARIANTS[vn],
                out_dir=compare_root / vn,
                args=args,
            )
            variant_rows[vn] = {
                "all_pass": row["gt_eval"]["all_pass"],
                "n_hit": row["gt_eval"]["n_hit"],
                "n_gt": row["gt_eval"]["n_gt"],
                "overlay": row.get("overlay"),
                "gt_eval": row["gt_eval"],
                "selected": [
                    {
                        "id": e.get("event_id"),
                        "role": e.get("role"),
                        "seed": e.get("seed_cell"),
                        "n_blocks": e.get("n_rep_blocks"),
                        "bbox": e.get("rep_bbox"),
                        "t_sec": [e.get("valid_start_sec"), e.get("valid_end_sec")],
                        "reason": e.get("reason"),
                    }
                    for e in row.get("selected", [])
                ],
            }
            logger.info(
                "  %s HIT %s/%s pass=%s",
                vn,
                row["gt_eval"]["n_hit"],
                row["gt_eval"]["n_gt"],
                row["gt_eval"]["all_pass"],
            )

        # Pick primary: prefer configured if it passes; else best n_hit among variants
        primary = args.primary_variant
        if primary not in variant_rows or not variant_rows[primary]["all_pass"]:
            ranked = sorted(
                variant_rows.items(),
                key=lambda kv: (kv[1]["n_hit"], 1 if kv[0] == PRIMARY_VARIANT else 0),
                reverse=True,
            )
            primary = ranked[0][0]
        promote_primary(video_out, primary)

        report = {
            "video_name": name,
            "primary_variant": primary,
            "variants": variant_rows,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        (compare_root / "compare_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        batch.append(report)
        logger.info("Done %s primary=%s in %.1fs", name, primary, report["elapsed_sec"])

    out = output_root / "batch_summary_b_roi_compare.json"
    # Overall scoreboard
    scoreboard = []
    for vn in variant_names:
        hits = sum(r["variants"][vn]["n_hit"] for r in batch if vn in r["variants"])
        gts = sum(r["variants"][vn]["n_gt"] for r in batch if vn in r["variants"])
        scoreboard.append({"variant": vn, "hits": hits, "gts": gts, "pass_all_videos": hits == gts})
    payload = {"videos": batch, "scoreboard": scoreboard, "primary_default": PRIMARY_VARIANT}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Batch summary → %s", out)
    for s in scoreboard:
        logger.info("SCORE %s  %s/%s  all=%s", s["variant"], s["hits"], s["gts"], s["pass_all_videos"])


if __name__ == "__main__":
    main()

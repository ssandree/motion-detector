#!/usr/bin/env python3
"""Analyze existing Stage3 lifetime AABB tubes for Multi-ROI case mining.

Reads already-written ``*_roi_tracks.json`` (and Stage2 NPZ metadata for
frame size / sampling fps). Does not rerun Stage1–3, does not rewrite
pipeline outputs, and does not generate overlay video.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motion_analyzer.config import load_target_video_ids  # noqa: E402
from motion_analyzer.visualization import grid_bbox_to_pixels  # noqa: E402

DEFAULT_VIDEO_LIST = REPO_ROOT / "configs" / "virat_all.txt"
DEFAULT_TRACKS_ROOT = REPO_ROOT / "outputs" / "virat_all_light_uf"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "analysis" / "multi_roi_virat_all"


@dataclass
class LifetimeTube:
    tube_id: int
    t0: int
    t1: int
    spatial_bbox_grid: tuple[int, int, int, int]


@dataclass
class VideoIssues:
    missing_tracks: bool = False
    missing_npz: bool = False
    parse_error: str | None = None
    t_out_of_range: list[str] = field(default_factory=list)
    bbox_beyond_frame: list[str] = field(default_factory=list)
    bbox_invalid: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    def as_notes(self) -> str:
        parts: list[str] = []
        if self.missing_tracks:
            parts.append("missing_tracks_json")
        if self.missing_npz:
            parts.append("missing_fusion_npz")
        if self.parse_error:
            parts.append(f"parse_error:{self.parse_error}")
        parts.extend(self.t_out_of_range)
        parts.extend(self.bbox_beyond_frame)
        parts.extend(self.bbox_invalid)
        parts.extend(self.extra)
        return "; ".join(parts)


def _int_field(line: str, key: str) -> int | None:
    token = f'"{key}":'
    if token not in line:
        return None
    rest = line.split(token, 1)[1].strip().rstrip(",")
    return int(rest)


def load_lifetime_tubes(path: Path) -> tuple[dict[str, Any], list[LifetimeTube]]:
    """Parse pretty-printed Stage3 tracks JSON without loading per_frame/members.

    Stage3 writes identical ``tracks`` and ``tubes`` arrays; only ``tracks``
    is consumed so tubes are not double-counted.
    """
    header: dict[str, Any] = {}
    tubes: list[LifetimeTube] = []
    in_tracks = False
    skip_nested = False
    cur: dict[str, Any] = {}
    bbox_buf: list[str] | None = None

    def _flush_tube() -> bool:
        nonlocal cur
        bbox = cur.get("spatial_bbox_grid")
        if cur.get("tube_id") is None or cur.get("t0") is None or cur.get("t1") is None or bbox is None:
            cur = {}
            return False
        tubes.append(
            LifetimeTube(
                tube_id=int(cur["tube_id"]),
                t0=int(cur["t0"]),
                t1=int(cur["t1"]),
                spatial_bbox_grid=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            )
        )
        cur = {}
        expected = header.get("num_final_rois")
        return expected is not None and len(tubes) >= int(expected)

    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if skip_nested:
                if line.startswith('"tubes"'):
                    break
                if line.startswith('"tube_id"'):
                    skip_nested = False
                    cur = {"tube_id": _int_field(line, "tube_id")}
                continue

            if bbox_buf is not None:
                bbox_buf.append(line)
                if "]" in line:
                    payload = " ".join(bbox_buf)
                    if ":" in payload:
                        payload = payload.split(":", 1)[1]
                    bbox = tuple(int(v) for v in json.loads(payload.strip().rstrip(",")))
                    cur["spatial_bbox_grid"] = bbox
                    bbox_buf = None
                    if _flush_tube():
                        break
                continue

            if not in_tracks:
                if line.startswith('"video_id"'):
                    header["video_id"] = json.loads("{" + line.rstrip(",") + "}")["video_id"]
                elif line.startswith('"cell_px"'):
                    header["cell_px"] = _int_field(line, "cell_px")
                elif line.startswith('"num_frames"'):
                    header["num_frames"] = _int_field(line, "num_frames")
                elif line.startswith('"num_final_rois"'):
                    header["num_final_rois"] = _int_field(line, "num_final_rois")
                elif line.startswith('"fusion_npz"'):
                    header["fusion_npz"] = json.loads("{" + line.rstrip(",") + "}")["fusion_npz"]
                elif line.startswith('"tracks"'):
                    in_tracks = True
                continue

            if line.startswith('"tubes"'):
                break

            if line.startswith('"tube_id"'):
                cur = {"tube_id": _int_field(line, "tube_id")}
            elif line.startswith('"t0"') and cur:
                cur["t0"] = _int_field(line, "t0")
            elif line.startswith('"t1"') and cur:
                cur["t1"] = _int_field(line, "t1")
            elif line.startswith('"spatial_bbox_grid"') and cur:
                if "[" in line and "]" in line:
                    payload = line.split(":", 1)[1].strip().rstrip(",")
                    cur["spatial_bbox_grid"] = tuple(int(v) for v in json.loads(payload))
                    if _flush_tube():
                        break
                else:
                    bbox_buf = [line]
            elif line.startswith('"members"') or line.startswith('"per_frame"'):
                if cur.get("spatial_bbox_grid") is not None:
                    if _flush_tube():
                        break
                else:
                    cur = {}
                skip_nested = True

    return header, tubes


def load_npz_meta(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        fps = float(np.asarray(data["sampling_fps"]).item()) if "sampling_fps" in data.files else 5.0
        width = int(np.asarray(data["video_width"]).item()) if "video_width" in data.files else 0
        height = int(np.asarray(data["video_height"]).item()) if "video_height" in data.files else 0
        n_sampled = (
            int(np.asarray(data["num_sampled_frames"]).item())
            if "num_sampled_frames" in data.files
            else 0
        )
    return {
        "sampling_fps": fps,
        "video_width": width,
        "video_height": height,
        "num_sampled_frames": n_sampled,
    }


def clip_pixel_bbox(
    px: list[int],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in px)
    x0 = max(0, min(x0, int(frame_width)))
    y0 = max(0, min(y0, int(frame_height)))
    x1 = max(0, min(x1, int(frame_width)))
    y1 = max(0, min(y1, int(frame_height)))
    return x0, y0, x1, y1


def box_area(box: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def box_centroid(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def normalized_centroid_distance(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    frame_width: int,
    frame_height: int,
) -> float:
    cx1, cy1 = box_centroid(a)
    cx2, cy2 = box_centroid(b)
    denom = math.sqrt(float(frame_width) ** 2 + float(frame_height) ** 2)
    if denom <= 0:
        return float("nan")
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / denom


def overlap_frames(a: LifetimeTube, b: LifetimeTube) -> tuple[int, int] | None:
    t0 = max(int(a.t0), int(b.t0))
    t1 = min(int(a.t1), int(b.t1))
    if t0 <= t1:
        return t0, t1
    return None


def nanmean(values: list[float]) -> float:
    arr = [v for v in values if v is not None and math.isfinite(v)]
    if not arr:
        return float("nan")
    return float(sum(arr) / len(arr))


def percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def minmax_norm(values: list[float]) -> list[float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return [float("nan")] * len(values)
    lo, hi = min(finite), max(finite)
    out: list[float] = []
    for v in values:
        if v is None or not math.isfinite(v):
            out.append(float("nan"))
        elif hi == lo:
            out.append(0.5)
        else:
            out.append((v - lo) / (hi - lo))
    return out


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def analyze_video(
    video_id: str,
    *,
    tracks_root: Path,
    fusion_root: Path | None,
) -> dict[str, Any]:
    issues = VideoIssues()
    tracks_path = tracks_root / f"{video_id}_roi_tracks.json"
    result_path = tracks_root / f"{video_id}_result.json"

    result: dict[str, Any] = {}
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))

    if not tracks_path.is_file():
        issues.missing_tracks = True
        header: dict[str, Any] = {}
        tubes: list[LifetimeTube] = []
    else:
        try:
            header, tubes = load_lifetime_tubes(tracks_path)
        except Exception as exc:  # noqa: BLE001
            issues.parse_error = str(exc)
            header, tubes = {}, []

    npz_path = None
    if header.get("fusion_npz"):
        npz_path = Path(str(header["fusion_npz"]))
    elif result.get("fusion_npz"):
        npz_path = Path(str(result["fusion_npz"]))
    elif fusion_root is not None:
        cand = fusion_root / video_id / f"{video_id}_gap_fusion_mean.npz"
        if cand.is_file():
            npz_path = cand

    meta = {
        "sampling_fps": 5.0,
        "video_width": 0,
        "video_height": 0,
        "num_sampled_frames": 0,
    }
    if npz_path is None or not npz_path.is_file():
        issues.missing_npz = True
    else:
        meta = load_npz_meta(npz_path)

    fps = float(meta["sampling_fps"] or 5.0)
    fw = int(meta["video_width"] or 0)
    fh = int(meta["video_height"] or 0)
    num_frames = int(header.get("num_frames") or result.get("visualization_frames") or 0)
    if num_frames <= 0 and meta["num_sampled_frames"]:
        num_frames = int(meta["num_sampled_frames"])
        issues.extra.append("num_frames_from_npz")

    cell_px = int(header.get("cell_px") or result.get("cell_px") or 64)
    duration_sec = float(num_frames) / fps if fps > 0 and num_frames > 0 else float("nan")

    if header.get("num_final_rois") is not None and int(header["num_final_rois"]) != len(tubes):
        issues.extra.append(
            f"num_final_rois={header['num_final_rois']} parsed_tubes={len(tubes)}"
        )

    pixel_boxes: list[tuple[int, int, int, int]] = []
    area_ratios: list[float] = []
    lifetimes: list[float] = []
    frame_area = float(fw * fh) if fw > 0 and fh > 0 else float("nan")

    for tube in tubes:
        if tube.t0 < 0 or (num_frames > 0 and tube.t1 >= num_frames):
            issues.t_out_of_range.append(
                f"tube{tube.tube_id}_t=[{tube.t0},{tube.t1}] num_frames={num_frames}"
            )
        gx0, gy0, gx1, gy1 = tube.spatial_bbox_grid
        if gx1 <= gx0 or gy1 <= gy0:
            issues.bbox_invalid.append(f"tube{tube.tube_id}_grid={list(tube.spatial_bbox_grid)}")
        unclipped = [
            int(gx0 * cell_px),
            int(gy0 * cell_px),
            int(gx1 * cell_px),
            int(gy1 * cell_px),
        ]
        if fw > 0 and fh > 0 and (
            unclipped[0] < 0
            or unclipped[1] < 0
            or unclipped[2] > fw
            or unclipped[3] > fh
            or unclipped[0] > fw
            or unclipped[1] > fh
        ):
            issues.bbox_beyond_frame.append(
                f"tube{tube.tube_id}_unclipped_px={unclipped} frame={fw}x{fh}"
            )
        if fw > 0 and fh > 0:
            vis = grid_bbox_to_pixels(
                tube.spatial_bbox_grid,
                unit_pixel_size=cell_px,
                frame_width=fw,
                frame_height=fh,
            )
            box = clip_pixel_bbox(vis, frame_width=fw, frame_height=fh)
        else:
            box = (unclipped[0], unclipped[1], unclipped[2], unclipped[3])
        pixel_boxes.append(box)
        if box[2] <= box[0] or box[3] <= box[1]:
            issues.bbox_invalid.append(f"tube{tube.tube_id}_empty_clipped_px={list(box)}")
        if math.isfinite(frame_area) and frame_area > 0:
            ratio = box_area(box) / frame_area
            area_ratios.append(ratio)
            if ratio < 0.0 or ratio > 1.0:
                issues.extra.append(f"tube{tube.tube_id}_area_ratio={ratio:.6f}")
        lifetimes.append((int(tube.t1) - int(tube.t0) + 1) / fps if fps > 0 else float("nan"))

    active = np.zeros(max(num_frames, 0), dtype=np.int32)
    for tube in tubes:
        if num_frames <= 0:
            continue
        t0 = max(0, int(tube.t0))
        t1 = min(num_frames - 1, int(tube.t1))
        if t0 <= t1:
            active[t0 : t1 + 1] += 1

    max_concurrent = int(active.max()) if active.size else 0
    multi_frames = int(np.sum(active >= 2)) if active.size else 0
    multi_sec = float(multi_frames) / fps if fps > 0 else float("nan")
    multi_ratio = float(multi_frames) / float(num_frames) if num_frames > 0 else float("nan")
    mean_active = float(active.mean()) if active.size else 0.0
    has_concurrent = max_concurrent >= 2

    pair_seps: list[float] = []
    pair_ious: list[float] = []
    pair_weights: list[int] = []
    skipped_nonoverlap = 0
    for (i, ta), (j, tb) in combinations(enumerate(tubes), 2):
        ov = overlap_frames(ta, tb)
        if ov is None:
            skipped_nonoverlap += 1
            continue
        o0, o1 = ov
        weight = int(o1 - o0 + 1)
        sep = normalized_centroid_distance(
            pixel_boxes[i],
            pixel_boxes[j],
            frame_width=fw,
            frame_height=fh,
        )
        iou = box_iou(pixel_boxes[i], pixel_boxes[j])
        if sep < 0.0 or sep > 1.0:
            issues.extra.append(f"pair_{ta.tube_id}_{tb.tube_id}_sep={sep:.6f}")
        pair_seps.append(sep)
        pair_ious.append(iou)
        pair_weights.append(weight)

    if pair_weights:
        w = np.asarray(pair_weights, dtype=np.float64)
        mean_sep = float(np.average(np.asarray(pair_seps, dtype=np.float64), weights=w))
        mean_iou = float(np.average(np.asarray(pair_ious, dtype=np.float64), weights=w))
        max_sep = float(np.max(pair_seps))
    else:
        mean_sep = max_sep = mean_iou = float("nan")

    area_arr = np.asarray(area_ratios, dtype=np.float64) if area_ratios else np.asarray([], dtype=np.float64)
    life_arr = np.asarray(lifetimes, dtype=np.float64) if lifetimes else np.asarray([], dtype=np.float64)

    row = {
        "video_id": video_id,
        "duration_sec": duration_sec,
        "num_frames": num_frames,
        "sampling_fps": fps,
        "frame_width": fw,
        "frame_height": fh,
        "cell_px": cell_px,
        "num_roi_tubes": len(tubes),
        "max_concurrent_rois": max_concurrent,
        "has_concurrent_multi_roi": int(has_concurrent),
        "num_tubes_ge2_sequential_only": int(len(tubes) >= 2 and not has_concurrent),
        "multi_roi_duration_sec": multi_sec,
        "multi_roi_duration_ratio": multi_ratio,
        "mean_active_rois": mean_active,
        "mean_roi_area_ratio": float(np.mean(area_arr)) if area_arr.size else float("nan"),
        "median_roi_area_ratio": float(np.median(area_arr)) if area_arr.size else float("nan"),
        "min_roi_area_ratio": float(np.min(area_arr)) if area_arr.size else float("nan"),
        "max_roi_area_ratio": float(np.max(area_arr)) if area_arr.size else float("nan"),
        "mean_roi_lifetime_sec": float(np.mean(life_arr)) if life_arr.size else float("nan"),
        "max_roi_lifetime_sec": float(np.max(life_arr)) if life_arr.size else float("nan"),
        "min_roi_lifetime_sec": float(np.min(life_arr)) if life_arr.size else float("nan"),
        "num_overlapping_pairs": len(pair_weights),
        "num_nonoverlapping_pairs_skipped": skipped_nonoverlap,
        "mean_pair_separation": mean_sep,
        "max_pair_separation": max_sep,
        "mean_pair_iou": mean_iou,
        "notes": issues.as_notes(),
        "_issues": issues,
    }
    if 0.0 <= multi_ratio <= 1.0 or not math.isfinite(multi_ratio):
        pass
    else:
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + f"multi_ratio_oob={multi_ratio}"
    return row


def summarize_array(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {k: float("nan") for k in ("n", "mean", "median", "p25", "p75", "p90", "min", "max")}
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": percentile(arr, 25),
        "p75": percentile(arr, 75),
        "p90": percentile(arr, 90),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def md_stats_table(title: str, stats: dict[str, float]) -> list[str]:
    keys = ("n", "mean", "median", "p25", "p75", "p90", "min", "max")
    header = "| metric | " + " | ".join(keys) + " |"
    sep = "|---|---" + "|---" * (len(keys) - 1) + "|"
    cells = []
    for k in keys:
        v = stats[k]
        if k == "n":
            cells.append(str(int(v)) if math.isfinite(v) else "")
        else:
            cells.append(f"{v:.4f}" if math.isfinite(v) else "")
    return [f"### {title}", "", header, sep, "| value | " + " | ".join(cells) + " |", ""]


def write_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    video_ids: list[str],
    tracks_root: Path,
    output_root: Path,
    score_notes: list[str],
    sanity: list[str],
) -> None:
    n = len(rows)
    n_roi = sum(1 for r in rows if int(r["num_roi_tubes"]) > 0)
    n_ge2 = sum(1 for r in rows if int(r["num_roi_tubes"]) >= 2)
    n_conc = sum(1 for r in rows if int(r["has_concurrent_multi_roi"]) == 1)
    n_seq = sum(1 for r in rows if int(r["num_tubes_ge2_sequential_only"]) == 1)
    conc_dist = Counter(int(r["max_concurrent_rois"]) for r in rows)

    conc_rows = [r for r in rows if int(r["has_concurrent_multi_roi"]) == 1]
    ratio_all = summarize_array([float(r["multi_roi_duration_ratio"]) for r in rows])
    ratio_conc = summarize_array([float(r["multi_roi_duration_ratio"]) for r in conc_rows])
    area_all = summarize_array([float(r["mean_roi_area_ratio"]) for r in rows])
    area_conc = summarize_array([float(r["mean_roi_area_ratio"]) for r in conc_rows])
    sep_conc = summarize_array([float(r["mean_pair_separation"]) for r in conc_rows])
    iou_conc = summarize_array([float(r["mean_pair_iou"]) for r in conc_rows])
    life_all = summarize_array([float(r["mean_roi_lifetime_sec"]) for r in rows])

    lines: list[str] = []
    lines.append("# VIRAT Stage3 Multi-ROI case mining")
    lines.append("")
    lines.append(
        "기존 Stage1–3 결과를 재계산하지 않고, Stage3 `*_roi_tracks.json`의 "
        "**lifetime AABB** (`t0`,`t1`,`spatial_bbox_grid`)만 사용했다. "
        "heatmap overlay MP4 / VLM inference는 생성하지 않았다."
    )
    lines.append("")
    lines.append(f"- 실행 시각: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}`")
    lines.append(f"- video list: `{DEFAULT_VIDEO_LIST}`")
    lines.append(f"- Stage3 tracks root: `{tracks_root}`")
    lines.append(f"- output: `{output_root}`")
    lines.append("- ROI 기준: Stage3-4 fixed lifetime AABB (grid → pixel, overlay와 동일 clip)")
    lines.append(
        "- concurrent pair: lifetime 구간 `[t0, t1]`이 한 프레임 이상 겹치는 unordered pair만 "
        "centroid distance / IoU에 포함"
    )
    lines.append(
        "- pair 통계는 overlap frame 수로 weighted average "
        "(AABB가 lifetime 동안 고정이므로 pair별 기하 값은 상수)"
    )
    lines.append("")
    lines.append("## 1. 분석 규모")
    lines.append("")
    lines.append(f"1. 분석된 전체 video 수: **{n}** (목록 {len(video_ids)})")
    lines.append(f"2. ROI가 존재하는 video 수 (`num_roi_tubes > 0`): **{n_roi}**")
    lines.append(f"3. `num_roi_tubes >= 2`인 video 수: **{n_ge2}**")
    lines.append(f"4. 실제 concurrent Multi-ROI video 수 (`max_concurrent_rois >= 2`): **{n_conc}**")
    lines.append(f"   - 이 중 tubes≥2 이지만 temporal overlap이 없는 sequential-only: **{n_seq}**")
    lines.append("")
    lines.append("`num_roi_tubes >= 2`와 실제 concurrent overlap은 구분한다. sequential-only는 Multi-ROI candidate가 아니다.")
    lines.append("")
    lines.append("## 2. max_concurrent_rois 분포")
    lines.append("")
    lines.append("| max_concurrent_rois | videos |")
    lines.append("|---:|---:|")
    for k in sorted(conc_dist):
        lines.append(f"| {k} | {conc_dist[k]} |")
    lines.append("")
    lines.append("## 3. 주요 통계")
    lines.append("")
    lines.append("아래 통계는 유한 값만 집계한다. pair separation / IoU는 concurrent pair가 있는 video만 해당한다.")
    lines.append("")
    lines.extend(md_stats_table("multi_roi_duration_ratio (전체 329)", ratio_all))
    lines.extend(md_stats_table("multi_roi_duration_ratio (concurrent만)", ratio_conc))
    lines.extend(md_stats_table("mean_roi_area_ratio (ROI 있는 video)", area_all))
    lines.extend(md_stats_table("mean_roi_area_ratio (concurrent만)", area_conc))
    lines.extend(md_stats_table("mean_pair_separation (concurrent pair, duration-weighted)", sep_conc))
    lines.extend(md_stats_table("mean_pair_iou (concurrent pair, duration-weighted)", iou_conc))
    lines.extend(md_stats_table("mean_roi_lifetime_sec (ROI 있는 video)", life_all))

    lines.append("## 4. Candidate score")
    lines.append("")
    lines.append("heuristic (방법론이 아님):")
    lines.append("")
    lines.append("```")
    lines.append("candidate_score =")
    lines.append("  minmax(multi_roi_duration_ratio)")
    lines.append("  × minmax(mean_pair_separation)")
    lines.append("  × (1 - minmax(mean_roi_area_ratio))")
    lines.append("```")
    lines.append("")
    lines.append("minmax는 **전체 329 video 중 해당 feature가 유한한 값**의 min/max로 계산했다.")
    lines.append("concurrent가 없는 video는 pair_separation이 비어 있어 candidate CSV에서 제외된다.")
    lines.append("")
    for note in score_notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## 5. Top 20 candidates")
    lines.append("")
    top = candidates[:20]
    cols = [
        "rank",
        "video_id",
        "candidate_score",
        "num_roi_tubes",
        "max_concurrent_rois",
        "multi_roi_duration_sec",
        "multi_roi_duration_ratio",
        "mean_pair_separation",
        "max_pair_separation",
        "mean_pair_iou",
        "mean_roi_area_ratio",
        "mean_roi_lifetime_sec",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in top:
        cells = []
        for c in cols:
            v = r[c]
            if c in {"rank", "num_roi_tubes", "max_concurrent_rois"}:
                cells.append(str(int(v)))
            elif c == "video_id":
                cells.append(str(v))
            else:
                cells.append(fmt(v, 4))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 6. Top candidate 핵심 통계")
    lines.append("")
    for r in top:
        lines.append(
            f"- **{r['video_id']}** (rank {int(r['rank'])}, score {fmt(r['candidate_score'], 4)}): "
            f"tubes={int(r['num_roi_tubes'])}, max_conc={int(r['max_concurrent_rois'])}, "
            f"multi={fmt(r['multi_roi_duration_sec'], 2)}s "
            f"({fmt(r['multi_roi_duration_ratio'], 3)}), "
            f"sep mean/max={fmt(r['mean_pair_separation'], 3)}/{fmt(r['max_pair_separation'], 3)}, "
            f"iou={fmt(r['mean_pair_iou'], 3)}, "
            f"area mean/med={fmt(r['mean_roi_area_ratio'], 3)}/{fmt(r['median_roi_area_ratio'], 3)}, "
            f"lifetime mean={fmt(r['mean_roi_lifetime_sec'], 2)}s"
        )
    lines.append("")
    lines.append("## 7. 이상치 / 특이한 video")
    lines.append("")

    large_area = sorted(
        [r for r in rows if math.isfinite(float(r["max_roi_area_ratio"]))],
        key=lambda r: float(r["max_roi_area_ratio"]),
        reverse=True,
    )[:8]
    lines.append("### 큰 ROI (full-frame에 가까움)")
    lines.append("")
    for r in large_area:
        lines.append(
            f"- `{r['video_id']}` max_area={fmt(r['max_roi_area_ratio'], 3)} "
            f"mean_area={fmt(r['mean_roi_area_ratio'], 3)} tubes={int(r['num_roi_tubes'])} "
            f"concurrent={int(r['has_concurrent_multi_roi'])}"
        )
    lines.append("")

    seq_only = [r for r in rows if int(r["num_tubes_ge2_sequential_only"]) == 1]
    lines.append(f"### sequential-only (tubes≥2, overlap 없음): {len(seq_only)}")
    lines.append("")
    for r in seq_only[:15]:
        lines.append(
            f"- `{r['video_id']}` tubes={int(r['num_roi_tubes'])} "
            f"max_conc={int(r['max_concurrent_rois'])}"
        )
    if len(seq_only) > 15:
        lines.append(f"- … 외 {len(seq_only) - 15}개 (`multi_roi_statistics.csv` 참고)")
    lines.append("")

    high_conc = [r for r in rows if int(r["max_concurrent_rois"]) >= 3]
    lines.append(f"### max_concurrent_rois ≥ 3: {len(high_conc)}")
    lines.append("")
    for r in sorted(high_conc, key=lambda x: int(x["max_concurrent_rois"]), reverse=True):
        lines.append(
            f"- `{r['video_id']}` max_conc={int(r['max_concurrent_rois'])} "
            f"tubes={int(r['num_roi_tubes'])} multi_ratio={fmt(r['multi_roi_duration_ratio'], 3)}"
        )
    lines.append("")

    tiny_sep = sorted(
        [r for r in conc_rows if math.isfinite(float(r["mean_pair_separation"]))],
        key=lambda r: float(r["mean_pair_separation"]),
    )[:8]
    lines.append("### concurrent이지만 공간적으로 거의 붙은 pair (낮은 separation)")
    lines.append("")
    for r in tiny_sep:
        lines.append(
            f"- `{r['video_id']}` mean_sep={fmt(r['mean_pair_separation'], 4)} "
            f"mean_iou={fmt(r['mean_pair_iou'], 3)} score={fmt(r.get('candidate_score'), 4)}"
        )
    lines.append("")

    far = sorted(
        [r for r in conc_rows if math.isfinite(float(r["mean_pair_separation"]))],
        key=lambda r: float(r["mean_pair_separation"]),
        reverse=True,
    )[:8]
    lines.append("### 가장 멀리 떨어진 concurrent pair")
    lines.append("")
    for r in far:
        lines.append(
            f"- `{r['video_id']}` mean_sep={fmt(r['mean_pair_separation'], 4)} "
            f"max_sep={fmt(r['max_pair_separation'], 4)} mean_iou={fmt(r['mean_pair_iou'], 3)}"
        )
    lines.append("")

    lines.append("## 8. Stage3 data-format / missing / invalid")
    lines.append("")
    missing_tracks = [r["video_id"] for r in rows if r["_issues"].missing_tracks]
    missing_npz = [r["video_id"] for r in rows if r["_issues"].missing_npz]
    parse_err = [r for r in rows if r["_issues"].parse_error]
    beyond = [r for r in rows if r["_issues"].bbox_beyond_frame]
    invalid = [r for r in rows if r["_issues"].bbox_invalid]
    tor = [r for r in rows if r["_issues"].t_out_of_range]
    extras = [r for r in rows if r["_issues"].extra or r["_issues"].parse_error]
    lines.append(f"- statistics CSV 행 수: {n} / 목록 {len(video_ids)}")
    lines.append(f"- missing `*_roi_tracks.json`: {len(missing_tracks)}" + (f" ({', '.join(missing_tracks[:10])})" if missing_tracks else ""))
    lines.append(f"- missing fusion NPZ metadata: {len(missing_npz)}" + (f" ({', '.join(missing_npz[:10])})" if missing_npz else ""))
    lines.append(f"- tracks parse error: {len(parse_err)}")
    lines.append(
        f"- lifetime bbox가 frame을 넘는 경우 (원본 JSON은 수정하지 않음, overlay clip과 동일하게 통계만 clip): "
        f"{len(beyond)} videos. 64px unit이 1080/720에 안 나누떨어져 마지막 행/열이 잘리는 경우가 대부분이다."
    )
    if beyond:
        sample = beyond[0]
        lines.append(f"  - 예: `{sample['video_id']}` — {sample['_issues'].bbox_beyond_frame[0]}")
    lines.append(f"- invalid/empty bbox: {len(invalid)}")
    lines.append(f"- t0/t1이 num_frames 밖: {len(tor)}")
    if extras:
        lines.append(f"- 기타 notes가 있는 video: {len(extras)}")
    lines.append("")
    lines.append("원본 Stage3 JSON/NPZ/MP4는 수정하지 않았다.")
    lines.append("")
    lines.append("## 9. Sanity check")
    lines.append("")
    for item in sanity:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 10. heatmap overlay를 확인할 가치가 높은 후보")
    lines.append("")
    lines.append(
        "이미 존재하는 Stage3 overlay MP4를 볼 때, concurrent + 충분히 지속 + 공간 분리 + "
        "과대 ROI가 아닌 쪽을 score 순으로 우선한다. 아래는 Top 20 video_id이다."
    )
    lines.append("")
    for r in top:
        mp4 = tracks_root / f"{r['video_id']}_roi_tube_light_h0.2_g2t0.25_cmp1.5r3.mp4"
        exists = "exists" if mp4.is_file() else "MISSING_MP4"
        lines.append(f"- `{r['video_id']}`  ({exists})")
    lines.append("")
    if seq_only:
        lines.append(
            f"참고: sequential-only {len(seq_only)}개는 tubes가 여러 개여도 동시에 나타나지 않으므로 "
            "Multi-ROI 확인 우선순위에서 제외했다."
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


STAT_COLUMNS = [
    "video_id",
    "duration_sec",
    "num_frames",
    "sampling_fps",
    "frame_width",
    "frame_height",
    "num_roi_tubes",
    "max_concurrent_rois",
    "has_concurrent_multi_roi",
    "num_tubes_ge2_sequential_only",
    "multi_roi_duration_sec",
    "multi_roi_duration_ratio",
    "mean_active_rois",
    "mean_roi_area_ratio",
    "median_roi_area_ratio",
    "min_roi_area_ratio",
    "max_roi_area_ratio",
    "mean_roi_lifetime_sec",
    "max_roi_lifetime_sec",
    "min_roi_lifetime_sec",
    "num_overlapping_pairs",
    "num_nonoverlapping_pairs_skipped",
    "mean_pair_separation",
    "max_pair_separation",
    "mean_pair_iou",
    "norm_multi_roi_duration_ratio",
    "norm_mean_pair_separation",
    "norm_mean_roi_area_ratio",
    "candidate_score",
    "notes",
]

CAND_COLUMNS = [
    "rank",
    "video_id",
    "candidate_score",
    "num_roi_tubes",
    "max_concurrent_rois",
    "multi_roi_duration_sec",
    "multi_roi_duration_ratio",
    "mean_pair_separation",
    "max_pair_separation",
    "mean_pair_iou",
    "mean_roi_area_ratio",
    "median_roi_area_ratio",
    "mean_roi_lifetime_sec",
    "duration_sec",
    "mean_active_rois",
    "num_overlapping_pairs",
    "notes",
]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    out[col] = "" if not math.isfinite(val) else f"{val:.8f}".rstrip("0").rstrip(".")
                else:
                    out[col] = val
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_list", type=Path, default=DEFAULT_VIDEO_LIST)
    parser.add_argument("--tracks_root", type=Path, default=DEFAULT_TRACKS_ROOT)
    parser.add_argument("--fusion_root", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_ids = load_target_video_ids(args.video_list)
    tracks_root = args.tracks_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fusion_root = args.fusion_root.resolve() if args.fusion_root else None

    rows: list[dict[str, Any]] = []
    for idx, video_id in enumerate(video_ids, start=1):
        row = analyze_video(video_id, tracks_root=tracks_root, fusion_root=fusion_root)
        rows.append(row)
        if idx % 25 == 0 or idx == len(video_ids):
            print(f"[{idx}/{len(video_ids)}] {video_id} tubes={row['num_roi_tubes']} conc={row['max_concurrent_rois']}", flush=True)

    score_notes: list[str] = []
    n_ratio = minmax_norm([float(r["multi_roi_duration_ratio"]) for r in rows])
    n_sep = minmax_norm([float(r["mean_pair_separation"]) for r in rows])
    n_area = minmax_norm([float(r["mean_roi_area_ratio"]) for r in rows])

    finite_ratio = [float(r["multi_roi_duration_ratio"]) for r in rows if math.isfinite(float(r["multi_roi_duration_ratio"]))]
    finite_sep = [float(r["mean_pair_separation"]) for r in rows if math.isfinite(float(r["mean_pair_separation"]))]
    finite_area = [float(r["mean_roi_area_ratio"]) for r in rows if math.isfinite(float(r["mean_roi_area_ratio"]))]
    if finite_ratio and min(finite_ratio) == max(finite_ratio):
        score_notes.append("multi_roi_duration_ratio가 상수라 minmax가 모두 0.5가 됨 (공식 유지).")
    if finite_sep and min(finite_sep) == max(finite_sep):
        score_notes.append("mean_pair_separation이 상수라 minmax가 모두 0.5가 됨 (공식 유지).")
    if finite_area and min(finite_area) == max(finite_area):
        score_notes.append("mean_roi_area_ratio가 상수라 minmax가 모두 0.5가 됨 (공식 유지).")
    if finite_ratio:
        score_notes.append(
            f"multi_roi_duration_ratio minmax 범위 = [{min(finite_ratio):.6f}, {max(finite_ratio):.6f}] "
            f"(n={len(finite_ratio)})"
        )
    if finite_sep:
        score_notes.append(
            f"mean_pair_separation minmax 범위 = [{min(finite_sep):.6f}, {max(finite_sep):.6f}] "
            f"(n={len(finite_sep)}; concurrent 없는 video는 NaN으로 제외)"
        )
    if finite_area:
        score_notes.append(
            f"mean_roi_area_ratio minmax 범위 = [{min(finite_area):.6f}, {max(finite_area):.6f}] "
            f"(n={len(finite_area)})"
        )
    score_notes.append(
        "pair_separation minmax의 분모는 concurrent video만의 분포이므로, "
        "0 duration_ratio video가 min을 0으로 끌어내리는 것과 비대칭이다. 공식은 바꾸지 않았다."
    )

    for row, nr, ns, na in zip(rows, n_ratio, n_sep, n_area):
        row["norm_multi_roi_duration_ratio"] = nr
        row["norm_mean_pair_separation"] = ns
        row["norm_mean_roi_area_ratio"] = na
        if all(math.isfinite(v) for v in (nr, ns, na)):
            row["candidate_score"] = nr * ns * (1.0 - na)
        else:
            row["candidate_score"] = float("nan")

    candidates = [
        r for r in rows if int(r["has_concurrent_multi_roi"]) == 1 and math.isfinite(float(r["candidate_score"]))
    ]
    candidates.sort(key=lambda r: float(r["candidate_score"]), reverse=True)
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank
    zero_score = [r["video_id"] for r in candidates if float(r["candidate_score"]) == 0.0]
    if zero_score:
        score_notes.append(
            f"minmax(mean_pair_separation)의 최솟값 video는 normalized_sep=0이 되어 "
            f"candidate_score=0이 된다 ({len(zero_score)}개: {', '.join(zero_score)}). "
            "동시 ROI가 있어도 랭킹에서 바닥에 붙는다. 공식은 바꾸지 않았다."
        )

    sanity: list[str] = []
    ids = [r["video_id"] for r in rows]
    sanity.append(f"CSV video 수 {len(rows)} == 목록 {len(video_ids)}: {len(rows) == len(video_ids)}")
    sanity.append(f"목록과 CSV ID 집합 동일: {set(ids) == set(video_ids)}")
    n_zero = sum(1 for r in rows if int(r["num_roi_tubes"]) == 0)
    sanity.append(f"ROI 없는 video도 행으로 포함 (num_roi_tubes==0: {n_zero})")
    ratios = [float(r["multi_roi_duration_ratio"]) for r in rows if math.isfinite(float(r["multi_roi_duration_ratio"]))]
    sanity.append(
        f"multi_roi_duration_ratio ∈ [0,1]: {all(0.0 <= v <= 1.0 for v in ratios)} "
        f"(n={len(ratios)} min={min(ratios) if ratios else 'nan'} max={max(ratios) if ratios else 'nan'})"
    )
    areas = []
    for r in rows:
        for key in ("mean_roi_area_ratio", "min_roi_area_ratio", "max_roi_area_ratio"):
            v = float(r[key])
            if math.isfinite(v):
                areas.append(v)
    sanity.append(
        f"ROI area ratio ∈ [0,1]: {all(0.0 <= v <= 1.0 for v in areas)} "
        f"(n={len(areas)} min={min(areas) if areas else 'nan'} max={max(areas) if areas else 'nan'})"
    )
    seps = []
    for r in rows:
        for key in ("mean_pair_separation", "max_pair_separation"):
            v = float(r[key])
            if math.isfinite(v):
                seps.append(v)
    sanity.append(
        f"normalized centroid distance ∈ [0,1]: {all(0.0 <= v <= 1.0 for v in seps)} "
        f"(n={len(seps)} min={min(seps) if seps else 'nan'} max={max(seps) if seps else 'nan'})"
    )
    leak = sum(1 for r in rows if int(r["num_nonoverlapping_pairs_skipped"]) < 0)
    sanity.append(f"non-overlap pair skip count 음수 없음: {leak == 0}")
    no_pair_but_sep = [
        r["video_id"]
        for r in rows
        if int(r["num_overlapping_pairs"]) == 0 and math.isfinite(float(r["mean_pair_separation"]))
    ]
    sanity.append(f"overlap 없는 video에 pair_separation이 채워진 경우: {len(no_pair_but_sep)}")
    conc_without_pair = [
        r["video_id"]
        for r in rows
        if int(r["has_concurrent_multi_roi"]) == 1 and int(r["num_overlapping_pairs"]) == 0
    ]
    sanity.append(f"concurrent인데 overlapping pair 0개인 모순: {len(conc_without_pair)}")
    n_ge2 = sum(1 for r in rows if int(r["num_roi_tubes"]) >= 2)
    n_conc = sum(1 for r in rows if int(r["has_concurrent_multi_roi"]) == 1)
    n_seq = sum(1 for r in rows if int(r["num_tubes_ge2_sequential_only"]) == 1)
    sanity.append(f"tubes>=2 ({n_ge2}) = concurrent ({n_conc}) + sequential-only ({n_seq}): {n_ge2 == n_conc + n_seq}")
    sanity.append("bbox/frame 이상은 원본 파일을 수정하지 않고 notes/report에만 기록")

    stats_csv = output_root / "multi_roi_statistics.csv"
    cand_csv = output_root / "multi_roi_candidates.csv"
    report_md = output_root / "multi_roi_analysis.md"
    write_csv(stats_csv, rows, STAT_COLUMNS)
    write_csv(cand_csv, candidates, CAND_COLUMNS)
    write_report(
        report_md,
        rows=rows,
        candidates=candidates,
        video_ids=video_ids,
        tracks_root=tracks_root,
        output_root=output_root,
        score_notes=score_notes,
        sanity=sanity,
    )
    meta = {
        "num_videos": len(rows),
        "num_candidates": len(candidates),
        "tracks_root": str(tracks_root),
        "output_root": str(output_root),
        "statistics_csv": str(stats_csv),
        "candidates_csv": str(cand_csv),
        "report_md": str(report_md),
    }
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

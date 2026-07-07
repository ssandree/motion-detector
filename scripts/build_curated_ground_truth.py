"""Build user-curated motion/ROI ground truth (thin motion bboxes)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_virat_ground_truth import (  # noqa: E402
    ROI_MARGIN,
    TARGET_FPS,
    VIRAT_EVENT_TYPES,
    VIRAT_OBJECT_TYPES,
    _expand_bbox,
    _sample_step,
    load_object_tracks,
    native_to_sampled_frames,
)

ANNOTATION_DIR = Path(r"C:\Datasets\VIRAT\annotations")
RESULT_V2 = PROJECT_ROOT / "Result" / "motion_analyzer_v2"


def _motion_box(x1: int, y1: int, w: int, h: int) -> list[int]:
    return [x1, y1, x1 + w, y1 + h]


def _entry(frame_idx: int, ts: float, bbox: list[int]) -> dict[str, Any]:
    return {
        "frame_idx": int(frame_idx),
        "timestamp_sec": round(ts, 4),
        "bbox": bbox,
        "object_bbox": bbox,
    }


def _entry_thick_crop(
    frame_idx: int,
    ts: float,
    object_bbox: list[int],
    *,
    frame_w: int,
    frame_h: int,
    margin: float = ROI_MARGIN,
) -> dict[str, Any]:
    crop = _expand_bbox(
        object_bbox,
        frame_w=frame_w,
        frame_h=frame_h,
        margin=margin,
    )
    return {
        "frame_idx": int(frame_idx),
        "timestamp_sec": round(ts, 4),
        "bbox": crop,
        "object_bbox": object_bbox,
    }


def _seq_from_virat_object(
    objects_path: Path,
    object_id: int,
    *,
    start_frame: int,
    end_frame: int,
    native_fps: float,
    dense: bool = True,
) -> list[dict[str, Any]]:
    tracks = load_object_tracks(objects_path)
    track = tracks.get(object_id, {})
    if not track:
        return []

    if dense:
        frames = [f for f in range(start_frame, end_frame + 1) if f in track]
    else:
        frames = native_to_sampled_frames(start_frame, end_frame, native_fps)

    seq: list[dict[str, Any]] = []
    for f in frames:
        o = track[f]
        bbox = [o["x1"], o["y1"], o["x2"], o["y2"]]
        seq.append(_entry(f, f / native_fps, bbox))
    return seq


def _seq_from_track_csv(
    csv_path: Path,
    track_id: int,
    *,
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    sub = df[
        (df["track_id"] == track_id)
        & (df["timestamp_sec"] >= start_sec)
        & (df["timestamp_sec"] <= end_sec)
    ].sort_values("frame_idx")
    return [
        _entry(
            int(r.frame_idx),
            float(r.timestamp_sec),
            [int(r.x1), int(r.y1), int(r.x2), int(r.y2)],
        )
        for r in sub.itertuples(index=False)
    ]


def _moving_car_track_ids(
    csv_path: Path,
    *,
    start_sec: float,
    end_sec: float,
    min_center_x: float,
    min_displacement_px: float = 80.0,
) -> list[int]:
    df = pd.read_csv(csv_path)
    sub = df[
        (df["timestamp_sec"] >= start_sec)
        & (df["timestamp_sec"] <= end_sec)
        & (df["class_name"] == "car")
        & (df["center_x"] >= min_center_x)
    ]
    ids: list[int] = []
    for tid, grp in sub.groupby("track_id"):
        if float(grp["x1"].max() - grp["x1"].min()) >= min_displacement_px:
            ids.append(int(tid))
    return sorted(ids)


def _seq_union_right_cars(
    csv_path: Path,
    *,
    start_sec: float,
    end_sec: float,
    min_center_x: float = 1100.0,
    track_ids: list[int] | None = None,
    frame_w: int | None = None,
    frame_h: int | None = None,
    thick_crop: bool = False,
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    sub = df[
        (df["timestamp_sec"] >= start_sec)
        & (df["timestamp_sec"] <= end_sec)
        & (df["class_name"].isin(["car", "truck"]))
        & (df["center_x"] >= min_center_x)
    ]
    if track_ids is not None:
        sub = sub[sub["track_id"].isin(track_ids)]
    by_frame: dict[int, list[list[int]]] = defaultdict(list)
    ts_map: dict[int, float] = {}
    for r in sub.itertuples(index=False):
        fi = int(r.frame_idx)
        by_frame[fi].append([int(r.x1), int(r.y1), int(r.x2), int(r.y2)])
        ts_map[fi] = float(r.timestamp_sec)

    seq: list[dict[str, Any]] = []
    for fi in sorted(by_frame):
        boxes = by_frame[fi]
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        obj_bbox = [x1, y1, x2, y2]
        if thick_crop and frame_w and frame_h:
            seq.append(
                _entry_thick_crop(
                    fi,
                    ts_map[fi],
                    obj_bbox,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )
            )
        else:
            seq.append(_entry(fi, ts_map[fi], obj_bbox))
    return seq


def _seq_from_track_csv_filtered(
    csv_path: Path,
    track_id: int,
    *,
    start_sec: float,
    end_sec: float,
    class_name: str | None = None,
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    sub = df[
        (df["track_id"] == track_id)
        & (df["timestamp_sec"] >= start_sec)
        & (df["timestamp_sec"] <= end_sec)
    ]
    if class_name:
        sub = sub[sub["class_name"] == class_name]
    sub = sub.sort_values("frame_idx")
    return [
        _entry(
            int(r.frame_idx),
            float(r.timestamp_sec),
            [int(r.x1), int(r.y1), int(r.x2), int(r.y2)],
        )
        for r in sub.itertuples(index=False)
    ]


def _merge_seq_prepend(
    early: list[dict[str, Any]],
    late: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not early:
        return late
    if not late:
        return early
    cutoff = int(late[0]["frame_idx"])
    return [e for e in early if int(e["frame_idx"]) < cutoff] + late


def _seq_from_blobs(
    blobs_csv: Path,
    video_id: str,
    *,
    start_sec: float,
    end_sec: float,
    native_fps: float,
    max_area_ratio: float = 0.35,
    center_x_range: tuple[float, float] | None = None,
    center_y_range: tuple[float, float] | None = None,
    pick: str = "top_importance",
) -> list[dict[str, Any]]:
    by_frame: dict[int, list[tuple[float, list[int], float]]] = defaultdict(list)
    with blobs_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("video_name") != video_id:
                continue
            t = float(row["timestamp_sec"])
            if t < start_sec or t > end_sec:
                continue
            if float(row["bbox_area_ratio"]) > max_area_ratio:
                continue
            x1, y1, x2, y2 = map(lambda k: int(float(row[k])), ("x1", "y1", "x2", "y2"))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if center_x_range and not (center_x_range[0] <= cx <= center_x_range[1]):
                continue
            if center_y_range and not (center_y_range[0] <= cy <= center_y_range[1]):
                continue
            fi = int(row["frame_idx"])
            imp = float(row["blob_importance"])
            by_frame[fi].append((t, [x1, y1, x2, y2], imp))

    seq: list[dict[str, Any]] = []
    for fi in sorted(by_frame):
        items = by_frame[fi]
        if pick == "top_importance":
            _, bbox, _ = max(items, key=lambda x: x[2])
        else:
            t0, bbox, _ = items[0]
        seq.append(_entry(fi, fi / native_fps, bbox))
    return seq


def _merge_track(
    roi_id: int,
    label: str,
    seq: list[dict[str, Any]],
    *,
    linked_event_id: int | None = None,
    source: str = "curated",
    notes: str = "",
    roi_display: str = "thin",
) -> dict[str, Any]:
    if not seq:
        return {}
    seq = sorted(seq, key=lambda e: e["frame_idx"])
    return {
        "roi_id": roi_id,
        "label": label,
        "linked_event_id": linked_event_id,
        "source": source,
        "roi_display": roi_display,
        "notes": notes,
        "start_frame_idx": seq[0]["frame_idx"],
        "end_frame_idx": seq[-1]["frame_idx"],
        "start_timestamp_sec": seq[0]["timestamp_sec"],
        "end_timestamp_sec": seq[-1]["timestamp_sec"],
        "duration_sec": round(seq[-1]["timestamp_sec"] - seq[0]["timestamp_sec"], 4),
        "bbox_sequence_len": len(seq),
        "bbox_sequence": seq,
    }


def build_000201() -> dict[str, Any]:
    vid = "VIRAT_S_000201_06_001354_001397"
    fps = 30.0
    fw, fh = 1280, 720
    ann = ANNOTATION_DIR / f"{vid}.viratdata.objects.txt"
    tracks_csv = RESULT_V2 / vid / "_data" / "object_tracks.csv"
    # Person only, exit event ends at frame 701 (~23.37s). No ROI after exit.
    seq = _seq_from_virat_object(
        ann, 1, start_frame=449, end_frame=701, native_fps=fps, dense=True
    )
    moving_right_ids = _moving_car_track_ids(
        tracks_csv, start_sec=16.0, end_sec=42.6, min_center_x=800
    )
    right_pass_thick = _seq_union_right_cars(
        tracks_csv,
        start_sec=16.0,
        end_sec=42.6,
        min_center_x=800,
        track_ids=moving_right_ids,
        frame_w=fw,
        frame_h=fh,
        thick_crop=True,
    )
    return {
        "video_id": vid,
        "native_fps": fps,
        "target_fps": TARGET_FPS,
        "frame_size": [fw, fh],
        "motion_type": "event_motion",
        "source": "curated_virat_annotation",
        "display_style": "mixed",
        "notes": [
            "Person (obj 1) getting out of vehicle only; ROI ends when VIRAT event ends (~23.4s).",
            "No tracking after exit — stationary period is intentionally excluded.",
            "Right-side passing cars: thick crop ROI (union of moving tracks "
            f"{moving_right_ids}).",
        ],
        "expected_motions": [
            {
                "event_id": 0,
                "virat_event_type_id": 6,
                "virat_event_type": "Person getting out of a Vehicle",
                "start_frame_native": 449,
                "end_frame_native": 701,
                "start_timestamp_sec": 14.9667,
                "end_timestamp_sec": 23.3667,
                "duration_sec": 8.4,
                "participant_object_ids": [1],
                "participant_object_types": ["person"],
            },
            {
                "event_id": 1,
                "virat_event_type": "Right-side passing cars",
                "start_timestamp_sec": 16.0,
                "end_timestamp_sec": 42.6,
                "track_ids": moving_right_ids,
            },
        ],
        "roi_tracks": [
            _merge_track(
                1,
                "person_exit_vehicle",
                seq,
                linked_event_id=0,
                roi_display="thin",
                notes="Thin motion bbox on person only (obj 1).",
            ),
            _merge_track(
                2,
                "right_passing_cars",
                right_pass_thick,
                linked_event_id=1,
                roi_display="thick",
                notes="Thick crop ROI for right-side passing cars.",
            ),
        ],
    }


def build_010207() -> dict[str, Any]:
    vid = "VIRAT_S_010207_08_001308_001332"
    fps = 23.97
    ann = ANNOTATION_DIR / f"{vid}.viratdata.objects.txt"
    # Person emerging from left bushes/trees and walking — VIRAT obj 0.
    seq = _seq_from_virat_object(
        ann, 0, start_frame=185, end_frame=428, native_fps=fps, dense=True
    )
    return {
        "video_id": vid,
        "native_fps": fps,
        "target_fps": TARGET_FPS,
        "frame_size": [1280, 720],
        "motion_type": "event_motion",
        "source": "curated_virat_annotation",
        "display_style": "motion_bbox_thin",
        "notes": [
            "Single ROI: person walking out from left bushes (VIRAT object 0).",
            "Previous ROI 1/2/3 (facility exit / carrying) removed per review.",
        ],
        "expected_motions": [
            {
                "event_id": 0,
                "virat_event_type": "Person walking from left bushes",
                "start_frame_native": 185,
                "end_frame_native": 428,
                "start_timestamp_sec": round(185 / fps, 4),
                "end_timestamp_sec": round(428 / fps, 4),
                "participant_object_ids": [0],
                "participant_object_types": ["person"],
            }
        ],
        "roi_tracks": [
            _merge_track(
                1,
                "person_from_left_bushes",
                seq,
                linked_event_id=0,
            )
        ],
    }


def build_050201() -> dict[str, Any]:
    vid = "VIRAT_S_050201_06_001168_001240"
    fps = 29.97002997002997
    blobs = RESULT_V2 / vid / "_data" / "motion_blobs.csv"
    tracks_csv = RESULT_V2 / vid / "_data" / "object_tracks.csv"

    top_left = _seq_from_blobs(
        blobs,
        vid,
        start_sec=22.0,
        end_sec=28.0,
        native_fps=fps,
        max_area_ratio=0.3,
        center_x_range=(0, 700),
        center_y_range=(0, 400),
    )
    moto_det = _seq_from_track_csv(tracks_csv, 197, start_sec=35.8, end_sec=38.5)
    moto_blob = _seq_from_blobs(
        blobs,
        vid,
        start_sec=32.0,
        end_sec=48.0,
        native_fps=fps,
        max_area_ratio=0.25,
        center_x_range=(800, 1300),
        center_y_range=(200, 650),
    )
    # Prefer detector boxes when available; fill gaps with blob boxes.
    moto_by_frame = {e["frame_idx"]: e for e in moto_blob}
    for e in moto_det:
        moto_by_frame[e["frame_idx"]] = e
    moto_seq = sorted(moto_by_frame.values(), key=lambda e: e["frame_idx"])

    return {
        "video_id": vid,
        "native_fps": fps,
        "target_fps": TARGET_FPS,
        "frame_size": [1920, 1080],
        "motion_type": "event_motion",
        "source": "curated_motion_pipeline",
        "display_style": "motion_bbox_thin",
        "notes": [
            "No ROI before 20s.",
            "22–28s: top-left motion only.",
            "32–48s: motorcycle motion only (ByteTrack 197 + motion blobs).",
            "No ROI after 48s.",
        ],
        "expected_motions": [
            {
                "event_id": 0,
                "virat_event_type": "top_left_motion",
                "start_timestamp_sec": 22.0,
                "end_timestamp_sec": 28.0,
            },
            {
                "event_id": 1,
                "virat_event_type": "motorcycle_motion",
                "start_timestamp_sec": 32.0,
                "end_timestamp_sec": 48.0,
            },
        ],
        "roi_tracks": [
            t
            for t in [
                _merge_track(1, "top_left_motion", top_left, linked_event_id=0),
                _merge_track(2, "motorcycle_motion", moto_seq, linked_event_id=1),
            ]
            if t
        ],
    }


def build_050300() -> dict[str, Any]:
    vid = "VIRAT_S_050300_07_001623_001690"
    fps = 29.97002997002997
    ann = ANNOTATION_DIR / f"{vid}.viratdata.objects.txt"
    tracks_csv = RESULT_V2 / vid / "_data" / "object_tracks.csv"

    right_pass = _seq_union_right_cars(tracks_csv, start_sec=0.0, end_sec=66.8, min_center_x=1100)
    person_entry = _seq_from_virat_object(
        ann, 122, start_frame=int(21.0 * fps), end_frame=819, native_fps=fps, dense=True
    )
    # Bottom red car entry 24–33s: person 122 + car 60 union per frame
    p122 = load_object_tracks(ann).get(122, {})
    c60 = load_object_tracks(ann).get(60, {})
    bottom_seq: list[dict[str, Any]] = []
    for f in range(int(24 * fps), int(33 * fps) + 1):
        boxes = []
        if f in p122:
            o = p122[f]
            boxes.append([o["x1"], o["y1"], o["x2"], o["y2"]])
        if f in c60:
            o = c60[f]
            boxes.append([o["x1"], o["y1"], o["x2"], o["y2"]])
        if not boxes:
            continue
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        bottom_seq.append(_entry(f, f / fps, [x1, y1, x2, y2]))

    mid_car = _seq_from_track_csv(tracks_csv, 11, start_sec=28.0, end_sec=57.0)
    early_left_car = _seq_from_virat_object(
        ann, 59, start_frame=int(28.0 * fps), end_frame=int(31.6 * fps), native_fps=fps, dense=True
    )
    late_left_car = _seq_from_track_csv_filtered(
        tracks_csv, 635, start_sec=31.6, end_sec=66.6
    )
    left_depart = _merge_seq_prepend(early_left_car, late_left_car)

    return {
        "video_id": vid,
        "native_fps": fps,
        "target_fps": TARGET_FPS,
        "frame_size": [1920, 1080],
        "motion_type": "event_motion",
        "source": "curated_mixed",
        "display_style": "motion_bbox_thin",
        "notes": [
            "ROI1: right-side passing cars (union) for full clip.",
            "ROI2: person getting into car from 21s (obj 122).",
            "ROI3: bottom red car + person entry 24–33s (obj 122 + car 60).",
            "ROI4: middle-left red car motion 28–57s (track 11).",
            "ROI5: left-departing red car (VIRAT obj 59 from 28s + ByteTrack 635).",
        ],
        "expected_motions": [
            {
                "event_id": 0,
                "virat_event_type": "Right-side passing cars",
                "start_timestamp_sec": 0.0,
                "end_timestamp_sec": 66.8,
            },
            {
                "event_id": 1,
                "virat_event_type_id": 5,
                "virat_event_type": "Person getting into a Vehicle",
                "start_timestamp_sec": 21.0,
                "end_timestamp_sec": 27.3,
                "participant_object_ids": [122],
            },
            {
                "event_id": 2,
                "virat_event_type": "Person walks to bottom red car and boards",
                "start_timestamp_sec": 24.0,
                "end_timestamp_sec": 33.0,
            },
            {
                "event_id": 3,
                "virat_event_type": "Middle-left red car motion",
                "start_timestamp_sec": 28.0,
                "end_timestamp_sec": 57.0,
            },
            {
                "event_id": 4,
                "virat_event_type": "Left-departing red car",
                "start_timestamp_sec": 28.0,
                "end_timestamp_sec": 66.6,
                "participant_object_ids": [59],
                "track_id": 635,
            },
        ],
        "roi_tracks": [
            t
            for t in [
                _merge_track(1, "right_passing_cars", right_pass, linked_event_id=0),
                _merge_track(
                    2,
                    "person_getting_into_car",
                    person_entry,
                    linked_event_id=1,
                ),
                _merge_track(3, "bottom_red_car_entry", bottom_seq, linked_event_id=2),
                _merge_track(4, "middle_left_red_car", mid_car, linked_event_id=3),
                _merge_track(
                    5,
                    "left_departing_red_car",
                    left_depart,
                    linked_event_id=4,
                ),
            ]
            if t
        ],
    }


BUILDERS = {
    "VIRAT_S_000201_06_001354_001397": build_000201,
    "VIRAT_S_010207_08_001308_001332": build_010207,
    "VIRAT_S_050201_06_001168_001240": build_050201,
    "VIRAT_S_050300_07_001623_001690": build_050300,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build curated ground-truth JSON.")
    parser.add_argument("--out-dir", type=Path, default=Path("ground_truth"))
    parser.add_argument("--video", action="append", default=[])
    args = parser.parse_args()

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = args.video or list(BUILDERS.keys())
    for vid in selected:
        gt = BUILDERS[vid]()
        out_path = out_dir / f"{vid}.ground_truth.json"
        out_path.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{vid}: {len(gt['roi_tracks'])} motion ROIs -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

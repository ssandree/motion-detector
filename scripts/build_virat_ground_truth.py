"""Build motion/ROI ground truth from VIRAT annotations or motion pipeline inference."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

VIRAT_EVENT_TYPES: dict[int, str] = {
    1: "Person loading an Object to a Vehicle",
    2: "Person Unloading an Object from a Car/Vehicle",
    3: "Person Opening a Vehicle/Car Trunk",
    4: "Person Closing a Vehicle/Car Trunk",
    5: "Person getting into a Vehicle",
    6: "Person getting out of a Vehicle",
    7: "Person gesturing",
    8: "Person digging",
    9: "Person carrying an object",
    10: "Person running",
    11: "Person entering a facility",
    12: "Person exiting a facility",
}

VIRAT_OBJECT_TYPES: dict[int, str] = {
    1: "person",
    2: "car",
    3: "vehicles",
    4: "object",
    5: "bike",
}

TARGET_FPS = 5.0
ROI_MARGIN = 0.15


def _sample_step(native_fps: float) -> int:
    return max(1, round(native_fps / TARGET_FPS))


def native_to_sampled_frames(
    start_native: int,
    end_native: int,
    native_fps: float,
) -> list[int]:
    step = _sample_step(native_fps)
    frames: list[int] = []
    fi = step
    while fi <= end_native:
        if fi >= start_native:
            frames.append(fi)
        fi += step
    return frames


def _union_bbox(boxes: list[tuple[int, int, int, int]]) -> list[int] | None:
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


def _expand_bbox(
    bbox: list[int],
    *,
    frame_w: int,
    frame_h: int,
    margin: float = ROI_MARGIN,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    mx = int(w * margin)
    my = int(h * margin)
    return [
        max(0, x1 - mx),
        max(0, y1 - my),
        min(frame_w, x2 + mx),
        min(frame_h, y2 + my),
    ]


def parse_mapping_line(line: str) -> dict:
    parts = line.split()
    event_id = int(parts[0])
    event_type = int(parts[1])
    duration = int(parts[2])
    start_frame = int(parts[3])
    end_frame = int(parts[4])
    num_objects = int(parts[5])
    flags = [int(x) for x in parts[6:]]
    participant_object_ids = [oid for oid, flag in enumerate(flags) if flag == 1]
    return {
        "event_id": event_id,
        "virat_event_type_id": event_type,
        "virat_event_type": VIRAT_EVENT_TYPES.get(event_type, f"unknown_{event_type}"),
        "duration_frames_native": duration,
        "start_frame_native": start_frame,
        "end_frame_native": end_frame,
        "num_objects": num_objects,
        "participant_object_ids": participant_object_ids,
    }


def load_object_tracks(objects_path: Path) -> dict[int, dict[int, dict]]:
    """object_id -> native_frame -> bbox dict."""
    tracks: dict[int, dict[int, dict]] = defaultdict(dict)
    with objects_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            oid = int(parts[0])
            frame = int(parts[2])
            x, y, w, h = map(int, parts[3:7])
            obj_type = int(parts[7])
            tracks[oid][frame] = {
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "object_type_id": obj_type,
                "object_type": VIRAT_OBJECT_TYPES.get(obj_type, f"type_{obj_type}"),
            }
    return tracks


def nearest_bbox(track: dict[int, dict], frame: int, *, native_fps: float) -> dict | None:
    if frame in track:
        return track[frame]
    if not track:
        return None
    nearest = min(track.keys(), key=lambda f: abs(f - frame))
    if abs(nearest - frame) > _sample_step(native_fps):
        return None
    return track[nearest]


def build_roi_from_objects(
    event: dict,
    object_tracks: dict[int, dict[int, dict]],
    *,
    native_fps: float,
    frame_w: int,
    frame_h: int,
    roi_id: int,
) -> dict:
    sampled_frames = native_to_sampled_frames(
        event["start_frame_native"],
        event["end_frame_native"],
        native_fps,
    )
    bbox_sequence: list[dict] = []
    for fi in sampled_frames:
        boxes: list[tuple[int, int, int, int]] = []
        for oid in event["participant_object_ids"]:
            track = object_tracks.get(oid, {})
            obs = nearest_bbox(track, fi, native_fps=native_fps)
            if obs is None:
                continue
            boxes.append((obs["x1"], obs["y1"], obs["x2"], obs["y2"]))
        union = _union_bbox(boxes)
        if union is None:
            continue
        crop = _expand_bbox(union, frame_w=frame_w, frame_h=frame_h)
        bbox_sequence.append(
            {
                "frame_idx": fi,
                "timestamp_sec": round(fi / native_fps, 4),
                "bbox": crop,
                "object_bbox": union,
            }
        )
    start_sec = event["start_frame_native"] / native_fps
    end_sec = event["end_frame_native"] / native_fps
    participant_types = []
    for oid in event["participant_object_ids"]:
        track = object_tracks.get(oid, {})
        if track:
            any_frame = next(iter(track.values()))
            participant_types.append(any_frame["object_type"])
    return {
        "roi_id": roi_id,
        "linked_event_id": event["event_id"],
        "source": "virat_annotation",
        "participant_object_ids": event["participant_object_ids"],
        "participant_object_types": sorted(set(participant_types)),
        "start_frame_idx": sampled_frames[0] if sampled_frames else event["start_frame_native"],
        "end_frame_idx": sampled_frames[-1] if sampled_frames else event["end_frame_native"],
        "start_timestamp_sec": round(start_sec, 4),
        "end_timestamp_sec": round(end_sec, 4),
        "duration_sec": round(end_sec - start_sec, 4),
        "bbox_sequence_len": len(bbox_sequence),
        "bbox_sequence": bbox_sequence,
    }


def build_from_virat_annotation(
    video_id: str,
    annotation_dir: Path,
    *,
    native_fps: float,
    frame_w: int,
    frame_h: int,
) -> dict:
    mapping_path = annotation_dir / f"{video_id}.viratdata.mapping.txt"
    objects_path = annotation_dir / f"{video_id}.viratdata.objects.txt"
    events = [parse_mapping_line(line) for line in mapping_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    object_tracks = load_object_tracks(objects_path)

    for ev in events:
        ev["start_timestamp_sec"] = round(ev["start_frame_native"] / native_fps, 4)
        ev["end_timestamp_sec"] = round(ev["end_frame_native"] / native_fps, 4)
        ev["duration_sec"] = round(
            (ev["end_frame_native"] - ev["start_frame_native"]) / native_fps, 4
        )
        ev["participant_object_types"] = sorted(
            {
                object_tracks[oid][next(iter(object_tracks[oid]))]["object_type"]
                for oid in ev["participant_object_ids"]
                if oid in object_tracks and object_tracks[oid]
            }
        )

    roi_tracks = [
        build_roi_from_objects(
            ev,
            object_tracks,
            native_fps=native_fps,
            frame_w=frame_w,
            frame_h=frame_h,
            roi_id=i,
        )
        for i, ev in enumerate(events, start=1)
    ]

    return {
        "video_id": video_id,
        "native_fps": native_fps,
        "target_fps": TARGET_FPS,
        "frame_size": [frame_w, frame_h],
        "motion_type": "event_motion" if events else "background_motion",
        "source": "virat_annotation",
        "expected_motions": events,
        "roi_tracks": roi_tracks,
    }


def _load_motion_blobs(blobs_csv: Path, video_id: str) -> list[dict]:
    rows: list[dict] = []
    with blobs_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("video_name") != video_id:
                continue
            rows.append(
                {
                    "frame_idx": int(row["frame_idx"]),
                    "timestamp_sec": float(row["timestamp_sec"]),
                    "x1": int(float(row["x1"])),
                    "y1": int(float(row["y1"])),
                    "x2": int(float(row["x2"])),
                    "y2": int(float(row["y2"])),
                    "bbox_area_ratio": float(row["bbox_area_ratio"]),
                    "blob_importance": float(row["blob_importance"]),
                    "persistence_track_id": int(row["persistence_track_id"]),
                }
            )
    return rows


def _load_fallback_tubes(tubes_csv: Path) -> list[dict]:
    tubes: list[dict] = []
    with tubes_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tubes.append(
                {
                    "tube_id": int(row["motion_tube_id"]),
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame"]),
                    "start_time_sec": float(row["start_time_sec"]),
                    "end_time_sec": float(row["end_time_sec"]),
                    "duration_sec": float(row["duration_sec"]),
                    "tube_importance": float(row["tube_importance"]),
                }
            )
    return tubes


def build_inferred_ground_truth(
    video_id: str,
    *,
    native_fps: float,
    frame_w: int,
    frame_h: int,
    result_dir: Path,
    min_importance: float = 0.45,
    min_duration_sec: float = 4.0,
    max_rois: int = 3,
) -> dict:
    blobs = _load_motion_blobs(result_dir / "_data" / "motion_blobs.csv", video_id)
    tubes = _load_fallback_tubes(result_dir / "_data" / "motion_fallback_tubes.csv")
    tubes = sorted(
        [t for t in tubes if t["tube_importance"] >= min_importance and t["duration_sec"] >= min_duration_sec],
        key=lambda t: (-t["tube_importance"], -t["duration_sec"]),
    )[:max_rois]

    expected_motions: list[dict] = []
    roi_tracks: list[dict] = []

    for i, tube in enumerate(tubes, start=1):
        motion = {
            "event_id": i - 1,
            "virat_event_type_id": None,
            "virat_event_type": "inferred_local_motion",
            "inference_source": "motion_fallback_tube",
            "motion_tube_id": tube["tube_id"],
            "start_frame_native": tube["start_frame"],
            "end_frame_native": tube["end_frame"],
            "start_timestamp_sec": round(tube["start_time_sec"], 4),
            "end_timestamp_sec": round(tube["end_time_sec"], 4),
            "duration_sec": round(tube["duration_sec"], 4),
            "tube_importance": round(tube["tube_importance"], 6),
            "participant_object_ids": [],
            "participant_object_types": [],
            "note": "No VIRAT event annotation; inferred from motion fallback tube.",
        }
        expected_motions.append(motion)

        sampled_frames = native_to_sampled_frames(
            tube["start_frame"], tube["end_frame"], native_fps
        )
        blobs_by_frame: dict[int, list[dict]] = defaultdict(list)
        for b in blobs:
            if tube["start_frame"] <= b["frame_idx"] <= tube["end_frame"]:
                if b["bbox_area_ratio"] < 0.45:
                    blobs_by_frame[b["frame_idx"]].append(b)

        bbox_sequence: list[dict] = []
        for fi in sampled_frames:
            candidates = blobs_by_frame.get(fi, [])
            if not candidates:
                # nearest sampled blob frame within one step
                step = _sample_step(native_fps)
                near = [
                    b
                    for b in blobs
                    if tube["start_frame"] <= b["frame_idx"] <= tube["end_frame"]
                    and abs(b["frame_idx"] - fi) <= step
                    and b["bbox_area_ratio"] < 0.45
                ]
                candidates = near
            if not candidates:
                continue
            best = max(candidates, key=lambda b: b["blob_importance"])
            union = [best["x1"], best["y1"], best["x2"], best["y2"]]
            crop = _expand_bbox(union, frame_w=frame_w, frame_h=frame_h)
            bbox_sequence.append(
                {
                    "frame_idx": fi,
                    "timestamp_sec": round(fi / native_fps, 4),
                    "bbox": crop,
                    "object_bbox": union,
                }
            )

        roi_tracks.append(
            {
                "roi_id": i,
                "linked_event_id": i - 1,
                "source": "motion_pipeline_inferred",
                "motion_tube_id": tube["tube_id"],
                "participant_object_ids": [],
                "participant_object_types": [],
                "start_frame_idx": sampled_frames[0] if sampled_frames else tube["start_frame"],
                "end_frame_idx": sampled_frames[-1] if sampled_frames else tube["end_frame"],
                "start_timestamp_sec": round(tube["start_time_sec"], 4),
                "end_timestamp_sec": round(tube["end_time_sec"], 4),
                "duration_sec": round(tube["duration_sec"], 4),
                "bbox_sequence_len": len(bbox_sequence),
                "bbox_sequence": bbox_sequence,
            }
        )

    return {
        "video_id": video_id,
        "native_fps": native_fps,
        "target_fps": TARGET_FPS,
        "frame_size": [frame_w, frame_h],
        "motion_type": "event_motion" if expected_motions else "background_motion",
        "source": "motion_pipeline_inferred",
        "expected_motions": expected_motions,
        "roi_tracks": roi_tracks,
        "notes": [
            "VIRAT annotation is not available for this clip.",
            "Motions/ROIs are inferred from motion_fallback_tubes + motion_blobs at 5fps sampling.",
        ],
    }


VIDEO_SPECS: dict[str, dict] = {
    "VIRAT_S_010207_08_001308_001332": {
        "native_fps": 23.97,
        "frame_w": 1280,
        "frame_h": 720,
        "annotated": True,
    },
    "VIRAT_S_050300_07_001623_001690": {
        "native_fps": 29.97002997002997,
        "frame_w": 1920,
        "frame_h": 1080,
        "annotated": True,
    },
    "VIRAT_S_000201_06_001354_001397": {
        "native_fps": 30.0,
        "frame_w": 1280,
        "frame_h": 720,
        "annotated": True,
    },
    "VIRAT_S_050201_06_001168_001240": {
        "native_fps": 29.97002997002997,
        "frame_w": 1920,
        "frame_h": 1080,
        "annotated": False,
        "result_dir": Path("Result/motion_analyzer_v2/VIRAT_S_050201_06_001168_001240"),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build VIRAT motion/ROI ground truth JSON.")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path(r"C:\Datasets\VIRAT\annotations"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ground_truth"),
    )
    parser.add_argument("--video", action="append", default=[])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = args.video or list(VIDEO_SPECS.keys())
    for video_id in selected:
        spec = VIDEO_SPECS[video_id]
        if spec.get("annotated", True):
            gt = build_from_virat_annotation(
                video_id,
                args.annotation_dir,
                native_fps=spec["native_fps"],
                frame_w=spec["frame_w"],
                frame_h=spec["frame_h"],
            )
        else:
            result_dir = repo_root / spec["result_dir"]
            gt = build_inferred_ground_truth(
                video_id,
                native_fps=spec["native_fps"],
                frame_w=spec["frame_w"],
                frame_h=spec["frame_h"],
                result_dir=result_dir,
            )
        out_path = out_dir / f"{video_id}.ground_truth.json"
        out_path.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"{video_id}: motion_type={gt['motion_type']}, "
            f"motions={len(gt['expected_motions'])}, rois={len(gt['roi_tracks'])} -> {out_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

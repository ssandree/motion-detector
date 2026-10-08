#!/usr/bin/env python3
"""Rebuild VLM ``metadata.json`` for all sampling methods under /data/datasets.

Applies the shared schema from ``motion_analyzer.sampling_metadata``:
  - proposed_ROI_w_fps / single_motion_ROI_w_fps: global + rois
  - fps_sampling_frames: global + rois=[]
  - proposed_ROI / proposed_ROI_margin / single_motion_ROI: global=null + rois
  - no empty ROI stubs
  - time_range / num_frames from sampled frames
  - source_region from 3×3 overlap coverage
  - single_motion normalized_bbox = union of all motion ROIs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from motion_analyzer.sampling_metadata import (  # noqa: E402
    RoiMetaInput,
    SavedFrameRef,
    build_sampling_metadata,
    normalize_bbox,
    source_region_from_normalized_bbox,
    union_normalized_bboxes,
    validate_sampling_metadata,
    write_sampling_metadata,
)
from motion_analyzer.visualization import grid_bbox_to_pixels  # noqa: E402

DATA = Path("/data/datasets")
GLOBAL_RE = re.compile(r"^global_frame_(?P<seq>\d+)_t(?P<time>\d+(?:\.\d+)?)$")
TUBE_RE = re.compile(r"^tube(?P<tube>\d+)_frame_(?P<seq>\d+)_t(?P<time>\d+(?:\.\d+)?)$")

TRACKS = {
    "gaptype4": REPO_ROOT / "outputs" / "stage3" / "3_roi_tube" / "gaptype4_9vid",
    "gaptype5": REPO_ROOT / "outputs" / "stage3" / "3_roi_tube" / "vlm_9vid_tracks",
}
VIDEO_ROOTS = [
    Path("/data/datasets/VIRAT/videos-00"),
    Path("/data/datasets/VIRAT/videos-01"),
    Path("/data/datasets/VIRAT/videos-04"),
    Path("/data/datasets/VIRAT/videos-05"),
    Path("/data/datasets/VIRAT"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def move_path(src: Path, dst: Path, *, dry_run: bool) -> None:
    if not src.exists() or dst.exists():
        return
    print(f"  MOVE {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


def promote_single_motion_to_gaptype5(*, dry_run: bool) -> None:
    for root_name in ("single_motion_ROI", "single_motion_ROI_w_fps"):
        root = DATA / root_name
        if not root.is_dir():
            continue
        g5 = root / "gaptype5"
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.startswith("VIRAT_"):
                move_path(child, g5 / child.name, dry_run=dry_run)


def resolve_video_size(video_id: str) -> tuple[int, int]:
    for root in VIDEO_ROOTS:
        path = root / f"{video_id}.mp4"
        if not path.is_file():
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if w > 0 and h > 0:
            return w, h
    return 1280, 720


def load_tracks(gaptype: str, video_id: str) -> dict | None:
    root = TRACKS.get(gaptype)
    if root is None:
        return None
    path = root / f"{video_id}_roi_tracks.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tube_norm_bboxes(
    tracks: dict | None,
    *,
    frame_width: int,
    frame_height: int,
) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    if not tracks:
        return out
    cell_px = int(tracks.get("cell_px") or 64)
    fw = frame_width
    fh = frame_height
    for tube in tracks.get("tracks") or tracks.get("tubes") or []:
        tid = int(tube.get("tube_id") or 0)
        bbox_grid = tube.get("spatial_bbox_grid")
        if tid <= 0 or not bbox_grid:
            continue
        px = grid_bbox_to_pixels(
            tuple(int(v) for v in bbox_grid),
            unit_pixel_size=cell_px,
            frame_width=fw,
            frame_height=fh,
        )
        out[tid] = normalize_bbox(px, frame_width=fw, frame_height=fh)
    return out


def list_jpgs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.jpg") if p.is_file())


def method_kind(video_dir: Path) -> str:
    """Classify leaf folder by path."""
    parts = video_dir.parts
    joined = "/".join(parts)
    if "fps_sampling_frames" in parts:
        return "fps_only"
    if "proposed_ROI_margin" in parts:
        return "roi_only"
    if "single_motion_ROI_w_fps" in parts:
        return "single_motion_w_fps"
    if "single_motion_ROI" in parts:
        return "single_motion_roi"
    if "proposed_ROI_w_fps" in parts:
        return "proposed_w_fps"
    if "proposed_ROI" in parts:
        return "roi_only"
    return "unknown"


def gaptype_of(video_dir: Path) -> str:
    for part in video_dir.parts:
        if part.startswith("gaptype"):
            return part
    return "gaptype5"


def discover_leaves() -> list[Path]:
    leaves: list[Path] = []
    seen: set[Path] = set()

    def maybe_add(path: Path) -> None:
        if path in seen or not path.is_dir():
            return
        if not path.name.startswith("VIRAT_"):
            return
        if list_jpgs(path) or (path / "metadata.json").is_file():
            seen.add(path)
            leaves.append(path)

    for gap in ("gaptype4", "gaptype5"):
        for root_name in (
            "proposed_ROI",
            "proposed_ROI_w_fps",
            "single_motion_ROI",
            "single_motion_ROI_w_fps",
        ):
            base = DATA / root_name / gap
            if base.is_dir():
                for child in base.iterdir():
                    maybe_add(child)

        fps_base = DATA / "fps_sampling_frames" / gap
        if fps_base.is_dir():
            for fps_name in ("fps0_25", "fps_2"):
                fps_dir = fps_base / fps_name
                if fps_dir.is_dir():
                    for child in fps_dir.iterdir():
                        maybe_add(child)

        margin_base = DATA / "proposed_ROI_margin" / gap
        if margin_base.is_dir():
            for margin_name in ("margin_25", "margin_50", "always_margin"):
                mdir = margin_base / margin_name
                if mdir.is_dir():
                    for child in mdir.iterdir():
                        maybe_add(child)

    return sorted(leaves)


def reorganize_layout(video_dir: Path, *, dry_run: bool) -> None:
    """Ensure global/ + roi_XX/ layout; keep filenames."""
    globals_: list[Path] = []
    tubes: dict[int, list[Path]] = {}
    for path in list_jpgs(video_dir):
        g = GLOBAL_RE.fullmatch(path.stem)
        if g:
            globals_.append(path)
            continue
        t = TUBE_RE.fullmatch(path.stem)
        if t:
            tubes.setdefault(int(t.group("tube")), []).append(path)

    if dry_run:
        return

    if globals_:
        gdir = video_dir / "global"
        gdir.mkdir(parents=True, exist_ok=True)
        for path in globals_:
            dest = gdir / path.name
            if path.resolve() != dest.resolve():
                if dest.exists():
                    path.unlink()
                else:
                    path.rename(dest)

    for tid, paths in tubes.items():
        rdir = video_dir / f"roi_{tid:02d}"
        rdir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            dest = rdir / path.name
            if path.resolve() != dest.resolve():
                if dest.exists():
                    path.unlink()
                else:
                    path.rename(dest)


def collect_frames(video_dir: Path) -> tuple[list[SavedFrameRef], dict[int, list[SavedFrameRef]]]:
    g_refs: list[SavedFrameRef] = []
    gdir = video_dir / "global"
    if gdir.is_dir():
        for path in sorted(gdir.glob("*.jpg")):
            m = GLOBAL_RE.fullmatch(path.stem)
            ts = float(m.group("time")) if m else 0.0
            g_refs.append(SavedFrameRef(timestamp=ts, file=f"global/{path.name}"))

    tubes: dict[int, list[SavedFrameRef]] = {}
    for rdir in sorted(video_dir.glob("roi_*")):
        if not rdir.is_dir():
            continue
        try:
            tid = int(rdir.name.split("_", 1)[1])
        except ValueError:
            continue
        frames: list[SavedFrameRef] = []
        for path in sorted(rdir.glob("*.jpg")):
            m = TUBE_RE.fullmatch(path.stem)
            ts = float(m.group("time")) if m else 0.0
            frames.append(SavedFrameRef(timestamp=ts, file=f"{rdir.name}/{path.name}"))
        if frames:
            tubes[tid] = frames
    return g_refs, tubes


def rebuild_one(video_dir: Path, *, dry_run: bool) -> list[str]:
    kind = method_kind(video_dir)
    gaptype = gaptype_of(video_dir)
    video_id = video_dir.name
    reorganize_layout(video_dir, dry_run=dry_run)
    if dry_run:
        return []

    g_refs, tubes = collect_frames(video_dir)
    fw, fh = resolve_video_size(video_id)
    bbox_map = tube_norm_bboxes(load_tracks(gaptype, video_id), frame_width=fw, frame_height=fh)

    if kind in ("single_motion_roi", "single_motion_w_fps"):
        # One (or more) tube folders, but bbox = union of all motion ROIs from tracks.
        union = union_normalized_bboxes(list(bbox_map.values()) or [[0.0, 0.0, 1.0, 1.0]])
        region = source_region_from_normalized_bbox(union)
        roi_inputs = [
            RoiMetaInput(
                roi_id=tid,
                normalized_bbox=union,
                frames=frames,
                source_region=region,
            )
            for tid, frames in sorted(tubes.items())
            if frames
        ]
    else:
        roi_inputs = []
        for tid, frames in sorted(tubes.items()):
            if not frames:
                continue
            bbox = bbox_map.get(tid) or [0.0, 0.0, 1.0, 1.0]
            roi_inputs.append(
                RoiMetaInput(
                    roi_id=tid,
                    normalized_bbox=bbox,
                    frames=frames,
                    source_region=None,
                )
            )

    if kind in ("roi_only", "single_motion_roi"):
        global_frames: list[SavedFrameRef] | None = None
    else:
        global_frames = g_refs

    if kind == "fps_only":
        roi_inputs = []

    meta = build_sampling_metadata(
        video_id=video_id,
        global_frames=global_frames,
        rois=roi_inputs,
    )
    write_sampling_metadata(video_dir / "metadata.json", meta)
    return validate_sampling_metadata(meta, video_dir=video_dir)


def main() -> int:
    args = parse_args()
    print("== promote single_motion root -> gaptype5 ==")
    promote_single_motion_to_gaptype5(dry_run=args.dry_run)

    leaves = discover_leaves()
    print(f"== rebuild metadata for {len(leaves)} leaf folders ==")
    failed = 0
    for i, video_dir in enumerate(leaves, start=1):
        errs = rebuild_one(video_dir, dry_run=args.dry_run)
        if errs:
            failed += 1
            print(f"[{i}/{len(leaves)}] FAIL {video_dir}")
            for e in errs[:8]:
                print(f"    - {e}")
        elif i % 20 == 0 or i == len(leaves):
            print(f"[{i}/{len(leaves)}] ok …")

    # Final validation sweep over every metadata.json under datasets sampling roots
    print("== final validation ==")
    roots = [
        DATA / "proposed_ROI",
        DATA / "proposed_ROI_w_fps",
        DATA / "fps_sampling_frames",
        DATA / "proposed_ROI_margin",
        DATA / "single_motion_ROI",
        DATA / "single_motion_ROI_w_fps",
    ]
    n_meta = 0
    n_bad = 0
    for root in roots:
        if not root.is_dir():
            continue
        for meta_path in sorted(root.rglob("metadata.json")):
            n_meta += 1
            video_dir = meta_path.parent
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            errs = validate_sampling_metadata(meta, video_dir=video_dir)
            # Extra: global-only must have rois=[]
            kind = method_kind(video_dir)
            if kind == "fps_only" and meta.get("rois"):
                errs.append("fps_only has non-empty rois")
            if kind in ("roi_only", "single_motion_roi") and meta.get("global") is not None:
                # allow empty global object? require null
                errs.append("roi_only global must be null")
            if any(int(r.get("num_frames", 0)) <= 0 for r in meta.get("rois") or []):
                errs.append("empty ROI stub present")
            if errs:
                n_bad += 1
                print(f"FAIL {meta_path}")
                for e in errs[:6]:
                    print(f"  - {e}")

    print(f"metadata={n_meta} failed={n_bad} rebuild_failed={failed}")
    return 0 if n_bad == 0 and failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

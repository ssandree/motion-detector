#!/usr/bin/env python3
"""Move 5-gap sampling trees under ``gaptype5/`` and write ``metadata.json``.

1) Root-level trees (sibling to ``gaptype4/``) are renamed into ``gaptype5/``
   so the layout matches gaptype4.
2) Every leaf video folder that contains JPEG frames is reorganized into
   ``global/`` + ``roi_XX/`` and gets a ``metadata.json``.

Does not re-run Stage3 / sampling — only filesystem + metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
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
    write_sampling_metadata,
)
from motion_analyzer.visualization import grid_bbox_to_pixels  # noqa: E402

GLOBAL_RE = re.compile(r"^global_frame_(?P<seq>\d+)_t(?P<time>\d+(?:\.\d+)?)$")
TUBE_RE = re.compile(r"^tube(?P<tube>\d+)_frame_(?P<seq>\d+)_t(?P<time>\d+(?:\.\d+)?)$")

DATA = Path("/data/datasets")
TRACKS_GAPTYPE4 = REPO_ROOT / "outputs" / "stage3" / "3_roi_tube" / "gaptype4_9vid"
TRACKS_GAPTYPE5 = REPO_ROOT / "outputs" / "stage3" / "3_roi_tube" / "vlm_9vid_tracks"
DEFAULT_VIDEO_ROOTS = [
    Path("/data/datasets/VIRAT/videos-00"),
    Path("/data/datasets/VIRAT/videos-01"),
    Path("/data/datasets/VIRAT/videos-04"),
    Path("/data/datasets/VIRAT/videos-05"),
    Path("/data/datasets/VIRAT"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_move", action="store_true", help="Only metadata / restructure.")
    return p.parse_args()


def move_path(src: Path, dst: Path, *, dry_run: bool) -> None:
    if not src.exists():
        return
    if dst.exists():
        print(f"  SKIP move (dst exists): {src} -> {dst}")
        return
    print(f"  MOVE {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


def promote_root_to_gaptype5(*, dry_run: bool) -> None:
    """Mirror gaptype4 layout by parking 5-gap roots under gaptype5/."""
    print("== promote root trees -> gaptype5 ==")

    # proposed_ROI / proposed_ROI_w_fps: VIRAT_* dirs at root
    for root_name in ("proposed_ROI", "proposed_ROI_w_fps"):
        root = DATA / root_name
        g5 = root / "gaptype5"
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith("gaptype"):
                continue
            if not child.name.startswith("VIRAT_"):
                continue
            move_path(child, g5 / child.name, dry_run=dry_run)

    # fps_sampling_frames: fps0_25 / fps_2 at root
    fps_root = DATA / "fps_sampling_frames"
    for sub in ("fps0_25", "fps_2"):
        src = fps_root / sub
        if src.is_dir() and not (fps_root / "gaptype5" / sub).exists():
            move_path(src, fps_root / "gaptype5" / sub, dry_run=dry_run)

    # proposed_ROI_margin: margin_* / always_margin at root
    margin_root = DATA / "proposed_ROI_margin"
    for sub in ("margin_25", "margin_50", "always_margin"):
        src = margin_root / sub
        if src.is_dir() and not (margin_root / "gaptype5" / sub).exists():
            move_path(src, margin_root / "gaptype5" / sub, dry_run=dry_run)


def resolve_video_size(video_id: str) -> tuple[int, int]:
    for root in DEFAULT_VIDEO_ROOTS:
        path = root / f"{video_id}.mp4"
        if path.is_file():
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                continue
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            if w > 0 and h > 0:
                return w, h
    return 1280, 720


def load_tracks(tracks_root: Path, video_id: str) -> dict | None:
    path = tracks_root / f"{video_id}_roi_tracks.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tracks_bbox_map(
    tracks: dict | None,
    *,
    frame_width: int,
    frame_height: int,
) -> dict[int, dict]:
    """tube_id -> {normalized_bbox, time_range, source_region}."""
    out: dict[int, dict] = {}
    if not tracks:
        return out
    cell_px = int(tracks.get("cell_px") or 64)
    fw = int(tracks.get("video_width") or frame_width)
    fh = int(tracks.get("video_height") or frame_height)
    # Prefer timeline from NPZ-backed fields if present in tracks — else use
    # sampling timestamps only later.
    rows = tracks.get("tracks") or tracks.get("tubes") or []
    for tube in rows:
        tid = int(tube.get("tube_id") or 0)
        bbox_grid = tube.get("spatial_bbox_grid")
        if not bbox_grid or tid <= 0:
            continue
        px = grid_bbox_to_pixels(
            tuple(int(v) for v in bbox_grid),
            unit_pixel_size=cell_px,
            frame_width=fw,
            frame_height=fh,
        )
        norm = normalize_bbox(px, frame_width=fw, frame_height=fh)
        region = (
            str(tube["source_region"])
            if tube.get("source_region")
            else source_region_from_normalized_bbox(norm)
        )
        # time_range filled later from frames if NPZ timeline unavailable
        out[tid] = {
            "normalized_bbox": norm,
            "source_region": region,
            "t0_index": tube.get("t0"),
            "t1_index": tube.get("t1"),
        }
    return out


def list_jpgs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.jpg") if p.is_file())


def is_leaf_frame_dir(folder: Path) -> bool:
    """True if folder (or its immediate children) hold sampling JPEGs."""
    if not folder.is_dir():
        return False
    # Already structured
    if (folder / "global").is_dir() or any(folder.glob("roi_*")):
        return True
    # Flat jpgs
    if any(folder.glob("*.jpg")):
        return True
    return False


def discover_leaf_video_dirs() -> list[tuple[Path, str]]:
    """Return (video_dir, gaptype) for every leaf that has frames."""
    found: list[tuple[Path, str]] = []

    def add(video_dir: Path, gaptype: str) -> None:
        if is_leaf_frame_dir(video_dir):
            found.append((video_dir, gaptype))

    for gaptype in ("gaptype4", "gaptype5"):
        # proposed_ROI / proposed_ROI_w_fps
        for root_name in ("proposed_ROI", "proposed_ROI_w_fps"):
            base = DATA / root_name / gaptype
            if not base.is_dir():
                continue
            for child in sorted(base.iterdir()):
                if child.is_dir() and child.name.startswith("VIRAT_"):
                    add(child, gaptype)

        # fps_sampling_frames
        fps_base = DATA / "fps_sampling_frames" / gaptype
        if fps_base.is_dir():
            for fps_name in ("fps0_25", "fps_2"):
                fps_dir = fps_base / fps_name
                if not fps_dir.is_dir():
                    continue
                for child in sorted(fps_dir.iterdir()):
                    if child.is_dir() and child.name.startswith("VIRAT_"):
                        add(child, gaptype)

        # proposed_ROI_margin
        margin_base = DATA / "proposed_ROI_margin" / gaptype
        if margin_base.is_dir():
            for margin_name in ("margin_25", "margin_50", "always_margin"):
                mdir = margin_base / margin_name
                if not mdir.is_dir():
                    continue
                for child in sorted(mdir.iterdir()):
                    if child.is_dir() and child.name.startswith("VIRAT_"):
                        add(child, gaptype)

    return found


def reorganize_and_metadata(video_dir: Path, gaptype: str, *, dry_run: bool) -> None:
    video_id = video_dir.name
    tracks_root = TRACKS_GAPTYPE4 if gaptype == "gaptype4" else TRACKS_GAPTYPE5
    tracks = load_tracks(tracks_root, video_id)
    fw, fh = resolve_video_size(video_id)
    if tracks:
        fw = int(tracks.get("video_width") or fw) or fw
        fh = int(tracks.get("video_height") or fh) or fh
        # tracks json may not store video_width — keep resolved
    bbox_by_tube = tracks_bbox_map(tracks, frame_width=fw, frame_height=fh)

    # Collect current frames (flat or already nested)
    globals_: list[tuple[Path, int, float]] = []
    tubes: dict[int, list[tuple[Path, int, float]]] = {}

    for path in list_jpgs(video_dir):
        stem = path.stem
        g = GLOBAL_RE.fullmatch(stem)
        if g:
            globals_.append((path, int(g.group("seq")), float(g.group("time"))))
            continue
        t = TUBE_RE.fullmatch(stem)
        if t:
            tid = int(t.group("tube"))
            tubes.setdefault(tid, []).append(
                (path, int(t.group("seq")), float(t.group("time")))
            )

    globals_.sort(key=lambda x: (x[2], x[1]))
    for tid in tubes:
        tubes[tid].sort(key=lambda x: (x[2], x[1]))

    if not globals_ and not tubes:
        print(f"  SKIP empty {video_dir}")
        return

    print(f"  FIX {video_dir}  global={len(globals_)} tubes={sorted(tubes)}")

    if dry_run:
        return

    # Move into global/ and roi_XX/
    global_dir = video_dir / "global"
    if globals_:
        global_dir.mkdir(parents=True, exist_ok=True)
        for path, _seq, _ts in globals_:
            dest = global_dir / path.name
            if path.resolve() != dest.resolve():
                if dest.exists():
                    path.unlink()
                else:
                    path.rename(dest)

    for tid, frames in tubes.items():
        roi_dir = video_dir / f"roi_{tid:02d}"
        roi_dir.mkdir(parents=True, exist_ok=True)
        for path, _seq, _ts in frames:
            dest = roi_dir / path.name
            if path.resolve() != dest.resolve():
                if dest.exists():
                    path.unlink()
                else:
                    path.rename(dest)

    # Drop leftover empty dirs / stray flat files already moved
    for stray in sorted(video_dir.glob("*.jpg")):
        # Should be none; keep if somehow unmatched
        pass

    # Rebuild file lists from final locations
    g_refs: list[SavedFrameRef] = []
    for i, path in enumerate(sorted((video_dir / "global").glob("*.jpg")), start=1):
        m = GLOBAL_RE.fullmatch(path.stem)
        ts = float(m.group("time")) if m else 0.0
        g_refs.append(SavedFrameRef(timestamp=ts, file=f"global/{path.name}"))

    roi_inputs: list[RoiMetaInput] = []
    for tid in sorted(tubes):
        roi_dir = video_dir / f"roi_{tid:02d}"
        frame_refs: list[SavedFrameRef] = []
        for path in sorted(roi_dir.glob("*.jpg")):
            m = TUBE_RE.fullmatch(path.stem)
            ts = float(m.group("time")) if m else 0.0
            frame_refs.append(
                SavedFrameRef(timestamp=ts, file=f"roi_{tid:02d}/{path.name}")
            )
        if not frame_refs:
            continue
        info = bbox_by_tube.get(tid)
        if info:
            norm = info["normalized_bbox"]
        else:
            norm = [0.0, 0.0, 1.0, 1.0]
        roi_inputs.append(
            RoiMetaInput(
                roi_id=tid,
                normalized_bbox=norm,
                frames=frame_refs,
                source_region=None,
            )
        )

    # Do not invent empty ROI stubs for track tubes without saved frames.
    meta = build_sampling_metadata(
        video_id=video_id,
        global_frames=g_refs if g_refs else None,
        rois=sorted(roi_inputs, key=lambda r: r.roi_id),
    )
    write_sampling_metadata(video_dir / "metadata.json", meta)


def main() -> int:
    args = parse_args()
    if not args.skip_move:
        promote_root_to_gaptype5(dry_run=args.dry_run)

    print("== reorganize + metadata ==")
    leaves = discover_leaf_video_dirs()
    print(f"leaf folders: {len(leaves)}")
    for video_dir, gaptype in leaves:
        try:
            reorganize_and_metadata(video_dir, gaptype, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {video_dir}: {exc}")
            raise

    if not args.dry_run:
        n = len(list(DATA.rglob("metadata.json")))
        print(f"done. metadata.json count under /data/datasets = {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

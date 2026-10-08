"""Where the existing Stage3 results and source videos live, per dataset.

All paths point at artifacts that already exist. Nothing here writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT.parent / "vlm-evaluation"
VLM_INFERENCE_ROOT = REPO_ROOT.parent / "vlm-inference"

OVERLAY_SUFFIX = "_roi_tube_light_h0.2_g2t0.25_cmp1.5r3.mp4"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    video_list: Path
    tracks_root: Path
    npz_root: Path
    overlay_root: Path
    raw_video_globs: tuple[str, ...]
    raw_video_root: Path
    gt_root: Path | None

    def tracks_path(self, video_id: str) -> Path:
        return self.tracks_root / f"{video_id}_roi_tracks.json"

    def npz_path(self, video_id: str) -> Path:
        return self.npz_root / video_id / f"{video_id}_gap_fusion_mean.npz"

    def overlay_path(self, video_id: str) -> Path:
        return self.overlay_root / f"{video_id}{OVERLAY_SUFFIX}"

    def raw_video_path(self, video_id: str) -> Path | None:
        for pattern in self.raw_video_globs:
            hits = sorted(self.raw_video_root.glob(pattern.format(video_id=video_id)))
            if hits:
                return hits[0]
        return None

    def video_ids(self) -> list[str]:
        return [
            line.strip()
            for line in self.video_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


VIRAT = DatasetSpec(
    name="VIRAT",
    video_list=REPO_ROOT / "configs" / "virat_all.txt",
    tracks_root=REPO_ROOT / "outputs" / "virat_all_light_uf" / "roi_tracks_json",
    npz_root=REPO_ROOT / "outputs" / "virat_all_light_uf" / "stage2_npz",
    overlay_root=REPO_ROOT / "outputs" / "virat_all_light_uf" / "mp4",
    raw_video_root=Path("/data/datasets/VIRAT"),
    raw_video_globs=("videos-*/{video_id}.mp4",),
    gt_root=EVAL_ROOT / "experiments" / "virat" / "caption_event_gt_191" / "gt",
)

MEVA = DatasetSpec(
    name="MEVA",
    video_list=REPO_ROOT / "configs" / "meva_drop4_hadcv22_all.txt",
    tracks_root=REPO_ROOT / "outputs" / "meva_drop4_hadcv22" / "json",
    npz_root=REPO_ROOT / "outputs" / "meva_drop4_hadcv22" / "stage2_npz",
    overlay_root=REPO_ROOT / "outputs" / "meva_drop4_hadcv22" / "video",
    raw_video_root=Path("/data/datasets/MEVA/drop-4-hadcv22"),
    raw_video_globs=("*/*/{video_id}.avi",),
    # drop-4-hadcv22 is not covered by the kitware MEVA annotation release.
    gt_root=None,
)

DATASETS = {"VIRAT": VIRAT, "MEVA": MEVA}


def get_dataset(name: str) -> DatasetSpec:
    key = name.strip().upper()
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[key]

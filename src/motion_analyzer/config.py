"""Pipeline constants for 3-stage Gap1/5/10/20/50 → Stage3 ROI tubes (exp_v2c)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_VIDEO_SEARCH_ROOTS = [
    Path("/data/datasets/VIRAT/videos-00"),
    Path("/data/datasets/VIRAT/videos-01"),
    Path("/data/datasets/VIRAT/videos-04"),
    Path("/data/datasets/VIRAT/videos-05"),
    Path("/data/datasets/VIRAT"),
]

GAPS = (1, 5, 10, 20, 50)  # Stage2 multi-gap maps + fusion
# Hierarchical mean fusion: S_short=(S1+S5)/2, S_long=(S10+S20+S50)/3, then mean.
SHORT_GAPS = (1, 5)
LONG_GAPS = (10, 20, 50)
STAGE1_GAPS = (1,)  # Stage1: Gap1 Farneback only
VEC_KEYS = {gap: f"U{gap}" for gap in GAPS}
MAG_KEYS = {gap: f"M{gap}" for gap in GAPS}
FUSION_SCHEME_CHOICES = ("flat", "short_long")
DEFAULT_FUSION_SCHEME = "flat"
STAGE1_BASE_MOTION_TAG = "_".join(f"U{g}" for g in STAGE1_GAPS)  # "U1"
BASE_MOTION_TAG = STAGE1_BASE_MOTION_TAG

# Official Stage2 (locked):
#   long gaps = sum of G consecutive filtered Gap1 vectors (no extra Farneback)
#   per-gap scale = ÷√G before fusion (random-walk / Brownian magnitude)
#   temporal fusion = mean @16px
#   spatial aggregation = 4×4 max → 64px
DEFAULT_FUSION = "mean"  # mean | max | median | rms
DEFAULT_SPATIAL_AGG = "max"  # 4×4 unit agg: mean | max | median | rms
FUSION_CHOICES = ("mean", "max", "median", "rms")
# Divide the Gap-G integral by √G so Gap1..Gap50 share a comparable mag scale.
# Ballistic motion then grows as √G; uncorrelated residual stays ~flat.
GAP_NORM_DIV = {int(g): float(g) ** 0.5 for g in GAPS}
# Turbo heatmap absolute scale for Stage2/Stage3 overlays.
HEAT_VMIN = 0.8
HEAT_VMAX = 3.5

# Stage3 official: hysteresis events on 64px MU, then attractive Union-Find
# → AABB compactness composition.
# Fixed MU gate (no hysteresis band): start/hold while MU ≥ thr, end after
# max_gap frames below thr.
ROI_THRESHOLD = 0.2
ROI_TAU_HIGH = ROI_THRESHOLD
ROI_TAU_LOW = ROI_THRESHOLD
ROI_MAX_GAP = 5  # keep event open while below-thr streak ≤ this many frames
ROI_MIN_BLOCK_EVENT = 3  # frames
ROI_MIN_TUBE_CELLS = 2
ROI_MIN_TUBE_DURATION = 2  # frames
ROI_NEIGH_RADIUS = 2  # Chebyshev ≤2 (CLI/tag; clustering uses merge_spatial_dist)
ROI_MERGE_SPATIAL_DIST = 2  # block Chebyshev for proximity merge
ROI_MERGE_TEMPORAL_GAP = 20  # frames between tube intervals
# Clipped pixel bbox must be ≥ one unit block (64px) in both width and height.
# Drops last-row strips on 720p (12×64=768 > 720 → last row is 16px).
ROI_MIN_BLOCK_PX = True
# Drop tube A if its [t0,t1]×spatial_bbox is strictly inside another tube B.
ROI_SUPPRESS_CONTAINED = True

INPUT_SCALE = 0.25
FARNEBACK_WINSIZE = 9
FARNEBACK_POLY_N = 7
FARNEBACK_POLY_SIGMA = 1.5
FARNEBACK_USE_GAUSSIAN = False  # cv2.OPTFLOW_FARNEBACK_GAUSSIAN
# "cuda" requires OpenCV built with cudaoptflow (see scripts/setup/build_opencv_cuda.sh).
FARNEBACK_DEVICE = "cuda"
FARNEBACK_PYR_SCALE = 0.5
FARNEBACK_LEVELS = 3
FARNEBACK_ITERATIONS = 2
# Before Stage1 spatial R-mean: zero dense flow vectors with ‖v‖ below this (px).
PRE_AGG_MAG_THRESHOLD = 0.5
RESIZED_BASE_BLOCK = 4  # output stride on 1/4-scale flow → 16 px base cell
STAGE1_SPATIAL_WIN = 6  # 6×6 dense neighborhood per base block
STAGE1_TEMPORAL_RADIUS = 2  # T5 after persistence: window [f−2, f+2]
# T5: False = mean every frame in [f−2,f+2], including exact zeros.
STAGE1_T5_ACTIVE_ONLY = False
ORIGINAL_CELL_PX = 16  # Stage1 base cell
AGGREGATION_BLOCK = 4  # Stage2 unit block → 64 px
UNIT_CELL_PX = ORIGINAL_CELL_PX * AGGREGATION_BLOCK

# Official Stage1 (locked):
#   Farneback Gap1 @1/4, winsize=9
#   mag < 0.5 → 0
#   v -= p_strip (v·t) t
#     p_strip = clip(λ1/τ_edge, 0, 1) * edgeness * alignment^γ
#   6×6 R-mean, R = (1-α P_ap) × (1-exp(-λ2/τ_str))
#   P13 ≥ STAGE1_P13_MIN → T5 (mean all frames in [f−2,f+2], zeros included)
#   → cell mag ≤ 0.5 → 0 (Stage1 only; Stage2 does not re-cut)
# P_ap hard-zero (v=0 where P_ap>τ) is removed. R_ap still downweights.
STRUCTURE_TENSOR_SIGMA = 1.5
STRUCTURE_TENSOR_STRENGTH_TAU = 400.0
STRUCTURE_TENSOR_EPS = 1e-6
APERTURE_TAU_EDGE = 950.0
APERTURE_GAMMA = 2.0
APERTURE_ALPHA = 1.0
APERTURE_R_TIMES_STRENGTH = True
APERTURE_STRIP_TANGENT = True
# <0 disables leftover helpers; pipeline no longer hard-zeros on P_ap.
APERTURE_P_HARD_TAU = -1.0
APERTURE_P_HARD_DILATE_PX = 0

# Stage1 after 6×6×1 spatial R-mean, before T5.
# keep = P_window ≥ τ_P. Default window is 13 frames (P13).
STAGE1_P15_WINDOW = 13  # P13 at 5fps = 2.6 s
STAGE1_P13_MIN = 0.8  # keep if directional persistence ≥ this (P13 window)
# After P13→T5: hard-zero Stage1 cell vectors with ‖v‖ ≤ this.
# Distinct from PRE_AGG_MAG_THRESHOLD (dense jitter, before 6×6).
STAGE1_CELL_MAG_FLOOR = 0.5
# Stage2 Gap1 integral: 0 = use Stage1 U1 as stored (no second mag cut).
STAGE2_CELL_MAG_FLOOR = 0.0

TARGET_VIDEO_IDS = (
    "VIRAT_S_000200_01_000226_000268",
    "VIRAT_S_000201_02_000590_000623",
    "VIRAT_S_010201_00_000000_000053",
    "VIRAT_S_010200_08_000838_000867",
    "VIRAT_S_050202_04_000690_000750",
    "VIRAT_S_040000_05_000668_000703",
    "VIRAT_S_040000_00_000000_000036",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "cache"
DEFAULT_TARGET_LIST = REPO_ROOT / "configs" / "target_videos.txt"


@dataclass
class PipelineConfig:
    sampling_fps: float = 5.0
    video_search_roots: list[Path] = field(
        default_factory=lambda: [p for p in DEFAULT_VIDEO_SEARCH_ROOTS]
    )
    fusion: str = DEFAULT_FUSION
    # Kept for CLI compatibility with scripts/3_roi_tube.py (--tau_high mirror).
    roi_threshold: float = ROI_TAU_HIGH

    def validate(self) -> None:
        if self.sampling_fps <= 0:
            raise ValueError("sampling_fps must be > 0")
        if self.roi_threshold < 0:
            raise ValueError("roi_threshold must be >= 0")
        if str(self.fusion).lower() not in FUSION_CHOICES:
            raise ValueError(f"fusion must be one of {FUSION_CHOICES}")


def load_target_video_ids(list_path: Path | None = None) -> list[str]:
    path = Path(list_path) if list_path is not None else DEFAULT_TARGET_LIST
    if path.is_file():
        rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [row for row in rows if row and not row.startswith("#")]
    return list(TARGET_VIDEO_IDS)

"""Pipeline constants for 3-stage Gap1/5/10/20/50 → ROI tube."""

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
STAGE1_GAPS = (1,)  # Stage1: Gap1 Farneback only
VEC_KEYS = {gap: f"U{gap}" for gap in GAPS}
MAG_KEYS = {gap: f"M{gap}" for gap in GAPS}
STAGE1_BASE_MOTION_TAG = "_".join(f"U{g}" for g in STAGE1_GAPS)  # "U1"
BASE_MOTION_TAG = STAGE1_BASE_MOTION_TAG

# Stage2 gap fusion across M1..M50.
DEFAULT_FUSION = "rms"  # mean | max | median | rms
FUSION_CHOICES = ("mean", "max", "median", "rms")
# Per-gap magnitude/vector scale factors before temporal fusion
# (bring Gap5..50 mean levels toward Gap1; from 30-video calibration).
GAP_NORM_DIV = {
    1: 1.0,
    5: 3.0,
    10: 3.5,
    20: 3.8,
    50: 4.2,
}
# Turbo heatmap absolute scale for Stage2/Stage3 overlays.
HEAT_VMIN = 0.8
HEAT_VMAX = 3.5

# Stage3 hysteresis block-tube → spatial merge.
# Per-block event: start MU≥τ_high, end when MU≤τ_low (max_gap=0), min 3 frames.
# Merge: Chebyshev ≤2 and tube frame-gap ≤20.
ROI_TAU_HIGH = 0.7
ROI_TAU_LOW = 0.4
ROI_MAX_GAP = 0  # end immediately once MU drops below τ_low
ROI_MIN_BLOCK_EVENT = 3  # frames
ROI_MIN_TUBE_CELLS = 2
ROI_MIN_TUBE_DURATION = 3  # frames (match min block-event length)
ROI_NEIGH_RADIUS = 2  # Chebyshev ≤2 → 24-connected (same-frame merge)
ROI_MERGE_SPATIAL_DIST = 2  # block Chebyshev for proximity merge
ROI_MERGE_TEMPORAL_GAP = 20  # frames between tube intervals
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
PRE_AGG_MAG_THRESHOLD = 0.6
RESIZED_BASE_BLOCK = 4  # output stride on 1/4-scale flow → 16 px base cell
STAGE1_SPATIAL_WIN = 8  # 8×8 dense neighborhood per base block
STAGE1_TEMPORAL_RADIUS = 2  # T5 after P15: frames [f−2, f+2]
ORIGINAL_CELL_PX = 16  # Stage1 base cell
AGGREGATION_BLOCK = 4  # Stage2 unit block → 64 px
UNIT_CELL_PX = ORIGINAL_CELL_PX * AGGREGATION_BLOCK

# Structure-tensor / aperture reliability (R_ap weight for 8×8 spatial mean).
STRUCTURE_TENSOR_SIGMA = 1.5
STRUCTURE_TENSOR_STRENGTH_TAU = 400.0
STRUCTURE_TENSOR_EPS = 1e-6
# tau_edge ≈ λ1 P25 on VIRAT 1/4 gray (~950); gates flat regions without orientation.
APERTURE_TAU_EDGE = 950.0
APERTURE_GAMMA = 2.0
APERTURE_ALPHA = 1.0

# Stage1 after 8×8×1 spatial R-mean, before T5.
# keep = P15 ≥ τ_P.
STAGE1_P15_WINDOW = 15  # P15 at 5fps = 3 s
STAGE1_P15_MIN = 0.85  # keep if directional persistence ≥ this

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

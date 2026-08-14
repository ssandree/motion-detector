# motion-detector

VIRAT 영상에서 motion base-block → gap fusion → ROI tube 를 만드는 3-stage 파이프라인.

표기: `Stage1-1`, `Stage2-1` … = 각 Stage 안의 단계 번호.

Stage1 → Stage2 → Stage3 순서로 실행한다. Stage2는 Stage1 NPZ(`U1`/`M1`)를 입력으로 쓴다.


## Stage1. Gap1 base-block (16px)

목표: Gap=1 Farneback → 16px base-block (`U1` / `M1`).

기본 경로: GPU에서 gray 업로드 → ¼ 유지 → Farneback + Sobel `R_ap` + 8×8 spatial mean → 16px 그리드 다운로드 → CPU에서 P15 + T5.

### Stage1-1. Sampling + resize + Farneback

- 입력 video → **fps=5** sampling
- **¼ resize** → 480×270 (원본 1920×1080 기준)
- Farneback optical flow, **Gap=1** (약 0.2s)
- `‖v‖ < 0.6` 인 dense flow는 0으로 처리

### Stage1-2. 8×8×1 R-mean → P15 → T5

- 8×8 spatial **aperture-weighted mean** (`w_i = R_ap`, 시간 혼합 없음), stride 4 → **16px** cell
- **P15** 게이트: 15프레임(5fps에서 3초) 방향 지속성 `≥ 0.85` 인 cell만 유지
- **T5**: 프레임 `[f−2, f+2]` 시간 평균
- 출력: Gap1 base vector map **U1** / magnitude **M1** @ 16px  
  (`data/cache/<video_id>/*_base_motion_U1.npz`)


## Stage2. 5-gap motion fusion (64px)

목표: Stage1 Gap1 + 같은 방식으로 만든 Gap 5/10/20/50 → 정규화·fusion → 64px unit block.

Stage2는 Stage1 NPZ가 없으면 실패한다.

### Stage2-1. Gap maps

- **Gap1**: Stage1 NPZ의 `U1`/`M1`을 로드한 뒤, Gap50 타임라인에 맞춰 slice
- **Gap 5/10/20/50**: Stage1과 동일하게 Farneback @ 1/4 → 8×8×1 `R_ap`-mean → P15 → T5

### Stage2-2. Gap 정규화 + temporal fusion (16px)

- gap별 스케일 (30영상 캘리브레이션): Gap1÷1, Gap5÷3.0, Gap10÷3.5, Gap20÷3.8, Gap50÷4.2
- 정규화된 Gap1/5/10/20/50 magnitude를 temporal fusion (기본 **RMS**)
- 출력: fused motion map `M_fused` @ **16px**

### Stage2-3. 4×4 magnitude max (64px)

- fused 16px map에 **4×4 spatial max**
- 출력: **64px** unit grid `MU_fused`


## Stage3. ROI tube

목표: 64px `MU` 위에서 hysteresis event → Chebyshev / proximity merge.

### Stage3-1. Block별 hysteresis event

- 각 block(**64px**)에서 `MU ≥ τ_high(0.7)` 이면 시작, `MU ≤ τ_low(0.4)` 이면 즉시 종료 (`max_gap=0`)
- 길이 **≥ 3** 프레임만 유지

### Stage3-2. Merge + 필터

- 같은 프레임: Chebyshev **≤ 2** (24-connected)
- proximity merge: 공간 Chebyshev **≤ 2**, 시간 gap **≤ 20**
- tube 유지 조건: cell **≥ 2** 그리고 duration **≥ 3** (1×64px block은 ROI 아님)
- 다른 tube의 시공간 AABB에 완전히 포함된 tube는 제거
- 출력: MP4 overlay, `roi_tracks` JSON, (x, y, t) 3D figure


## 한눈에 보기

```
Video
 │
 ├─ Stage1-1  fps=5 sample + ¼ resize(480×270) + Farneback Gap=1
 ├─ Stage1-2  8×8×1 R_ap-mean → P15≥0.85 → T5 → 16px U1/M1
 │              └── NPZ: data/cache/<id>/*_base_motion_U1.npz
 │
 ├─ Stage2-1  Gap1 ← Stage1 NPZ
 │            Gap 5/10/20/50 ← 같은 8×8×1 → P15 → T5
 ├─ Stage2-2  gap-normalize + RMS fusion @16px
 ├─ Stage2-3  4×4 magnitude max → 64px MU
 │
 ├─ Stage3-1  hysteresis block event (τ_high=0.7, τ_low=0.4)
 └─ Stage3-2  Chebyshev≤2 / proximity merge → ROI tubes
```


## 약어

| 약어 | 의미 |
|------|------|
| Gap *N* | 5fps 샘플에서 *N* 프레임 떨어진 Farneback (Gap1 ≈ 0.2s) |
| `R_ap` | aperture reliability. 가장자리·구멍 난 영역 가중치 (0~1) |
| 8×8×1 | 축소 해상도에서 8×8 공간 평균, 시간 혼합 없음. stride 4 → 16px cell |
| P15 | 15프레임 창에서 방향 지속성 `‖Σv‖ / Σ‖v‖` |
| T5 | 프레임 `[f−2, f+2]` 시간 평균 |
| `U1`/`M1` | Gap1 벡터 / magnitude @ 16px |
| `MU` | 64px unit-grid magnitude (Stage2 4×4 max) |


## 해상도 / 좌표

원본 프레임(보통 1920×1080) → **¼** (480×270)에서 Farneback. 벡터 단위는 **원본 px**.  
그리드는 축소 해상도 기준: 16px cell (Stage1/2) → 64px unit (Stage2/3).


## 디렉터리

```
motion-detector/
├── configs/                 # video list (기본: target_videos.txt)
├── src/motion_analyzer/     # Stage1~2 + shared
│   ├── motion_map.py        # Stage1 (Gap1 8×8×1 → P15 → T5)
│   ├── aggregation.py       # Stage2 (Stage1 NPZ + extra gaps → fusion)
│   └── stage1_gpu.py        # CUDA Farneback / R_ap / 8×8 mean
├── src/stage3/              # Stage3 ROI tubes
│   ├── hysteresis_tube.py
│   ├── roi_tube.py
│   └── tube_3d_viz.py
├── scripts/
│   ├── 1_base_motion.py     # Stage1
│   ├── 2_gap_fusion.py      # Stage2 (Stage1 NPZ 필요)
│   ├── 3_roi_tube.py        # Stage3
│   └── setup/               # OpenCV CUDA Farneback 빌드
├── data/cache/              # Stage1 NPZ
└── outputs/                 # summary · MP4 · JSON · 3D PNG
```


## 실행

영상은 기본으로 `/data/datasets/VIRAT/` 아래에서 `<video_id>.mp4` 를 찾는다. 대상 목록은 `configs/target_videos.txt`.

```bash
cd /path/to/motion-detector
export PYTHONPATH=src
pip install -r requirements.txt

# (최초 1회) OpenCV CUDA Farneback
bash scripts/setup/build_opencv_cuda.sh
source ~/.local/opencv-cuda/env.sh

# Stage1  → data/cache/<id>/*_base_motion_U1.npz
python scripts/1_base_motion.py

# Stage2  → outputs/stage2/2_gap_fusion/<stamp>/
python scripts/2_gap_fusion.py

# Stage3  → outputs/stage3/3_roi_tube/<stamp>/
python scripts/3_roi_tube.py \
  --fusion_root outputs/stage2/2_gap_fusion/<stamp>
```

단일 영상: `--video_id VIRAT_S_...`  
목록 지정: `--video_list configs/virat_videos_le60s_sample30.txt`


## Stage ↔ 스크립트

| Stage  | 단계                         | 스크립트           | 입력 | 기본 출력 |
|--------|------------------------------|--------------------|------|-----------|
| Stage1 | Stage1-1, Stage1-2           | `1_base_motion.py` | video | `data/cache/`, `outputs/stage1/` |
| Stage2 | Stage2-1, Stage2-2, Stage2-3 | `2_gap_fusion.py`  | Stage1 NPZ + video | `outputs/stage2/2_gap_fusion/` |
| Stage3 | Stage3-1, Stage3-2           | `3_roi_tube.py`    | Stage2 fusion root | `outputs/stage3/3_roi_tube/` |

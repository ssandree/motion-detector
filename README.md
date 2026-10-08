# motion-detector

VIRAT 영상에서 motion base-block → gap fusion → ROI tube 를 만드는 3-stage 파이프라인.

표기: `Stage1-1`, `Stage2-1` … = 각 Stage 안의 단계 번호.

Stage1 → Stage2 → Stage3 순서로 실행한다. Stage2는 Stage1 NPZ(`U1`/`M1`)를 입력으로 쓴다.


## Stage1. Gap1 base-block (16px)

목표: Gap=1 Farneback → 16px base-block (`U1` / `M1`).

기본 경로: GPU에서 gray 업로드 → ¼ 유지 → Farneback + Sobel `R_ap` + 6×6 spatial mean → 16px 그리드 다운로드 → CPU에서 P13 + T5.

### Stage1-1. Sampling + resize + Farneback

- 입력 video → **fps=5** sampling
- **¼ resize** → 480×270 (원본 1920×1080 기준)
- Farneback optical flow, **Gap=1** (약 0.2s)
- dense `‖v‖ < 0.5` → 0 (`PRE_AGG_MAG_THRESHOLD`)

### Stage1-2. 6×6×1 R-mean → P13 → T5

- 6×6 spatial **aperture-weighted mean** (`w_i = R_ap`, 시간 혼합 없음), stride 4 → **16px** cell
- **P13** 게이트: 13프레임(5fps에서 2.6초) 방향 지속성 `≥ 0.8` 인 cell만 유지
- **T5**: 프레임 `[f−2, f+2]` 시간 평균 (0도 포함)
- **Stage1 cell floor**: 16px `‖U1‖ ≤ 0.5` → 0 (`STAGE1_CELL_MAG_FLOOR`). Stage2는 다시 자르지 않음.
- 출력: Gap1 base vector map **U1** / magnitude **M1** @ 16px  
  (`data/cache/<video_id>/*_base_motion_U1.npz`)


## Stage2. 5-gap motion fusion (64px)

목표: Stage1 Gap1 + 같은 방식으로 만든 Gap 5/10/20/50 → 정규화·fusion → 64px unit block.

Stage2는 Stage1 NPZ가 없으면 실패한다.

### Stage2-1. Gap maps

- **Gap1**: Stage1 NPZ의 `U1`/`M1`을 로드한 뒤, Gap50 타임라인에 맞춰 slice. **cell mag floor 재적용 없음.**
- **Gap 5/10/20/50**: filtered Gap1 연속 합 (추가 Farneback 없음)

### Stage2-2. Temporal fusion (16px)

- Gap1 integrals (Gap 5/10/20/50 = consecutive Gap1 합). fusion 전 각 gap을 **÷√G**
- temporal fusion 잠금 **mean**
- 출력: fused motion map `M_fused` @ **16px**

### Stage2-3. 4×4 magnitude max (64px)

- fused 16px map에 **4×4 spatial max**
- 출력: **64px** unit grid `MU_fused`


## Stage3. ROI tube (attractive Union-Find → AABB composition)

목표: 물체 세그멘테이션이 아니라 VLM용 contextual ROI 1~2개(많아야 3개).
Graph-cut / coarse grouping / Kernighan–Lin 없음.

### Stage3-1. Block temporal event

- 각 block(**64px**)에서 `MU ≥ 0.2` 이면 시작, thr 아래로 `max_gap=5` 프레임이면 종료
- 길이 **≥ 3** 프레임만 유지; 가장자리 partial / 고립 1칸 제거
- 각 이벤트: `(x,y,t0,t1)` + 평균 flow `(u,v)` + coherence

### Stage3-2. Attractive Union-Find split

- 그리드 해시로 Chebyshev ≤ 2 이웃만 후보 (spatial / temporal bridge gap 1–10)
- `c = 0.5 S_xy + 0.5 S_t − 1.0 R_dir − 0.25` → `c>0` union, `c≤0` 유지
- 전역 multicut 없음; 방향이 충돌하면만 끊음

### Stage3-3. ROI composition (AABB)

- 모션 재검사 없음; `merged_area/(area_A+area_B) ≤ 1.5`, temporal_gap ≤ 15
- active frame당 ROI > 3이면 강제 merge

### Stage3-4. Filter + fixed lifetime bbox

- 짧은/작은 tube 제거, contained 억제, IoU NMS
- `[t0,t1]` 동안 동일 공간 박스 유지


## 한눈에 보기

```
Video
 │
 ├─ Stage1-1  fps=5 sample + ¼ resize(480×270) + Farneback Gap=1
 ├─ Stage1-2  6×6×1 R_ap-mean → P13≥0.8 → T5(0 포함) → ‖U1‖≤0.5→0 → 16px U1/M1
 │              └── NPZ: data/cache/<id>/*_base_motion_U1.npz
 │
 ├─ Stage2-1  Gap1 ← Stage1 NPZ (no extra cell floor)
 │            Gap 5/10/20/50 ← Gap1 integrals
 ├─ Stage2-2  temporal mean fusion @16px
 ├─ Stage2-3  4×4 magnitude max → 64px MU
 │
 ├─ Stage3-1  block event (thr=0.2, max_gap=5)
 ├─ Stage3-2  attractive Union-Find (c>0 merge, local hash)
 ├─ Stage3-3  AABB composition (compactness≤1.5)
 └─ Stage3-4  filter + fixed lifetime bbox → ROI tubes
```


## 약어

| 약어 | 의미 |
|------|------|
| Gap *N* | 5fps 샘플에서 *N* 프레임 떨어진 Farneback (Gap1 ≈ 0.2s) |
| `R_ap` | aperture reliability. 가장자리·구멍 난 영역 가중치 (0~1) |
| 6×6×1 | 축소 해상도에서 6×6 공간 평균, 시간 혼합 없음. stride 4 → 16px cell |
| P13 | 13프레임 창에서 방향 지속성 `‖Σv‖ / Σ‖v‖` (설정 키는 `STAGE1_P15_WINDOW`) |
| T5 | 프레임 `[f−2, f+2]` 시간 평균 |
| `U1`/`M1` | Gap1 벡터 / magnitude @ 16px |
| `MU` | 64px unit-grid magnitude (Stage2 4×4 max) |


## 해상도 / 좌표

원본 프레임(보통 1920×1080) → **¼** (480×270)에서 Farneback. 벡터 단위는 **원본 px**.  
그리드는 축소 해상도 기준: 16px cell (Stage1/2) → 64px unit (Stage2/3).


## 디렉터리

```
motion-detector/
├── configs/                 # video list
├── src/motion_analyzer/     # Stage1~2 + shared
│   ├── motion_map.py        # Stage1 (Gap1 6×6×1 → P13 → T5)
│   ├── aggregation.py       # Stage2 (Stage1 NPZ + extra gaps → fusion)
│   └── stage1_gpu.py        # CUDA Farneback / R_ap / spatial mean
├── src/stage3/              # Stage3 ROI tubes
│   ├── hysteresis_tube.py   # block events + RoiTube
│   ├── v2/                  # official: UF split → AABB composition
│   ├── roi_tube.py          # Stage2 loaders + overlay
│   └── tube_3d_viz.py
├── src/stage3_graph/        # signed multicut primitives (edges/nodes/solver)
├── scripts/
│   ├── 1_base_motion.py     # Stage1
│   ├── 2_gap_fusion.py      # Stage2 (Stage1 NPZ 필요)
│   ├── 3_roi_tube.py        # Stage3
│   └── setup/               # OpenCV CUDA Farneback
├── data/cache/              # Stage1 NPZ
├── data/stage2/             # Stage2 NPZ
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

# Stage2  NPZ → data/stage2/2_gap_fusion/<stamp>/
python scripts/2_gap_fusion.py

# Stage3  → outputs/stage3/3_roi_tube/<stamp>/
python scripts/3_roi_tube.py \
  --fusion_root data/stage2/2_gap_fusion/<stamp>
```

단일 영상: `--video_id VIRAT_S_...`  
목록 지정: `--video_list configs/virat_videos_le60s_sample30.txt`


## Stage ↔ 스크립트

| Stage  | 단계                         | 스크립트           | 입력 | 기본 출력 |
|--------|------------------------------|--------------------|------|-----------|
| Stage1 | Stage1-1, Stage1-2           | `1_base_motion.py` | video | `data/cache/`, `outputs/stage1/` |
| Stage2 | Stage2-1, Stage2-2, Stage2-3 | `2_gap_fusion.py`  | Stage1 NPZ + video | `data/stage2/`, `outputs/stage2/` |
| Stage3 | Stage3-1..4                  | `3_roi_tube.py`    | Stage2 fusion root | `outputs/stage3/3_roi_tube/` |

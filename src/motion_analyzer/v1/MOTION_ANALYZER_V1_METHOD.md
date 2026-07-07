# Motion Analyzer v1 — 방법론 상세 설명



이 문서는 `src/motion_analyzer/v1/` 파이프라인에서 사용하는 motion 분석 방법을 단계별로 설명합니다.



---



## 1. 개요



Motion Analyzer v1은 surveillance 영상(기본: VIRAT)에서 **프레임 → 그리드 셀(blob) → 시간축 tube → 영상 전체 특성** 순으로 motion을 계층적으로 집계합니다. 최종 목표는 VLM(Visual Language Model) 입력 후보를 생성하는 **Adaptive Input Plan**을 만드는 것입니다.



### 설계 원칙



- **큰 motion blob만 선택하지 않음**: energy(면적×강도)가 아닌 density, novelty, scale, persistence, object motion 등을 반영한 importance score를 별도로 계산합니다.

- **Pixel-level은 내부 중간 표현**: motion map과 frame feature는 box-level 이후 단계의 입력으로만 사용하며, 사용자-facing 시각화는 box-level overlay video(`motion_blobs_overlay.mp4`) 하나로 통합합니다.

- **공간 단위는 20×20 grid**: motion 검출은 픽셀 connected component가 아니라 **고정 그리드 셀** 단위로 수행합니다. 인접 active 셀은 **4-connected region**으로 병합한 뒤, **근접 blob 추가 병합**으로 사람의 팔·다리·머리 등 분할을 줄입니다.

- **False positive 억제 3종**: (1) global motion compensation, (2) texture flicker suppression, (3) optical flow coherence filter + 2프레임 persistence.



### 파이프라인 구조



```

원본 mp4

    │

    ▼  Stage 1: Pixel-level (내부)

    motion_maps.npz + sampled_gray.npz + frame_motion_features.csv

    │

    ▼  Stage 2: Box-level (20×20 grid)

    motion_blobs_raw/merged.csv + motion_blobs.csv + motion_blobs_overlay.mp4

    │

    ▼  Stage 3: Video-level

    motion_tubes.csv + video_category.json + adaptive_input_plan.json

```



### 입력 영상 경로

기본 `--video_dir`: `C:\Datasets\VIRAT\video\`

```
C:\Datasets\VIRAT\video\
├── videos-00\*.mp4
├── videos-01\*.mp4
└── videos-05\*.mp4
```

`--video_name`은 위 하위 폴더 전체에서 `{name}.mp4`를 자동 검색합니다. 특정 split만 쓰려면 `--video_dir C:\Datasets\VIRAT\video\videos-01`처럼 지정하세요.

### 실행 방법 (영상 1개)



```powershell

cd c:\Project\motion_analyzer

conda activate motion



# overlay만 필요하면 Stage 3 생략 (권장)

python scripts/run_motion_pipeline_v1.py `

  --video_name VIRAT_S_010000_00_000000_000165 `

  --skip_video



# box만 재실행 (pixel 캐시 있을 때)

python scripts/run_motion_box_level_v1.py `

  --video_name VIRAT_S_010000_00_000000_000165

```




---



## 2. Stage 1 — Pixel-level Motion Aggregation



**스크립트**: `scripts/run_motion_pixel_level_v1.py`  

**출력**: `Result/motion_analyzer_v1/{video_name}/_cache/`, `_data/`



### 2.1 프레임 샘플링



| 파라미터 | 기본값 | 설명 |

|---------|--------|------|

| `target_fps` | 5.0 | 분석용 샘플링 FPS |



원본 영상의 native FPS(예: 23.97)에서 `step = round(native_fps / target_fps)` 간격으로 프레임을 추출합니다.  

연속된 두 sampled frame `(t-1, t)` 쌍마다 motion map **1장**이 생성됩니다.



165초 VIRAT 영상 기준: 약 **791 frame pair** (5fps).



### 2.2 Global Motion Compensation (기본 ON)



프레임 전체가 같은 방향으로 흔들리는 camera/global motion을 object motion에서 분리합니다.



**알고리즘 (프레임 쌍 `(t-1, t)`):**



1. `goodFeaturesToTrack` + `calcOpticalFlowPyrLK`로 배경 feature 매칭

2. RANSAC `estimateAffinePartial2D`로 partial affine 추정 (translation + rotation + uniform scale)

3. prev frame을 current frame에 warp alignment

4. **aligned frame diff**를 motion map 기본 신호로 사용

5. Farneback flow에서 **global affine flow를 차감** → `residual_flow_magnitude`



**Frame-level 추가 feature** (`frame_motion_features.csv`):



| Feature | 설명 |

|---------|------|

| `global_motion_dx`, `global_motion_dy` | affine translation (px) |

| `global_motion_scale` | uniform scale |

| `global_motion_rotation` | rotation (rad) |

| `global_motion_score` | inlier ratio + transform magnitude ∈ [0,1] |

| `residual_motion_mean` | compensation 후 residual map 평균 |

| `residual_motion_area_ratio` | residual > 0.05 픽셀 비율 |

| `camera_motion_flag` | global score 높고 residual area 낮으면 true |



`camera_motion_flag == true`인 프레임은 Stage 2에서 `cell_motion_threshold`를 **2배**로 올려 대형 false bbox를 억제합니다.



| 파라미터 | 기본값 | 설명 |

|---------|--------|------|

| `use_global_compensation` | true | `--no-use_global_compensation`으로 비활성화 |



### 2.3 Motion Map 계산



1. BGR → Grayscale

2. (기본) global compensation 후 **residual** absolute difference

3. Gaussian blur (ksize=7, σ=1.5)

4. min-max normalization → [0, 1]



선택적으로 residual flow magnitude를 합성할 수 있습니다 (`--use_flow`).



```

motion_map_t = normalize( α × residual_frame_diff + β × residual_flow_mag )

```



| 파라미터 | 기본값 | 설명 |

|---------|--------|------|

| `use_flow` | false | optical flow 합성 여부 |

| `flow_scale` | 0.5 | flow 계산 해상도 축소 (0.5 ≈ 4× 빠름) |

| `alpha` | 1.0 | frame diff 가중치 |

| `beta` | 0.5 | flow magnitude 가중치 |



### 2.4 Frame-level Feature (기존 + global)



| Feature | 설명 |

|---------|------|

| `motion_magnitude_mean` | motion map 평균 |

| `motion_magnitude_sum` | motion map 합 |

| `motion_area_ratio` | threshold(0.05) 초과 픽셀 비율 |

| `motion_entropy` | 공간 분포 Shannon entropy (64 bins) |

| `motion_concentration` | 상위 10% 픽셀 energy 비율 |

| `small_motion_energy` | connected component ≤100px 영역 energy 합 |

| `new_motion_region_score` | 최근 5프레임 quiet 영역에 새 motion 발생 비율 |

| `scene_change_score` | frame diff mean 급증 (ratio − 1) |

| *(global)* | §2.2 표 참조 |



### 2.5 Stage 1 출력 파일



| 파일 | 설명 |

|------|------|

| `_cache/motion_maps.npz` | `(N, H, W)` float32 motion map 배열 |

| `_cache/sampled_gray.npz` | sampled frame grayscale (box 재실행 시 영상 재디코딩 방지) |

| `_data/frame_motion_features.csv` | 프레임별 feature |

| `_data/summary_pixel_v1.json` | Stage 1 처리 메타데이터 |



---



## 3. Stage 2 — Box-level Grid Motion Aggregation



**스크립트**: `scripts/run_motion_box_level_v1.py`  

**출력**: `Result/motion_analyzer_v1/{video_name}/`



### 3.1 20×20 Grid 분할 및 Region 병합



| 파라미터 | 기본값 | 설명 |

|---------|--------|------|

| `grid_rows` | 20 | 세로 그리드 수 |

| `grid_cols` | 20 | 가로 그리드 수 |

| `cell_motion_threshold` | 0.05 | 셀 평균 motion threshold |



**검출 파이프라인:**



1. 셀 평균 motion ≥ threshold → candidate (`camera_motion_flag` 시 threshold ×2)

2. optical flow filter 통과 → active cell

3. active cell **4-connected merge** → region bbox

4. *(신규)* **Nearby blob merge** — 같은 frame 내 bbox 거리 ≤ `nearby_merge_distance_ratio × frame_height`이면 추가 병합. 세로로 가깝고 작은 blob은 더 적극적으로 merge (팔·다리·머리 분할 완화)

5. **2프레임 persistence filter** — tube matching으로 길이 ≥2 track만 유지



| 파라미터 | 기본값 | 설명 |

|---------|--------|------|

| `use_nearby_blob_merge` | true | 근접 blob 병합 |

| `nearby_merge_distance_ratio` | 0.06 | frame height 대비 merge 거리 |

| `min_persistence_frames` | 2 | 최소 지속 프레임 수 |



### 3.2 Optical Flow 방향 일관성 필터



연속 sampled frame `(t-1, t)`에 Farneback flow (`flow_scale` 적용)를 계산하고 셀별 통계:



| 통계 | 설명 |

|------|------|

| `mean_flow_magnitude` | 셀 내 flow 크기 평균 (px) |

| `flow_direction_coherence` | flow 방향 circular mean resultant ∈ [0, 1] |



| 파라미터 | 기본값 | 설명 |

|---------|--------|------|

| `use_flow_filter` | true | `--no_flow_filter`로 비활성화 |

| `min_flow_magnitude` | 0.5 | 이보다 작으면 제외 |

| `min_direction_coherence` | 0.45 | magnitude ≥ threshold일 때 방향 불일치 제외 |

| `flow_scale` | 0.5 | box flow 계산 해상도 (pixel summary와 동기화) |



Gray frame은 `_cache/sampled_gray.npz`에서 우선 로드합니다 (없으면 영상에서 seek-read).



### 3.3 Texture Flicker / Static Detail Suppression



의자·난간·나무·그림자 등 **제자리 고주파 flicker**를 importance에서 감점합니다.



**추가 blob feature:**



| Feature | 설명 |

|---------|------|

| `edge_density` | bbox 내 Canny edge pixel ratio |

| `centroid_displacement` | 이전 프레임 matched blob 대비 중심 이동 (frame height 정규화) |

| `temporal_position_stability` | 위치 고정 정도 (displacement 낮을수록 ↑) |

| `texture_flicker_score` | edge↑ + displacement↓ + stability↑ + motion 반복 |

| `object_motion_score` | displacement + flow coherence + (1 − texture) |

| `global_motion_score` | 해당 frame의 global motion score (blob에 복사) |



| 파라미터 | 기본값 | 설명 |

|---------|--------|------|

| `use_texture_suppression` | true | texture/object score 및 importance 감점 |

| `texture_flicker_penalty` | 0.20 | importance 감점 가중치 |

| `global_motion_penalty` | 0.15 | importance 감점 가중치 |



### 3.4 Blob Importance Score



프레임 내 min-max 정규화 후:



```

blob_importance =

  0.22 × norm(motion_density)

+ 0.18 × norm(spatial_novelty_score)

+ 0.15 × norm(scale_score)

+ 0.15 × norm(local_peak_score)

+ 0.15 × norm(object_motion_score)

+ 0.10 × norm(flow_direction_coherence)

− 0.20 × norm(texture_flicker_score)    # texture_flicker_penalty

− 0.15 × norm(global_motion_score)      # global_motion_penalty

```



motion energy만 큰 grid region보다 **실제 object motion**이 상위에 오도록 설계했습니다.



### 3.5 Stage 2 출력 파일



| 파일 | 설명 |

|------|------|

| `motion_blobs_overlay.mp4` | merged blob overlay (5fps) |

| `_data/motion_blobs_raw.csv` | 4-connected merge 직후 (nearby merge 전) |

| `_data/motion_blobs_merged.csv` | nearby merge + texture enrichment 후 (persistence 전) |

| `_data/motion_blobs.csv` | persistence filter 적용 최종본 |

| `_data/frame_top_blobs.csv` | 프레임별 top-k |

| `_data/summary_box_v1.json` | Stage 2 요약 |



**Overlay 라벨 형식:** `R{rank} n={셀 수} imp={importance} tex={texture} gm={global}`



**튜닝 예시:**



```powershell

python scripts/run_motion_box_level_v1.py --min_direction_coherence 0.55

python scripts/run_motion_box_level_v1.py --nearby_merge_distance_ratio 0.08

python scripts/run_motion_box_level_v1.py --no-use_texture_suppression

python scripts/run_motion_box_level_v1.py --skip_overlay_video   # overlay 생략, 더 빠름

```



---



## 4. Stage 3 — Video-level Motion Tube Aggregation



**스크립트**: `scripts/run_motion_video_level_v1.py`  

**출력**: `Result/motion_analyzer_v1/{video_name}/_data/`



Stage 2의 `motion_blobs.csv`를 입력으로 tube matching, video feature, adaptive input plan을 생성합니다. (Stage 2 변경과 독립적으로 동작)



### 4.1 Temporal Tube Matching



```

score = 0.45 × IoU + 0.35 × center_sim + 0.20 × size_sim

```



Grid shortcut: 동일 `(grid_row, grid_col)` 且 동일 `num_grid_cells` → score = 1.0



| 파라미터 | 기본값 | 설명 |

|---------|--------|------|

| `max_gap` | 2 | sampled frame gap |

| `match_threshold` | 0.25 | 최소 score |

| `min_tube_length` | 2 | 최소 tube 길이 |



### 4.2–4.6 Tube Features / Motion Type / Adaptive Input Plan

**모듈**: `tube_features.py`, `video_features.py`, `input_plan_builder.py`, `input_plan_visualize.py`

Stage 3는 motion tube를 만들고, 비디오 전체의 **motion_type**을 결정한 뒤 **adaptive input plan**을 생성합니다.

#### Motion Type (최종 비디오 분류)

| motion_type | 의미 | VLM 입력 |
|-------------|------|----------|
| `background_motion` | 추가 ROI 불필요. 정적·반복 모션·카메라/글로벌 모션·일관된 translation | **Sparse Global만** |
| `event_motion` | 이벤트 관련 공간 디테일 보존을 위해 motion ROI 필요 | **Sparse Global + Motion ROI (최대 2)** |

출력: `video_category.json` (`motion_type`, `confidence`, `primary_reason`)

#### Adaptive Input Plan

출력: `adaptive_input_plan.json`

```json
{
  "motion_type": "event_motion",
  "global_inputs": [{ "type": "sparse_global", "fps": 0.25, "always_include": true }],
  "roi_inputs": [{ "type": "motion_roi", "roi_id": 1, ... }],
  "diagnostics": { "roi_pattern": "single_roi", ... }
}
```

- **Sparse Global**은 모든 비디오에 항상 포함 (`fps=0.25` 기본)
- `event_motion`일 때만 motion tube importance 순으로 ROI track 선택 (최대 2, 인접 tube 병합 가능)
- ROI sampling fps는 tube 내부 `mean_motion_speed`에 따라 0.5 / 1.0 / 2.0

#### Diagnostic ROI Patterns (비디오 카테고리 아님)

`diagnostics.roi_pattern`은 입력 구성 참고용이며 **최종 motion_type을 결정하지 않습니다**.

| roi_pattern | 설명 |
|-------------|------|
| `none` | ROI track 없음 (`background_motion`에서 흔함) |
| `single_roi` | 단일 ROI track |
| `multi_blob_merge` | 여러 tube/blob가 하나의 ROI로 병합 |
| `large_small_blobs` | 크기가 다른 2개 ROI 그룹 |

#### 시각화

**모듈**: `input_plan_visualize.py`  
**출력**: `{video_name}/input_plan_overlay.mp4`

| 요소 | 표시 |
|------|------|
| HUD | `motion_type`, `roi_pattern` (diagnostic) |
| SG 마커 | Sparse Global 샘플 프레임 |
| ROI 1 / ROI 2 | 선택된 motion ROI bbox (색상: 녹색 / 주황) |
| Raw blobs | `--debug` 시에만 (`motion_blobs_raw.csv`) |

Stage 2 blob overlay (`motion_blobs_overlay.mp4`)도 `--debug` 시에만 생성됩니다.

*(Tube matching, tube importance, motion_type 분류 규칙 상세는 `tube_features.py`, `video_features.py` 참조)*



---



## 5. 출력 디렉터리 구조



```

Result/motion_analyzer_v1/

├── MOTION_ANALYZER_V1_METHOD.md

└── {video_name}/

    ├── input_plan_overlay.mp4          ← Human output (motion_type + ROI + sparse global)
    ├── motion_blobs_overlay.mp4        ← Debug only (--debug on box stage)

    ├── _cache/

    │   ├── motion_maps.npz

    │   └── sampled_gray.npz          ← gray frame 캐시

    └── _data/

        ├── frame_motion_features.csv

        ├── summary_pixel_v1.json

        ├── motion_blobs_raw.csv

        ├── motion_blobs_merged.csv

        ├── motion_blobs.csv

        ├── frame_top_blobs.csv

        ├── summary_box_v1.json

        ├── adaptive_input_plan.json
        ├── video_category.json
        ├── video_summary.md
        └── summary_video.json

```



---



## 6. 성능 — 왜 오래 걸리나? / 줄이는 방법



### 6.1 시간이 걸리는 이유 (165초 VIRAT 1개 기준)



| 구간 | 대략적 소요 | 원인 |

|------|------------|------|

| Stage 1 pixel | ~2–3분 | 791 frame pair × (global comp + Farneback flow @ 1280×720) + 전체 영상 1회 디코딩 |

| Stage 2 box | ~2–3분 | 791회 Farneback flow + blob feature + overlay MP4 인코딩 |

| **smoke comparison** | **~15–20분/영상** | before pixel + before box + after pixel + after box = **4회** 전체 파이프라인 |



즉 **20분**은 단일 pass가 아니라 **비교 스크립트가 4배** 돌아서입니다. 영상 1개 + `--skip_video`면 **약 4–6분**이 일반적입니다.



### 6.2 속도 개선 (v1.1 적용)



| 방법 | 효과 |

|------|------|

| `sampled_gray.npz` 캐시 | box 단계에서 영상 **재디코딩 제거** |

| `flow_scale=0.5` (기본) | Farneback **약 4× 빠름** (품질 trade-off) |

| `--skip_video` | Stage 3 생략 |

| `--skip_overlay_video` | MP4 인코딩 생략 (box 재튜닝 시) |

| pixel 캐시 유지 + box만 재실행 | 파라미터 튜닝 반복 시 pixel 생략 (`--skip_pixel`) |

**빠른 반복 예시:**



```powershell

# 최초 1회만 pixel

python scripts/run_motion_pixel_level_v1.py --video_name VIRAT_S_010000_00_000000_000165



# 이후 box만 여러 번 (overlay 없이)

python scripts/run_motion_box_level_v1.py `

  --video_name VIRAT_S_010000_00_000000_000165 `

  --skip_overlay_video



# 더 빠른 flow (품질↓)

python scripts/run_motion_pipeline_v1.py `

  --video_name VIRAT_S_010000_00_000000_000165 `

  --skip_video --target_fps 3.0

```



---



## 7. 주요 파라미터 요약



| 단계 | 파라미터 | 기본값 |

|------|---------|--------|

| Pixel | target_fps | 5.0 |

| Pixel | use_global_compensation | true |

| Pixel | flow_scale | 0.5 |

| Pixel | use_flow | false |

| Box | grid_rows × grid_cols | 20 × 20 |

| Box | cell_motion_threshold | 0.05 |

| Box | use_flow_filter | true |

| Box | min_flow_magnitude | 0.5 px |

| Box | min_direction_coherence | 0.45 |

| Box | use_nearby_blob_merge | true |

| Box | nearby_merge_distance_ratio | 0.06 |

| Box | use_texture_suppression | true |

| Box | min_persistence_frames | 2 |

| Tube | max_gap | 2 |

| Plan | sparse_global_fps | 0.25 |
| Plan | max_roi_tracks | 2 |
| Overlay | debug raw blobs | `--debug` (box/video stage) |

---



## 8. 코드 ↔ 모듈 매핑



| 단계 | 스크립트 | 핵심 모듈 |

|------|---------|----------|

| Pixel | `run_motion_pixel_level_v1.py` | `v1/pixel_motion.py`, `v1/global_motion.py`, `v1/features.py` |

| Box | `run_motion_box_level_v1.py` | `v1/blob_extractor.py`, `v1/blob_merge.py`, `v1/blob_features.py`, `v1/blob_visualize.py` |

| Video | `run_motion_video_level_v1.py` | `v1/tube_builder.py`, `v1/tube_features.py`, `v1/video_features.py`, `v1/input_plan_builder.py`, `v1/input_plan_visualize.py` |

| 전체 | `run_motion_pipeline_v1.py` | 위 3단계 순차 실행 (`motion_analyzer.v1`) |

---



## 9. 참고: v1의 한계



- **Grid 고정 해상도**: 20×20은 해상도에 따라 셀 크기가 달라집니다.

- **flow_scale < 1**: 속도↑, 미세 motion 검출↓ trade-off.

- **Token budget**: Adaptive Input Plan은 예상 count만 저장, VLM token clipping 미구현.

- **Category 분류**: rule-based 임시 분류.



---



*문서 최종 갱신: global motion compensation, texture flicker suppression, nearby blob merge, persistence filter, gray cache, flow_scale 최적화*



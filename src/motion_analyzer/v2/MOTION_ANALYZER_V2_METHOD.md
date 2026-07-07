# Motion Analyzer v2 — 방법론 (상세)

`src/motion_analyzer/v2/` 파이프라인의 **현재 구현**을 단계별로 기술합니다.  
핵심은 **motion-first**: VLM에 넘길 ROI의 identity는 **motion blob group**이고, YOLO+ByteTrack은 **보조(bbox 확장·라벨·진단)** 용도입니다.

---

## 목차

1. [개요](#1-개요)
2. [전체 파이프라인](#2-전체-파이프라인)
3. [Motion 감지 — Stage 1 Pixel](#3-motion-감지--stage-1-pixel)
4. [Motion Blob — Stage 2 Box](#4-motion-blob--stage-2-box)
5. [영상 분류 — Stage 3 Video](#5-영상-분류--stage-3-video)
6. [객체 트래킹 — ByteTrack](#6-객체-트래킹--bytetrack)
7. [Motion Evidence & Moving/Stationary 분류](#7-motion-evidence--movingstationary-분류)
8. [Event ROI 선정 (핵심)](#8-event-roi-선정-핵심)
9. [Adaptive Input Plan 확정](#9-adaptive-input-plan-확정)
10. [Input Builder 소비 규칙](#10-input-builder-소비-규칙)
11. [설정값 표](#11-설정값-표)
12. [산출물·모듈 매핑](#12-산출물모듈-매핑)

---

## 1. 개요

### 1.1 목적

CCTV 영상에서 VLM 입력용 **Adaptive Input Plan**을 생성합니다.

| `motion_type` | VLM 입력 |
|---------------|----------|
| `background_motion` | **Sparse Global만** (`roi_inputs = []`) |
| `event_motion` | Sparse Global + **Motion ROI 최대 2개** (`plan["roi_inputs"]`) |

### 1.2 설계 원칙

| 원칙 | 내용 |
|------|------|
| Motion 주도 | ROI·tube의 1차 근거는 motion map / motion blob |
| Tracker 보조 | ByteTrack `track_id`는 ROI identity가 **아님**. bbox 확장·근접 객체 흡수만 |
| Plan 단일 소스 | 최종 crop/overlay/Input Builder는 **`plan["roi_inputs"]`만** 사용 (`frames[]` track crop 사용 안 함) |
| 구간 제한 | ROI bbox는 `[start_frame_idx, end_frame_idx]` 안에서만 표시 (forward-fill 금지) |
| 부분 재실행 | pixel / box / tracking / event ROI를 독립 refresh 가능 |

### 1.3 실행

```powershell
python scripts/run_motion_pipeline_v2.py --video_name <STEM> --target_fps 5.0
python scripts/run_event_roi_refresh.py --output_dir Result/motion_analyzer_v2 --video_name <STEM> --write_overlays
python scripts/run_adaptive_input_builder.py --output_dir Result/motion_analyzer_v2 --video_name <STEM>
```

---

## 2. 전체 파이프라인

```
mp4
 ├─ Stage 1 Pixel     → motion_maps.npz (suppressed), frame_motion_features.csv
 ├─ Stage 2 Box       → motion_blobs.csv
 ├─ Stage 3 Video     → motion_tubes.csv, video_category.json (motion_type)
 │
 ├─ v2 ByteTrack      → object_tracks.csv
 ├─ Motion evidence   → moving/stationary_object_tracks.csv
 │
 ├─ Event ROI         → event_roi_tracks.json  (event_motion만)
 ├─ Plan finalize     → adaptive_input_plan.json, adaptive_input_plan_debug.json
 │
 └─ Overlays / Input Builder
```

**진입점**: `v2/pipeline.py` → `process_video_v2()`  
**모듈**: `event_roi_grouping.py`, `track_motion_features.py`, `plan_finalize.py`, `adaptive_input/plan_resolver.py`

---

## 3. Motion 감지 — Stage 1 Pixel

**모듈**: `v1/pixel_motion.py`, `v1/global_motion.py`, `v1/background_jitter.py`  
**스크립트**: `run_motion_pixel_level_v1.py`

### 3.1 샘플링

- 원본 영상에서 `target_fps`(기본 5.0)로 프레임 균등 샘플링
- `frame_idx`는 **원본 영상의 native frame index** (5fps@30fps 영상이면 간격 6)
- `frame_motion_features.csv`에 프레임별 메타 저장

### 3.2 프레임 쌍 motion map

연속 샘플 프레임 `(t-1, t)`에 대해:

1. **Grayscale 변환**
2. **Global motion compensation** (`use_global_compensation=True`, 기본)
   - Good features to track + RANSAC affine 추정 (`global_motion.py`)
   - 실패 시 median residual flow fallback
   - 레이어 분해:
     - `original_map`: 보상 전 frame diff
     - `global_motion_map`: 카메라 움직임에 해당하는 성분
     - `residual_map`: original − global
     - `suppressed_map`: residual에 **background jitter 억제** 적용
3. **Background jitter 억제** (`background_jitter.py`)
   - residual/flow의 방향 일관성·넓은 영역 coverage로 흔들림 점수 산출
   - `background_jitter_flag`, `background_jitter_score`를 frame feature에 기록
   - 억제 후 map → **`motion_maps.npz`에 저장** (box stage 입력)

### 3.3 프레임 수준 feature (`frame_motion_features.csv`)

주요 컬럼:

| 컬럼 | 의미 |
|------|------|
| `motion_magnitude_mean` | suppressed map 평균 |
| `motion_area_ratio` | motion 픽셀 비율 |
| `global_motion_score` | 카메라/전역 움직임 강도 |
| `residual_motion_area_ratio` | residual 영역 비율 |
| `camera_motion_flag` | 카메라 움직임 프레임 여부 |
| `background_jitter_flag` | 배경 흔들림 프레임 |
| `scene_change_score` | 장면 전환 점수 |
| `motion_concentration` | motion 공간 집중도 |

### 3.4 진단 오버레이

`motion_map_layers.npz` 기반: `original_`, `global_`, `residual_`, `suppressed_motion_overlay.mp4`

---

## 4. Motion Blob — Stage 2 Box

**모듈**: `v1/blob_extractor.py`, `v1/blob_features.py`, `v1/blob_merge.py` 등  
**스크립트**: `run_motion_box_level_v1.py`

### 4.1 Grid blob 추출

각 샘플 프레임의 suppressed motion map에 대해:

1. **20×20 고정 그리드** 분할 (`BlobExtractConfig`)
2. 셀 평균 motion ≥ `cell_motion_threshold` (0.05) → active cell
3. **Optical flow 필터** (Farneback, `flow_scale=0.5`)
   - `min_flow_magnitude`, `min_direction_coherence` (0.45) 미달 셀 제거
4. active 셀 **4-connected** 병합 → `RawBlob` (bbox)
5. **blob feature** 산출: `blob_importance`, `motion_density`, `flow_direction_coherence`, `bbox_area_ratio` 등

### 4.2 후처리

| 단계 | 설명 | 플래그 |
|------|------|--------|
| Texture flicker 억제 | 정적 텍스처 깜빡임 blob 제거 | `use_texture_suppression` |
| Nearby blob merge | 인접 blob 병합 (사람 신체 분할 완화) | `use_nearby_blob_merge`, `nearby_merge_distance_ratio=0.06` |
| Persistence | 2프레임 연속성 필터 | v1 기본 |

### 4.3 산출물

| 파일 | 내용 |
|------|------|
| `motion_blobs_raw.csv` | grid 추출 직후 |
| `motion_blobs_merged.csv` | 인접 병합 후 |
| `motion_blobs.csv` | **최종 blob** (Event ROI 입력) |

**blob 1행 예시 필드**: `frame_idx`, `timestamp_sec`, `x1,y1,x2,y2`, `blob_importance`, `bbox_area_ratio`, `flow_direction_coherence`, `blob_id_in_frame`

---

## 5. 영상 분류 — Stage 3 Video

**모듈**: `v1/video_features.py`, `v1/tube_builder.py`  
**스크립트**: `run_motion_video_level_v1.py`

### 5.1 Tube (v1)

- blob을 프레임 간 IoU로 연결 → `motion_tubes.csv`
- tube importance, `small_tube_flag`, `num_frames` 등 산출
- **v2 최종 ROI에는 직접 사용하지 않음** (motion_type 분류·legacy plan에만 영향)

### 5.2 `motion_type` 분류 (`classify_video_category`)

**출력은 오직 두 가지**: `background_motion` | `event_motion`

규칙 기반 점수 경쟁 (`MotionTypeConfig` 임계값):

**background_motion 후보 규칙** (예):
- 전역 motion·area 모두 낮고 strong tube 없음
- `mean_global_motion_score` 높고 residual area 낮음 (카메라 pan)
- 프레임 다수 `camera_motion_flag`
- tube 수 적거나 importance 약함
- 반복적 translation + local importance 낮음

**event_motion 후보 규칙** (예):
- `strong_tube_count ≥ 3` 또는 `max_tube_importance` 높음
- `mean_motion_concentration` 높음 (국소 peak)
- `local_motion_importance_ratio` 높음
- `multi_motion_score` 높고 tube 다수

최종: background vs event 규칙 중 **더 높은 confidence** 선택 → `primary_reason` 기록.

`legacy_category_for_input_plan()`은 **sparse global 보조 frame 선택**용 legacy 라벨만 생성하며, `motion_type` 결정과 분리됨.

---

## 6. 객체 트래킹 — ByteTrack

**모듈**: `v2/object_tracker.py`  
**조건**: `tracker ∈ {bytetrack, botsort}` (기본 `bytetrack`)

### 6.1 알고리즘

1. 샘플 프레임을 **시간 순**으로 순회
2. Ultralytics `YOLO(model).track(persist=True, tracker=bytetrack.yaml)`
3. `DEFAULT_TARGET_CLASSES`: person, car, bicycle, motorcycle, bus, truck, bag류 등
4. `conf_threshold=0.15` (기본)
5. 각 detection에 `track_id` 부여 (샘플 타임라인 기준 연속 ID)

### 6.2 산출물 `object_tracks.csv`

프레임별 observation: `track_id`, `class_name`, `x1,y1,x2,y2`, `confidence`, `center_x/y`, `bbox_area_ratio`

### 6.3 역할 (중요)

| 용도 | 사용 여부 |
|------|-----------|
| ROI identity / group_id | **사용 안 함** |
| ROI bbox 확장 (auxiliary) | **사용** (`load_auxiliary_track_obs`) |
| Moving track 진단·overlay | 사용 |
| `plan["frames"]` track crop | 생성되지만 **최종 roi_inputs에 반영 안 함** |

캐시: `object_tracks.csv` 존재 시 재사용. `--force_retrack`으로 재실행.

---

## 7. Motion Evidence & Moving/Stationary 분류

**모듈**: `v2/track_motion_features.py`

ByteTrack은 **모든 검출 객체를 track**하지만, 이후 단계에서 motion map 증거로 **moving / stationary**를 나눕니다.

### 7.1 프레임별 observation 분류 (`_classify_observation`)

각 `(track_id, frame_idx)` observation에 대해 motion map·blob을 조회:

| 타입 | 의미 |
|------|------|
| `observed_motion` | bbox 내부 motion density/coverage 충분, blob 연관 |
| `weak_motion` | expanded 영역에서 약한 motion (차량 단독으로는 moving 확정에 불충분) |
| `detector_only` | 검출만 있고 motion 증거 부족 (카메라 흔들림·정적 배경 등) |

**억제 휴리스틱**:
- `_is_widespread_camera_shake`: 넓은 영역 diffuse motion
- `_is_static_background_pattern`: 주변은 움직이나 bbox 내부 flat
- `_is_uniform_flicker`: peak-to-mean 낮은 깜빡임
- `shake_score` 높을 때 threshold 가중 (`SHAKE_DISCOUNT=0.55`)

**Blob 연관**: track bbox와 blob IoU ≥ 0.05 또는 center inside → `associated_motion_bbox`

### 7.2 Track 수준 feature (`aggregate_track_features`)

트랙별 집계:
- `observed_motion_ratio`, `mean_motion_density`, `mean_motion_coverage`
- `normalized_track_displacement` (대각선 대비 net displacement)
- `mean_speed`, `bbox_jitter_score` (trajectory / net disp)
- `track_motion_importance` (moving track만 weighted sum 정규화)

### 7.3 Moving vs Stationary (`_classify_moving_track`)

**차량** (`VEHICLE_CLASSES`):
- `_is_stationary_vehicle`: 장시간 visible + displacement·speed 낮음, 또는 jitter만 큼
- `_vehicle_moving_criteria`: displacement ≥ 0.06 **또는** (고 density + observed frames + duration) + condition A/B/C
  - A: `observed_motion_ratio ≥ 0.15`, density ≥ 0.55, coverage ≥ 0.035
  - B: displacement ≥ 0.06, speed ≥ 15, jitter ≤ 4
  - C: max density ≥ 1.8, observed frames ≥ 2, duration ≥ 0.4s

**사람**: criteria 더 관대 (`PERSON_OBSERVED_RATIO=0.08`, displacement ≥ 0.015 등)

**결과**: `moving_object_tracks.csv` / `stationary_object_tracks.csv`  
→ **최종 Event ROI 선택에는 사용하지 않음** (진단·plan frames 보조).

---

## 8. Event ROI 선정 (핵심)

**모듈**: `v2/event_roi_grouping.py`  
**적용**: `motion_type == "event_motion"` 일 때만 (`build_event_roi_pipeline`)

### 8.1 입력·출력

| 입력 | 역할 |
|------|------|
| `motion_blobs.csv` | **Primary** — group 형성·점수·bbox |
| `frame_motion_features.csv` | jitter/scene/global penalty |
| `object_tracks.csv` | **Auxiliary** — bbox 흡수만 (`load_auxiliary_track_obs`) |

| 출력 | 내용 |
|------|------|
| `event_roi_candidates.json` | 전체 후보 + `bbox_sequence` |
| `event_roi_tracks.json` | 최종 선택 ≤ 2 |
| `event_roi_debug.json` | build/selection rejected, component scores |

### 8.2 Phase A — Blob 필터 (`filter_jitter_blobs`)

제거 조건:
- `bbox_area_ratio ≥ 0.55` (전체 화면·scene-change blob)
- `background_jitter_flag` 프레임에서 대형+고 coherence blob
- `mean_area ≥ 0.40` & `mean_coh ≥ 0.68` 그룹 → `background_jitter_group`

### 8.3 Phase B — Spatiotemporal Grouping (`_link_spatiotemporal_groups`)

Union-Find로 blob을 **event group**으로 묶음. `group_id`가 ROI identity.

**Step B1 — 같은 프레임 flow 병합** (`flow_group_merge.group_frame_blobs_by_flow`):
- 동일 `frame_idx` 내 flow 방향 유사 blob 먼저 union

**Step B2 — cross-frame linking**:
- **시간**: sampled timeline **position** 차이 ≤ `round(temporal_window_sec × sampled_fps)` (기본 0.8s × 5 = 4 frames)
  - ⚠ native `frame_idx` 차이가 아님 (30fps 영상 5fps 샘플 시 idx 간격 6, position 간격 1)
- **공간**: 정규화 중심거리 ≤ `spatial_merge_dist` (0.10)
- **Flow**: `|coh_a - coh_b| ≤ flow_coherence_delta` (0.35), 둘 다 낮으면 통과

### 8.4 Phase C — Timeline & 최소 길이 (`_timeline_indices_for_group`)

blob이 active인 sampled frame 집합에 대해:

1. `_bridge_active_positions`: 샘플 position gap ≤ `max_active_gap_frames`(2) 이면 중간 채움
2. 앞뒤 `context_frames`(4) 확장
3. `min_roi_frames`(4) 미만이면 timeline 양끝으로 확장
4. `min_roi_duration_sec`(2.0) 미만이면 양끝으로 추가 확장

→ 짧은 통과 event도 ROI tube 최소 길이 충족.

### 8.5 Phase D — Bbox sequence (`_build_bbox_sequence`)

각 timeline position `fi`에 대해:

1. **Active frame** (blob 있음):
   - `group_union` = 그룹 전체 bbox union
   - `frame_union` = 해당 프레임 blob union
   - base = union(group_union, frame_union)
2. **Context frame** (blob 없음): base = `group_union` (그룹 공간 envelope 유지)
3. **Margin 확장** (`_expand_bbox_context`):
   - `mean_area ≤ small_event_area_ratio`(0.012) → `small_event_margin`(0.38)
   - 그 외 → `context_margin`(0.28)
4. **Auxiliary 흡수** (`_absorb_auxiliary_detections`):
   - IoU ≥ 0.02 또는 중심거리 ≤ 0.09 인 ByteTrack box를 union
   - `auxiliary_track_ids` 기록 (identity 아님)
5. **Smoothing**:
   - 프레임 간 `smooth_alpha=0.35` EMA
   - tube 전체 `_smooth_bbox_tube` (`tube_smooth_alpha=0.22`)

결과: `bbox_sequence[]` = `{frame_idx, timestamp_sec, bbox}` 리스트

### 8.6 Phase E — 점수 (`_score_group_components`)

컴포넌트별 decomposition (debug에 기록):

#### motion_score
```
motion_raw = mean(blob_importance)
persistence_sec = num_active_frames / sampled_fps
small_persistent = area ≤ 0.012 AND persistence ≥ 0.4s
motion_score = 0.45·min(1, 2·motion_raw)
             + 0.35·min(1, persistence/1.5)
             + 0.20·(small_persistent ? 1 : 0.35·min(1, persistence/2))
```

#### object_proximity_score & interaction_score
- `bbox_sequence` 각 프레임에서 확장 bbox 근처 ByteTrack 검사 (`PROXIMITY_DIST=0.14`)
- person / vehicle 클래스 카운트
- `interaction_frames`: 동일 프레임에 person+vehicle 동시 근접
```
object_proximity = min(1, (person_frames + vehicle_frames) / n_seq · 1.2)
interaction      = min(1, interaction_frames / n_seq · 2.5 + bonuses)
```
- 소형 event + person+vehicle → interaction +0.15
- 소형 + vehicle only → proximity/interaction 추가 가중

#### background_flow_penalty (감점)
다음이 겹치면 penalty ↑ (도로 전체 흐름·pan):
- 고 flow coherence + 큰 area + 긴 track duration
- vehicle_only + coherent flow
- jitter 프레임 비율
- 높은 `global_motion_score`
- 큰 union area + interaction 없음

`is_background_flow_region=True` → **ROI 후보에서 제외** (debug `background_flow_regions`에 기록)

#### event_score (최종 ranking)
```
event_score = 0.18·motion_score
            + 0.22·object_proximity_score
            + 0.42·interaction_score
            + 0.12·(small_localized AND (person OR vehicle nearby))
            - 0.55·background_flow_penalty
            - 0.15·scene_change_penalty
            - jitter_penalty (large area + high coh 그룹)
```

`event_score < min_event_score`(0.08) → build reject.

### 8.7 Phase F — 선택 (`select_event_roi_tracks`)

`event_score` 내림차순 greedy, 최대 `max_roi_tracks=2`:

각 후보에 대해 이미 선택된 ROI와:
1. **Spatial-Temporal NMS**: `temporal_iou ≥ 0.40` AND `spatial_iou ≥ 0.35` → 제외
2. **Shared auxiliary track**: 동일 `auxiliary_track_ids` 교집합 + temporal_iou ≥ 0.2 → 제외

선택된 track → `event_roi_tracks.json`:
```json
{
  "roi_id": 1,
  "primary_source": "blob_group",
  "group_id": 25,
  "auxiliary_track_ids": [14, 19],
  "event_score": 0.837,
  "motion_score": 0.71,
  "interaction_score": 0.45,
  "bbox_sequence": [...],
  "start_frame_idx": ..., "end_frame_idx": ...
}
```

### 8.8 ROI 표시 규칙 (`roi_bbox_at_frame`)

- `frame_idx < start` 또는 `frame_idx > end` → **None** (박스 없음)
- 구간 내: exact match 또는 **이전 keyframe hold** (구간 밖 forward-fill 없음)

---

## 9. Adaptive Input Plan 확정

**모듈**: `v2/plan_finalize.py`, `v2/track_input_plan.py`, `v2/pipeline.py`

### 9.1 Plan 생성 흐름

1. `build_track_adaptive_input_plan()`:
   - sparse global frames (~0.2 fps)
   - moving track / fallback tube 기반 `plan["frames"]` entry 생성
   - **`roi_inputs = []`** (track에서 ROI 미생성)
2. `build_event_roi_pipeline()` → `event_tracks`
3. `finalize_adaptive_input_plan()`:

| motion_type | `roi_inputs` | `roi_source` |
|-------------|--------------|--------------|
| `background_motion` | `[]` | `none` |
| `event_motion` | `event_tracks` (≤2) | `blob_group` or `none` |

4. `canonicalize_plan_roi_tracks()`:
   - `background_motion` → 항상 `[]`
   - `roi_source ∈ {blob_group, none}` → `frames[]`에서 crop 재구성 **금지**

### 9.2 Debug (`adaptive_input_plan_debug.json`)

```json
{
  "final_motion_type": "event_motion",
  "reason": "persistent valid motion tubes",
  "num_roi_inputs": 2,
  "roi_source": "blob_group",
  "sparse_global_count": 18
}
```

---

## 10. Input Builder 소비 규칙

**모듈**: `adaptive_input/plan_resolver.py`, `adaptive_input/builder.py`

`resolve_input_plan(plan)`:

| motion_type | sparse_global | roi_frames | event_anchors |
|-------------|---------------|------------|---------------|
| `background_motion` | ✓ | **없음** | 없음 |
| `event_motion` | ✓ | `plan["roi_inputs"]`의 `bbox_sequence`만 | plan anchors |

ROI crop은 **`_collect_roi_frames` → `canonicalize_plan_roi_tracks`** 경로만 사용.  
`plan["frames"]`의 `crop_bbox` (track/fallback)는 ROI crop에 **사용하지 않음**.

산출: `adaptive_input/sparse_global/`, `roi1/`, `roi2/`, `input_builder_visualization.mp4`

---

## 11. 설정값 표

### EventRoiConfig (ROI 선정)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `temporal_window_sec` | 0.8 | 그룹 linking 시간 창 |
| `spatial_merge_dist` | 0.10 | 정규화 중심거리 |
| `flow_coherence_delta` | 0.35 | flow 호환 허용 차이 |
| `min_roi_duration_sec` | 2.0 | ROI 최소 길이 |
| `min_roi_frames` | 4 | ROI 최소 sampled frame |
| `context_frames` | 4 | 짧은 event 전후 확장 |
| `context_margin` | 0.28 | bbox 확장 |
| `small_event_margin` | 0.38 | 소형 event 추가 확장 |
| `small_event_area_ratio` | 0.012 | 소형 event 면적 기준 |
| `nms_temporal_iou` | 0.40 | 후보 NMS |
| `nms_spatial_iou` | 0.35 | 후보 NMS |
| `min_event_score` | 0.08 | 후보 최소 점수 |
| `max_active_gap_frames` | 2 | active frame gap bridge |
| `tube_smooth_alpha` | 0.22 | bbox tube smoothing |
| `background_flow_area_ratio` | 0.06 | 도로 흐름 penalty |
| `background_flow_coherence` | 0.72 | 도로 흐름 penalty |

### Track motion (주요 임계값)

| 상수 | 값 | 용도 |
|------|-----|------|
| `OBSERVED_MIN_DENSITY` | 0.35 | observed_motion |
| `COND_B_NORM_DISPLACEMENT` | 0.06 | 차량 이동 |
| `VEHICLE_MOVING_MIN_DISPLACEMENT` | 0.06 | 차량 이동 |
| `VEHICLE_STATIONARY_NORM_DISP` | 0.02 | 정지 차량 |
| `SHAKE_GLOBAL_SCORE` | 0.42 | camera shake gate |

### PipelineV2Config

| 파라미터 | 기본값 |
|----------|--------|
| `target_fps` | 5.0 |
| `tracker` | `bytetrack` |
| `detector_conf` | 0.15 |
| `use_global_compensation` | true |
| `use_texture_suppression` | true |
| `use_nearby_blob_merge` | true |

---

## 12. 산출물·모듈 매핑

### 핵심 데이터 흐름

```
motion_blobs.csv ─────────────────────────► Event ROI (primary)
frame_motion_features.csv ────────────────► jitter/scene penalty
object_tracks.csv ────────────────────────► bbox expansion only
                                              (NOT roi identity)

event_roi_tracks.json ──finalize──► plan["roi_inputs"]
plan_resolver ────────────────────► adaptive_input/ crops
```

### 모듈表

| 단계 | 모듈 |
|------|------|
| Pixel motion | `v1/pixel_motion.py`, `global_motion.py`, `background_jitter.py` |
| Blob | `v1/blob_extractor.py`, `blob_features.py` |
| motion_type | `v1/video_features.py` |
| Tracking | `v2/object_tracker.py` |
| Motion evidence | `v2/track_motion_features.py` |
| **Event ROI** | `v2/event_roi_grouping.py`, `flow_group_merge.py` |
| Plan finalize | `v2/plan_finalize.py`, `v1/input_plan_visualize.py` |
| Input Builder | `adaptive_input/plan_resolver.py`, `builder.py` |
| Pipeline | `v2/pipeline.py` |

### Refresh 스크립트

| 스크립트 | 범위 |
|----------|------|
| `run_motion_pipeline_v2.py` | 전체 |
| `run_event_roi_refresh.py` | ROI만 (tracking 없음) |
| `run_tracking_refresh_v2.py` | tracking + plan |
| `run_adaptive_input_builder.py` | VLM crop/시각화 |

---

## 부록: 알고리즘 관계 요약

```
[Motion 감지]  grid blob + global compensation + jitter 억제
      ↓
[Blob group]   Union-Find (시간·공간·flow) → group_id
      ↓
[ROI tube]     context 확장 + union bbox + ByteTrack 흡수 + smooth
      ↓
[점수]         motion + proximity + interaction − background_flow
      ↓
[선택]         top-2 + NMS → plan["roi_inputs"]

[ByteTrack]    병렬 실행 — motion 증거로 moving/stationary 분류 (진단용)
               ROI identity와 분리됨
```

---

*문서 갱신: motion 감지 · ByteTrack · blob-group Event ROI · plan finalize · Input Builder 소비 규칙 — 현재 코드 기준*

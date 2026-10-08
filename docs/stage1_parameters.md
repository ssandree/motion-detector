# Stage1 공식 파라미터

소스: `src/motion_analyzer/config.py`, `motion_map.py`, `stage1_gpu.py`.  
실행: `scripts/1_base_motion.py`. GPU 기본, `--no_gpu` 시 CPU 동일 식.

Stage1 상수(Farneback / aperture / P13 / T5 / floor)는 2026-08-19 이후 잠금. 2026-09-24 코드와 다시 대조했고 **알고리즘은 그대로**다.

기본 출력: `data/cache/<video_id>/*_base_motion_U1.npz`.  
실험 캐시 예: `data/stage1/p13_t5_sample30_6x6_tanstrip_floor05/`  
(`spatial_win=6`, `p15_window=13`, `p13_min=0.8`, 하드킬 없음, **셀 mag floor 0.5**).

설정 키 이름 `STAGE1_P15_WINDOW` / NPZ 키 `p15_window` / `P15`는 예전 P15에서 온 이름이다. **기본 창 길이는 13프레임 (P13)** 이다. 문턱 키는 `STAGE1_P13_MIN`.

---

## 파이프라인 순서

Gap1만 Farneback 한다. 긴 gap은 Stage2에서 Stage1 `U1`을 더한다.

1. 원본 비디오 → **5 fps** 샘플 (BGR → gray)
2. gray를 **1/4** (`INTER_AREA`, 추가 Gaussian blur 없음)
3. CUDA Farneback Gap1 (연속 샘플 프레임)
4. flow를 원본 픽셀 단위로 스케일 (`×4`)
5. dense **`‖v‖ < 0.5 → 0`** (tanstrip 전)
6. structure tensor → **tanstrip** (접선 성분 제거) + **R 가중** (벡터를 0으로 만들지는 않음)
7. **6×6** R-가중 공간 평균, stride 4 → **16 px** 셀. 시간 혼합 없음. mag=0 픽셀도 분모에 들어감 (all-pixel)
8. **P13 ≥ 0.8** 아니면 그 프레임 셀 벡터를 0
9. **T5**: `[f−2, f+2]` 평균, **0 포함**
10. 셀 **`‖U1‖ ≤ 0.5 → 0`**

출력 그리드: 원본 해상도 / 16 px. `T = n_sampled − 1` (`align_start=1`).  
예: 1920×1080 → ¼는 480×270, 16 px 셀은 `(T, 68, 120, 2)` 근처(마지막 행은 잘릴 수 있음). 720×1280이면 `(T, 45, 80, 2)`.

---

## 샘플 / 해상도

| 파라미터 | 값 | 설정 키 | 설명 |
|---|---|---|---|
| sampling fps | 5.0 | `PipelineConfig.sampling_fps` | Gap1 ≈ 0.2 s |
| Farneback 입력 스케일 | 0.25 | `INPUT_SCALE` | 720p → 180×320 |
| 리사이즈 | `INTER_AREA` | — | `blur_sigma=None` |
| 출력 stride (1/4 그리드) | 4 | `RESIZED_BASE_BLOCK` | 셀 한 칸 = 원본 16 px |
| 출력 셀 | 16 px | `ORIGINAL_CELL_PX` | |

---

## Farneback (`cv2.cuda.FarnebackOpticalFlow`)

| 파라미터 | 값 | 설정 키 |
|---|---|---|
| device | `cuda` | `FARNEBACK_DEVICE` |
| numLevels | 3 | `FARNEBACK_LEVELS` |
| pyrScale | 0.5 | `FARNEBACK_PYR_SCALE` |
| fastPyramids | False | (하드코드) |
| winSize | 9 | `FARNEBACK_WINSIZE` |
| numIters | 2 | `FARNEBACK_ITERATIONS` |
| polyN | 7 | `FARNEBACK_POLY_N` |
| polySigma | 1.5 | `FARNEBACK_POLY_SIGMA` |
| Gaussian window | False | `FARNEBACK_USE_GAUSSIAN` |

1/4에서 winSize 9 ≈ 원본 **36×36**. flow는 계산 후 `scale = full/scaled` (=4)를 곱해 원본 px/프레임으로 둔다.

---

## Dense mag floor (tanstrip 전)

| 파라미터 | 값 | 설정 키 |
|---|---|---|
| 문턱 | 0.5 px | `PRE_AGG_MAG_THRESHOLD` |

`‖v‖ < 0.5`인 dense 벡터를 0. 공간 평균 **이전**, tanstrip **이전**. 픽셀 지터용. 자전거처럼 mag가 이미 큰 대상은 여기서 안 죽는다.

---

## Aperture: tanstrip + R (하드킬 없음)

structure tensor는 1/4 gray 위에서 σ-Gaussian.

| 파라미터 | 값 | 설정 키 | 역할 |
|---|---|---|---|
| tensor σ | 1.5 | `STRUCTURE_TENSOR_SIGMA` | J 스무딩 |
| τ_edge | 950 | `APERTURE_TAU_EDGE` | 강한 에지 (`λ1`) |
| γ | 2.0 | `APERTURE_GAMMA` | 접선 정렬 지수 |
| α | 1.0 | `APERTURE_ALPHA` | `R_ap = clip(1 − α P_ap, 0, 1)` |
| R × strength | True | `APERTURE_R_TIMES_STRENGTH` | `R *= 1 − exp(−λ2 / τ_str)` |
| τ_str | 400 | `STRUCTURE_TENSOR_STRENGTH_TAU` | 평탄 픽셀 투표 억제 |
| ε | 1e-6 | `STRUCTURE_TENSOR_EPS` | |
| tanstrip | True | `APERTURE_STRIP_TANGENT` | 접선 성분 제거 |
| P_ap 하드킬 τ | **−1.0 (꺼짐)** | `APERTURE_P_HARD_TAU` | `< 0` 이면 `P_ap > τ → v=0` 안 함 |
| 하드킬 dilate | 0 px | `APERTURE_P_HARD_DILATE_PX` | 미사용 |

식:

- 에지도 `edgeness = 1 − λ2 / (λ1 + ε)`
- 접선 `t ⟂ n`, 정렬 `|(v · t)| / ‖v‖`
- `P_ap = 1[λ1 > τ_edge] · edgeness · alignment^γ`
- `R_ap = clip(1 − α P_ap, 0, 1) · (1 − exp(−λ2 / τ_str))`
- tanstrip: `p_strip = clip(λ1 / τ_edge, 0, 1) · edgeness · alignment^γ`,  
  `v ← v − p_strip (v · t) t`

**R은 평균 가중치만** 바꾼다. 벡터를 통째로 0으로 만들지 않는다.

---

## 공간 평균 (6×6 all-pixel)

| 파라미터 | 값 | 설정 키 |
|---|---|---|
| window | 6 | `STAGE1_SPATIAL_WIN` |
| stride | 4 | `RESIZED_BASE_BLOCK` |
| 방식 | `Σ(R v) / Σ(R)` | mag=0도 R>0이면 분모에 포함 |
| 시간 창 | 없음 (×1) | GPU `temporal_radius=0` |

1/4에서 6×6 ≈ 원본 **24×24**. 출력은 셀 중심 샘플, 원본 16 px 격자.

 mag>0만 평균하지 않는다.

---

## Directional persistence (P13)

공간 평균 뒤 셀 시계열. 0 프레임은 분모에 안 넣는다 (`Σ‖v‖ = 0` 이면 P=0).

| 파라미터 | 값 | 설정 키 |
|---|---|---|
| 창 길이 | 13 프레임 (≈ 2.6 s) | `STAGE1_P15_WINDOW` |
| 문턱 | 0.8 | `STAGE1_P13_MIN` |
| 창 정렬 | 중심. 왼쪽 6 / 오른쪽 6 | `_centered_window_bounds` |

\[
P_t = \frac{\lVert \sum_{k \in W(t)} v_k \rVert}{\sum_{k \in W(t)} \lVert v_k \rVert}
\]

`P < 0.8` 이면 그 시각 셀 `v = 0`. 벡터를 스케일하지 않고 게이트만 한다.

---

## T5 + 셀 floor

| 파라미터 | 값 | 설정 키 |
|---|---|---|
| 반경 R | 2 → 창 5프레임 | `STAGE1_TEMPORAL_RADIUS` |
| 0 포함 | True (`active_only=False`) | `STAGE1_T5_ACTIVE_ONLY` |
| 셀 mag floor | 0.5 | `STAGE1_CELL_MAG_FLOOR` |

T5는 게이트된 셀을 `[f−2, f+2]`에서 **산술 평균**한다. 이웃이 0이면 mag가 내려간다. 그다음 `‖v‖ ≤ 0.5`를 0.

dense `PRE_AGG_MAG_THRESHOLD = 0.5`와 숫자는 같지만 대상이 다르다. 앞은 6×6 전 픽셀 지터, 뒤는 6×6·P13·T5 이후 16 px 셀의 T5 꼬리·희석 잔여.

Stage2는 이 floor를 다시 적용하지 않는다 (`STAGE2_CELL_MAG_FLOOR = 0`).

---

## 출력 NPZ

경로: `{data_root}/{video_id}/{video_id}_base_motion_U1.npz`

주요 배열: `U1`, `M1`, `stage1_keep`, `P15` (P13 점수), 프레임 인덱스/타임스탬프.

메타에 위 파라미터가 그대로 들어간다. `aggregation` 예: `stage1_gap1_gpu_6x6x1_P13_T5`.

CLI 오버라이드: `--spatial_win`, `--p_window`, `--block_size`, `--temporal_radius`, `--no_gpu`. 문턱·floor·Farneback·aperture는 config 상수.

---

## 쓰지 않는 것 (공식 Stage1)

| 항목 | 상태 |
|---|---|
| `P_ap > τ` 하드킬 | 꺼짐 (`APERTURE_P_HARD_TAU = -1`) |
| 공간 mag>0-only 평균 | 안 함 |
| T5 mag>0-only | 안 함 |
| Gap5–50 Farneback | Stage1 없음 (Stage2가 Gap1 적분) |
| 8×8 공간 창 | 예전 기본. 지금 6×6 |
| P15 (15프레임) | 예전 기본. 지금 13프레임 |

---

## 시각화 (계산과 별개)

히트맵 기본 `HEAT_VMIN=0.8`, `HEAT_VMAX=3.5`는 **오버레이 스케일**이지 Stage1 컷이 아니다. 계산된 `M1`은 셀 floor 0.5 위면 0.8 미만도 NPZ에 있다.

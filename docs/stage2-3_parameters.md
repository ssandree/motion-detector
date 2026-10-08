# Stage2–3 공식 파라미터

Stage2 소스: `src/motion_analyzer/aggregation.py`, `config.py`.  
실행: `scripts/2_gap_fusion.py`. Stage1 NPZ가 없으면 실패한다.

Stage3 소스: `src/stage3/v2/light_pipeline.py`, `light_split.py`, `roi_composition.py`,  
`hysteresis_tube.py`, `components.py`, `src/stage3_graph/{nodes,edges,config,filter_nodes}.py`.  
실행: `scripts/3_roi_tube.py --fusion_root data/stage2/2_gap_fusion/<stamp>`.

2026-09-24 코드 기준. Graph-cut / coarse grouping / Kernighan–Lin 없음.  
공식 Stage3는 attractive Union-Find → AABB composition.

---

## 한눈에 보기

```
Stage1 NPZ  U1/M1 @16px
 │
 ├─ Stage2-1  Gap G = Σ(연속 G개 U1)   (Farneback 없음, floor 재적용 없음)
 ├─ Stage2-2  각 gap ÷√G → 5-gap nanmean → M_fused / U_fused @16px
 └─ Stage2-3  4×4 max(M) → MU_fused @64px
              4×4 mean(U) → UU_fused @64px
 │
 ├─ Stage3-1  MU≥0.2 블록 이벤트 (max_gap=5, 길이≥3) + 가장자리/고립 제거
 ├─ Stage3-2  Chebyshev≤2 로컬 해시 → c>0 이면 Union-Find merge
 ├─ Stage3-3  AABB compactness≤1.5, tgap≤15, frame당 ROI>3 이면 강제 merge
 └─ Stage3-4  짧은/작은/contained 제거 + IoU NMS 0.5 → lifetime 고정 bbox
```

벡터 단위는 **원본 px / 샘플 프레임**. 5 fps에서 Gap1 ≈ 0.2 s.

---

# Stage2. 5-gap fusion (16px → 64px)

목표: Stage1의 필터된 Gap1만 적분해서 여러 시간 스케일을 맞춘 뒤, 64px unit에서 Stage3가 쓸 `MU`를 만든다.

## 파이프라인 순서

1. Stage1 NPZ에서 `U1` 로드. `STAGE2_CELL_MAG_FLOOR = 0` 이라 **셀 floor를 다시 자르지 않는다.**
2. 타임라인은 Stage1과 같다 (`align_start`, 보통 샘플 인덱스 1). Gap50에 맞춰 앞을 자르지 않는다.
3. Gap \(G \in \{1,5,10,20,50\}\): 연속 \(G\)개 Gap1 벡터의 합. 추가 Farneback / P13 / T5 없음.
4. 각 gap을 \(\sqrt{G}\)로 나눈다.
5. 5개 magnitude를 **nanmean** 해서 `M_fused` @16px. 벡터도 같은 방식으로 `U_fused`.
6. `M_fused`를 4×4 **max** → `MU_fused` @64px. `U_fused`는 4×4 **mean** → `UU_fused`.
7. fusion 뒤 남은 NaN은 0으로 채운다.

Gap \(G\)를 아직 못 만드는 앞쪽 프레임은 그 gap만 NaN. `nanmean`이 있는 gap만 쓴다.

## Stage2-1. Gap 적분

샘플 인덱스 \(s\)에서 Gap \(G\)는 Stage1 벡터의 누적합이다.

\[
U_G(s) = \sum_{k=s-G+1}^{s} U_1(k)
\]

`u1[i]`는 샘플 인덱스 `align_start + i`로 들어오는 Gap1 (이전 샘플 → 현재).  
\(s < G\) 이면 창이 안 채워져 `U_G = NaN`.

| 파라미터 | 값 | 설정 키 | 설명 |
|---|---|---|---|
| gaps | 1, 5, 10, 20, 50 | `GAPS` | 5 fps에서 0.2 / 1 / 2 / 4 / 10 s |
| 입력 | Stage1 `U1` | `DEFAULT_DATA_ROOT` = `data/cache` | |
| Stage2 cell floor | **0 (끔)** | `STAGE2_CELL_MAG_FLOOR` | Stage1에서 이미 0.5로 자름 |
| 타임라인 | Stage1 `align_start` | — | 예전처럼 Gap50에 맞춰 slice하지 않음 |

## Stage2-2. \(\div\sqrt{G}\) + temporal mean

\[
\tilde U_G = U_G / \sqrt{G},\qquad
\tilde M_G = \|U_G\| / \sqrt{G}
\]

이유: 무상관(브라운) 잔여는 mag가 \(\sqrt{G}\)로 커지므로 나누면 스케일이 비슷해진다.  
한 방향으로 쌓이는 실제 이동은 \(\sqrt{G}\)로 남는다.

공식 fusion은 **flat + mean**: 다섯 gap을 같은 가중으로 `nanmean`.

\[
M_\text{fused} = \operatorname{nanmean}_G(\tilde M_G),\qquad
U_\text{fused} = \operatorname{nanmean}_G(\tilde U_G)
\]

| 파라미터 | 값 | 설정 키 |
|---|---|---|
| \(\sqrt{G}\) | 1, √5, √10, √20, √50 | `GAP_NORM_DIV` |
| fusion | `mean` | `DEFAULT_FUSION` |
| scheme | `flat` (5 gap 동등) | `DEFAULT_FUSION_SCHEME` |

실험용 `short_long` (`S_short=(G1+G5)/2`, `S_long=(G10+G20+G50)/3` 후 평균)과 max/median/rms는 코드에 남아 있지만 공식 경로가 아니다.

## Stage2-3. 64px unit

16px 셀 4×4 = 원본 **64 px**.

| 맵 | 방법 | 설정 키 | Stage3에서 |
|---|---|---|---|
| `MU_fused` | 4×4 **max** (`nanmax`) | `DEFAULT_SPATIAL_AGG=max` | 이벤트 게이트 |
| `UU_fused` | 4×4 **mean** (mag 컷 없음) | `aggregate_mean_flow_vector` | 이벤트 \((u,v)\), coherence |

가장자리 부분 블록은 pad NaN을 무시한다.

1920×1080이면 대략 `(T, 17, 30)` unit. 1080이 64로 안 나누떨어져 마지막 행은 16 px 스트립이 된다 (Stage3 `ROI_MIN_BLOCK_PX`가 이걸 버린다).

## 출력 NPZ

경로: `data/stage2/2_gap_fusion/<stamp>/<video_id>/<video_id>_gap_fusion_mean.npz`

| 키 | 내용 |
|---|---|
| `M_fused`, `U_fused` | 16px fused mag / vector |
| `MU_fused`, `UU_fused` | 64px unit mag / vector |
| `M{G}`, `U{G}` | 적분 직후 (÷√G 전) |
| `M{G}_norm`, `U{G}_norm` | ÷√G 후 |
| `aggregation` | `stage2_gap1_integrate_divsqrtg_fusion_mean_then_4x4max` |

히트맵 `HEAT_VMIN=0.8`, `HEAT_VMAX=3.5`는 오버레이용. `MU` 계산 컷이 아니다.

---

# Stage3. ROI tube (Union-Find → AABB)

목표: 물체 마스크가 아니라 VLM용 contextual ROI **1~2개** (많아야 3개).  
입력: Stage2 `MU_fused` (이벤트) + `UU_fused` (방향).

## 파이프라인 순서

1. 각 64px 블록에서 `MU ≥ 0.2` 구간을 이벤트로 뽑는다 (`max_gap=5`, 길이 ≥ 3).
2. 프레임 밖 잘린 가장자리 칸·시간 겹치는 이웃이 없는 1칸은 제거.
3. 남은 이벤트를 노드로 만들고, 로컬 해시 이웃만 보고 `c>0`이면 union.
4. 컴포넌트 AABB를 compactness로 합쳐 ROI 수를 줄인다.
5. 짧은/작은/완전 포함 tube 제거 + 시공간 IoU NMS → `[t0,t1]` 동안 같은 bbox.

## Stage3-1. Block temporal event

각 unit 셀의 1-D `MU` 시계열.

- `MU ≥ τ` 이면 시작 (`τ = 0.2`). `tau_high = tau_low` 이라 **히스테리시스 밴드 없음**.
- `τ` 아래로 내려가도 `max_gap=5` 프레임까지는 이벤트를 유지.
- 그보다 길면 `t1 = 마지막 지지 프레임`에서 끊는다.
- 길이 `< 3` 프레임은 버린다.

각 이벤트: `(x, y, t0, t1)` + `UU` 위 평균 flow \((u,v)\) + coherence

\[
C_i = \frac{\|\operatorname{mean} v\|}{\operatorname{mean}\|v\| + \varepsilon}
\]

| 파라미터 | 값 | 설정 키 |
|---|---|---|
| 게이트 | 0.2 | `ROI_THRESHOLD` / `ROI_TAU_HIGH` / `ROI_TAU_LOW` |
| below-thr 허용 | 5 프레임 | `ROI_MAX_GAP` |
| 최소 길이 | 3 프레임 | `ROI_MIN_BLOCK_EVENT` |

그다음 두 필터:

| 필터 | 동작 |
|---|---|
| partial edge | 잘린 칸(< 64×64 px)은 **꽉 찬 이웃**(Chebyshev≤1, 시간 겹침)이 있을 때만 유지 |
| isolated 1칸 | 시간 겹치는 Chebyshev≤2 이웃이 없으면 제거 (`filter_isolated_single_block_nodes`) |

`ROI_MERGE_SPATIAL_DIST` / `ROI_MERGE_TEMPORAL_GAP` 은 예전 proximity tube용. **공식 Stage3는 안 쓴다.**

## Stage3-2. Attractive Union-Find

전역 multicut 없음. 그리드 해시로 Chebyshev ≤ 2 쌍만 본다.

후보 종류:

- **spatial**: 시간 구간이 겹침
- **temporal_bridge**: 겹치지 않고 사이 프레임 수 \(1 \le \text{gap} \le 10\)

쌍 점수 (공식):

\[
\begin{aligned}
S_{xy} &= \exp\!\bigl(-d_\infty^2 / (2\sigma_{xy}^2)\bigr) \\
S_t &=
\begin{cases}
|\Delta t| / \min(\mathrm{dur}_i,\mathrm{dur}_j) & \text{겹치면} \\
\exp(-\mathrm{gap}/\sigma_t) & \text{bridge}
\end{cases} \\
S_v &= \tfrac12(1 + \cos\theta) \\
R_\mathrm{dir} &= \min(C_i,C_j)\,(1-S_v) \\
c &= w_{xy} S_{xy} + w_t S_t - w_\mathrm{dir} R_\mathrm{dir} - \tau
\end{aligned}
\]

\(c > 0\) 이면 union, \(c \le 0\) 이면 그대로 둔다.  
방향이 둘 다 뚜렷하고 반대일 때만 강하게 끊긴다.

| 파라미터 | 값 | 설정 키 |
|---|---|---|
| Chebyshev | 2 | `CONSERVATIVE_CHEBYSHEV` |
| bridge gap | 10 프레임 | `CONSERVATIVE_TEMPORAL_GAP` |
| \(\sigma_{xy}\) | 1.5 블록 | `SIGMA_XY` |
| \(\sigma_t\) | 3.0 프레임 | `SIGMA_T` |
| \(w_{xy}, w_t\) | 0.5, 0.5 | `W_XY`, `W_T` |
| \(w_\mathrm{dir}\) | **1.0** | `CONSERVATIVE_W_DIR` |
| \(\tau\) | **0.25** | `CONSERVATIVE_TAU_GRAPH` |

즉 \(c = 0.5\,S_{xy} + 0.5\,S_t - 1.0\,R_\mathrm{dir} - 0.25\).

`stage3_graph.config`의 `W_DIR=0.5`, `TAU_SPATIAL=0.5`, `TEMPORAL_GAP_MAX=6` 은 예전 그래프 기본값. **UF 공식 경로는 위 conservative 값을 쓴다.**

## Stage3-3. ROI composition (AABB)

모션을 다시 보지 않는다. Union-Find 컴포넌트의 lifetime AABB만 본다.

두 후보 A, B:

1. temporal gap \(\le 15\)
2. \(A_\text{merged} / (A_A + A_B) \le 1.5\)  (박스가 너무 커지면 거부)
3. 둘 다 되면 **합쳐진 면적이 작은** 쌍부터 agglomerative merge

그다음 active frame마다 ROI가 **3개 초과**면 면적·gap이 제일 작은 쌍을 강제 merge.

| 파라미터 | 값 | 설정 키 |
|---|---|---|
| max temporal gap | 15 | `COMP_MAX_TEMPORAL_GAP` |
| compactness | 1.5 | `COMP_COMPACTNESS_FACTOR` |
| 목표 / 상한 | 2 / 3 per frame | `COMP_TARGET_ROI_PER_FRAME`, `COMP_MAX_ROI_PER_FRAME` |

목표는 frame당 1~2개 contextual box.

## Stage3-4. Filter + fixed lifetime bbox

`apply_stage3_roi_filters` (`iou_nms_tau=0.5` 고정):

| 필터 | 값 | 설정 키 |
|---|---|---|
| 최소 셀 수 | 2 | `ROI_MIN_TUBE_CELLS` |
| 최소 duration | 2 프레임 | `ROI_MIN_TUBE_DURATION` |
| 부분 블록 tube | 클립된 px가 64×64 미만이면 삭제 | `ROI_MIN_BLOCK_PX` |
| contained | A의 \([t0,t1]\times\)bbox가 B 안에 완전 포함되면 A 삭제 | `ROI_SUPPRESS_CONTAINED` |
| 시공간 IoU NMS | IoU ≥ 0.5 이면 작은 쪽 삭제 | `light_pipeline` 하드코드 0.5 |

살아남은 tube는 `[t0, t1]` 동안 **같은 공간 bbox**를 쓴다 (`RoiTube.spatial_bbox`). 프레임마다 박스가 따라가지 않는다.

## 출력

`outputs/stage3/3_roi_tube/<stamp>/`

- `<id>_roi_tracks.json` — `variant: stage3_light_uf_compose`
- `<id>_roi_tube_*.mp4` — turbo `MU` + ROI
- 3D PNG는 `outputs/stage3/3_roi_tube_3dviz/<stamp>/`

---

## 쓰지 않는 것 (공식 Stage2–3)

| 항목 | 상태 |
|---|---|
| Stage2에서 Gap5–50 Farneback | 안 함 (Gap1 적분만) |
| Stage2 cell mag floor 재적용 | 꺼짐 (0) |
| Gap50에 맞춘 타임라인 slice | 예전. 지금은 Stage1 타임라인 |
| `short_long` / max / median / rms fusion | CLI에 있음. 기본 아님 |
| Graph-cut, coarse group, KL | 삭제됨 |
| `ROI_MERGE_*` proximity tube | 예전 `build_roi_tubes` 전용 |
| `stage3_graph.pipeline.process_video` | 비교용 엔트리. `3_roi_tube.py`는 안 탐 |

---

## 해상도 / 좌표 다시

원본 (보통 1920×1080) → ¼에서 Farneback → 16px 셀 (Stage1/2) → 64px unit (Stage2-3 / Stage3).  
벡터는 항상 원본 px. 그리드 인덱스는 축소 해상도 기준.

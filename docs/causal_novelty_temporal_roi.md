# Causal Novelty Temporal ROI — `B3_hop1_long` 기준 설명

본 문서는 권장 overlay  
`sliding_window_overlay_B3_hop1_long.mp4` (= `sliding_window_overlay_B.mp4`)에  
적용된 알고리즘을 설명한다.

| 항목 | 값 |
|------|-----|
| Variant | **`B3_hop1_long`** |
| Novelty | `sliding_window_regions.py` (A/B 공용) |
| Peak → ROI | `block_peak_roi.py` |
| 실행 | `scripts/run_b_roi_compare.py` / `run_causal_roi_ab.py --methods B` |

---

## 0. Overlay 차이 (한 표)

| 파일 | 무엇 |
|------|------|
| **`sliding_window_overlay_B3_hop1_long.mp4`** / **`_B.mp4`** | **권장 B3** (동일 내용). hop1 이웃 성장 + 긴 span + WHY 라벨 |
| `method_B_compare/B1_…` | 같은 B지만 **seed 1셀만** → ROI가 잘림 |
| `method_B_compare/B2`/`B4`/`B5` | B3와 같은 파이프, **넓이·span 파라미터만** 다름 (B4=고정 3×3로 더 큼) |
| `sliding_window_overlay_A.mp4` | Approach A (UF/Jaccard 등) — box 선택 로직이 다름 |
| `sliding_window_overlay.mp4` / heatmap / temporal_linking / sub_block_agg | 예전·중간 실험용 — **B3와 동일시하면 안 됨** |

---

## 1. 핵심 아이디어 (질문에 적힌 것과 동일하게 구현됨)

**맞다.** spatial block에서 “새로 생긴 motion”을 잡기 위해 **causal baseline**을 쓴다.  
코드: `_collect_causal_features` / `_causal_baseline_at` (`sliding_window_regions.py`).

각 sliding window의 시작 시각을 \(t_0\), 길이를 \(L\)이라 하고,  
**aggregation grid의 각 셀 \((r,c)\)** 마다 아래를 계산한다.  
(미래 프레임은 쓰지 않는다.)

### 1.1 과거 baseline 통계 — 프레임 `[0, t₀)`

| 이름 (코드) | 정의 |
|-------------|------|
| **`baseline_activity` (\(ba\))** | 과거 프레임 중 해당 block이 **active**였던 **비율**. `active_mask[:t0].mean(axis=0)` |
| **`baseline_mean_mag` (\(bm\))** | 과거 **전체 frame**의 pooled RMS magnitude **평균**. `pooled_rms[:t0].mean(axis=0)` |

> 문서/대화에서 “과거 window 비율”이라고 해도 의미는 같고,  
> 구현은 **window 단위가 아니라 과거 frame 스택 평균**이다.

### 1.2 현재·직전 구간 통계

| 이름 (코드) | 정의 |
|-------------|------|
| **`current_mean_mag` (\(cur\))** | **현재 window** `[t0, t0+L)` 구간의 평균 pooled RMS |
| **`previous_mean_mag` (\(prev\))** | 현재 window **바로 앞**, **같은 길이 \(L\)**인 구간 `[t0−L, t0)` 의 평균 pooled RMS (시작 근처면 가능한 과거만) |

### 1.3 Novelty → `block_score`

\[
\begin{aligned}
\text{magnitude\_novelty} &= \max(cur - bm,\ 0) \\
\text{onset\_score} &= \max(cur - prev,\ 0)
\end{aligned}
\]

- **mag novelty**: “예전에 비해 이 블록 RMS가 커졌는가” (chronic baseline 대비)  
- **onset**: “직전 구간보다 지금 갑자기 커졌는가” (단기 변화)

정규화 후 (대략):

\[
block\_score =
confidence
\cdot (0.6\,\mathrm{norm}(mag)+0.4\,\mathrm{norm}(onset))
\cdot \max(0.05,\ 1-ba)^{2}
\]

- \((1-ba)^2\): 예전에 자주 active이던 도로 등은 억제 (**quiet boost**)  
- `confidence`: 과거 프레임이 warmup보다 적을 때 낮춤  

이 `block_score` 맵이 **윈도우마다** 한 장씩 쌓인다 →  
\(S[w,r,c]\) = w번째 window에서 셀 \((r,c)\)의 novelty 점수.

A/B 모두 여기까지 **공유**. B3는 이 다음에 **시계열 peak**로 사건을 고른다.

---

## 2. “시계열 peak”가 뭔가 (B3의 본체)

“시계열 peak”는 **사람이 보이는 peak가 아니라**,  
**한 spatial block의 `block_score`가 시간에 따라 그리는 1D 곡선에서 찾은 local maximum**이다.

### 2.1 셀마다 1D 시계열

고정된 셀 \((r,c)\)에 대해:

\[
s(w) = S[w, r, c],\quad w = 0,1,2,\ldots
\]

예: 왼쪽 하차 자리 셀이 26–32초 근처에서만 커지면, \(s(w)\)도 그 구간에 봉우리가 생긴다.

### 2.2 Peak 정의 (`_temporal_peaks_1d`)

\(s(w)\)가 **좌우 이웃보다 크거나 같고**, \(s(w) \ge min\_peak(0.05)\) 이면 **peak**.

즉 “그 셀 novelty가 이 순간 **한동안의 극대**”를 사건 후보로 본다.  
(전역 argmax 하나가 아니라, **셀별로** 여러 peak가 날 수 있다.)

### 2.3 Peak → 시간 span → 사건 점수

peak 인덱스 \(w^\*\) 주변에서:

- \(s(w) \ge 0.12 \cdot s(w^\*)\) 인 연속 구간으로 span 확장 (`span_frac=0.12`)
- 양옆 **+2 window pad** (`span_pad=2`) → overlay에 더 길게 보임
- 같은 셀·겹치는 span은 NMS

\[
\text{event\_score} = peak \cdot (1-ba_{peak})^{2} \cdot \log(1+\text{persist})
\]

### 2.4 Peak seed → ROI (B3 = hop1_long)

- **seed** = peak가 난 셀 \((r,c)\)  
- span 구간의 평균 score map에서, Moore 이웃 점수가 seed의 **22% 이상**이면 포함  
- 최대 **8 cells** → bbox = union (맥락용으로 192×192보다 넓어짐)

이게 이름 **`hop1_long`**:  
공간은 1-hop 이웃, 시간은 span+pad로 길게.

### 2.5 Top-2

후보 사건 중 최대 2개:

- **G1**: late / top / interior를 가중한 강한 novelty (`late_primary` 등)  
- **G2**: G1이 late면 상단·quiet onset (`early_top_onset` / `mid_top_onset`), 아니면 spatial–temporal partner  

Overlay WHY 란에 이 **role / reason**이 찍힌다.

---

## 3. 입력·윈도우 (짧게)

```text
<sub_block_agg_12x12>/active_mask.npy, aggregated_rms_mag.npy, …
```

- 셀 ≈ 192×192 px (12×12 aggregation)  
- window=10, stride=5 (샘플 프레임 기준)  
- 샘플↔초: \(t \approx (6+6s)/30\)

---

## 4. B3 고정 파라미터

| 키 | 값 | 의미 |
|----|-----|------|
| `spatial_mode` | `hop1` | seed + 1-hop 이웃 |
| `neighbor_ratio` | `0.22` | 이웃 ≥ 0.22×seed |
| `max_cells` | `8` | ROI 상한 |
| `span_frac` | `0.12` | peak의 12%로 시간 확장 |
| `span_pad` | `2` | ±2 window pad |
| `min_peak` | `0.05` | 1D peak 하한 |
| `max_rois` | `2` | Top-2 |

---

## 5. GT / 최근 결과

튜닝 3클립 (하차·상단 탑승/출발·우측 탑승).  
Strict HIT = 시간 겹침 + (`gt_cov≥0.08` 또는 `roi_in≥0.5`).

B3 최근: **4/4 HIT**  
(예: alight seed `(1,1)` ~384×384; board/depart는 이웃으로 넓힌 박스).

---

## 6. 실행

```bash
conda activate motion
cd /home/vailab02/vlm_motion/motion-detector
PYTHONPATH=src:scripts python scripts/run_b_roi_compare.py \
  --video_names \
    VIRAT_S_000201_06_001354_001397 \
    VIRAT_S_000201_02_000590_000623 \
    VIRAT_S_050000_08_001235_001295
```

---

## 7. 코드 맵

| 단계 | 파일 |
|------|------|
| \(ba,bm,cur,prev\) → mag/onset → `block_score` | `temporal/sliding_window_regions.py` |
| 셀 시계열 peak → span → hop1 → Top-2 → WHY overlay | `temporal/block_peak_roi.py` |
| Approach A (비교) | `temporal/temporal_group_roi.py` |
| B3 primary 배치 | `scripts/run_b_roi_compare.py` |

---

## 8. 한 문장

> **인과 baseline(\(ba,bm\))과 현재/직전 mag(\(cur,prev\))로 셀별 novelty `block_score`를 만들고**,  
> 그 **1D 시계열의 local peak**를 사건으로 잡아 hop1로 넓힌 뒤 Top-2를 그린다 (`B3_hop1_long`).

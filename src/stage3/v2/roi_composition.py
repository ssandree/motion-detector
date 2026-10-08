"""Step 4: ROI Composition — AABB 기반 agglomerative merge.

GraphCut 이후 component들의 lifetime fixed AABB를 기준으로
spatial compactness + temporal compatibility를 보고 합쳐서
active time당 1~2개의 contextual ROI를 만든다.

Merge 기준 (component pair):
  A. Temporal compatibility:
     - temporal_gap ≤ max_temporal_gap

  B. Spatial compactness (AABB 기준):
     - merged_AABB_area ≤ compactness_factor × (area_A + area_B)
       → AABB가 지나치게 커지면 merge 거부
       → compactness_factor=1.5: 두 박스가 합쳐져서 1.5배 이하면 merge

  C. 결과 ROI 수 제어:
     - active frame마다 ROI 수를 세어서 target_rois_per_frame 초과 시 merge 강제

목표: 각 active frame에서 ROI 1~2개.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stage3.components import GraphComponent, node_to_event
from stage3.hysteresis_tube import RoiTube, finalize_tubes, tube_from_events

# ROI Composition 파라미터
COMP_MAX_TEMPORAL_GAP = 15          # 최대 temporal gap (프레임)
COMP_COMPACTNESS_FACTOR = 1.5       # exp_v2c / official: merged_area / (area_A+area_B)
COMP_TARGET_ROI_PER_FRAME = 2       # active frame당 목표 ROI 수
COMP_MAX_ROI_PER_FRAME = 3          # 이 이상이면 강제 merge


@dataclass
class _RoiCandidate:
    cid: int
    comps: list[GraphComponent] = field(default_factory=list)

    @property
    def t0(self) -> int:
        return min(min(int(n.t0) for n in c.nodes) for c in self.comps)

    @property
    def t1(self) -> int:
        return max(max(int(n.t1) for n in c.nodes) for c in self.comps)

    def aabb(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) grid-cell AABB."""
        xs = [int(n.x) for c in self.comps for n in c.nodes]
        ys = [int(n.y) for c in self.comps for n in c.nodes]
        return min(xs), min(ys), max(xs) + 1, max(ys) + 1

    def area(self) -> int:
        x0, y0, x1, y1 = self.aabb()
        return max(0, x1 - x0) * max(0, y1 - y0)

    def all_events(self):
        return [node_to_event(n) for c in self.comps for n in c.nodes]

    def active_frames(self) -> set[int]:
        frames: set[int] = set()
        for c in self.comps:
            for n in c.nodes:
                for t in range(int(n.t0), int(n.t1) + 1):
                    frames.add(t)
        return frames


def _time_gap_candidates(a: _RoiCandidate, b: _RoiCandidate) -> int:
    t1a, t0a = a.t1, a.t0
    t1b, t0b = b.t1, b.t0
    if t1a >= t0b and t1b >= t0a:
        return 0
    if t1a < t0b:
        return int(t0b) - int(t1a) - 1
    return int(t0a) - int(t1b) - 1


def _merged_aabb(a: _RoiCandidate, b: _RoiCandidate) -> tuple[int, int, int, int]:
    ax0, ay0, ax1, ay1 = a.aabb()
    bx0, by0, bx1, by1 = b.aabb()
    return min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)


def _merged_area(a: _RoiCandidate, b: _RoiCandidate) -> int:
    x0, y0, x1, y1 = _merged_aabb(a, b)
    return max(0, x1 - x0) * max(0, y1 - y0)


def _compactness_ok(
    a: _RoiCandidate,
    b: _RoiCandidate,
    *,
    factor: float,
) -> bool:
    """Dead-space ratio: merged_area / (area_a + area_b) ≤ factor.

    factor=1.0 : 두 박스가 정확히 나란히 붙어 있어야 merge 허용
    factor=1.5 : 합쳐진 박스가 두 박스 면적 합의 1.5배까지 허용
    factor=2.0 : 두 박스 사이에 빈 공간이 절반 이하까지 허용

    핵심: 두 박스가 멀리 떨어져 있으면 merged_area가 크게 증가 → ratio 높아짐 → 거부.
    """
    area_sum = a.area() + b.area()
    if area_sum <= 0:
        return True
    merged = _merged_area(a, b)
    ratio = float(merged) / float(area_sum)
    return ratio <= float(factor)


def _roi_counts_per_frame(candidates: list[_RoiCandidate]) -> dict[int, int]:
    """각 frame t에서 active한 ROI 수."""
    counts: dict[int, int] = {}
    for roi in candidates:
        for t in roi.active_frames():
            counts[t] = counts.get(t, 0) + 1
    return counts


def _max_rois_per_frame(candidates: list[_RoiCandidate]) -> int:
    counts = _roi_counts_per_frame(candidates)
    return max(counts.values()) if counts else 0


def _best_merge_pair(
    candidates: list[_RoiCandidate],
    *,
    max_temporal_gap: int,
    compactness_factor: float,
) -> tuple[int, int, str] | None:
    """가장 좋은 merge pair를 찾아 반환. (i, j, reason)"""
    best_score: float = 1e18
    best_pair: tuple[int, int, str] | None = None

    for ia in range(len(candidates)):
        for ib in range(ia + 1, len(candidates)):
            a, b = candidates[ia], candidates[ib]
            gap = _time_gap_candidates(a, b)
            if gap > max_temporal_gap:
                continue
            if not _compactness_ok(a, b, factor=compactness_factor):
                continue
            # score: merged area (작을수록 compact), 이차적으로 gap
            merged = _merged_area(a, b)
            score = float(merged) * 1000 + float(gap)
            if score < best_score:
                best_score = score
                best_pair = (ia, ib, "compact_merge")

    return best_pair


def _force_merge_pair_by_overlap(
    candidates: list[_RoiCandidate],
) -> tuple[int, int, str] | None:
    """강제 merge: 가장 많이 겹치는 (시간 겹침 + 공간 가까운) pair."""
    best: tuple[int, int, str] | None = None
    best_score: float = 1e18

    for ia in range(len(candidates)):
        for ib in range(ia + 1, len(candidates)):
            a, b = candidates[ia], candidates[ib]
            gap = _time_gap_candidates(a, b)
            merged = _merged_area(a, b)
            score = float(merged) * 1000 + float(gap)
            if score < best_score:
                best_score = score
                best = (ia, ib, "force_merge")

    return best


def compose_rois(
    comps: list[GraphComponent],
    *,
    max_temporal_gap: int = COMP_MAX_TEMPORAL_GAP,
    compactness_factor: float = COMP_COMPACTNESS_FACTOR,
    target_rois: int = COMP_TARGET_ROI_PER_FRAME,
    max_rois: int = COMP_MAX_ROI_PER_FRAME,
) -> tuple[list[RoiTube], dict[str, Any]]:
    """GraphCut component들을 AABB compactness 기준으로 agglomerative merge.

    Returns:
        tubes: list[RoiTube] — 최종 ROI tubes (finalize_tubes 완료)
        log: 통계 dict
    """
    if not comps:
        return [], {
            "num_input_comps": 0,
            "num_output_rois": 0,
            "num_merges": 0,
            "merge_log": [],
        }

    next_cid = max(int(c.component_id) for c in comps) + 1
    candidates: list[_RoiCandidate] = [
        _RoiCandidate(cid=int(c.component_id), comps=[c])
        for c in comps
    ]
    merge_log: list[dict] = []

    # Phase 1: compact merge (temporal + spatial compactness)
    changed = True
    while changed and len(candidates) >= 2:
        changed = False
        pair = _best_merge_pair(
            candidates,
            max_temporal_gap=max_temporal_gap,
            compactness_factor=compactness_factor,
        )
        if pair is None:
            break
        ia, ib, reason = pair
        a, b = candidates[ia], candidates[ib]
        merged = _RoiCandidate(
            cid=next_cid,
            comps=a.comps + b.comps,
        )
        next_cid += 1
        merge_log.append({
            "reason": reason,
            "cid_a": a.cid, "cid_b": b.cid,
            "merged_cid": merged.cid,
            "area_a": a.area(), "area_b": b.area(),
            "merged_area": merged.area(),
            "gap": _time_gap_candidates(a, b),
        })
        candidates = [c for k, c in enumerate(candidates) if k != ia and k != ib]
        candidates.append(merged)
        changed = True

    # Phase 2: 강제 merge — active frame당 ROI가 max_rois 초과 시
    force_merges = 0
    while len(candidates) > 1:
        max_count = _max_rois_per_frame(candidates)
        if max_count <= max_rois:
            break
        pair = _force_merge_pair_by_overlap(candidates)
        if pair is None:
            break
        ia, ib, reason = pair
        a, b = candidates[ia], candidates[ib]
        merged = _RoiCandidate(
            cid=next_cid,
            comps=a.comps + b.comps,
        )
        next_cid += 1
        merge_log.append({
            "reason": "force_" + reason,
            "cid_a": a.cid, "cid_b": b.cid,
            "merged_cid": merged.cid,
            "area_a": a.area(), "area_b": b.area(),
            "merged_area": merged.area(),
            "gap": _time_gap_candidates(a, b),
            "max_rois_at_merge": max_count,
        })
        candidates = [c for k, c in enumerate(candidates) if k != ia and k != ib]
        candidates.append(merged)
        force_merges += 1

    # RoiTube 생성
    tubes_raw: list[RoiTube] = []
    for i, cand in enumerate(candidates, start=1):
        events = cand.all_events()
        tube = tube_from_events(events, tube_id=i)
        tubes_raw.append(tube)

    tubes = finalize_tubes(tubes_raw)

    log: dict[str, Any] = {
        "num_input_comps": len(comps),
        "num_output_rois": len(tubes),
        "num_merges": len(merge_log),
        "num_force_merges": force_merges,
        "max_rois_per_frame_final": _max_rois_per_frame(candidates),
        "merge_log": merge_log,
        "params": {
            "max_temporal_gap": max_temporal_gap,
            "compactness_factor": compactness_factor,
            "target_rois": target_rois,
            "max_rois": max_rois,
        },
    }
    return tubes, log

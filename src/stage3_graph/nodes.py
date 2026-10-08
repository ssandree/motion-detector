"""Graph nodes from Stage3 block-level temporal events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stage3.hysteresis_tube import BlockEvent, build_block_events
from stage3_graph.config import FLOW_EPS


@dataclass(frozen=True)
class GraphNode:
    """One Stage3 temporal event as a graph node."""

    node_id: int
    event_id: int
    x: int
    y: int
    t0: int
    t1: int
    u: float  # mean flow dx (pixels / frame on unit grid)
    v: float  # mean flow dy
    mean_mag: float
    coherence: float = 0.0  # C_i = ||mean(v)|| / (mean(||v||)+eps) in [0,1]

    @property
    def duration(self) -> int:
        return int(self.t1 - self.t0 + 1)

    @property
    def flow(self) -> tuple[float, float]:
        return float(self.u), float(self.v)

    def active_at(self, t: int) -> bool:
        return int(self.t0) <= int(t) <= int(self.t1)


def _mean_flow_mag_coherence(
    flow: np.ndarray,
    mag: np.ndarray | None,
    *,
    y: int,
    x: int,
    t0: int,
    t1: int,
) -> tuple[float, float, float, float]:
    """Average UU flow, magnitude, and directional coherence over [t0, t1]."""
    t0 = max(0, int(t0))
    t1 = min(int(flow.shape[0]) - 1, int(t1))
    if t1 < t0:
        return 0.0, 0.0, 0.0, 0.0
    patch = np.asarray(flow[t0 : t1 + 1, int(y), int(x)], dtype=np.float32)
    finite = np.isfinite(patch).all(axis=1)
    if not bool(finite.any()):
        return 0.0, 0.0, 0.0, 0.0
    vec = patch[finite]
    u = float(np.mean(vec[:, 0]))
    v = float(np.mean(vec[:, 1]))
    frame_mags = np.linalg.norm(vec, axis=1)
    mean_speed = float(np.mean(frame_mags))
    coher = float(np.hypot(u, v) / (mean_speed + FLOW_EPS))
    coher = float(np.clip(coher, 0.0, 1.0))
    if mag is not None:
        mseq = np.asarray(mag[t0 : t1 + 1, int(y), int(x)], dtype=np.float32)
        m_ok = np.isfinite(mseq)
        mean_mag = float(np.mean(mseq[m_ok])) if bool(m_ok.any()) else mean_speed
    else:
        mean_mag = mean_speed
    if abs(u) < FLOW_EPS and abs(v) < FLOW_EPS:
        u, v = 0.0, 0.0
    return u, v, mean_mag, coher


def nodes_from_block_events(
    events: list[BlockEvent],
    flow_unit: np.ndarray,
    mag_unit: np.ndarray | None = None,
) -> list[GraphNode]:
    """Build nodes from BlockEvent list + unit-grid flow (T,H,W,2)."""
    flow = np.asarray(flow_unit, dtype=np.float32)
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError(f"expected flow (T,H,W,2), got {flow.shape}")
    mag = None if mag_unit is None else np.asarray(mag_unit, dtype=np.float32)
    nodes: list[GraphNode] = []
    for index, ev in enumerate(events):
        u, v, mean_mag, coher = _mean_flow_mag_coherence(
            flow, mag, y=ev.y, x=ev.x, t0=ev.t0, t1=ev.t1
        )
        nodes.append(
            GraphNode(
                node_id=index,
                event_id=int(ev.event_id),
                x=int(ev.x),
                y=int(ev.y),
                t0=int(ev.t0),
                t1=int(ev.t1),
                u=u,
                v=v,
                mean_mag=mean_mag,
                coherence=coher,
            )
        )
    return nodes


def load_events_json(path: Path) -> tuple[list[BlockEvent], dict]:
    """Load Stage3 ``*_motion_events.json`` (or tubes json with events)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("events")
    if raw is None:
        raise KeyError(f"{path}: missing 'events'")
    events: list[BlockEvent] = []
    for item in raw:
        events.append(
            BlockEvent(
                y=int(item["y"]),
                x=int(item["x"]),
                t0=int(item["start_frame"]),
                t1=int(item["end_frame"]),
                event_id=int(item["event_id"]),
            )
        )
    return events, data


def resolve_events_json(events_root: Path, video_id: str) -> Path | None:
    root = Path(events_root)
    candidates = (
        root / f"{video_id}_motion_events.json",
        root / video_id / f"{video_id}_motion_events.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(root.glob(f"**/{video_id}_motion_events.json"))
    return matches[0] if matches else None


def build_or_load_events(
    *,
    video_id: str,
    mag: np.ndarray,
    events_root: Path | None,
    tau_high: float,
    tau_low: float,
    max_gap: int,
    min_block_event: int,
) -> tuple[list[BlockEvent], dict | None]:
    """Prefer existing Stage3 events JSON; else rebuild with same hysteresis."""
    if events_root is not None:
        path = resolve_events_json(Path(events_root), video_id)
        if path is not None:
            events, meta = load_events_json(path)
            meta = dict(meta)
            meta["_events_json"] = str(path)
            return events, meta
    events = build_block_events(
        mag,
        tau_high=float(tau_high),
        tau_low=float(tau_low),
        max_gap=int(max_gap),
        min_block_event=int(min_block_event),
    )
    return events, None

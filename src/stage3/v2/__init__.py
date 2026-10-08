"""Official Stage3: events → attractive Union-Find → AABB composition.

No Graph-cut, no coarse grouping, no Kernighan–Lin.

  ① Block temporal event (tau=0.2, max_gap=5, 64px)
  ② Attractive Union-Find split (local hashed edges, c>0 → union)
  ③ ROI composition (AABB compactness ≤ 1.5, tgap ≤ 15, max 3/frame)
  ④ Size filter + fixed lifetime bbox
"""

from stage3.v2.light_pipeline import process_video, process_video_light

__all__ = ["process_video", "process_video_light"]

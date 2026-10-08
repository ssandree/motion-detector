"""Defaults for Stage3_graph (signed event graph → multicut ROI)."""

from __future__ import annotations

# Candidate edges
SPATIAL_CHEBYSHEV_MAX = 2
TEMPORAL_GAP_MAX = 6  # frames between non-overlapping intervals
# Legacy (unused): flow prediction gate removed; temporal uses Chebyshev ≤ SPATIAL_CHEBYSHEV_MAX.
TEMPORAL_PRED_RADIUS = 2.0

# Edge features
SIGMA_XY = 1.5  # block units
SIGMA_T = 3.0  # frames

# Temporal signed cost (unchanged): c_ij = S_ij - tau, S = (S_xy+S_t+S_v)/3
EDGE_TAU = 0.5

# Spatial signed cost (noaff_cohdir):
#   c = w_xy*S_xy + w_t*S_t - w_dir*R_dir - tau_spatial
#   R_dir = min(C_i, C_j) * (1 - S_v)
W_XY = 0.5
W_T = 0.5
W_DIR = 0.5
TAU_SPATIAL = 0.5

# Multicut (GAEC + KL)
KL_MAX_PASSES = 20
FLOW_EPS = 1e-6

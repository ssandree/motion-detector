# Adaptive Input Plan Validation — 4 Representative Videos

Generated after `run_event_roi_refresh.py --write_overlays` + `run_adaptive_input_builder.py`.

## Summary Table

| Video | final_motion_type | num_roi_inputs | roi_source | sparse_global | overlay full | builder viz | Status |
|-------|-------------------|----------------|------------|---------------|--------------|-------------|--------|
| VIRAT_S_050201_06_001168_001240 | event_motion | 2 | blob_group | 18 | yes (71.4s/71.4s) | yes | **PASS** |
| VIRAT_S_010207_08_001308_001332 | background_motion | 0 | none | 5 | yes (17.0s/17.0s) | yes | **PASS** |
| VIRAT_S_050300_07_001623_001690 | event_motion | 2 | blob_group | 17 | yes (66.6s/66.6s) | yes | **PASS** |
| VIRAT_S_000201_06_001354_001397 | event_motion | 2 | blob_group | 11 | yes (42.6s/42.6s) | yes | **PASS** |

## Per-Video Details

### VIRAT_S_050201_06_001168_001240

- **final_motion_type**: `event_motion`
- **reason**: persistent valid motion tubes
- **num_roi_inputs**: 2
- **roi_source**: `blob_group`
- **sparse_global_count**: 18
- **input_plan_overlay.mp4**: exists, duration 71.4s (expected 71.4s), full duration: **OK**
- **input_builder_visualization.mp4**: exists, duration 61.5s
- **ROI policy**: ROI from roi_inputs only (2 tracks, source=blob_group)
- **Failures**: none

### VIRAT_S_010207_08_001308_001332

- **final_motion_type**: `background_motion`
- **reason**: valid motion tubes are weak or absent
- **num_roi_inputs**: 0
- **roi_source**: `none`
- **sparse_global_count**: 5
- **input_plan_overlay.mp4**: exists, duration 17.0s (expected 17.0s), full duration: **OK**
- **input_builder_visualization.mp4**: exists, duration 5.0s
- **ROI policy**: No ROI (correct for background_motion)
- **Failures**: none

### VIRAT_S_050300_07_001623_001690

- **final_motion_type**: `event_motion`
- **reason**: persistent valid motion tubes
- **num_roi_inputs**: 2
- **roi_source**: `blob_group`
- **sparse_global_count**: 17
- **input_plan_overlay.mp4**: exists, duration 66.6s (expected 66.6s), full duration: **OK**
- **input_builder_visualization.mp4**: exists, duration 45.5s
- **ROI policy**: ROI from roi_inputs only (2 tracks, source=blob_group)
- **Failures**: none

### VIRAT_S_000201_06_001354_001397

- **final_motion_type**: `event_motion`
- **reason**: persistent valid motion tubes
- **num_roi_inputs**: 2
- **roi_source**: `blob_group`
- **sparse_global_count**: 11
- **input_plan_overlay.mp4**: exists, duration 42.6s (expected 42.6s), full duration: **OK**
- **input_builder_visualization.mp4**: exists, duration 47.5s
- **ROI policy**: ROI from roi_inputs only (2 tracks, source=blob_group)
- **Failures**: none

## Overall

**ALL PASS** — 4/4 videos passed validation.

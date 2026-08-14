"""Gap1/5/10/20/50 Farneback → RMS fusion → ROI tube."""

from motion_analyzer.opencv_cuda_bootstrap import bootstrap_opencv_cuda, reload_cv2_if_needed

bootstrap_opencv_cuda()
reload_cv2_if_needed()

from motion_analyzer.config import PipelineConfig, TARGET_VIDEO_IDS

__all__ = ["PipelineConfig", "TARGET_VIDEO_IDS"]

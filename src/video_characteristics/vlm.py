"""Qwen3-VL wrapper, reusing the ``vlm-inference`` model loading path.

The model, processor and generation plumbing all come from ``vlm_inference`` so this
runs in the same environment as the rest of the project. Only the per-video frame
budget is handled here.

Sampling: 1 fps like the benchmark conditions, but capped at ``MAX_FRAMES`` frames.
Without the cap, a 10-minute clip would spread ``total_pixels`` over 600 frames and
each frame would fall to roughly 180x180, which makes the OD judgement meaningless.
With the cap, a frame never drops below ``total_pixels / MAX_FRAMES`` pixels.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .datasets import VLM_INFERENCE_ROOT

VLM_SRC = VLM_INFERENCE_ROOT / "src"
if str(VLM_SRC) not in sys.path:
    sys.path.insert(0, str(VLM_SRC))

from qwen_vl_utils import process_vision_info  # noqa: E402
from qwen_vl_utils.vision_process import FRAME_FACTOR  # noqa: E402
from vlm_inference.config import (  # noqa: E402
    AppConfig,
    GenerationConfig,
    ModelConfig,
    VideoConfig,
)
from vlm_inference.model import (  # noqa: E402
    Qwen3VLCaptioner,
    _generation_kwargs,
    count_processor_tokens,
)
from vlm_inference.video_io import inspect_video  # noqa: E402

from .video_reader import install as install_video_reader

# vlm_inference.model installs a reader that decodes every frame first, which OOMs on
# the long clips in this target set. Override it with the seeking reader.
install_video_reader()

MODEL_PATH = "/data/models/Qwen3-VL-8B-Instruct-BF16"
MODEL_NAME = "8b"
MODEL_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "sdpa"

SAMPLE_FPS = 1.0
MAX_FRAMES = 160
MIN_PIXELS = 4 * 32 * 32
# The processor clamps max_pixels to total_pixels / 80; stay under it to avoid the
# per-call warning while still giving short clips a high per-frame resolution.
MAX_PIXELS = 400 * 32 * 32
TOTAL_PIXELS = 32768 * 32 * 32

MAX_NEW_TOKENS = 768
DO_SAMPLE = False
REPETITION_PENALTY = 1.05

# Pass B describes every ROI tube before answering, so its output grows with the tube
# count; a flat budget truncates the JSON of many-ROI videos mid-object.
PASS_B_TOKENS_PER_ROI = 200
PASS_B_MAX_NEW_TOKENS_CAP = 2560


def pass_b_max_new_tokens(num_roi_tubes: int) -> int:
    """Token budget for one pass-B answer, scaled to the number of ROI tubes."""
    return min(
        PASS_B_MAX_NEW_TOKENS_CAP,
        max(MAX_NEW_TOKENS, 384 + PASS_B_TOKENS_PER_ROI * max(num_roi_tubes, 1)),
    )


@dataclass
class VlmResult:
    response: str
    requested_frames: int | None
    requested_fps: float | None
    sampled_frames: int | None
    decoder_fps: float | None
    visual_tokens: int | None
    text_tokens: int | None
    input_tokens: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "requested_frames": self.requested_frames,
            "requested_fps": self.requested_fps,
            "sampled_frames": self.sampled_frames,
            "decoder_fps": self.decoder_fps,
            "visual_tokens": self.visual_tokens,
            "text_tokens": self.text_tokens,
            "input_tokens": self.input_tokens,
        }


def build_captioner() -> Qwen3VLCaptioner:
    cfg = AppConfig(
        model=ModelConfig(
            path=MODEL_PATH,
            dtype=MODEL_DTYPE,
            device_map="auto",
            attn_implementation=ATTN_IMPLEMENTATION,
            name=MODEL_NAME,
        ),
        generation=GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            repetition_penalty=REPETITION_PENALTY,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        ),
        video=VideoConfig(
            fps=SAMPLE_FPS,
            nframes=None,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            total_pixels=TOTAL_PIXELS,
        ),
        mode="default",
    )
    return Qwen3VLCaptioner(cfg)


def _frame_budget(video_path: Path) -> tuple[int | None, float | None]:
    """``(nframes, fps)``: uniform ``nframes`` for long clips, plain fps otherwise."""
    try:
        info = inspect_video(video_path)
    except Exception:  # noqa: BLE001
        return None, SAMPLE_FPS
    duration_sec = info.frame_count / info.native_fps if info.native_fps > 0 else 0.0
    wanted = duration_sec * SAMPLE_FPS
    if wanted <= MAX_FRAMES:
        return None, SAMPLE_FPS
    nframes = int(MAX_FRAMES // FRAME_FACTOR) * FRAME_FACTOR
    return max(FRAME_FACTOR, nframes), None


def _video_message(video_path: Path, prompt: str) -> tuple[list[dict[str, Any]], int | None, float | None]:
    nframes, fps = _frame_budget(video_path)
    item: dict[str, Any] = {
        "type": "video",
        "video": video_path.resolve().as_uri(),
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "total_pixels": TOTAL_PIXELS,
    }
    if nframes is not None:
        item["nframes"] = nframes
    else:
        item["fps"] = fps
    messages = [{"role": "user", "content": [item, {"type": "text", "text": prompt}]}]
    return messages, nframes, fps


def generate(
    captioner: Qwen3VLCaptioner,
    video_path: Path,
    prompt: str,
    *,
    max_new_tokens: int | None = None,
    repetition_penalty: float | None = None,
) -> VlmResult:
    messages, nframes, fps = _video_message(video_path, prompt)
    text = captioner.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    sampled_frames = None
    native_fps = None
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
        if video_metadatas and isinstance(video_metadatas[0], dict):
            indices = video_metadatas[0].get("frames_indices") or []
            sampled_frames = len(indices)
            native_fps = video_metadatas[0].get("fps")
    else:
        video_metadatas = None

    inputs = captioner.processor(
        text=[text],
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **(video_kwargs or {}),
    )
    token_stats = count_processor_tokens(captioner.processor, getattr(inputs, "input_ids", None))
    device = next(captioner.model.parameters()).device
    inputs = inputs.to(device)
    gc = captioner.model.generation_config
    gen_kwargs = _generation_kwargs(
        captioner.cfg.generation,
        eos_token_id=gc.eos_token_id,
        pad_token_id=gc.pad_token_id,
    )
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    output_ids = captioner.model.generate(**inputs, **gen_kwargs)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
    decoded = captioner.processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return VlmResult(
        response=decoded[0].strip(),
        requested_frames=nframes,
        requested_fps=fps,
        sampled_frames=sampled_frames,
        decoder_fps=native_fps,
        visual_tokens=None if token_stats is None else token_stats.get("num_visual_tokens"),
        text_tokens=None if token_stats is None else token_stats.get("num_text_tokens"),
        input_tokens=None if token_stats is None else token_stats.get("num_total_input_tokens"),
    )


def extract_json(text: str) -> dict[str, Any]:
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", body, flags=re.DOTALL)
    if fence:
        body = fence.group(1)
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model response")
    return json.loads(body[start : end + 1])


def model_settings() -> dict[str, Any]:
    return {
        "model_name": "Qwen3-VL-8B-Instruct",
        "model_path": MODEL_PATH,
        "dtype": MODEL_DTYPE,
        "device_map": "auto",
        "attn_implementation": ATTN_IMPLEMENTATION,
        "do_sample": DO_SAMPLE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
        "video_fps": SAMPLE_FPS,
        "max_frames": MAX_FRAMES,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "total_pixels": TOTAL_PIXELS,
    }

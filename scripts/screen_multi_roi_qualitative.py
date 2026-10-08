#!/usr/bin/env python3
"""Qualitative screening of the existing Multi-ROI Top-20 overlay MP4s.

Reads ``outputs/analysis/multi_roi_virat_all/multi_roi_candidates.csv`` and the
already-written Stage3 overlay MP4s. Does not rerun Stage1–3 and does not
rewrite ranking scores. Qwen3-VL only describes what is visible.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
VLM_SRC = REPO_ROOT.parent / "vlm-inference" / "src"
if str(VLM_SRC) not in sys.path:
    sys.path.insert(0, str(VLM_SRC))

from qwen_vl_utils import process_vision_info  # noqa: E402
from vlm_inference.config import (  # noqa: E402
    AppConfig,
    GenerationConfig,
    ModelConfig,
    VideoConfig,
)
from vlm_inference.inputs import Sample  # noqa: E402
from vlm_inference.model import (  # noqa: E402
    Qwen3VLCaptioner,
    _generation_kwargs,
    count_processor_tokens,
)
from vlm_inference.video_io import inspect_video  # noqa: E402

CANDIDATES_CSV = (
    REPO_ROOT / "outputs" / "analysis" / "multi_roi_virat_all" / "multi_roi_candidates.csv"
)
TRACKS_ROOT = REPO_ROOT / "outputs" / "stage3" / "3_roi_tube" / "virat_all"
OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "multi_roi_virat_all"
OVERLAY_SUFFIX = "_roi_tube_light_h0.2_g2t0.25_cmp1.5r3.mp4"
TOP_N = 20

MODEL_PATH = "/data/models/Qwen3-VL-8B-Instruct-BF16"
MODEL_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "sdpa"
SAMPLE_FPS = 1.0
MIN_PIXELS = 4 * 32 * 32
MAX_PIXELS = 512 * 32 * 32
TOTAL_PIXELS = 20480 * 32 * 32
MAX_NEW_TOKENS = 768
DO_SAMPLE = False
REPETITION_PENALTY = 1.0

TRI = ("yes", "no", "uncertain")
TRI_FIELDS = (
    "distinct_events_across_rois",
    "small_target_present",
    "multi_roi_semantically_meaningful",
    "suitable_feasibility_case",
)

PROMPT = """너는 감시 영상 overlay를 보고 화면에 보이는 것만 적는 기록자다.
연구 결론, 방법의 우열, 알고리즘 평가, 실험이 성공했는지에 대한 판단은 쓰지 마라.
보이지 않거나 구분되지 않으면 지어내지 말고 uncertain이라고 해라.

이 영상에는 원본 프레임 위에 두 가지가 덧그려져 있다.
- 색이 번진 영역은 움직임 크기 heatmap이다. 물체나 사건이 아니다.
- 색 테두리 박스와 그 안의 E1, E2, E3 같은 글자가 제안된 ROI다. ROI는 화면에 적힌 그 라벨로만 불러라.

질문:
1. 각 ROI 박스 안에 어떤 주요 object, person, vehicle이 보이는가?
2. 각 ROI 박스 안에서 어떤 action 또는 event가 일어나는가?
3. 서로 다른 ROI가 실제로 서로 다른 대상이나 행동의 증거를 담고 있는가?
4. 한 개의 큰 박스로 합치면 지금 박스들이 따로 보여주고 있는 내용이 섞이거나 잘릴 것처럼 보이는가?
5. 프레임 전체에서 보면 중요한 object나 action이 작게 보이는가?
6. 위 관찰만 놓고, 여러 ROI가 서로 다른 내용을 보여 줘서 사람이 이어서 볼 만한 영상인가?

JSON만 출력해라. 설명 문장이나 코드 펜스는 붙이지 마라.
yes, no, uncertain 외의 값은 쓰지 마라.
short_reason은 화면에 보인 내용 한두 문장만 써라.

{
  "rois": [{"id": "E1", "objects": "", "actions": ""}],
  "distinct_events_across_rois": "yes",
  "small_target_present": "uncertain",
  "multi_roi_semantically_meaningful": "uncertain",
  "suitable_feasibility_case": "uncertain",
  "short_reason": ""
}
"""

CSV_COLUMNS = [
    "rank",
    "video_id",
    "roi_descriptions",
    "distinct_events_across_rois",
    "small_target_present",
    "multi_roi_semantically_meaningful",
    "suitable_feasibility_case",
    "short_reason",
]


def overlay_path(video_id: str) -> Path:
    return TRACKS_ROOT / f"{video_id}{OVERLAY_SUFFIX}"


def load_top_candidates(path: Path, n: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["rank"]))
    top = rows[:n]
    if len(top) != n:
        raise RuntimeError(f"expected {n} candidates, found {len(top)} in {path}")
    return top


def _extract_json(text: str) -> dict[str, Any]:
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", body, flags=re.DOTALL)
    if fence:
        body = fence.group(1)
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model response")
    return json.loads(body[start : end + 1])


def _tri(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in TRI:
        return text
    return "uncertain"


def _roi_descriptions(payload: dict[str, Any]) -> str:
    rois = payload.get("rois")
    if not isinstance(rois, list) or not rois:
        return ""
    parts: list[str] = []
    for roi in rois:
        if not isinstance(roi, dict):
            continue
        label = str(roi.get("id") or roi.get("label") or "").strip() or "?"
        objects = " ".join(str(roi.get("objects") or "").split())
        actions = " ".join(str(roi.get("actions") or "").split())
        parts.append(f"{label}: objects={objects}; actions={actions}")
    return " || ".join(parts)


def parse_screening(text: str) -> dict[str, str]:
    payload = _extract_json(text)
    row = {field: _tri(payload.get(field)) for field in TRI_FIELDS}
    row["roi_descriptions"] = _roi_descriptions(payload)
    row["short_reason"] = " ".join(str(payload.get("short_reason") or "").split())
    return row


def generate_observed(captioner: Qwen3VLCaptioner, sample: Sample, prompt: str) -> dict[str, Any]:
    messages = captioner._build_messages(sample, prompt)
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
        if video_metadatas:
            meta0 = video_metadatas[0]
            if isinstance(meta0, dict):
                indices = meta0.get("frames_indices") or []
                sampled_frames = len(indices)
                native_fps = meta0.get("fps")
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
    token_stats = count_processor_tokens(
        captioner.processor, getattr(inputs, "input_ids", None)
    )
    device = next(captioner.model.parameters()).device
    inputs = inputs.to(device)
    gen = captioner.cfg.generation
    gc = captioner.model.generation_config
    gen_kwargs = _generation_kwargs(
        gen,
        eos_token_id=gc.eos_token_id,
        pad_token_id=gc.pad_token_id,
    )
    output_ids = captioner.model.generate(**inputs, **gen_kwargs)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
    texts = captioner.processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "response": texts[0].strip(),
        "sampled_frames": sampled_frames,
        "decoder_fps": native_fps,
        "visual_tokens": None if token_stats is None else token_stats.get("num_visual_tokens"),
        "text_tokens": None if token_stats is None else token_stats.get("num_text_tokens"),
        "input_tokens": None if token_stats is None else token_stats.get("num_total_input_tokens"),
    }


def settings_block() -> dict[str, Any]:
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
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "total_pixels": TOTAL_PIXELS,
        "input": "Stage3 heatmap + lifetime ROI overlay MP4",
        "overlay_suffix": OVERLAY_SUFFIX,
        "prompt": PROMPT,
    }


def load_raw(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"settings": settings_block(), "videos": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    data["settings"] = settings_block()
    data.setdefault("videos", [])
    return data


def save_raw(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def human_priority(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Pick 8–10 videos for a person to watch. Not a new score and not a final set."""

    def rank_of(row: dict[str, Any]) -> int:
        return int(row["rank"])

    both_yes = [
        row
        for row in rows
        if row["distinct_events_across_rois"] == "yes"
        and row["multi_roi_semantically_meaningful"] == "yes"
    ]
    mixed = [
        row
        for row in rows
        if row not in both_yes
        and "no" not in (
            row["distinct_events_across_rois"],
            row["multi_roi_semantically_meaningful"],
        )
        and (
            row["distinct_events_across_rois"] == "yes"
            or row["multi_roi_semantically_meaningful"] == "yes"
        )
    ]
    both_yes.sort(key=rank_of)
    mixed.sort(key=rank_of)
    ordered = both_yes + mixed
    note = (
        "distinct_events_across_rois와 multi_roi_semantically_meaningful이 둘 다 yes인 영상을 "
        "기존 rank 순으로 먼저 두었다. 8개보다 적으면, 둘 중 하나는 yes이고 나머지는 uncertain인 "
        "영상을 같은 순서로 뒤에 붙였다. no가 하나라도 있으면 이 목록에 넣지 않았다. "
        "이 순서는 읽기 순서일 뿐이며 ranking score와 곱하거나 합치지 않았다."
    )
    if len(ordered) > 10:
        overflow = ", ".join(
            f"rank {int(row['rank'])} `{row['video_id']}`" for row in ordered[10:]
        )
        return (
            ordered[:10],
            note
            + " 해당 영상이 10개를 넘어 앞의 10개만 여기에 적었다. "
            + f"같은 조건으로 빠진 영상: {overflow}.",
        )
    return ordered, note


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def write_md(path: Path, rows: list[dict[str, Any]], raw_videos: list[dict[str, Any]]) -> None:
    by_id = {item["video_id"]: item for item in raw_videos}
    priority, priority_note = human_priority(rows)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = [
        "# Multi-ROI qualitative screening (Top 20)",
        "",
        "기존 Stage1–3 결과와 candidate ranking은 바꾸지 않았다. "
        "Qwen3-VL은 이미 만들어 둔 heatmap/ROI overlay MP4만 보고, 화면에 보이는 내용을 적었다. "
        "이 문서는 사람이 5~6개를 고르기 전의 1차 screening이다. "
        "ranking score와 qualitative 답을 합친 새 점수는 없다.",
        "",
        f"- 실행 시각: `{now}`",
        f"- 모델: `Qwen3-VL-8B-Instruct` (`{MODEL_PATH}`)",
        f"- dtype / attention: `{MODEL_DTYPE}` / `{ATTN_IMPLEMENTATION}`, `device_map=auto`",
        f"- 생성: `do_sample={DO_SAMPLE}`, `max_new_tokens={MAX_NEW_TOKENS}`, "
        f"`repetition_penalty={REPETITION_PENALTY}`",
        f"- 비디오 샘플링: overlay MP4를 `fps={SAMPLE_FPS}`로 샘플. "
        f"`min_pixels={MIN_PIXELS}`, `max_pixels={MAX_PIXELS}`, `total_pixels={TOTAL_PIXELS}`",
        "- 입력: Stage3 lifetime ROI overlay (`E1`, `E2`, … 박스 + turbo heatmap). "
        "원본 MP4를 다시 인코딩하지 않았다.",
        "",
        "## Prompt",
        "",
        "```text",
        PROMPT.rstrip(),
        "```",
        "",
        "## Top 20",
        "",
        "| rank | video_id | distinct_events | small_target | multi_roi_meaningful | feasibility_case | short_reason |",
        "|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | `{video_id}` | {distinct} | {small} | {meaningful} | {suitable} | {reason} |".format(
                rank=int(row["rank"]),
                video_id=row["video_id"],
                distinct=row["distinct_events_across_rois"],
                small=row["small_target_present"],
                meaningful=row["multi_roi_semantically_meaningful"],
                suitable=row["suitable_feasibility_case"],
                reason=str(row["short_reason"]).replace("|", "/"),
            )
        )
    lines.extend(["", "## ROI별 관찰", ""])
    for row in rows:
        lines.append(f"### {int(row['rank'])}. `{row['video_id']}`")
        lines.append("")
        lines.append(row["roi_descriptions"] or "(ROI description 없음)")
        lines.append("")
        meta = by_id.get(row["video_id"], {})
        lines.append(
            f"- overlay `{meta.get('overlay_width')}x{meta.get('overlay_height')}` "
            f"@ {meta.get('overlay_fps')} fps, {meta.get('overlay_frames')} frames; "
            f"Qwen sampled frames={meta.get('sampled_frames')}, "
            f"visual_tokens={meta.get('visual_tokens')}"
        )
        lines.append("")

    lines.extend(
        [
            "## 사람이 먼저 볼 영상",
            "",
            "최종 5~6개를 여기서 확정하지 않는다. 아래는 overlay를 이어서 볼 우선순위일 뿐이다.",
            "",
            priority_note,
            "",
        ]
    )
    if not priority:
        lines.append("yes 조합이 없어 우선 확인 목록을 만들지 않았다. 전체 표를 직접 보면 된다.")
        lines.append("")
    else:
        lines.append("| order | rank | video_id | distinct_events | multi_roi_meaningful | short_reason |")
        lines.append("|---:|---:|---|---|---|---|")
        for order, row in enumerate(priority, start=1):
            lines.append(
                "| {order} | {rank} | `{video_id}` | {distinct} | {meaningful} | {reason} |".format(
                    order=order,
                    rank=int(row["rank"]),
                    video_id=row["video_id"],
                    distinct=row["distinct_events_across_rois"],
                    meaningful=row["multi_roi_semantically_meaningful"],
                    reason=str(row["short_reason"]).replace("|", "/"),
                )
            )
        lines.append("")
    counts = {field: {"yes": 0, "no": 0, "uncertain": 0} for field in TRI_FIELDS}
    for row in rows:
        for field in TRI_FIELDS:
            counts[field][row[field]] += 1
    lines.extend(["## 판단 분포", ""])
    lines.append("| field | yes | no | uncertain |")
    lines.append("|---|---:|---:|---:|")
    for field in TRI_FIELDS:
        lines.append(
            f"| {field} | {counts[field]['yes']} | {counts[field]['no']} | {counts[field]['uncertain']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "multi_roi_qualitative_screening.csv"
    md_path = OUTPUT_DIR / "multi_roi_qualitative_screening.md"
    raw_path = OUTPUT_DIR / "multi_roi_qualitative_screening_raw.json"

    candidates = load_top_candidates(CANDIDATES_CSV, TOP_N)
    raw = load_raw(raw_path)
    done = {
        item["video_id"]: item
        for item in raw["videos"]
        if item.get("response") and not item.get("error")
    }

    cfg = AppConfig(
        model=ModelConfig(
            path=MODEL_PATH,
            dtype=MODEL_DTYPE,
            device_map="auto",
            attn_implementation=ATTN_IMPLEMENTATION,
            name="8b",
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
    captioner = Qwen3VLCaptioner(cfg)

    ordered_raw: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        video_id = cand["video_id"]
        rank = int(cand["rank"])
        path = overlay_path(video_id)
        if video_id in done:
            item = done[video_id]
            print(f"[skip] rank {rank} {video_id}", flush=True)
        else:
            info = inspect_video(path)
            sample = Sample(
                job_name="multi_roi_qualitative_screening",
                sample_id=video_id,
                input_type="video",
                paths=[path.resolve().as_uri()],
            )
            print(f"[run] rank {rank} {video_id}", flush=True)
            try:
                observed = generate_observed(captioner, sample, PROMPT)
                error = None
            except Exception as exc:  # noqa: BLE001
                observed = {
                    "response": "",
                    "sampled_frames": None,
                    "decoder_fps": None,
                    "visual_tokens": None,
                    "text_tokens": None,
                    "input_tokens": None,
                }
                error = str(exc)
            item = {
                "rank": rank,
                "video_id": video_id,
                "overlay_path": str(path),
                "overlay_fps": info.native_fps,
                "overlay_frames": info.frame_count,
                "overlay_width": info.width,
                "overlay_height": info.height,
                "error": error,
                **observed,
            }
            if error:
                print(f"[error] {video_id}: {error}", flush=True)
            else:
                print(
                    f"[done] {video_id} frames={item['sampled_frames']} "
                    f"visual_tokens={item['visual_tokens']}",
                    flush=True,
                )
        ordered_raw.append(item)
        if item.get("error") or not item.get("response"):
            parsed = {field: "uncertain" for field in TRI_FIELDS}
            parsed["roi_descriptions"] = ""
            parsed["short_reason"] = f"model error: {item.get('error') or 'empty response'}"
        else:
            try:
                parsed = parse_screening(item["response"])
            except Exception as exc:  # noqa: BLE001
                parsed = {field: "uncertain" for field in TRI_FIELDS}
                parsed["roi_descriptions"] = ""
                parsed["short_reason"] = f"JSON parse failed: {exc}"
                item["parse_error"] = str(exc)
        rows.append({"rank": rank, "video_id": video_id, **parsed})
        raw["videos"] = ordered_raw
        raw["settings"] = settings_block()
        save_raw(raw_path, raw)
        write_csv(csv_path, rows)
        write_md(md_path, rows, ordered_raw)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(json.dumps({"csv": str(csv_path), "md": str(md_path), "n": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

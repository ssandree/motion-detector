#!/usr/bin/env python3
"""Build /data/datasets/VIRAT/virat_scenes.md + PDF (+ first-frame thumbs)."""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

import cv2
from fpdf import FPDF
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
VIRAT_ROOT = Path("/data/datasets/VIRAT")
THUMB_DIR = VIRAT_ROOT / "virat_scenes" / "thumbs"
MD_PATH = VIRAT_ROOT / "virat_scenes.md"
PDF_PATH = VIRAT_ROOT / "virat_scenes.pdf"
LE60 = REPO / "configs" / "virat_videos_le60s.txt"
FONT_REG = Path("/tmp/NotoSansCJK-KR.ttf")
FONT_BOLD = Path("/tmp/NotoSansCJK-KR-Bold.ttf")

GROUPS: list[dict] = [
    {
        "camera": "벽돌모서리",
        "ids": "000001–000006",
        "blurb": "빨간 벽돌건물 옆 작은 아스팔트. 차 한 대와 사람 위주.",
        "scenes": [
            ("000001", "은색차 벽돌모서리"),
            ("000002", "사람들 지나는 벽돌모서리"),
            ("000003", "사람 모여있는 벽돌모서리"),
            ("000004", "사람들 지나는 벽돌모서리"),
            ("000006", "트렁크 연 흰색차"),
        ],
    },
    {
        "camera": "자전거거치대",
        "ids": "000101–000102",
        "blurb": "건물 사이 보행로. 노란 볼라드와 자전거 랙.",
        "scenes": [
            ("000101", "자전거거치대"),
            ("000102", "자전거거치대 (사람 셋)"),
        ],
    },
    {
        "camera": "경비실 큰주차장",
        "ids": "000200–000207",
        "blurb": "흰 경비실과 벽돌 주차장·차고. 같은 구도에서 빈↔찬만 달라진다.",
        "scenes": [
            ("000200", "빈주차장"),
            ("000201", "중간주차장"),
            ("000202", "중간주차장"),
            ("000203", "찬주차장"),
            ("000205", "더 찬주차장"),
            ("000206", "한산해진 주차장"),
            ("000207", "거의 빈주차장"),
        ],
    },
    {
        "camera": "파라솔",
        "ids": "010000–010005",
        "blurb": "빨간 파라솔·테이블, 잔디 언덕, 왼쪽 넓은 계단. 카메라 구도는 같다.",
        "scenes": [
            ("010000", "파라솔"),
            ("010001", "파라솔"),
            ("010002", "파라솔"),
            ("010003", "파라솔"),
            ("010004", "파라솔"),
            ("010005", "파라솔"),
        ],
    },
    {
        "camera": "옥상주차장",
        "ids": "010106–010116",
        "blurb": "건물에서 내려다본 넓은 롯. 왼쪽 아래 쓰레기장과 건물 그림자.",
        "scenes": [
            ("010106", "한산한 옥상주차장"),
            ("010107", "한산한 옥상주차장"),
            ("010108", "반찬 옥상주차장"),
            ("010109", "찬 옥상주차장"),
            ("010110", "찬 옥상주차장"),
            ("010111", "찬 옥상주차장"),
            ("010112", "중간 옥상주차장"),
            ("010113", "중간 옥상주차장"),
            ("010114", "한산한 옥상주차장"),
            ("010115", "한산한 옥상주차장"),
            ("010116", "한산한 옥상주차장"),
        ],
    },
    {
        "camera": "초록차양",
        "ids": "010200–010208",
        "blurb": "Student Services / CHICANO 초록 차양, 오른쪽 빨간 벽과 흰 화살표. 보행 광장.",
        "scenes": [
            ("010200", "초록차양"),
            ("010201", "초록차양"),
            ("010202", "초록차양"),
            ("010203", "초록차양"),
            ("010204", "초록차양"),
            ("010205", "초록차양"),
            ("010206", "초록차양"),
            ("010207", "초록차양"),
            ("010208", "초록차양"),
        ],
    },
    {
        "camera": "큰잔디밭",
        "ids": "040000–040005",
        "blurb": "줄무늬 잔디, 작은 주차면, 교차로.",
        "scenes": [
            ("040000", "큰잔디밭"),
            ("040001", "큰잔디밭"),
            ("040002", "큰잔디밭"),
            ("040003", "큰잔디밭"),
            ("040004", "큰잔디밭"),
            ("040005", "큰잔디밭"),
        ],
    },
    {
        "camera": "파란트럭 주차장",
        "ids": "040103–040104",
        "blurb": "작은 주차면. 파란 픽업과 건물 그림자가 눈에 띈다.",
        "scenes": [
            ("040103", "파란트럭 주차장"),
            ("040104", "파란트럭 주차장"),
        ],
    },
    {
        "camera": "공사거리",
        "ids": "050000",
        "blurb": "다차로 거리. 주황 콘과 바리케이드.",
        "scenes": [("050000", "공사거리")],
    },
    {
        "camera": "빗길 벽돌주차장",
        "ids": "050200–050203",
        "blurb": "젖은 노면, 노란 화살표, 오른쪽 벽돌 모서리.",
        "scenes": [
            ("050200", "빗길 벽돌주차장"),
            ("050201", "빗길 벽돌주차장"),
            ("050202", "빨간차 있는 빗길주차장"),
            ("050203", "빨간차 있는 빗길주차장"),
        ],
    },
    {
        "camera": "고가도로 주차장",
        "ids": "050300–050301",
        "blurb": "롯 너머 큰 도로와 고가.",
        "scenes": [
            ("050300", "고가도로 주차장"),
            ("050301", "고가도로 주차장"),
        ],
    },
]


def list_videos() -> dict[str, list[Path]]:
    by_scene: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(VIRAT_ROOT.rglob("*.mp4")):
        parts = path.stem.split("_")
        if len(parts) >= 3:
            by_scene[parts[2]].append(path)
    return by_scene


def pick_clip(clips: list[Path], prefer: set[str]) -> Path:
    for clip in clips:
        if clip.stem in prefer:
            return clip
    return clips[0]


def extract_thumbs(
    by_scene: dict[str, list[Path]], prefer: set[str]
) -> dict[str, dict]:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}
    wanted = [sid for group in GROUPS for sid, _ in group["scenes"]]
    for scene_id in wanted:
        clips = by_scene.get(scene_id, [])
        if not clips:
            raise SystemExit(f"no clips for scene {scene_id}")
        chosen = pick_clip(clips, prefer)
        cap = cv2.VideoCapture(str(chosen))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise SystemExit(f"failed to read first frame: {chosen}")
        height, width = frame.shape[:2]
        scale = 960 / width
        thumb = cv2.resize(
            frame,
            (960, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        out = THUMB_DIR / f"{scene_id}.jpg"
        cv2.imwrite(str(out), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        meta[scene_id] = {
            "video": chosen.name,
            "n_clips": len(clips),
            "size": f"{width}×{height}",
            "thumb": out,
        }
    return meta


def write_markdown(meta: dict[str, dict]) -> None:
    lines = [
        "# VIRAT 장면 별칭",
        "",
        "VIRAT Ground (`/data/datasets/VIRAT/`) 장면 ID에 카메라 구도용 별칭을 붙인 목록이다.",
        "각 이미지는 해당 6자리 ID에서 고른 **대표 클립의 첫 프레임**이다.",
        "",
        "- 같은 6자리 ID는 **같은 카메라**다. 클립마다 차·사람 밀도만 달라지는 경우가 많다.",
        "- Google Drive에서 이미지와 함께 보려면 같은 폴더의 [`virat_scenes.pdf`](virat_scenes.pdf)를 연다. Drive는 `.md` 미리보기에 이미지를 넣지 않는다.",
        "",
        "## 카메라 11곳",
        "",
        "| 장면 ID | 별칭 | 한 줄 | 클립 수 |",
        "|---------|------|--------|---------|",
    ]
    for group in GROUPS:
        n = sum(meta[sid]["n_clips"] for sid, _ in group["scenes"])
        lines.append(
            f"| `{group['ids']}` | **{group['camera']}** | {group['blurb']} | {n} |"
        )
    lines += [
        "",
        "## 장면 ID별",
        "",
    ]
    for group in GROUPS:
        lines += [
            f"### {group['ids']} · {group['camera']}",
            "",
            group["blurb"],
            "",
        ]
        for scene_id, name in group["scenes"]:
            info = meta[scene_id]
            rel = f"virat_scenes/thumbs/{scene_id}.jpg"
            lines += [
                f"#### `{scene_id}` — {name}",
                "",
                f"![{scene_id} {name}]({rel})",
                "",
                f"- 대표 클립: `{info['video']}` ({info['size']}, 이 ID 클립 {info['n_clips']}개)",
                "",
            ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


class CatalogPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Noto", size=9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 6, "VIRAT 장면 별칭  ·  카메라 구도 참고", align="L")
        self.ln(8)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Noto", size=8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"{self.page_no()}", align="C")


def _jpeg_bytes(path: Path, max_w: int = 960) -> bytes:
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        h = int(im.height * max_w / im.width)
        im = im.resize((max_w, h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()


def write_pdf(meta: dict[str, dict]) -> None:
    pdf = CatalogPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_font("Noto", "", str(FONT_REG))
    pdf.add_font("Noto", "B", str(FONT_BOLD))
    pdf.set_title("VIRAT 장면 별칭")

    pdf.add_page()
    pdf.set_font("Noto", "B", 20)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 12, "VIRAT 장면 별칭", ln=True)
    pdf.set_font("Noto", size=11)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(
        0,
        6,
        "첫 프레임으로 카메라 구도를 보는 참고 문서다. "
        "같은 6자리 ID는 같은 카메라이고, 클립마다 차·사람 밀도만 다른 경우가 많다.",
    )
    pdf.ln(2)
    pdf.set_font("Noto", size=10)
    pdf.multi_cell(
        0,
        5.5,
        "이미지는 각 장면 ID의 대표 클립 0번째 프레임이다. "
        "원본은 /data/datasets/VIRAT/ 아래 mp4다.",
    )
    pdf.ln(3)
    pdf.set_font("Noto", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 8, "카메라 11곳", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    for group in GROUPS:
        n = sum(meta[sid]["n_clips"] for sid, _ in group["scenes"])
        pdf.set_font("Noto", "B", 10)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(
            0,
            6,
            f"{group['ids']}    {group['camera']}    · 클립 {n}개",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_font("Noto", size=9)
        pdf.set_text_color(70, 70, 70)
        pdf.multi_cell(0, 5, group["blurb"])
        pdf.ln(1.5)

    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    gap = 4.0
    card_w = (usable_w - gap) / 2.0

    for group in GROUPS:
        pdf.add_page()
        pdf.set_font("Noto", "B", 14)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 8, f"{group['ids']}  ·  {group['camera']}", ln=True)
        pdf.set_font("Noto", size=10)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 5.5, group["blurb"])
        pdf.ln(2)

        col = 0
        x_left = pdf.l_margin
        y_row = pdf.get_y()
        for scene_id, name in group["scenes"]:
            info = meta[scene_id]
            jpg = _jpeg_bytes(info["thumb"])
            im = Image.open(io.BytesIO(jpg))
            img_h = card_w * im.height / im.width
            caption_h = 11.0
            block_h = img_h + caption_h
            if y_row + block_h > pdf.h - 16:
                pdf.add_page()
                pdf.set_font("Noto", "B", 11)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(0, 7, f"{group['camera']} (이어서)", ln=True)
                pdf.ln(1)
                y_row = pdf.get_y()
                col = 0
            x = x_left if col == 0 else x_left + card_w + gap
            pdf.set_xy(x, y_row)
            pdf.image(io.BytesIO(jpg), x=x, y=y_row, w=card_w)
            pdf.set_xy(x, y_row + img_h + 0.8)
            pdf.set_font("Noto", "B", 9)
            pdf.set_text_color(20, 20, 20)
            pdf.cell(card_w, 4.5, f"{scene_id}  {name}", ln=True)
            pdf.set_x(x)
            pdf.set_font("Noto", size=7.5)
            pdf.set_text_color(90, 90, 90)
            pdf.cell(card_w, 4, f"{info['video']}  ·  클립 {info['n_clips']}개", ln=True)
            if col == 0:
                col = 1
            else:
                col = 0
                y_row += block_h + 3.5
        if col == 1:
            y_row += block_h + 3.5

    pdf.output(str(PDF_PATH))


def main() -> None:
    VIRAT_ROOT.mkdir(parents=True, exist_ok=True)
    prefer = set(LE60.read_text().split()) if LE60.exists() else set()
    by_scene = list_videos()
    meta = extract_thumbs(by_scene, prefer)
    write_markdown(meta)
    write_pdf(meta)
    print(f"wrote {MD_PATH}")
    print(f"wrote {PDF_PATH} ({PDF_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"thumbs {len(list(THUMB_DIR.glob('*.jpg')))} in {THUMB_DIR}")


if __name__ == "__main__":
    main()

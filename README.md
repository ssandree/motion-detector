# Motion Analyzer

CCTV 영상에서 VLM(Vision-Language Model) 입력용 **Adaptive Input Plan**을 생성하는 파이프라인입니다.

- **v2 (권장)**: motion-first ROI 선정 + ByteTrack 보조 + Adaptive Input Builder
- **v1**: pixel → box → video 3단계 motion 분석 (v2의 motion stage에서 재사용)

상세 방법론:

- [v2 방법론](src/motion_analyzer/v2/MOTION_ANALYZER_V2_METHOD.md)
- [v1 방법론](src/motion_analyzer/v1/MOTION_ANALYZER_V1_METHOD.md)

## 요구 사항

- Python 3.10+
- 입력 영상: `.mp4` (기본 데이터셋은 [VIRAT](https://viratdata.org/) 형식 가정)
- GPU는 선택 사항 (YOLO/ByteTrack 가속에 유리)

## 설치

```powershell
cd motion_analyzer
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

첫 YOLO 실행 시 `yolov8n.pt`가 자동 다운로드됩니다 (`.gitignore`에 포함).

## 입력 영상 준비

기본 `--video_dir`는 `C:/Datasets/VIRAT/video`입니다. 다른 경로를 쓰려면 모든 실행 스크립트에 `--video_dir`를 넘기세요.

VIRAT 트리 예시:

```
C:/Datasets/VIRAT/video/
├── videos-00/
│   └── VIRAT_S_000201_06_001354_001397.mp4
├── videos-01/
│   └── VIRAT_S_010207_08_001308_001332.mp4
└── videos-05/
    └── VIRAT_S_050201_06_001168_001240.mp4
```

`--video_name`에는 **확장자 없는 stem**을 사용합니다 (예: `VIRAT_S_050201_06_001168_001240`). 스크립트가 하위 폴더를 재귀 탐색해 mp4를 찾습니다.

## 빠른 시작 (v2)

프로젝트 루트에서 실행합니다. `PYTHONPATH` 설정 없이 `scripts/` 진입점이 `src/`를 자동으로 추가합니다.

### 1) 전체 파이프라인

```powershell
python scripts/run_motion_pipeline_v2.py `
  --video_dir D:/data/VIRAT/video `
  --video_name VIRAT_S_050201_06_001168_001240 `
  --target_fps 5.0
```

Stage 1–3 (pixel/box/video) → ByteTrack → event ROI → plan 확정까지 한 번에 실행합니다.

### 2) Event ROI만 재빌드 (tracking 생략)

motion blob 파라미터 튜닝 후 ROI만 갱신할 때:

```powershell
python scripts/run_event_roi_refresh.py `
  --output_dir Result/motion_analyzer_v2 `
  --video_name VIRAT_S_050201_06_001168_001240 `
  --write_overlays
```

### 3) VLM 입력 프레임·시각화 생성

```powershell
python scripts/run_adaptive_input_builder.py `
  --output_dir Result/motion_analyzer_v2 `
  --video_name VIRAT_S_050201_06_001168_001240
```

## 출력 구조

기본 결과 루트: `Result/motion_analyzer_v2/<video_stem>/`

```
<video_stem>/
├── _cache/                  # motion map 캐시 (용량 큼, git 제외)
├── _data/                   # CSV/JSON 메타데이터 (git 포함 가능)
│   ├── adaptive_input_plan.json      # 최종 VLM 입력 계획
│   ├── event_roi_tracks.json
│   ├── motion_blobs.csv
│   └── ...
├── adaptive_input/          # VLM crop 프레임 (git 제외)
├── input_plan_overlay.mp4   # git 제외
└── input_builder_visualization.mp4
```

| `motion_type` | VLM 입력 |
|---------------|----------|
| `background_motion` | Sparse Global만 (`roi_inputs = []`) |
| `event_motion` | Sparse Global + Motion ROI 최대 2개 |

## 스크립트 목록

### v2 파이프라인

| 스크립트 | 용도 |
|----------|------|
| `run_motion_pipeline_v2.py` | 전체 v2 파이프라인 |
| `run_tracking_refresh_v2.py` | tracking + plan만 재실행 |
| `run_event_roi_refresh.py` | event ROI + plan 재빌드 (tracking 없음) |
| `run_adaptive_input_builder.py` | plan → VLM crop/시각화 |

### v1 motion stage (v2에서도 호출)

| 스크립트 | 용도 |
|----------|------|
| `run_motion_pixel_level_v1.py` | Stage 1: motion map |
| `run_motion_box_level_v1.py` | Stage 2: motion blob |
| `run_motion_video_level_v1.py` | Stage 3: tube / video 분류 |
| `run_motion_pipeline_v1.py` | v1 전체 (독립 실행) |

### 검증·ground truth

| 스크립트 | 용도 |
|----------|------|
| `validate_4videos_plan.py` | 대표 4영상 plan 검증 리포트 생성 |
| `verify_event_roi_constraints.py` | event ROI 산출물 제약 검사 |
| `build_virat_ground_truth.py` | VIRAT annotation → ground truth JSON |
| `build_curated_ground_truth.py` | 수동 큐레이션 ground truth 생성 |
| `visualize_ground_truth.py` | ground truth 오버레이 영상 |

## 부분 재실행 팁

| 상황 | 권장 명령 |
|------|-----------|
| pixel 파라미터 변경 | `run_motion_pixel_level_v1.py` → box 이후 단계 재실행 |
| blob 파라미터만 변경 | `run_motion_box_level_v1.py` → video/ROI 재실행 |
| tracker만 변경 | `run_tracking_refresh_v2.py` |
| ROI 로직만 변경 | `run_event_roi_refresh.py` |

v1 중간 산출물이 있으면 v2가 `--v1_output_dir`에서 artifact를 재사용할 수 있습니다.

## 검증 데이터

`ground_truth/`에 대표 4영상의 큐레이션 ground truth JSON이 있습니다.

검증 요약: [Result/motion_analyzer_v2/validation_summary_4videos.md](Result/motion_analyzer_v2/validation_summary_4videos.md)

```powershell
python scripts/validate_4videos_plan.py
```

## Git에 포함·제외되는 항목

`.gitignore`로 대용량 산출물은 제외합니다:

- `Result/**/_cache/`, `*.npz`, `*.mp4`, `adaptive_input/`
- `yolov8n.pt` 등 모델 가중치
- `ground_truth/overlays/`

소형 메타데이터(`_data/*.json`, `*.csv`, `validation_summary_4videos.md`)와 소스 코드는 커밋 대상입니다. 로컬에서 파이프라인을 다시 돌리면 캐시·영상 산출물이 재생성됩니다.

## 프로젝트 구조

```
motion_analyzer/
├── scripts/              # CLI 진입점
├── src/motion_analyzer/
│   ├── v1/               # motion detection (pixel/box/video)
│   ├── v2/               # tracking, event ROI, plan finalize
│   └── adaptive_input/   # VLM input frame builder
├── ground_truth/         # 큐레이션 GT JSON
├── Result/               # 파이프라인 산출물 (대부분 git 제외)
└── requirements.txt
```

## 라이선스

저장소에 별도 LICENSE 파일이 없으면 사용 전 프로젝트 소유자에게 문의하세요.

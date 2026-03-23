# Mediapipe Image Pose Pipeline

`best example` 사진 1장과 `worst example` 사진 1장을 입력으로 받아, MediaPipe Pose로 랜드마크를 추출하고 각도를 저장하는 오프라인 분석 파이프라인입니다.

## 목표

1. Python + Conda + MediaPipe + OpenCV 기반 사진 오프라인 분석
2. CSV/JSON 결과 저장
3. 추후 FastAPI API 확장 가능 구조 유지

## 디렉터리 구조

```text
mediapipe_application/
├── config/
│   └── angle_definitions.example.json
├── data/
│   ├── input/
│   └── output/
├── scripts/
│   └── run_batch.py
├── src/
│   └── mp_glute_bridge/
│       ├── __init__.py
│       ├── analyzer.py
│       ├── angle_utils.py
│       └── io_utils.py
└── environment.yml
```

## 설치

```bash
conda env create -f environment.yml
conda activate mediapipe-youtube
```

## 입력 사진 준비

샘플별로 `best`와 `worst` 사진을 하위 폴더에 넣습니다.

예시:

```text
data/input/gb1/best.jpg
data/input/gb1/worst.jpg
data/input/gb2/best.png
data/input/gb2/worst.png
```

## 실행

기본 실행:

```bash
python scripts/run_image_batch.py \
  --input-dir data/input \
  --output-dir data/output \
  --angle-config config/glute_bridge_angles.json
```

새로 추가된 샘플만 처리:

```bash
python scripts/run_image_batch.py \
  --input-dir data/input \
  --output-dir data/output \
  --angle-config config/glute_bridge_angles.json \
  --only-new
```

특정 샘플만 좌우반전해서 처리:

```bash
python scripts/run_image_batch.py \
  --input-dir data/input \
  --output-dir data/output \
  --angle-config config/glute_bridge_angles.json \
  --flip-samples gb2
```

## 출력 결과

샘플마다 `data/output/<sample_name>/` 폴더가 생성되고, 해당 폴더 안에 아래 파일이 저장됩니다.

예시:

```text
data/output/gb1/gb1_landmarks.csv
data/output/gb1/gb1_angles.csv
data/output/gb1/gb1_summary.json
data/output/gb1/gb1_best_skeleton.jpg
data/output/gb1/gb1_worst_skeleton.jpg
```

- `*_landmarks.csv`: best/worst 두 사진의 랜드마크 좌표
- `*_angles.csv`: 샘플 최종 4개 각도
- `*_summary.json`: 처리 요약과 최종 각도
- `*_best_skeleton.jpg`, `*_worst_skeleton.jpg`: skeleton과 주요 landmark 라벨이 그려진 확인용 사진

## angle config 예시

`config/angle_definitions.example.json` 참고

삼점 각도는 `a-b-c` 형태로 계산됩니다. 예를 들어 hip angle이면 `shoulder-hip-knee`처럼 정의할 수 있습니다.

## 참고

사진 기반 파이프라인에서는 더 이상 `worst/best` 시간 구간 설정이 필요하지 않습니다.

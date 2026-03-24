# 🌟 Mediapipe Glute Bridge Image Pose Pipeline

![Mediapipe](https://img.shields.io/badge/MediaPipe-GluteBridge-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![Conda](https://img.shields.io/badge/Conda-supported-orange)

> `best` 이미지 1장과 `worst` 이미지 1장을 입력으로 받아 MediaPipe Pose를 이용해 랜드마크/각도/판정까지 출력을 생성하는 **오프라인 분석 파이프라인**입니다.

---

## 🎯 주요 목표

1. Python + Conda + MediaPipe + OpenCV 기반 이미지 분석
2. `CSV`/`JSON`/`이미지` 형태로 데이터 결과 저장
3. 파이프라인 재사용성, 추가 FastAPI API 확장 적합 구조

---

## 📁 프로젝트 구조

- `config/`
  - `angle_definitions.example.json`: 각도 정의 템플릿
  - `glute_bridge_angles.json`: 실제 각도 정의
  - `glute_bridge_windows.json`, `video_windows.json`: 판정 윈도우 설정
- `data/`
  - `input/`: 샘플 별 사진 (예: `gb1/best.jpg`, `gb1/worst.jpg`)
  - `output/`: 결과 저장, aggregate/summary 포함
- `scripts/`
  - `run_batch.py`, `run_image_batch.py`: 배치 실행 스크립트
  - `aggregate_image_angles.py`: 콜렉션 결과 병합 및 통계
- `src/mp_glute_bridge/`
  - `analyzer.py`: 사진 분석 메인 로직
  - `image_analyzer.py`: 이미지 기반 워크플로우
  - `angle_utils.py`: 각도 계산
  - `io_utils.py`: 파일 입출력
  - `judgment_utils.py`: 판정 로직
- `environment.yml`: Conda 환경 정의

---

## ⚙️ 설치 방법

```bash
conda env create -f environment.yml
conda activate mediapipe-youtube
pip install -r requirements.txt  # 존재 시
```

---

## 🖼 입력 데이터 형식

`data/input/{sample}` 폴더 하위에 `best`, `worst` 이미지 2개가 있어야 합니다.

예시:

```text
data/input/gb1/best.jpg
data/input/gb1/worst.jpg
```

---

## ▶️ 실행 예시

### 기본 실행

```bash
python scripts/run_image_batch.py \
  --input-dir data/input \
  --output-dir data/output \
  --angle-config config/glute_bridge_angles.json
```

### 새 샘플만 처리

```bash
python scripts/run_image_batch.py \
  --input-dir data/input \
  --output-dir data/output \
  --angle-config config/glute_bridge_angles.json \
  --only-new
```

### 특정 샘플 좌우 반전 처리

```bash
python scripts/run_image_batch.py \
  --input-dir data/input \
  --output-dir data/output \
  --angle-config config/glute_bridge_angles.json \
  --flip-samples gb2
```

### 판정 기준 포함 자동 분류

```bash
python scripts/run_image_batch.py \
  --input-dir data/input \
  --output-dir data/output \
  --angle-config config/glute_bridge_angles.json \
  --judgment-ranges data/output/aggregate/posture_judgment_ranges.json \
  --judgment-method min_max
```

---

## 📤 출력 결과

각 샘플의 출력 폴더 `data/output/<sample>/` 생성

- `*_landmarks.csv`: best/worst landmark CSV
- `*_angles.csv`: 계산된 각도 CSV
- `*_summary.json`: 샘플 처리 요약
- `*_classification.json`: 판정 결과
- `*_best_skeleton.jpg`, `*_worst_skeleton.jpg`: 시각화 이미지

`data/output/aggregate/` 에서 전체 집계 결과 제공:
- `all_samples_angles.csv`
- `angle_stats.csv`, `angle_ranges.json`, `mean_angles.csv`, `mean_angles.json`
- `posture_judgment_ranges.csv`, `posture_judgment_ranges.json`

---

## 📐 각도 정의

`config/angle_definitions.example.json` 참고 (3점 (`a-b-c`) 방식).

예: `hip angle`: `shoulder-hip-knee`

---

## 🧠 모듈 설명

- `src/mp_glute_bridge/analyzer.py`: Dataflow + 대상 샘플 처리 제어
- `src/mp_glute_bridge/image_analyzer.py`: 이미지 입력 / Mediapipe Pose API 연결
- `src/mp_glute_bridge/angle_utils.py`: 두 벡터 간 각도, 트리플 앵글 계산
- `src/mp_glute_bridge/io_utils.py`: JSON/CSV 읽기 쓰기 + 디렉터리 자동 생성
- `src/mp_glute_bridge/judgment_utils.py`: 기준값 비교 및 등급화

---

## 💡 주의 사항

- 입력 파일 이름은 `best`/`worst` 형태로 통일
- `environment.yml` 기반 패키지 설치 필수
- 출력 폴더 및 설정 파일 경로를 체크 후 실행

---

## 🌈 마무리

이 리포지토리는 이미지 기반 운동 자세 분석 파이프라인의 핵심을 담고 있으며, 현재 `glute bridge` 분석에 특화되어 있습니다. 이후 `FastAPI` 서버 확장 및 추가 운동(스쿼트 등) 등록이 쉽도록 설계되었습니다.


# Health Eat 발표자료 상세 설명서

> 각 슬라이드 항목에 대한 상세 설명 및 핵심 코드

---

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [EDA 데이터 분석](#2-eda-데이터-분석)
3. [자동화 파이프라인](#3-자동화-파이프라인)
4. [데이터 확장 시도](#4-데이터-확장-시도)
5. [시행착오 상세](#5-시행착오-상세)
6. [2-Stage Pipeline](#6-2-stage-pipeline)
7. [모델 아키텍처](#7-모델-아키텍처)
8. [W&B 실험 추적](#8-wb-실험-추적)
9. [데이터 정제](#9-데이터-정제)
10. [Kaggle 제출](#10-kaggle-제출)
11. [베스트 모델 상세 (0.96703)](#11-베스트-모델-상세-096703)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 목표
- **Kaggle Competition**: AI06 Level1 Project
- **평가 지표**: mAP@[0.75:0.95] - IoU 0.75~0.95 범위의 평균 AP
- **최종 성과**: Baseline 0.82 → Best **0.96703** (+18% 개선)

### 1.2 팀 구성

| 역할 | 담당자 | 핵심 책임 | 주요 산출물 |
|------|--------|----------|------------|
| **Leader** | 이진석 | 프로젝트 조율, 코드 통합 관리, 발표 자료 조율 | README.md, 회의록, 최종 발표 자료 |
| **Data Engineer** | 김민우, 김나연 | EDA 주도, 데이터 전처리/증강 파이프라인 구축 | `notebooks/eda.ipynb`, `src/data/` |
| **Model Architect** | 김보윤 | 모델 리서치/선정, 베이스라인 구현, 아키텍처 최적화 | `src/models/baseline.py`, 모델 체크포인트 |
| **Experimentation Lead** | 황유민 | 실험 추적 시스템(W&B), 하이퍼파라미터 튜닝, 성능 분석 | `configs/experiments/`, `logs/experiments/` |
| **Integration Specialist** | 이진석 | PR 리뷰, 코드 통합, 추론 파이프라인, Kaggle 제출 | `src/inference/`, submission.csv |

**프로젝트 중 실제 기여**:
| 담당자 | 실제 기여 내용 |
|--------|---------------|
| **이진석** | 2-Stage Pipeline 설계/구현, 데이터셋 정제, AIHub 데이터 추출 |
| **김민우** | EDA 분석, 클래스 분포 분석, B2C/B2B 비즈니스 모델 분석 |
| **김나연** | EDA 분석, 색상 증강 실험, 약국/응급구조대 니즈 조사 |
| **김보윤** | Ubuntu 자동화 파이프라인(exc.sh), W&B 콜백, Cloud 아키텍처 설계 |
| **황유민** | W&B 팀 워크스페이스 구축, 증강 실험 전략 설계, 실험 ID 표준화 |

---

## 2. EDA 데이터 분석

### 2.1 데이터 구조

**데이터 경로 설정** (`src/data/yolo_dataset/config.py`):
```python
# Kaggle 데이터
TRAIN_IMG_DIR = "data/train_images"
TRAIN_ANN_DIR = "data/train_annotations"
TEST_IMG_DIR = "data/test_images"

# AIHub 데이터
AIHUB_IMG_DIR = "data/aihub_single"
AIHUB_DETECTOR_IMG_DIR = "data/aihub_detector/images"
AIHUB_DETECTOR_ANN_DIR = "data/aihub_detector/annotations"

# YOLO 출력
YOLO_ROOT = "data/yolo"
VAL_RATIO = 0.2
SPLIT_SEED = 42
```

### 2.2 JSON-이미지 매핑 로직

**COCO JSON 파싱** (`src/data/yolo_dataset/coco_parser.py` 로직):
```python
def load_kaggle_annotations():
    """Kaggle annotation JSON 파일들을 로드"""
    json_files = glob.glob(os.path.join(TRAIN_ANN_DIR, "**/*.json"), recursive=True)

    all_images = []
    all_annotations = []

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # images 섹션에서 메타정보 추출
        for img_info in data.get("images", []):
            all_images.append({
                "id": img_info.get("id"),           # 이미지 고유 ID
                "file_name": img_info.get("file_name"),  # 파일명
                "width": img_info.get("width"),     # 이미지 너비
                "height": img_info.get("height"),   # 이미지 높이
            })

        # annotations 섹션에서 bbox 추출
        for ann in data.get("annotations", []):
            bbox = ann.get("bbox")  # [x, y, w, h] COCO 형식
            if isinstance(bbox, list) and len(bbox) == 4:
                all_annotations.append({
                    "id": ann.get("id"),
                    "image_id": ann.get("image_id"),  # images.id와 매칭
                    "bbox": bbox,
                    "category_id": ann.get("category_id")
                })

    return images_df, annotations_df
```

### 2.3 Classifier용 클래스 매핑 (dl_idx)

**74개 클래스 정의** (`src/data/aihub/extract_and_crop.py`):
```python
# 74개 타겟 클래스 (dl_idx 기준)
KAGGLE_56_DL_IDX = {
    1899, 2482, 3350, 3482, 3543, 3742, 3831, 4542,
    12080, 12246, 12777, 13394, 13899, 16231, 16261, 16547,
    16550, 16687, 18146, 18356, 19231, 19551, 19606, 19860,
    20013, 20237, 20876, 21324, 21770, 22073, 22346, 22361,
    24849, 25366, 25437, 25468, 27732, 27776, 27925, 27992,
    28762, 29344, 29450, 29666, 30307, 31862, 31884, 32309,
    33008, 33207, 33879, 34596, 35205, 36636, 38161, 41767
}  # Kaggle Train 데이터의 56개 클래스

MISSING_18_DL_IDX = {
    4377, 5093, 5885, 6191, 6562,
    10220, 10223, 12419, 18109, 21025, 22626,
    23202, 23222, 27652, 29870, 31704,
    33877, 44198
}  # Test 데이터에만 존재하는 18개 추가 클래스

ALL_74_DL_IDX = KAGGLE_56_DL_IDX | MISSING_18_DL_IDX  # 합집합 = 74개
TARGET_K_CODES = {f"K-{dl_idx + 1:06d}" for dl_idx in ALL_74_DL_IDX}


def k_code_to_dl_idx(k_code: str) -> int:
    """K-code를 dl_idx로 변환

    예시: K-001900 → 1899 (숫자 부분 - 1)
    """
    return int(k_code.split("-")[1]) - 1


def dl_idx_to_k_code(dl_idx: int) -> str:
    """dl_idx를 K-code로 변환

    예시: 1899 → K-001900 (숫자 + 1, 6자리 패딩)
    """
    return f"K-{dl_idx + 1:06d}"
```

**폴더 구조 기반 클래스 매칭**:
```python
def process_kaggle(class_counts: dict, class_data: dict):
    """Kaggle 데이터에서 추출 - 폴더명으로 클래스 결정"""

    for ann_path in KAGGLE_ANN_DIR.glob("**/*.json"):
        # ⭐ 핵심: 폴더 이름이 K-code
        k_code_folder = ann_path.parent.name  # 예: "K-001900"

        if k_code_folder not in TARGET_K_CODES:
            continue

        # K-code → dl_idx 변환
        dl_idx = k_code_to_dl_idx(k_code_folder)  # 1899

        # JSON에서 bbox 추출
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        bbox = annotations[0].get("bbox", [])

        class_data[dl_idx].append({
            'source': 'kaggle',
            'img_path': img_path,
            'bbox': tuple(bbox),
            'k_code': k_code_folder
        })
```

### 2.4 데이터 정합성 문제

**정합성 필터링** (`src/data/yolo_dataset/yolo_export_detector.py`):
```python
def filter_existing_images(images_df, annotations_df):
    """실제 존재하는 이미지만 필터링"""

    # Kaggle 이미지 확인
    kaggle_existing = set(
        os.path.basename(p) for p in glob.glob(
            os.path.join(TRAIN_IMG_DIR, "*.png")
        )
    )
    print(f"[Kaggle] 폴더 내 이미지 수: {len(kaggle_existing)}")

    # AIHub 이미지 확인
    aihub_existing = set()
    if os.path.exists(AIHUB_DETECTOR_IMG_DIR):
        aihub_existing = set(
            os.path.basename(p) for p in glob.glob(
                os.path.join(AIHUB_DETECTOR_IMG_DIR, "*.png")
            )
        )
    print(f"[AIHub] 폴더 내 이미지 수: {len(aihub_existing)}")

    # source별로 필터링
    valid_mask = []
    for _, row in images_df.iterrows():
        fn = row["file_name"]
        source = row.get("source", "kaggle")
        if source == "kaggle":
            valid_mask.append(fn in kaggle_existing)
        else:
            valid_mask.append(fn in aihub_existing)

    images_df = images_df[valid_mask]

    # 유효한 이미지 ID만 남기기
    valid_ids = set(images_df["id"])
    annotations_df = annotations_df[
        annotations_df["image_id"].isin(valid_ids)
    ]

    return images_df, annotations_df
```

**문제 발견 결과**:
- 폴더에는 있지만 JSON에 없는 이미지: **419개**
- JSON에는 있지만 폴더에 없는 이미지: **137개**
- 해결: 폴더 ∩ JSON 교집합만 사용
- 필터링 후 유효 학습 데이터: **232개 이미지, 763개 bbox**

---

## 3. 자동화 파이프라인

### 3.1 exc.sh - 파이프라인 설정 스크립트

**전체 코드** (`scripts/exc.sh`):
```bash
#!/bin/bash

# 현재 실행 위치 = 프로젝트 루트
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

# run.sh 생성
cat << 'EOF' > "$SCRIPT_DIR/run.sh"
#!/bin/bash

PY=python3
export KAGGLE_CONFIG_DIR=./data/.kaggle

# 필요한 패키지 설치
$PY -m pip install gdown
$PY -m pip install --upgrade pip
$PY -m pip install albumentations
$PY -m pip install ultralytics==8.3.235
$PY -m pip install kaggle==1.7.4.5
$PY -m pip install matplotlib
$PY -m pip install seaborn

# 기본 실행 모델
MODEL_FILE=${1:-baseline.py}

# 상대경로 실행 (프로젝트 루트 기준)
$PY -m src.data.data_load.data_loader    # 1. 데이터 로딩
$PY -m src.data.yolo_dataset.yolo_export # 2. YOLO 포맷 변환
$PY -m src.models.$(basename $MODEL_FILE .py)  # 3. 모델 학습
EOF

chmod +x "$SCRIPT_DIR/run.sh"

# alias 등록 (항상 프로젝트 루트에서 실행하는 전제)
echo 'alias exc_pip="bash ./scripts/run.sh \"\$@\""' >> ~/.bashrc

# 적용
source ~/.bashrc

echo "exc_pip 명령어가 등록되었습니다."
```

### 3.2 사용법

```bash
# 기본 실행 (baseline.py)
exc_pip

# 특정 모델 실행
exc_pip yolo11m.py
exc_pip yolo12m_aug.py
```

### 3.3 데이터 로더

**Kaggle API 연동** (`src/data/data_load/data_loader.py`):
```python
import subprocess
import os
import json

# 1. Kaggle 데이터 다운로드
zip_path = os.path.join(target_dir, "ai06-level1-project.zip")

if not os.path.exists(zip_path):
    print("[INFO] ZIP 파일이 없어서 Kaggle에서 다운로드합니다.")
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "ai06-level1-project",
        "-p", target_dir
    ])

# 2. 압축 해제
if not os.path.exists(extract_marker):
    subprocess.run(["unzip", "-o", zip_path, "-d", target_dir])

# 3. 추가 데이터 (Google Drive)
data_path = os.path.join(target_dir, "aihub_single")

if not os.path.exists(data_path):
    subprocess.run([
        "gdown",
        "--folder",
        "1wfrysUZwosthpMUCgi2ECvvCOjn4ziXh",
        "-O",
        target_dir
    ])
```

---

## 4. 데이터 확장 시도

### 4.1 AIHub 복합경구제 (실패)

**문제점**:
1. 이미지당 라벨 1개만 존재 → 학습 시 bbox 1개만 검출
2. 모든 라벨을 가져오면 클래스 폭증 → mAP 0.014 이하

### 4.2 AIHub 단일경구제 (미미)

**데이터 특성 불일치**:
- 이미지당 알약 1개
- 어두운 배경, 붉은색 조명
- Kaggle 테스트 데이터와 다름

**다운로드 과정**:
- 라벨만 먼저 다운로드 후 56개 클래스 필터링
- 1.5TB 다운로드 (23시간 소요)
- 클래스당 100개씩 추출하여 팀 공유 (6GB)

### 4.3 74개 클래스 확장

**12/17 강사님 디렉션**:
- 테스트 이미지에 **18개 추가 클래스** 존재 확인
- 복합경구제 TL/TS에서 추가 클래스 확보 제안

**ZIP 파일 처리** (`src/data/aihub/extract_and_crop.py`):
```python
def process_aihub_zip(class_counts: dict, class_data: dict):
    """AIHub ZIP 파일에서 타겟 클래스만 추출"""

    # ZIP 파일 쌍 찾기 (TL_*_조합.zip, TS_*_조합.zip)
    label_zips = sorted(AIHUB_LABEL_DIR.glob("TL_*_조합.zip"))

    for label_zip_path in label_zips:
        # 대응하는 이미지 ZIP 찾기
        zip_num = label_zip_path.name.split("_")[1]  # TL_1_조합.zip -> 1
        image_zip_path = AIHUB_IMAGE_DIR / f"TS_{zip_num}_조합.zip"

        with zipfile.ZipFile(label_zip_path, 'r') as label_zip, \
             zipfile.ZipFile(image_zip_path, 'r') as image_zip:

            # 이미지 ZIP 파일 목록 캐시
            image_files = {
                Path(f).name: f for f in image_zip.namelist()
                if f.endswith('.png')
            }

            # 라벨링 JSON 파일 순회
            for json_path in label_zip.namelist():
                if not json_path.endswith('.json'):
                    continue

                # K-code 폴더 확인
                parts = json_path.split('/')
                k_code_folder = parts[1]  # K-code 폴더

                if k_code_folder not in TARGET_K_CODES:
                    continue  # 74개 타겟 클래스가 아니면 스킵

                # JSON 읽기 및 bbox 추출
                with label_zip.open(json_path) as f:
                    data = json.load(f)

                bbox = data["annotations"][0]["bbox"]

                class_data[dl_idx].append({
                    'source': 'aihub_zip',
                    'image_zip': image_zip_path,
                    'img_path_in_zip': image_files[img_filename],
                    'bbox': tuple(bbox),
                    'k_code': k_code_folder
                })
```

---

## 5. 시행착오 상세

### 5.1 Phase 1: Baseline 구축

**보윤님 Ubuntu 자동화 파이프라인**:
```bash
exc_pip baseline.py
# → YOLO12m End-to-end → 0.82점
```

**Baseline 모델** (`src/models/baseline.py`):
```python
from ultralytics import YOLO
import wandb

# W&B 초기화
wandb.init(
    project="codeit_team8",
    entity="codeit_team8",
    config={
        "model": "yolov8n.pt",
        "data": "data/yolo/pills.yaml",
        "epochs": 50,
        "imgsz": 640,
    }
)

model = YOLO("yolov8n.pt")
model.add_callback("on_fit_epoch_end", wandb_train_logging)
model.add_callback("on_val_end", wandb_val_logging)

model.train(
    data="data/yolo/pills.yaml",
    epochs=50,
    imgsz=640,
    seed=42,
    save=True,
    save_period=5
)
```

### 5.2 Phase 2: 복합경구제 실패

**문제**: 이미지당 bbox 1개만 학습 → 예측도 1개만 검출
```
학습 데이터: 이미지 1개 = bbox 1개
→ 모델 학습: "이미지당 알약 1개"로 학습됨
→ 추론 결과: 테스트 이미지에서도 bbox 1개만 검출
→ 결과: mAP 0.014 이하 (테스트는 3~4개 알약)
```

### 5.3 Phase 3: 단일경구제 미미

**도메인 불일치**:
| 항목 | Kaggle 테스트 | AIHub 단일경구제 |
|------|--------------|-----------------|
| 배경 | 연회색 | 어두운 배경 |
| 조명 | 주백색 | 붉은색 조명 |
| 알약 수 | 3~4개 | 1개 |

결과: Kaggle + 단일경구제 합쳐도 **0.822** (유의미하지 않음)

### 5.4 Phase 4: ROI Crop 실패

**치명적 오류**:
```python
# 잘못된 코드 (당시)
ann_data = {
    "annotations": [{
        "bbox": [0, 0, crop_w, crop_h],  # ❌ bbox = 이미지 전체
    }]
}
```

→ bbox가 이미지 전체로 설정됨
→ Detector가 "전체 이미지 = 알약"으로 학습
→ 추론 시 개별 알약 bbox 검출 불가

### 5.5 Phase 5: 2-Stage Pipeline 성공

**핵심 아이디어**:
```
기존: 이미지 → YOLO → 74개 클래스 (End-to-end)
     ❌ 클래스 불균형, 데이터 부족

신규: 이미지 → Detector → bbox 검출 (단일 클래스 "Pill")
            → Classifier → 74개 클래스 분류
     ✅ 각 Task 독립 최적화 가능
```

결과: **0.822 → 0.963** (+0.141) 대폭 상승!

### 5.6 Phase 6: 데이터 정제

**cleanup_detector_data.py 적용**:
- bbox 개수 검증 (2~4개)
- IoU 기반 중복 제거
- bbox 크기/비율 검증

결과: **0.963 → 0.967** (Best!)

---

## 6. 2-Stage Pipeline

### 6.1 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    2-Stage Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: YOLO11m Detector                                  │
│  ├── Input: 원본 이미지 (1280x960)                          │
│  ├── Output: Bounding Boxes                                 │
│  ├── 클래스: 단일 ("Pill")                                  │
│  └── 데이터: Kaggle(232) + AIHub(~5,000)                    │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: ConvNeXt Classifier                               │
│  ├── Input: 크롭된 알약 이미지 (224x224)                    │
│  ├── Output: K-code + Confidence                            │
│  ├── 클래스: 74개                                           │
│  └── Pretrained: ImageNet                                   │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 파이프라인 구현

**전체 코드** (`src/inference/pill_pipeline.py`):
```python
class PillPipeline:
    """2-Stage 알약 인식 파이프라인"""

    def __init__(
        self,
        detector_path: str = "runs/detect/yolo11m_detector/weights/best.pt",
        classifier_path: str = "runs/classify/convnext/best.pt",
        detector_conf: float = 0.1,
        classifier_conf: float = 0.5,
        detector_iou: float = 0.5,
        agnostic_nms: bool = True,
    ):
        # 모델 로드
        self.detector = YOLO(detector_path)
        self.classifier = YOLO(classifier_path)

    def predict(self, image_path: str, padding_ratio: float = 0.1) -> list:
        """이미지에서 알약 검출 및 분류"""

        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size

        # Stage 1: Detection
        det_results = self.detector.predict(
            image_path,
            conf=self.detector_conf,
            iou=self.detector_iou,
            agnostic_nms=self.agnostic_nms,
            verbose=False
        )[0]

        results = []

        # 검출된 각 bbox에 대해
        for box in det_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            det_conf = float(box.conf[0])

            # 패딩 적용
            w, h = x2 - x1, y2 - y1
            pad_x, pad_y = w * padding_ratio, h * padding_ratio

            crop_x1 = max(0, int(x1 - pad_x))
            crop_y1 = max(0, int(y1 - pad_y))
            crop_x2 = min(img_w, int(x2 + pad_x))
            crop_y2 = min(img_h, int(y2 + pad_y))

            # Crop
            cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Stage 2: Classification
            cls_results = self.classifier.predict(cropped, verbose=False)[0]

            # Top-1 prediction
            probs = cls_results.probs
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)

            # ⭐ 핵심 수정: YOLO model의 names 직접 사용
            k_code = self.classifier.names[top1_idx]

            results.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "k_code": k_code,
                "detector_conf": round(det_conf, 3),
                "classifier_conf": round(top1_conf, 3)
            })

        return results
```

---

## 7. 모델 아키텍처

### 7.1 Stage 1: YOLO11m Detector

**학습 스크립트** (`src/models/yolo11m_detector.py`):
```python
# Seed 고정
def seed_fix(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_fix(42)

# 설정
CONFIG = {
    "model": "yolo11m.pt",
    "data": "data/yolo/pills.yaml",
    "epochs": 50,
    "imgsz": 640, 
    "batch": 8,   
    "patience": 15,
}

def train():
    model = YOLO(CONFIG["model"])

    model.train(
        data=CONFIG["data"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        seed=42,

        # 저장
        save=True,
        save_period=10,

        # Early stopping
        patience=CONFIG["patience"],

        # Augmentation
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=15, translate=0.1, scale=0.3,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, mixup=0.1,

        # 최적화
        optimizer="AdamW",
        lr0=0.01, lrf=0.01,
    )
```

**학습 결과**:
| Metric | 값 |
|--------|-----|
| mAP50 | 0.995 |
| mAP50-95 | 0.85 |
| Precision | 0.99 |
| Recall | 0.99 |

### 7.2 Stage 2: ConvNeXt Classifier

**학습 스크립트** (`src/models/convnext_classifier.py`):
```python
import timm

CONFIG = {
    "model_name": "convnext_tiny",
    "pretrained": True,
    "epochs": 50,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "img_size": 224,
    "val_split": 0.1,
    "patience": 10,
}

# 모델 생성
model = timm.create_model(
    CONFIG["model_name"],
    pretrained=CONFIG["pretrained"],
    num_classes=num_classes  # 74
)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

**학습 결과**:
- Val Accuracy: 98.5%
- Early Stop: Epoch 21

---

## 8. W&B 실험 추적

### 8.1 콜백 구현

**mAP@0.75:0.95 계산** (`src/models/callbacks.py`):
```python
import wandb
import torch
import numpy as np

_last_train_log = {}

def wandb_train_logging(trainer):
    """train loss를 모아두는 역할"""
    global _last_train_log
    log_dict = {}

    # Train loss
    loss = getattr(trainer, "loss", None)
    if isinstance(loss, torch.Tensor):
        if loss.numel() == 1:
            log_dict["train/loss_total"] = loss.item()
        else:
            for i, v in enumerate(loss):
                log_dict[f"train/loss_part{i}"] = v.item()

    # Learning Rate
    if hasattr(trainer, "optimizer"):
        for i, g in enumerate(trainer.optimizer.param_groups):
            log_dict[f"lr/group{i}"] = g["lr"]

    _last_train_log = log_dict


def wandb_val_logging(validator):
    """validation metrics와 함께 train loss를 한 번에 wandb로 로깅"""
    global _last_train_log
    log_dict = dict(_last_train_log)

    metrics = getattr(validator, "metrics", None)

    if metrics:
        results = metrics.results_dict
        for k, v in results.items():
            log_dict[f"val/{k}"] = float(v)

        # ⭐ mAP@0.75:0.95 계산 (Kaggle 평가 기준)
        # IoU thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        # index 5~9가 0.75~0.95에 해당
        if hasattr(metrics, 'box') and hasattr(metrics.box, 'ap'):
            ap = metrics.box.ap  # shape: (num_classes, 10)
            if ap is not None and len(ap) > 0:
                ap_75_95 = ap[:, 5:].mean()  # IoU 0.75~0.95 평균
                log_dict["val/mAP75-95"] = float(ap_75_95)

    # W&B 업로드
    if log_dict:
        wandb.log(log_dict)
```

### 8.2 모델에 콜백 연결

```python
from src.models.callbacks import wandb_train_logging, wandb_val_logging

model = YOLO("yolo11m.pt")
model.add_callback("on_fit_epoch_end", wandb_train_logging)
model.add_callback("on_val_end", wandb_val_logging)
```

---

## 9. 데이터 정제

### 9.1 정제 기준

**DetectorDataCleaner** (`src/data/cleanup_detector_data.py`):
```python
class DetectorDataCleaner:
    """COCO JSON Detector 데이터 정제"""

    def __init__(
        self,
        data_dir: str = "data/aihub_detector",
        min_bbox_count: int = 2,      # 최소 bbox 개수
        max_bbox_count: int = 4,      # 최대 bbox 개수 (Kaggle 테스트와 동일)
        min_iou_overlap: float = 0.7, # 중복 판정 IoU
        min_bbox_size: int = 30,      # 최소 bbox 크기 (px)
        max_bbox_ratio: float = 3.5,  # 최대 종횡비
        min_area_ratio: float = 0.003,# 이미지 대비 최소 면적 (0.3%)
        max_area_ratio: float = 0.15, # 이미지 대비 최대 면적 (15%)
    ):
        self.issues = {
            "wrong_bbox_count": [],
            "overlapping_bbox": [],
            "out_of_bounds": [],
            "invalid_bbox_size": [],
            "invalid_bbox_ratio": [],
        }
```

### 9.2 IoU 계산

```python
def _calculate_iou(self, box1, box2):
    """IoU 계산 (COCO: x, y, w, h)"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # xyxy로 변환
    ax1, ay1, ax2, ay2 = x1, y1, x1+w1, y1+h1
    bx1, by1, bx2, by2 = x2, y2, x2+w2, y2+h2

    # 교집합
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = w1*h1 + w2*h2 - inter

    return inter / union if union > 0 else 0.0
```

### 9.3 실행 방법

```bash
# 검사만 (dry run)
python src/data/cleanup_detector_data.py --data_dir data/aihub_detector

# 실제 삭제
python src/data/cleanup_detector_data.py --data_dir data/aihub_detector --delete

# 시각화 포함
python src/data/cleanup_detector_data.py --viz
```

---

## 10. Kaggle 제출

### 10.1 제출 파일 생성

**submit.py** (`src/inference/submit.py`):
```python
def run_submission(
    test_dir: str = "data/test_images",
    output_csv: str = "submission.csv",
    detector_conf: float = 0.05,
    classifier_conf: float = 0.3,
):
    # 파이프라인 초기화
    pipeline = PillPipeline(
        detector_conf=detector_conf,
        classifier_conf=classifier_conf,
    )

    results = []
    annotation_id = 1

    for img_path in tqdm(test_images):
        image_id = int(Path(img_path).stem)

        # 추론
        preds = pipeline.predict(img_path)

        for pred in preds:
            x1, y1, x2, y2 = pred["bbox"]

            # K-code → category_id (dl_idx)
            k_code = pred["k_code"]
            category_id = k_code_to_dl_idx(k_code)

            results.append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox_x": int(x1),
                "bbox_y": int(y1),
                "bbox_w": int(x2 - x1),
                "bbox_h": int(y2 - y1),
                "score": round(pred["classifier_conf"], 2),
            })
            annotation_id += 1

    # CSV 저장
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
```

### 10.2 실행

```bash
python -m src.inference.submit \
    --test_dir data/test_images \
    --output submission.csv \
    --det_conf 0.05 \
    --cls_conf 0.3
```

---

## 11. 베스트 모델 상세 (0.96703)

### 11.1 개요

| 항목 | 내용 |
|------|------|
| **최종 점수** | **0.96703** (Kaggle mAP@[0.75:0.95]) |
| **달성일** | 2025-12-19 |
| **담당자** | 이진석 |
| **Baseline 대비** | +18.7% 개선 (0.815 → 0.96703) |

### 11.2 핵심 성공 요인

#### 1. 데이터셋 정제 (가장 중요한 변경)

테스트 환경과 유사한 고품질 데이터만 선별하여 학습 데이터 품질 향상

#### 2. 테스트 환경과 유사한 데이터 필터링

```python
# cleanup_detector_data.py 정제 기준
BBOX_COUNT_RANGE = (3, 4)  # Kaggle 테스트는 3~4개 알약
IOU_THRESHOLD = 0.7        # 진짜 중복만 제거
MIN_BBOX_SIZE = 50         # 너무 작은 bbox 제외
MAX_BBOX_SIZE = 500        # 너무 큰 bbox 제외
ASPECT_RATIO_RANGE = (0.3, 3.0)  # 비정상 비율 제외
```

**정제 후 데이터 분포**:
| bbox 개수 | 이미지 수 | 비율 |
|-----------|----------|------|
| 4개 | ~5,833 | 83% |
| 3개 | ~1,167 | 17% |
| **합계** | **~7,000** | 100% |

#### 3. 2-Stage Pipeline 아키텍처

```
┌────────────────────────────────────────────────────────────────┐
│                    Best Model Pipeline                          │
├────────────────────────────────────────────────────────────────┤
│  Stage 1: YOLO11m Detector                                      │
│  ├── 역할: 알약 위치 검출 (단일 클래스 "Pill")                  │
│  ├── 학습 데이터: ~7,000개 이미지 (3~4개 bbox)                  │
│  └── 핵심: 모든 알약 bbox 수집으로 검출 누락 방지               │
├────────────────────────────────────────────────────────────────┤
│  Stage 2: ConvNeXt Classifier                                   │
│  ├── 역할: 74개 클래스 분류                                     │
│  ├── 학습 데이터: 크롭된 알약 이미지                            │
│  └── 핵심: ImageNet pretrained → 강력한 feature extraction     │
└────────────────────────────────────────────────────────────────┘
```

### 11.3 모델 설정

#### Stage 1: YOLO11m Detector

```python
CONFIG = {
    "model": "yolo11m.pt",
    "data": "data/yolo/pills.yaml",
    "epochs": 50,
    "imgsz": 640,
    "batch": 8,
    "patience": 15,
    "conf": 0.3,
    "iou": 0.5,
}

# Augmentation
hsv_h=0.015, hsv_s=0.5, hsv_v=0.4
degrees=15, translate=0.1, scale=0.3
fliplr=0.5, mosaic=1.0, mixup=0.1
optimizer="AdamW", lr0=0.01
```

**학습 결과**:
| Metric | 값 |
|--------|-----|
| mAP50 | **0.995** |
| mAP50-95 | 0.85 |
| Precision | **0.99** |
| Recall | **0.99** |

#### Stage 2: ConvNeXt Classifier

```python
CONFIG = {
    "model_name": "convnext_tiny",
    "pretrained": True,  # ImageNet
    "epochs": 50,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "img_size": 224,
    "val_split": 0.1,
    "patience": 10,
}

# Transforms
RandomHorizontalFlip()
RandomRotation(15)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**학습 결과**:
| Metric | 값 |
|--------|-----|
| Val Accuracy | **98.5%** |
| Early Stop | Epoch 21 |

### 11.4 추론 설정

```python
# submit.py - Best Score 재현 설정
detector_conf = 0.05    # 낮춰서 recall 확보 (누락 방지)
classifier_conf = 0.3   # 적당한 수준
detector_iou = 0.5      # NMS 임계값
agnostic_nms = True     # 클래스 무관 NMS
max_det = 4             # 이미지당 최대 검출 수
```

### 11.5 점수 개선 히스토리

| 버전 | Score | 주요 변경 | 개선폭 |
|------|-------|----------|--------|
| Baseline | 0.815 | End-to-end YOLO | - |
| v1 | 0.690 | 196 클래스 시도 | -0.125 |
| v2 | **0.920** | **2-Stage 도입 (YOLO-cls)** | **+0.230** |
| v3 | **0.963** | **데이터셋/Detector 개선** | **+0.043** |
| v4 | 0.965 | AIHub 추가 | +0.002 |
| **v5** | **0.96703** | **데이터셋 정제, ConvNeXt** | **+0.002** |

**핵심 개선 포인트**:
1. **2-Stage 분리** (+0.230): 가장 큰 개선
2. **ConvNeXt 유지**: YOLO-cls와 분류 정확도 비슷하여 변경 불필요
3. **데이터셋 정제** (+0.002): 마지막 미세 조정

### 11.6 재현 방법

```bash
# 1. 데이터 추출 (모든 알약 bbox 포함)
python src/data/aihub/extract_for_detector.py

# 2. YOLO 포맷 변환
python -m src.data.yolo_dataset.yolo_export_detector

# 3. 데이터 정제 (3~4개 bbox만)
python src/data/cleanup_detector_data.py --delete

# 4. Detector 학습
python src/models/yolo11m_detector.py

# 5. Classifier 학습
python src/models/convnext_classifier.py

# 6. 추론 및 제출
python -m src.inference.submit \
    --test_dir data/test_images \
    --output submission.csv \
    --det_conf 0.05 \
    --cls_conf 0.3
```

### 11.7 실패한 추가 시도

| 시도 | 결과 | 원인 분석 |
|------|------|----------|
| imgsz 1280 | 0.713 | 학습/추론 설정 불일치 |
| imgsz 1280 + TTA | 0.533 | bbox 정밀도 저하, NMS에서 정확한 bbox 손실 |

**결론**: 640 설정 유지가 최선

### 11.8 한계점 및 향후 개선 방향

**현재 한계**:
- mAP@[0.75:0.95] 기준에서 bbox 경계 정밀도가 매우 중요
- IoU 0.75~0.95에서 정확히 맞아야 점수 획득

**향후 개선 (미적용)**:
1. 추론 시 `imgsz` 명시적 설정
2. Classifier 앙상블 (ConvNeXt + EfficientNet)
3. bbox 후처리 정밀화
4. 더 정밀한 라벨링 데이터 확보

---

## 부록: 점수 히스토리

| # | Score | 설명 | 변화 |
|---|-------|------|------|
| 1 | 0.82 | Baseline (YOLO12m End-to-end) | - |
| 2 | 0.014 | 복합경구제 실패 (bbox 1개) | -0.806 |
| 3 | 0.822 | 단일경구제 추가 | +0.808 |
| 4 | **0.920** | **2-Stage (YOLO + YOLO-cls)** | **+0.098** |
| 5 | **0.963** | **데이터셋/Detector 개선** | **+0.043** |
| 6 | 0.965 | AIHub 데이터 추가 | +0.002 |
| **7** | **0.96703** | **데이터셋 정제 + ConvNeXt** | **+0.002** |

---

## Git 커밋 히스토리 (주요)

```
30408df feat: Add trained model weights for classifier and detector
2dda3d1 feat: Add YOLO11s model weights for pill detection
7bede34 feat: 2-Stage Pill Detection & Classification Pipeline
138181e [Week2] 추가 데이터셋(Zip) 추출 모듈 Json dl_idx 반영 로직 추가
692a839 [Week2] 추가 데이터셋(Zip) 추출 모듈
1911695 [Week2] 증강 v2, v3 추가
d046438 [Week2] Train 이미지 파일 이름 기준 dl_idx 추출
ae383d6 [Week2] Augmentation
b1051ea Large batch_size 11m added for high performance GPU
```

---

*문서 생성일: 2025-12-22*
*작성: Claude Code*

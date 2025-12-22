# Best Model Summary (0.96703)

> 최종 베스트 모델에 사용된 기법 및 설정 정리

---

## 최종 성과

| 항목 | 값 |
|------|-----|
| **Kaggle Score** | **0.96703** |
| **평가 지표** | mAP@[0.75:0.95] |
| **Baseline 대비** | +0.147 (0.82 → 0.96703) |
| **달성일** | 2025-12-19 |

---

## 1. 아키텍처: 2-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    2-Stage Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: YOLO11m Detector                                   │
│  ├── Task: 알약 위치 검출                                    │
│  ├── 클래스: 단일 ("Pill")                                   │
│  └── Output: Bounding Boxes                                  │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: ConvNeXt Classifier                                │
│  ├── Task: 알약 종류 분류                                    │
│  ├── 클래스: 74개                                            │
│  └── Output: K-code + Confidence                             │
└─────────────────────────────────────────────────────────────┘
```

**왜 2-Stage인가?**
- End-to-end (74개 클래스 직접 예측)는 클래스 불균형 + 데이터 부족으로 한계
- Detector: 위치만 담당 → 일반화 용이
- Classifier: 분류만 담당 → 독립 최적화 가능

---

## 2. Stage 1: YOLO11m Detector

### 모델 설정

| 설정 | 값 |
|------|-----|
| Base Model | yolo11m.pt |
| imgsz | 640 |
| batch | 8 |
| epochs | 50 |
| patience | 15 (Early Stopping) |
| optimizer | AdamW |
| lr0 | 0.01 |

### Augmentation

```python
hsv_h=0.015, hsv_s=0.5, hsv_v=0.4
degrees=15, translate=0.1, scale=0.3
fliplr=0.5, flipud=0.0
mosaic=1.0, mixup=0.1
```

### 학습 결과

| Metric | 값 |
|--------|-----|
| mAP50 | 0.995 |
| mAP50-95 | 0.85 |
| Precision | 0.99 |
| Recall | 0.99 |

### 학습 데이터
- Kaggle 원본: ~232개 이미지
- AIHub 추가: ~7,000개 이미지 (정제 후)
- **총 학습 데이터**: ~7,000개 이미지

---

## 3. Stage 2: ConvNeXt Classifier

### 모델 설정

| 설정 | 값 |
|------|-----|
| Base Model | convnext_tiny |
| Pretrained | ImageNet |
| img_size | 224 |
| batch_size | 32 |
| epochs | 50 |
| lr | 1e-4 |
| weight_decay | 0.01 |
| patience | 10 (Early Stopping) |

### Transforms

```python
# Train
transforms.Resize((224, 224))
transforms.RandomHorizontalFlip()
transforms.RandomRotation(15)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Val/Test
transforms.Resize((224, 224))
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### 학습 결과

| Metric | 값 |
|--------|-----|
| Val Accuracy | 98.5% |
| Early Stop | Epoch 21 |

### 학습 데이터
- 74개 클래스 크롭 이미지
- AIHub + Kaggle 데이터에서 추출

### 참고: YOLO-cls와 비교
- **분류 정확도는 비슷함**
- ConvNeXt 유지 이유: ImageNet pretrained로 안정적인 feature extraction

---

## 4. 데이터셋 정제 (핵심)

### 정제 기준 (cleanup_detector_data.py)

```python
BBOX_COUNT_RANGE = (3, 4)    # Kaggle 테스트와 동일
IOU_THRESHOLD = 0.7          # 중복 제거
MIN_BBOX_SIZE = 50           # 너무 작은 bbox 제외
MAX_BBOX_SIZE = 500          # 너무 큰 bbox 제외
ASPECT_RATIO_RANGE = (0.3, 3.0)  # 비정상 비율 제외
```

### 정제 결과

| 항목 | 정제 전 | 정제 후 |
|------|--------|--------|
| 이미지 수 | ~10,000개 | ~7,000개 |
| bbox 개수 분포 | 1~10개 혼재 | 3~4개만 |
| 품질 | 혼재 | 고품질 |

### 왜 중요한가?
- 테스트 이미지는 모두 3~4개 알약
- 학습 데이터도 동일한 분포로 맞춰야 함
- **데이터 품질 > 데이터 양**

---

## 5. 추론 설정

### submit.py / pill_pipeline.py

```python
detector_conf = 0.05      # 낮춰서 recall 확보 (누락 방지)
classifier_conf = 0.3     # 적당한 수준
detector_iou = 0.5        # NMS 임계값
agnostic_nms = True       # 클래스 무관 NMS
max_det = 4               # 이미지당 최대 검출 수
```

### 추론 흐름

```
1. 원본 이미지 입력 (1280x960)
2. Detector 추론 → bbox 검출
3. 패딩 적용 (10%) 후 크롭
4. Classifier 추론 → K-code 예측
5. K-code → dl_idx 변환
6. CSV 출력
```

---

## 6. 클래스 매핑

### 74개 타겟 클래스

```python
# Kaggle Train에 있는 56개
KAGGLE_56_DL_IDX = {1899, 2482, 3350, ...}  # 56개

# Test에만 있는 18개 추가
MISSING_18_DL_IDX = {4377, 5093, 5885, ...}  # 18개

# 총 74개
ALL_74_DL_IDX = KAGGLE_56_DL_IDX | MISSING_18_DL_IDX
```

### K-code ↔ dl_idx 변환

```python
def k_code_to_dl_idx(k_code: str) -> int:
    """K-001900 → 1899"""
    return int(k_code.split("-")[1]) - 1

def dl_idx_to_k_code(dl_idx: int) -> str:
    """1899 → K-001900"""
    return f"K-{dl_idx + 1:06d}"
```

---

## 7. 재현 명령어

```bash
# 1. Detector 데이터 추출
python src/data/aihub/extract_for_detector.py

# 2. 데이터 정제 (3~4개 bbox만)
python src/data/cleanup_detector_data.py --delete

# 3. Detector 학습
python src/models/yolo11m_detector.py

# 4. Classifier 데이터 추출
python src/data/aihub/extract_and_crop.py

# 5. Classifier 학습
python src/models/convnext_classifier.py

# 6. 추론 및 제출
python -m src.inference.submit_v2 \
    --det_conf 0.05 \
    --cls_conf 0.3
```

---

## 8. 점수 개선 히스토리

| 버전 | Score | 주요 변경 |
|------|-------|----------|
| Baseline | 0.82 | End-to-end YOLO |
| v1 | 0.920 | 2-Stage 도입 (+0.10) |
| v2 | 0.963 | 데이터셋/Detector 개선 |
| v3 | 0.965 | AIHub 데이터 추가 |
| **v4** | **0.96703** | **데이터셋 정제** |

**핵심 개선 포인트**:
1. **2-Stage 분리** (+0.10): 가장 큰 개선
2. **데이터셋 정제**: 테스트와 유사한 3~4개 bbox만 사용

---

## 9. 사용하지 않은 기법 (실패 or 미적용)

| 기법 | 결과 | 이유 |
|------|------|------|
| imgsz 1280 | 0.713 | 학습/추론 설정 불일치 |
| TTA | 0.533 | mAP@[0.75:0.95]에서 bbox 정밀도 저하 |
| End-to-end 74클래스 | ~0.6 | 클래스 불균형, 데이터 부족 |
| Ensemble | 미적용 | 시간 부족 |

---

## 10. 파일 위치

| 파일 | 경로 |
|------|------|
| Detector 학습 | `src/models/yolo11m_detector.py` |
| Classifier 학습 | `src/models/convnext_classifier.py` |
| 2-Stage Pipeline | `src/inference/pill_pipeline.py` |
| 제출 스크립트 | `src/inference/submit_v2.py` |
| 데이터 정제 | `src/data/cleanup_detector_data.py` |
| Detector 데이터 추출 | `src/data/aihub/extract_for_detector.py` |
| Classifier 데이터 추출 | `src/data/aihub/extract_and_crop.py` |
| 학습된 Detector | `models/yolo11m_detector.pt` |
| 학습된 Classifier | `models/convnext_classifier.pt` |

---

*문서 생성일: 2025-12-22*

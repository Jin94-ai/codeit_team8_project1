# Health Eat 프로젝트 타임라인

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | Health Eat - AI 알약 인식 |
| **기간** | 2025.12.04 ~ 12.23 (약 3주) |
| **목표** | Kaggle mAP@[0.75:0.95] 경쟁 |
| **최종 결과** | **0.96703** (Baseline 대비 +18.7%) |

---

## Phase 1: 프로젝트 셋업 (12/04 ~ 12/05)

### 주요 작업
- 프로젝트 구조 초기화
- 팀 역할 배정
- 협업 워크플로우 수립

### 커밋 하이라이트
```
12/04 [Week 0] Initialize project structure with visual documentation
12/05 [Week 0] Setup: Add project structure and leader tools
12/05 Update team roles with member names
```

### 팀 구성
| 역할 | 담당자 |
|------|--------|
| Leader / Integration | 이진석 |
| Data Engineer | 김민우 |
| Data Engineer | 김나연 |
| Model Architect | 김보윤 |
| Experimentation Lead | 황유민 |

---

## Phase 2: EDA & 데이터 전처리 (12/05 ~ 12/09)

### 주요 작업
- 데이터 탐색적 분석 (EDA)
- YOLO 데이터셋 포맷 변환
- 데이터 전처리 파이프라인 구축

### 커밋 하이라이트
```
12/05 [Week 0] data eda
12/08 first EDA
12/08 [Week 1] Yolo 데이터셋 ver1
12/08 [Week 1] Yolo 데이터셋 ver2
12/08 Refactor: Move yolo_dataset to src/data for better project structure
```

### 주요 발견
- 학습 데이터: 232개 이미지, 74 클래스
- 테스트 데이터: 843개 이미지, 196 클래스
- **클래스 불일치 문제 확인** → 2-Stage 접근 필요성 인식

---

## Phase 3: Baseline 모델 개발 (12/09 ~ 12/12)

### 주요 작업
- YOLO 베이스라인 모델 구현
- 초기 학습 파이프라인 구축
- AIHub 데이터 통합 시도

### 커밋 하이라이트
```
12/09 auto pipeline for data loading, YOLO export, and model training
12/10 feat: Add AIHub data integration with ID collision fix
12/11 Refactor AIHub integration: switch from combo to single data approach
12/12 M model added (YOLO11m)
```

### 첫 Kaggle 제출
| Submission | Score | 내용 |
|------------|-------|------|
| #1 | **0.815** | End-to-end YOLO Baseline |

---

## Phase 4: 모델 실험 & Augmentation (12/12 ~ 12/16)

### 주요 작업
- 다양한 YOLO 버전 실험 (8n, 8s, 8m, 11n, 11s, 11m, 12*)
- 데이터 증강 기법 적용
- WandB 모니터링 도입

### 커밋 하이라이트
```
12/12 Seed Fix Added
12/15 [Week2] 증강 모듈(개별 브랜치용)
12/16 11m added for Colab
12/16 feat: Add mAP@0.75:0.95 metric for Kaggle evaluation
12/16 [Week2] Augmentation
```

### 실험 모델
- YOLOv8: n, s, m 버전
- YOLO11: n, s, m 버전 + Augmentation 변형
- YOLO12: n, s, m 버전 + Augmentation 변형

### Kaggle 제출
| Submission | Score | 내용 |
|------------|-------|------|
| #2 | 0.690 | End-to-end 196 클래스 시도 (실패) |

---

## Phase 5: 2-Stage Pipeline 개발 (12/17 ~ 12/18)

### 주요 작업
- **핵심 전환점: 2-Stage 아키텍처 도입**
- Stage 1: YOLO Detector (단일 클래스 "Pill")
- Stage 2: ConvNeXt Classifier (74 클래스)

### 커밋 하이라이트
```
12/17 [Week2] 실험 스케줄링
12/17 [Week2] 추가 데이터셋(Zip) 추출 모듈
12/18 feat: 2-Stage Pill Detection & Classification Pipeline
12/18 feat: Add trained model weights for classifier and detector
```

### 아키텍처 변경
```
Before: YOLO (End-to-end) → 196 클래스 직접 예측
After:  YOLO Detector → 알약 검출 → ConvNeXt → 클래스 분류
```

### Kaggle 제출
| Submission | Score | 내용 |
|------------|-------|------|
| #3 | **0.920** | 2-Stage (YOLO + YOLO-cls) |
| #4 | **0.963** | 2-Stage (YOLO + ConvNeXt) |

---

## Phase 6: 성능 최적화 (12/18 ~ 12/19)

### 주요 작업
- AIHub 데이터 추출 로직 수정 (핵심 개선)
- 데이터 정제 (3~4개 bbox 필터링)
- 추론 파라미터 최적화

### 핵심 발견
**문제**: 기존 코드는 74개 타겟 클래스 폴더만 스캔
→ 이미지에 다른 클래스 알약이 있으면 bbox 누락

**해결**: 모든 K-code 폴더 스캔하여 bbox 수집
→ Detector가 모든 알약 위치 학습 가능

### Kaggle 제출
| Submission | Score | 내용 |
|------------|-------|------|
| #5 | 0.965 | AIHub 데이터 추가 |
| **#6** | **0.96703** | **bbox 추출 수정 + 데이터 정제** |

---

## Phase 7: 추가 실험 & 마무리 (12/19 ~ 12/22)

### 주요 작업
- imgsz 1280 실험 (실패)
- TTA (Test Time Augmentation) 실험 (실패)
- 저장소 정리 및 문서화

### 커밋 하이라이트
```
12/22 feat: 2-Stage Pipeline 최종 버전 (Best Score 0.96703)
12/22 docs: 협업일지/실험로그 날짜 정정
12/22 refactor: 저장소 구조 정리
```

### 추가 실험 결과
| Submission | Score | 내용 |
|------------|-------|------|
| #7 | 0.713 | imgsz 1280 (성능 저하) |
| #8 | 0.533 | imgsz 1280 + TTA (성능 저하) |

### 교훈
- imgsz를 키운다고 무조건 성능 향상 X
- 학습/추론 설정 일치의 중요성
- mAP@[0.75:0.95]는 bbox 정밀도에 민감

---

## 점수 개선 히스토리

```
0.815 ─────────────────────────────────────────────────── Baseline
  │
  │  ↓ End-to-end 196 클래스 (실패)
  │
0.690 ───────────────────────────────────────────────────
  │
  │  ↑ 2-Stage Pipeline 도입 (+0.230)
  │
0.920 ───────────────────────────────────────────────────
  │
  │  ↑ ConvNeXt Classifier (+0.043)
  │
0.963 ───────────────────────────────────────────────────
  │
  │  ↑ bbox 추출 수정 (+0.004)
  │
0.967 ─────────────────────────────────────────────────── Best Score
```

---

## 팀원별 기여

### 이진석 (Leader / Integration)
- 프로젝트 구조 설계
- 2-Stage Pipeline 구현
- AIHub 데이터 추출 로직 개선
- 최종 모델 통합

### 김민우 (Data Engineer)
- 데이터 EDA
- YOLOv8 실험
- 데이터 증강 (Augmentation)
- 추가 데이터셋 추출 모듈

### 김나연 (Data Engineer)
- 데이터 전처리
- YOLO11n 실험
- 증강 모듈 개발
- EDA 노트북

### 김보윤 (Model Architect)
- YOLO11/12 모델 실험
- Colab 학습 환경
- 시드 고정 & 체크포인트
- WandB 모니터링

### 황유민 (Experimentation Lead)
- 실험 스케줄링
- YOLO12 실험
- 학습 콜백 구현
- 실험 로그 관리

---

## 핵심 성공 요인

1. **2-Stage 분리**: Detection과 Classification 독립 최적화
2. **AIHub 데이터 활용**: bbox 어노테이션 품질 우수
3. **데이터 필터링**: 테스트 환경과 유사한 3~4개 bbox만 사용
4. **ConvNeXt 도입**: ImageNet pretrained → 강력한 분류 성능

---

## 프로젝트 통계

| 항목 | 수치 |
|------|------|
| 총 커밋 수 | ~180개 |
| PR 수 | 86개 |
| 협업일지 | 40개 |
| 실험 로그 | 12개 |
| 회의록 | 4개 |
| 모델 실험 | 30+ 버전 |
| Kaggle 제출 | 8회 |

---

## 결론

3주간의 프로젝트를 통해 **Baseline 0.815에서 0.96703까지 +18.7% 개선** 달성.

핵심은 **End-to-end 접근법의 한계를 인식**하고 **2-Stage Pipeline으로 전환**한 것.
데이터 품질 개선(AIHub bbox 추출 수정)이 최종 성능 향상의 결정적 요인이었음.

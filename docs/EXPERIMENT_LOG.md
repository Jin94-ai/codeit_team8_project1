# Experiment Log

## 2024-12-11 ~ 12-12: AIHub 데이터 통합 시도

### 배경
- Competition 데이터만으로 학습 시 mAP ~0.8 달성
- 클래스 불균형 해소를 위해 AIHub 경구약제 데이터 활용 시도

### 시도 1: Combo 데이터 통합

**방법:**
- AIHub "경구약제 이미지 데이터" 중 콤보(3-4개 알약 조합) 이미지 다운로드
- Competition 56개 TARGET_CLASSES에 해당하는 이미지만 필터링
- train_images, train_annotations에 통합

**결과:**
- mAP 0.8 → 0.6으로 **성능 하락**

**원인 분석:**
- 콤보 이미지에는 TARGET_CLASSES 외의 알약도 포함됨
- 이 알약들은 annotation에 없어서 **"배경"으로 학습됨**
- 모델이 실제 알약을 배경으로 오인식하는 문제 발생

```
예시: 콤보 이미지에 알약 A, B, C, D가 있고
      A만 TARGET_CLASS인 경우
      → B, C, D는 annotation 없음
      → 모델은 B, C, D를 "배경"으로 학습
      → 테스트 시 유사한 알약을 배경으로 오탐
```

### 결론: Combo 데이터 사용 불가

**이유:**
1. TARGET_CLASSES 외 알약이 배경으로 학습되어 성능 저하
2. 5000종 중 56종만 사용하므로 대부분의 콤보 이미지에 "미라벨링 객체" 존재

---

### 시도 2: Single 데이터로 전환 계획

**방법:**
- AIHub 단일 알약 이미지(싱글 데이터)만 사용
- 이미지당 1개 알약만 있으므로 "미라벨링 객체" 문제 없음

**진행 상황:**

1. **라벨링 데이터 분석** (완료)
   - TL_1~81 ZIP 파일 분석
   - 55/56개 TARGET_CLASSES 발견 (41767 미발견)
   - 32개 TS 폴더에 분산

2. **문제점 발견**
   - 32개 TS 이미지 폴더 총 용량: ~1TB
   - 로컬 저장 공간 부족 (<100GB)

3. **대안 검토**
   - aihubshell 선택적 다운로드: ZIP 단위라 개별 이미지 선택 불가
   - Kaggle/DACON 선례 검색: AIHub 외부데이터 활용 사례 찾지 못함

### 현재 상태

| 항목 | 상태 |
|-----|------|
| Combo 데이터 | 폐기 (성능 저하 확인) |
| Single 데이터 라벨링 분석 | 완료 |
| Single 데이터 이미지 다운로드 | 미진행 (용량 문제) |
| baseline.py | 원상복구 (epochs=50, augmentation 기본값) |

### 생성된 스크립트

```
src/data/aihub/
├── config.py              # TARGET_CLASSES 설정
├── analyze_annotations.py # TL ZIP 분석 (완료)
├── integrate_single.py    # 싱글 데이터 통합 (미사용)
└── README_AIHUB.md
```

### 분석 결과 파일

- `data/ts_analysis_result.json`: TS 폴더별 TARGET_CLASSES 매핑

---

## 교훈

1. **외부 데이터 통합 시 주의점**
   - Object Detection에서 "라벨링되지 않은 객체"는 배경으로 학습됨
   - 다중 객체 이미지 사용 시 모든 객체가 라벨링되어야 함

2. **데이터 품질 > 데이터 양**
   - 잘못된 데이터 추가는 성능을 오히려 저하시킴
   - 도메인 갭도 고려 필요 (스튜디오 vs 실제 환경)

3. **단일 객체 이미지의 장점**
   - 미라벨링 문제 없음
   - 깔끔한 학습 데이터

---

## 다음 단계 (검토 필요)

1. Competition 데이터 클래스 분포 분석
2. 부족한 클래스만 AIHub Single에서 보강 (용량 최소화)
3. 또는 Augmentation 강화로 데이터 불균형 해소

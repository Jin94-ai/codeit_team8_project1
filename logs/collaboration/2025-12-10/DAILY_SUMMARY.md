# 2025-12-10 일일 작업 요약

## 📊 오늘의 주요 성과

### 1. 🎯 Kaggle 제출 파이프라인 완성 (JIN)

**PR #42**: Fix submission format and add experiment tracking

**핵심 수정사항**:
- ✅ **Category ID 매핑 문제 해결** (0점 원인)
  - YOLO 0-based index → 원본 COCO category_id 변환
  - `class_mapping.json` 생성 (yolo_export.py)
  - Submission 생성 시 정확한 category_id 사용

- ✅ **파이프라인 안정성 개선**
  - run.sh 필수 패키지 추가 (scikit-learn, pandas, numpy, wandb)
  - 근본 원인 해결: yolo_export 실패 → pills.yaml 미생성 문제

- ✅ **Submission 자동 생성**
  - outputs/submissions/ 폴더에 타임스탬프 파일명 저장
  - 실험 히스토리 자동 보관

- ✅ **Inference 파이프라인 분리**
  - scripts/inference.py 추가
  - CLI로 쉽게 재사용 가능

**변경 파일**:
- scripts/run.sh
- src/data/yolo_dataset/yolo_export.py
- src/models/baseline.py
- scripts/inference.py (신규)
- .gitignore (outputs/ 추가)

---

### 2. 🔧 실행 스크립트 개선 (보윤님)

**PR #45**: modify scripts, model(wandb)

**주요 변경**:
- ✅ scripts/run.sh → exc.sh로 파이프라인 개선
- ✅ W&B callback 통합 (`add_wandb_callback`)
- ✅ Makefile 추가 (clean 명령어)

---

### 3. 📈 추가 데이터셋 EDA (민우님)

**PR #40, #41**:
- ✅ TL1 데이터셋 EDA (ver1, ver2)
- ✅ TS1 데이터셋 EDA
- ✅ TL4 데이터셋 EDA
- ✅ 시각화 업데이트

**노트북**:
- notebooks/ver1_mw_eda_add_TL1.ipynb
- notebooks/ver2_mw_eda_add_TL1.ipynb
- notebooks/TL_3.ipynb
- notebooks/TL_4.ipynb

---

### 4. 🧪 실험 추적 시스템 (유민님)

**PR #43**: experiment logs

**추가 파일**:
- logs/experiments/exp_001.md (업데이트)
- logs/experiments/exp_002.md (신규)
- logs/experiments/sceduling.md (업데이트)

**병합 방식**:
- baseline.py 변경사항 제외 (submission 생성 코드 보존)
- 실험 로그만 선택적 merge

---

### 5. 📝 협업일지 작성 (나연님)

**PR #46**:
- logs/collaboration/2025-12-10/2025-12-10_나연.md

---

## 🔥 긴급 이슈 해결

### Issue: PR 충돌 (유민님 - baseline.py)

**문제**:
- 유민님 브랜치가 submission 생성 코드 전체 삭제
- JIN의 핵심 수정사항(class_mapping.json)과 충돌

**해결**:
```bash
git merge --no-commit --no-ff origin/members/hwang-yumin
git checkout --ours src/models/baseline.py
git commit
```

**결과**:
- ✅ 실험 로그만 선택적으로 merge
- ✅ baseline.py는 main 버전 유지 (submission 기능 보존)

---

## 📦 최종 Main 브랜치 상태

**커밋 수**: 27개 (최근 업데이트)

**주요 기능**:
1. ✅ 정확한 Kaggle submission 생성 (category_id 매핑 완료)
2. ✅ W&B 통합 실험 추적
3. ✅ 추가 데이터셋 EDA 완료 (TL1, TS1, TL3, TL4)
4. ✅ 재사용 가능한 inference 파이프라인
5. ✅ 안정적인 패키지 관리

---

## 🎯 다음 단계

### 즉시 필요
1. **Kaggle 첫 제출** (category_id 수정본)
   - 0점 해결 여부 확인
   - Baseline mAP 점수 확보

2. **제출 전략 수립**
   - 1일 5회 제한 관리
   - 누가 언제 제출할지 조율

### 단기 목표
3. **파라미터 튜닝**
   - conf, iou, imgsz 최적화
   - W&B로 실험 추적

4. **추가 데이터 활용**
   - TL1, TL4 데이터 통합 여부 결정
   - 데이터 증강 전략 수립

---

## 📌 참고사항

### 실행 방법 (업데이트됨)

**기존**:
```bash
bash scripts/run.sh
```

**신규 (보윤님 버전)**:
```bash
./scripts/exc.sh
exc_pip
```

**Inference만 실행**:
```bash
python scripts/inference.py --model runs/detect/train/weights/best.pt
```

---

## 🏆 팀원별 기여

| 팀원 | 주요 기여 | PR 번호 |
|:-----|:----------|:--------|
| **이진석** | Submission 형식 수정, Inference 파이프라인 | #42 |
| **김보윤** | 실행 스크립트 개선, W&B 통합 | #45 |
| **김민우** | 추가 데이터셋 EDA (TL1/TS1/TL4) | #40, #41 |
| **황유민** | 실험 추적 시스템 구축 | #43 |
| **김나연** | 협업일지 작성 | #46 |

---

**작성일**: 2025-12-10
**작성자**: 이진석 (Leader & Integration Specialist)

# Git 워크플로우

## 브랜치 전략

### 브랜치 구조

```
main (배포 가능한 안정 버전)
├── JIN (이진석 작업 브랜치)
├── members/hwang-yumin (황유민 작업 브랜치)
├── members/kim-boyoon (김보윤 작업 브랜치)
├── members/kim-minwoo (김민우 작업 브랜치)
└── members/kim-nayeon (김나연 작업 브랜치)
```

### 브랜치 규칙

- **main**: 안정적이고 검증된 코드만 병합
- **개인 브랜치**: 각자 자유롭게 작업
- **브랜치명**: `members/이름` 또는 `이름` 형식

---

## 작업 흐름

### 1. 첫 설정 (최초 1회)

```bash
# 저장소 클론
git clone https://github.com/Jin94-ai/codeit_team8_project1.git
cd codeit_team8_project1

# 개인 브랜치 생성 및 전환
git checkout -b members/your-name

# 원격 저장소에 푸시
git push -u origin members/your-name
```

### 2. 일상 작업

```bash
# 1. 최신 코드 가져오기
git pull origin main

# 2. 작업 시작 (파일 수정/생성)

# 3. 변경사항 확인
git status

# 4. 변경사항 스테이징
git add .
# 또는 특정 파일만
git add src/models/yolo.py

# 5. 커밋
git commit -m "[Week X] 작업 내용"

# 6. 원격 저장소에 푸시
git push origin members/your-name
```

### 3. Pull Request (PR) 생성

```bash
# GitHub 웹사이트에서:
1. 본인 브랜치 선택
2. "Pull Request" 버튼 클릭
3. 제목: [Week X] 작업 요약
4. 설명: 변경사항, 테스트 결과, 리뷰 요청 사항
5. Reviewers에 Integration Specialist(이진석) 추가
6. "Create Pull Request" 클릭
```

### 4. 코드 리뷰 및 병합

```bash
# Integration Specialist가 리뷰 후 승인
# "Merge Pull Request" 버튼으로 main에 병합
# 병합 후 브랜치 삭제는 선택사항
```

---

## 커밋 메시지 규칙

### 형식

```
[Week X] 작업 카테고리: 간단한 설명

예:
[Week 0] Setup: Add project folder structure
[Week 1] Data: Complete EDA notebook
[Week 1] Model: Implement YOLOv8 baseline
[Week 2] Experiment: Add Albumentations augmentation
[Week 3] Docs: Update final presentation
```

### 카테고리

- `Setup`: 환경 설정, 폴더 구조
- `Data`: 데이터 처리, EDA, 증강
- `Model`: 모델 구현, 학습
- `Experiment`: 실험, 하이퍼파라미터 튜닝
- `Docs`: 문서, 로그, 회의록
- `Fix`: 버그 수정
- `Refactor`: 코드 리팩토링

---

## 금지 사항

### ❌ 절대 하지 말 것

1. **main 브랜치에 직접 푸시**
   ```bash
   # 이렇게 하지 마세요!
   git checkout main
   git add .
   git commit -m "..."
   git push origin main
   ```

2. **리뷰 없이 병합**
   - 반드시 Integration Specialist의 승인 필요

3. **큰 파일 커밋**
   - 데이터 파일 (`.csv`, `.jpg`, `.png`)
   - 모델 체크포인트 (`.pth`, `.pt`)
   - `.gitignore`에 이미 설정되어 있음

4. **민감 정보 커밋**
   - API 키, 비밀번호
   - `.env` 파일 사용 권장

---

## 충돌 해결

### 충돌 발생 시

```bash
# 1. main 최신 코드 가져오기
git checkout main
git pull origin main

# 2. 본인 브랜치로 돌아가기
git checkout members/your-name

# 3. main 병합
git merge main

# 4. 충돌 발생 시 파일 수정
# VSCode에서 충돌 부분 확인 및 수정

# 5. 충돌 해결 후 커밋
git add .
git commit -m "[Fix] Resolve merge conflict"

# 6. 푸시
git push origin members/your-name
```

### 도움 요청

- Integration Specialist(이진석)에게 연락
- 팀 Discord/Slack에 문의

---

## 유용한 명령어

```bash
# 현재 브랜치 확인
git branch

# 원격 브랜치 확인
git branch -r

# 변경사항 임시 저장
git stash

# 임시 저장 복원
git stash pop

# 마지막 커밋 수정 (푸시 전에만!)
git commit --amend

# 최근 커밋 이력 확인
git log --oneline -10

# 특정 파일 변경 이력
git log --oneline src/models/yolo.py
```

---

## 참고 자료

- [GitHub 공식 문서](https://docs.github.com/)
- [Git 간단 가이드](https://rogerdudler.github.io/git-guide/index.ko.html)
- 팀 Discord/Slack: #git-help 채널

---

**작성일**: 2025-12-05
**작성자**: 이진석 (Integration Specialist)

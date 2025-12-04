# GitHub 협업 완벽 가이드

5인 팀 프로젝트를 위한 GitHub 협업 필수 지식

---

## 목차

1. [Git & GitHub 기초](#1-git--github-기초)
2. [Branch 전략](#2-branch-전략)
3. [Pull Request (PR) 워크플로우](#3-pull-request-pr-워크플로우)
4. [Code Review 가이드](#4-code-review-가이드)
5. [Merge Conflict 해결](#5-merge-conflict-해결)
6. [Issue 기반 작업 흐름](#6-issue-기반-작업-흐름)
7. [팀 협업 규칙](#7-팀-협업-규칙)

---

## 1. Git & GitHub 기초

### Git이란?

**버전 관리 시스템** - 코드의 변경 이력을 추적하고 관리

```
Version 1 → Version 2 → Version 3
   ↓           ↓           ↓
 Initial    Add Feature  Bug Fix
```

### GitHub이란?

**Git 저장소 호스팅 서비스** - 협업을 위한 플랫폼

- 원격 저장소 제공
- Pull Request, Issue, Projects 등 협업 도구
- 코드 리뷰 및 토론

### 기본 용어

| 용어 | 설명 | 예시 |
|------|------|------|
| **Repository** | 프로젝트 저장소 | `codeit_test` |
| **Commit** | 변경사항 저장 단위 | "Add profile" |
| **Branch** | 독립적인 작업 공간 | `feature/member1` |
| **Merge** | 브랜치 합치기 | feature → main |
| **Pull** | 원격 → 로컬 가져오기 | `git pull` |
| **Push** | 로컬 → 원격 보내기 | `git push` |
| **Clone** | 원격 저장소 복사 | `git clone` |
| **Fork** | 다른 사람 저장소 복사 | (GitHub에서) |

---

## 2. Branch 전략

### 왜 Branch를 사용하나?

**문제 상황:**
```
Member 1: index.html 수정 중...
Member 2: 같은 파일 수정 시작
Member 1: 먼저 push → ✅
Member 2: push 시도 → ❌ Conflict!
```

**해결책: Branch 분리**
```
main
  ├─ feature/member1 (Member 1 작업)
  └─ feature/member2 (Member 2 작업)
```

### Branch 구조

```
main (배포용)
  ↓
develop (통합)
  ↓
feature/member-name (개인 작업)
```

**우리 프로젝트 전략:**
- `main`: 완성된 코드만 (절대 직접 수정 금지!)
- `feature/작업내용`: 개인 작업용

### Branch 명령어

```bash
# 현재 브랜치 확인
git branch

# 새 브랜치 생성
git branch feature/member1-profile

# 브랜치 이동
git checkout feature/member1-profile

# 생성 + 이동 (한 번에)
git checkout -b feature/member1-profile

# 브랜치 목록 (원격 포함)
git branch -a

# 브랜치 삭제
git branch -d feature/member1-profile
```

### ⚠️ 주의사항

1. **절대 main에서 직접 작업하지 말 것!**
   ```bash
   # ❌ 나쁜 예
   git checkout main
   # 파일 수정
   git commit -m "수정"

   # ✅ 좋은 예
   git checkout -b feature/my-work
   # 파일 수정
   git commit -m "수정"
   ```

2. **작업 시작 전 항상 최신 상태로**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/new-work
   ```

---

## 3. Pull Request (PR) 워크플로우

### PR이란?

**"내 코드를 main에 합쳐주세요!"** 요청

### 전체 흐름

```
1. Branch 생성
   ↓
2. 코드 작성
   ↓
3. Commit & Push
   ↓
4. PR 생성 (GitHub)
   ↓
5. Code Review (팀원)
   ↓
6. 수정 반영 (필요시)
   ↓
7. Merge (승인 후)
```

### 실습: PR 생성하기

**Step 1: 작업 준비**
```bash
# 최신 main 가져오기
git checkout main
git pull origin main

# 새 브랜치 생성
git checkout -b feature/member1-profile
```

**Step 2: 작업하기**
```bash
# 파일 수정
# - index.html에 자신의 카드 추가
# - members/member1.html 생성

# 변경사항 확인
git status

# 파일 추가
git add index.html members/member1.html

# Commit (의미 있는 메시지!)
git commit -m "feat: Add member1 profile card with skills and bio"
```

**Step 3: Push**
```bash
git push origin feature/member1-profile
```

**Step 4: PR 생성 (GitHub 웹)**
1. GitHub에서 Repository 접속
2. "Pull requests" 탭
3. "New pull request" 클릭
4. Base: `main` ← Compare: `feature/member1-profile`
5. 제목 및 설명 작성
   ```
   Title: Add member1 profile

   Description:
   - Added profile card in index.html
   - Created detailed profile page
   - Added skills: Python, JavaScript, Git

   Closes #1 (이슈 번호가 있으면)
   ```
6. "Create pull request"

### PR 템플릿 (권장)

```markdown
## 변경 사항
- [ ] index.html에 프로필 카드 추가
- [ ] members/member1.html 생성
- [ ] 스킬 뱃지 추가

## 스크린샷
(변경 사항 이미지)

## 체크리스트
- [ ] 로컬에서 테스트 완료
- [ ] 충돌(conflict) 없음
- [ ] 커밋 메시지 규칙 준수
```

---

## 4. Code Review 가이드

### 왜 Code Review?

- 버그 조기 발견
- 지식 공유
- 코드 품질 향상
- 팀 소통 증진

### 좋은 리뷰 예시

**✅ 구체적이고 건설적인 리뷰:**
```
Line 45:
"여기서 class 이름을 'member-card'로 통일하면
CSS 재사용이 쉬울 것 같아요!"

Suggestion:
<div class="member-card">
```

**✅ 칭찬도 잊지 말기:**
```
"프로필 카드 디자인 정말 깔끔하네요!
색상 조합이 특히 마음에 듭니다"
```

**❌ 나쁜 리뷰:**
```
"이거 잘못됨"
"다시 해"
```

### 리뷰 남기기

GitHub PR 페이지에서:
1. "Files changed" 탭
2. 코드 라인 클릭
3. "+" 버튼 클릭
4. 코멘트 작성
5. "Add review comment" 또는 "Start a review"

### 리뷰 받은 후

**Option 1: 수정**
```bash
# 같은 브랜치에서 수정
git add .
git commit -m "Apply review feedback: Change class name"
git push origin feature/member1-profile
# → PR에 자동 반영!
```

**Option 2: 토론**
- 리뷰 코멘트에 답글로 의견 교환
- 합의 후 수정

---

## 5. Merge Conflict 해결

### Conflict가 발생하는 이유

**상황:**
```
Member 1: index.html 10번째 줄 수정 → merge ✅
Member 2: index.html 10번째 줄 수정 → merge 시도 → ❌ Conflict!
```

Git: "어느 게 맞는지 모르겠어요. 직접 선택해주세요!"

### Conflict 발생 시 파일 모습

```html
<<<<<<< HEAD (현재 브랜치)
<h1>Member 1의 코드</h1>
=======
<h1>Member 2의 코드</h1>
>>>>>>> feature/member2-profile
```

### 해결 방법

**Step 1: 최신 main 가져오기**
```bash
git checkout main
git pull origin main
```

**Step 2: 내 브랜치로 돌아가기**
```bash
git checkout feature/member2-profile
```

**Step 3: main 내용 merge**
```bash
git merge main
# → Conflict 발생!
```

**Step 4: 충돌 파일 수정**
```html
<!-- ❌ 이 상태에서 -->
<<<<<<< HEAD
<h1>Member 1의 코드</h1>
=======
<h1>Member 2의 코드</h1>
>>>>>>> feature/member2-profile

<!-- ✅ 이렇게 수정 (마커 삭제 + 원하는 코드 선택) -->
<h1>Member 1의 코드</h1>
<h1>Member 2의 코드</h1>
```

**Step 5: Conflict 해결 완료**
```bash
git add .
git commit -m "Resolve merge conflict in index.html"
git push origin feature/member2-profile
```

### VSCode에서 해결하기

VSCode는 conflict를 시각적으로 표시:
```
Accept Current Change  |  Accept Incoming Change  |  Accept Both
```

클릭 한 번으로 선택 가능!

### Conflict 예방법

1. **자주 Pull 받기**
   ```bash
   git checkout main
   git pull origin main
   git checkout feature/my-work
   git merge main  # 최신 상태 유지
   ```

2. **작업 영역 분리**
   - Member 1: index.html 상단
   - Member 2: index.html 중간
   - Member 3: index.html 하단

3. **빠르게 PR 완료**
   - 작은 단위로 자주 merge
   - 오래 묵히지 않기

---

## 6. Issue 기반 작업 흐름

### Issue란?

**작업 항목 추적 시스템**
- 버그 리포트
- 기능 요청
- 할 일 목록

### Issue 생성

GitHub에서:
1. "Issues" 탭
2. "New issue"
3. 제목과 설명 작성

**예시:**
```
Title: 프로필 카드에 GitHub 링크 추가

Description:
각 멤버의 GitHub 프로필로 이동할 수 있는 링크를
카드에 추가하면 좋을 것 같습니다.

참고:
- FontAwesome 아이콘 사용
- 새 탭에서 열리도록
```

### Issue → Branch → PR 연결

**1. Issue 생성 (#12)**

**2. Branch 생성**
```bash
git checkout -b feature/add-github-link-#12
```

**3. PR 생성 시 Issue 연결**
```markdown
Title: Add GitHub link to profile cards

Closes #12
```

→ PR merge 시 Issue 자동 close!

### Labels 활용

- `bug`: 버그 수정
- `enhancement`: 기능 개선
- `good first issue`: 초보자 환영
- `help wanted`: 도움 필요

---

## 7. 팀 협업 규칙

### ✅ DO (해야 할 것)

1. **작업 전 Issue 생성**
   - 무엇을 할지 명확히
   - 중복 작업 방지

2. **의미 있는 Commit 메시지**
   ```bash
   # ✅ Good
   git commit -m "feat: Add profile card with hover animation"

   # ❌ Bad
   git commit -m "update"
   git commit -m "ㅇㅇ"
   ```

3. **자주 Pull & Merge**
   - 하루 최소 1회
   - 작업 시작 전 필수

4. **PR은 작고 자주**
   - 한 번에 1개 기능만
   - 리뷰 받기 쉽게

5. **적극적인 리뷰 참여**
   - 최소 1명 이상 Approve 필요
   - 모든 팀원이 리뷰어

### ❌ DON'T (하지 말 것)

1. **main에 직접 Push 금지**
   ```bash
   # ❌ 절대 금지
   git checkout main
   git push origin main
   ```

2. **리뷰 없이 Merge 금지**
   - 최소 1명 Approve 필요

3. **다른 사람 브랜치에 강제 Push 금지**
   ```bash
   # ❌ 위험
   git push --force
   ```

4. **대용량 파일 Commit 금지**
   - 이미지는 최적화
   - 불필요한 파일 제외 (.gitignore)

### 커뮤니케이션 규칙

1. **PR 생성 시 팀원 멘션**
   ```
   @member2 @member3 리뷰 부탁드립니다!
   ```

2. **질문은 Issue/Discussion에**
   - 기록 남기기
   - 나중에 참고 가능

3. **긴급한 문제는 즉시 공유**
   - Issue로 먼저 생성
   - 팀 채팅으로 알림

---

## 다음 단계

1. [실전 워크플로우 예시](WORKFLOW_EXAMPLE.md) - 단계별 실습
2. [Git 공식 문서](https://git-scm.com/doc) - 심화 학습
3. [GitHub Skills](https://skills.github.com/) - 인터랙티브 튜토리얼

---

## 도움이 필요하면?

- **GitHub Issues**: 질문 남기기
- **팀 채팅**: 즉시 도움 요청
- **이 문서**: 언제든 다시 참고!

**Good luck!**

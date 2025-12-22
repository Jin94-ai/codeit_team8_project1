# Health Eat - Expert Presentation

> Black & Blue 테마의 인터랙티브 웹 프레젠테이션

---

## 실행 방법

### 방법 1: 직접 열기
`index.html` 파일을 브라우저에서 직접 열기

### 방법 2: 로컬 서버 (권장)
```bash
# Python 3
cd docs/presentation
python -m http.server 8000

# 브라우저에서 접속
http://localhost:8000
```

### 방법 3: VS Code Live Server
1. VS Code에서 `index.html` 열기
2. 우클릭 → "Open with Live Server"

---

## 조작 방법

| 키 | 기능 |
|---|------|
| `→` / `Space` | 다음 슬라이드 |
| `←` | 이전 슬라이드 |
| `ESC` | 전체 보기 |
| `F` | 전체 화면 |
| `S` | 발표자 노트 |
| `O` | 개요 보기 |

---

## 슬라이드 구성 (12장)

1. **Title** - Health Eat 소개
2. **Executive Summary** - 핵심 지표 & 점수 그래프
3. **Problem Statement** - 문제 정의 & 데이터 분석
4. **2-Stage Architecture** - 파이프라인 구조
5. **Detector Details** - YOLO11m 설정 & 성능
6. **Data Refinement** - 데이터 정제 전략
7. **Experiment Journey** - 실험 타임라인
8. **Failed Experiments** - 실패 사례 & 교훈
9. **Best Model Config** - 최종 설정
10. **Team Contributions** - 팀 기여도
11. **Conclusion** - 결론
12. **Q&A** - 질의응답

---

## 기술 스택

- **Reveal.js 4.5** - 프레젠테이션 프레임워크
- **Chart.js** - 인터랙티브 차트
- **Google Fonts** - Noto Sans KR, JetBrains Mono

---

## 커스터마이징

### 색상 변경
```css
:root {
    --primary-blue: #0066FF;
    --secondary-blue: #00A3FF;
    --accent-blue: #00D4FF;
    --dark-bg: #0a0a0f;
}
```

### 슬라이드 추가
```html
<section>
    <h2>새 슬라이드</h2>
    <p>내용...</p>
</section>
```

---

## 파일 구조

```
docs/presentation/
├── index.html      # 메인 프레젠테이션
└── README.md       # 이 파일
```

---

*Generated: 2025-12-22*

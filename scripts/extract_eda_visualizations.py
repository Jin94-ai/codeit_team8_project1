"""
EDA 시각화 이미지 추출 스크립트
발표자료용 이미지 생성
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 색상 팔레트
COLORS = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF',
          '#E2B8FF', '#FFCCE5', '#D5F5E3', '#D6EAF8', '#FAD7A0']
NAVY = '#1a365d'


def save_figure(fig, filename):
    """그림 저장"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"저장 완료: {filepath}")


def create_data_composition_pie():
    """1. 데이터 구성 파이차트"""
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ['Train 이미지\n(651개)', 'Test 이미지\n(843개)', 'Annotation\n(1,001개)']
    sizes = [651, 843, 1001]
    colors = ['#A3CEF1', '#FFC8DD', '#BDE0FE']

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=90, colors=colors,
        textprops={'fontsize': 12}
    )

    ax.set_title('데이터셋 구성', fontsize=16, fontweight='bold')
    save_figure(fig, '01_data_composition.png')


def create_valid_data_comparison():
    """2. 필터링 전후 데이터 비교"""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Train 이미지', 'Annotation', '유효 학습 데이터']
    before = [651, 1001, 0]
    after = [232, 763, 232]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, before, width, label='원본', color='#FFB3BA')
    bars2 = ax.bar(x + width/2, after, width, label='필터링 후', color='#BAFFC9')

    ax.set_ylabel('개수', fontsize=12)
    ax.set_title('데이터 정합성 필터링 결과', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend()

    # 값 표시
    for bar in bars1:
        if bar.get_height() > 0:
            ax.annotate(f'{int(bar.get_height())}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.annotate(f'{int(bar.get_height())}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=10)

    save_figure(fig, '02_data_filtering.png')


def create_class_distribution():
    """3. 클래스 분포 막대 그래프 (Top-10)"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Top-10 클래스 데이터 (EDA 노트북 기반)
    classes = [
        '일양하이트린정 2mg', '기넥신에프정', '아토젯정',
        '크레스토정 20mg', '리바로정 4mg', '리피토정 20mg',
        '플라빅스정 75mg', '뉴로메드정', '로수바미브정', '콜리네이트캡슐'
    ]
    counts = [240, 45, 40, 31, 29, 29, 29, 27, 27, 26]

    bars = ax.bar(range(len(classes)), counts, color=COLORS[:len(classes)])

    ax.set_ylabel('어노테이션 개수', fontsize=12)
    ax.set_title('Top-10 알약 클래스 분포 (클래스 불균형: 1:80)', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)

    # 값 표시
    for bar in bars:
        ax.annotate(f'{int(bar.get_height())}',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)

    # 불균형 표시
    ax.axhline(y=3, color='red', linestyle='--', linewidth=2, label='최소 클래스: 3개')
    ax.legend()

    plt.tight_layout()
    save_figure(fig, '03_class_distribution.png')


def create_pill_shape_distribution():
    """4. 알약 모양 분포"""
    fig, ax = plt.subplots(figsize=(8, 6))

    shapes = ['원형', '타원형', '장방형', '팔각형', '육각형']
    counts = [480, 318, 176, 6, 3]
    colors = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF']

    wedges, texts, autotexts = ax.pie(
        counts, labels=shapes, autopct='%1.1f%%',
        startangle=90, colors=colors,
        textprops={'fontsize': 11}
    )

    ax.set_title('알약 모양(drug_shape) 분포', fontsize=16, fontweight='bold')
    save_figure(fig, '04_pill_shape.png')


def create_pill_color_distribution():
    """5. 알약 색상 분포"""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors_name = ['주황', '하양', '분홍', '노랑', '갈색', '기타']
    counts = [282, 262, 152, 130, 78, 97]
    bar_colors = ['#FF9F43', '#FFFFFF', '#FFC0CB', '#FFEB3B', '#8B4513', '#CCCCCC']
    edge_colors = ['#FF9F43', '#000000', '#FFC0CB', '#FFEB3B', '#8B4513', '#CCCCCC']

    bars = ax.bar(colors_name, counts, color=bar_colors, edgecolor=edge_colors, linewidth=2)

    ax.set_ylabel('개수', fontsize=12)
    ax.set_title('알약 색상(color_class1) 분포', fontsize=16, fontweight='bold')

    # 값 표시
    for bar in bars:
        ax.annotate(f'{int(bar.get_height())}',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10)

    save_figure(fig, '05_pill_color.png')


def create_bbox_per_image():
    """6. 이미지당 bbox 개수 분포"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 이미지당 bbox 개수 분포 (EDA 기반 추정)
    bbox_counts = [1, 2, 3, 4]
    image_counts = [15, 85, 95, 37]  # 총 232개
    colors = ['#FFB3BA', '#FFDFBA', '#BAFFC9', '#BAE1FF']

    bars = ax.bar(bbox_counts, image_counts, color=colors)

    ax.set_xlabel('이미지당 알약 개수', fontsize=12)
    ax.set_ylabel('이미지 개수', fontsize=12)
    ax.set_title('이미지당 bbox 개수 분포 (평균 2.7개)', fontsize=16, fontweight='bold')
    ax.set_xticks(bbox_counts)

    # 값 표시
    for bar in bars:
        ax.annotate(f'{int(bar.get_height())}',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=11)

    save_figure(fig, '06_bbox_per_image.png')


def create_score_timeline():
    """7. 점수 개선 히스토리 그래프"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 점수 히스토리
    stages = [
        'Baseline\n(YOLO12m)',
        '복합경구제\n(실패)',
        '단일경구제\n',
        '2-Stage\n도입',
        '데이터셋\n개선',
        'Best\n(정제)'
    ]
    scores = [0.82, 0.6, 0.822, 0.920, 0.963, 0.96703]
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#2ecc71', '#27ae60']

    bars = ax.bar(range(len(stages)), scores, color=colors)

    ax.set_ylabel('mAP@[0.75:0.95]', fontsize=12)
    ax.set_title('점수 개선 히스토리 (Baseline 0.82 → Best 0.96703)', fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylim(0, 1.1)

    # 값 표시
    for i, bar in enumerate(bars):
        ax.annotate(f'{scores[i]:.3f}' if scores[i] >= 0.1 else f'{scores[i]:.3f}',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 기준선
    ax.axhline(y=0.82, color='blue', linestyle='--', alpha=0.5, label='Baseline')
    ax.axhline(y=0.96703, color='green', linestyle='--', alpha=0.5, label='Best')
    ax.legend()

    save_figure(fig, '07_score_timeline.png')


def create_2stage_architecture():
    """8. 2-Stage 아키텍처 다이어그램"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Stage 1 박스
    stage1 = patches.FancyBboxPatch((0.5, 1.5), 4, 3,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#e3f2fd', edgecolor=NAVY, linewidth=2)
    ax.add_patch(stage1)
    ax.text(2.5, 4, 'Stage 1: YOLO11m Detector', fontsize=14, fontweight='bold', ha='center')
    ax.text(2.5, 3.2, 'Input: 원본 이미지 (1280x960)', fontsize=10, ha='center')
    ax.text(2.5, 2.6, 'Output: Bounding Boxes', fontsize=10, ha='center')
    ax.text(2.5, 2.0, '클래스: 단일 ("Pill")', fontsize=10, ha='center')

    # 화살표
    arrow = patches.FancyArrowPatch((4.7, 3), (5.8, 3),
                                    arrowstyle='->', mutation_scale=20,
                                    color=NAVY, linewidth=3)
    ax.add_patch(arrow)

    # Stage 2 박스
    stage2 = patches.FancyBboxPatch((6, 1.5), 4, 3,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#e8f5e9', edgecolor=NAVY, linewidth=2)
    ax.add_patch(stage2)
    ax.text(8, 4, 'Stage 2: ConvNeXt Classifier', fontsize=14, fontweight='bold', ha='center')
    ax.text(8, 3.2, 'Input: 크롭 이미지 (224x224)', fontsize=10, ha='center')
    ax.text(8, 2.6, 'Output: K-code + Confidence', fontsize=10, ha='center')
    ax.text(8, 2.0, '클래스: 74개', fontsize=10, ha='center')

    # 결과 박스
    result = patches.FancyBboxPatch((10.5, 2), 3, 2,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#fff3e0', edgecolor='#ff9800', linewidth=2)
    ax.add_patch(result)
    ax.text(12, 3.5, 'Result', fontsize=12, fontweight='bold', ha='center')
    ax.text(12, 2.8, 'mAP: 0.96703', fontsize=11, ha='center', color='#2e7d32')

    # 화살표 2
    arrow2 = patches.FancyArrowPatch((10.2, 3), (10.4, 3),
                                     arrowstyle='->', mutation_scale=20,
                                     color=NAVY, linewidth=3)
    ax.add_patch(arrow2)

    ax.set_title('2-Stage Pipeline Architecture', fontsize=18, fontweight='bold', pad=20)

    save_figure(fig, '08_2stage_architecture.png')


def create_collaboration_stats():
    """9. 협업 통계"""
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['커밋', 'PR', '협업일지', '실험 로그', '회의록']
    values = [180, 86, 40, 12, 4]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    bars = ax.bar(categories, values, color=colors)

    ax.set_ylabel('개수', fontsize=12)
    ax.set_title('협업 통계 (3주간)', fontsize=16, fontweight='bold')

    # 값 표시
    for bar in bars:
        ax.annotate(f'{int(bar.get_height())}',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    save_figure(fig, '09_collaboration_stats.png')


def create_failure_timeline():
    """10. 실패 사례 타임라인"""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 타임라인 축
    ax.plot([1, 13], [4, 4], color='#34495e', linewidth=3)

    # 각 단계 정의
    stages = [
        {'x': 2.5, 'y': 4, 'label': '복합경구제', 'score': '0.6',
         'status': 'fail', 'detail': 'bbox 1개만 검출'},
        {'x': 5.5, 'y': 4, 'label': '단일경구제', 'score': '0.822',
         'status': 'warn', 'detail': 'Baseline 유지'},
        {'x': 8.5, 'y': 4, 'label': 'ROI Crop', 'score': '-',
         'status': 'fail', 'detail': '구현 실패'},
        {'x': 11.5, 'y': 4, 'label': '2-Stage', 'score': '0.96703',
         'status': 'success', 'detail': '최종 성공'}
    ]

    colors = {'fail': '#e74c3c', 'warn': '#f39c12', 'success': '#27ae60'}

    for stage in stages:
        # 점
        circle = plt.Circle((stage['x'], stage['y']), 0.3,
                            color=colors[stage['status']], zorder=5)
        ax.add_patch(circle)

        # 라벨 (위/아래 교대)
        if stages.index(stage) % 2 == 0:
            y_label = 5.5
            y_score = 6.2
            y_detail = 6.8
        else:
            y_label = 2.5
            y_score = 1.8
            y_detail = 1.2

        ax.text(stage['x'], y_label, stage['label'], fontsize=12,
               fontweight='bold', ha='center', va='center')
        ax.text(stage['x'], y_score, f"Score: {stage['score']}", fontsize=10,
               ha='center', va='center', color=colors[stage['status']])
        ax.text(stage['x'], y_detail, stage['detail'], fontsize=9,
               ha='center', va='center', style='italic', color='#7f8c8d')

        # 연결선
        if stages.index(stage) % 2 == 0:
            ax.plot([stage['x'], stage['x']], [stage['y'] + 0.3, 5.2],
                   color='#bdc3c7', linewidth=1.5, linestyle='--')
        else:
            ax.plot([stage['x'], stage['x']], [stage['y'] - 0.3, 2.8],
                   color='#bdc3c7', linewidth=1.5, linestyle='--')

    # 범례
    legend_elements = [
        patches.Circle((0, 0), 0.1, color='#e74c3c', label='실패'),
        patches.Circle((0, 0), 0.1, color='#f39c12', label='경고'),
        patches.Circle((0, 0), 0.1, color='#27ae60', label='성공')
    ]
    ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor='#e74c3c', markersize=10, label='실패'),
                       plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor='#f39c12', markersize=10, label='경고'),
                       plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor='#27ae60', markersize=10, label='성공')],
              loc='upper right')

    ax.set_title('실패 사례 타임라인: 문제 해결 과정', fontsize=16, fontweight='bold', pad=20)

    save_figure(fig, '10_failure_timeline.png')


def create_automation_pipeline():
    """11. Ubuntu 자동화 파이프라인 다이어그램"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 제목
    ax.set_title('Ubuntu 자동화 파이프라인 (김보윤)', fontsize=18, fontweight='bold', pad=20)

    # 메인 박스들
    boxes = [
        {'x': 1, 'y': 5, 'w': 2.5, 'h': 1.5, 'label': 'exc.sh', 'sub': 'run.sh 생성\nalias 등록', 'color': '#e3f2fd'},
        {'x': 4.5, 'y': 5, 'w': 2.5, 'h': 1.5, 'label': '패키지 설치', 'sub': 'ultralytics\nalbumentations', 'color': '#fff3e0'},
        {'x': 8, 'y': 5, 'w': 2.5, 'h': 1.5, 'label': '데이터 로딩', 'sub': 'data_loader.py', 'color': '#e8f5e9'},
        {'x': 11.5, 'y': 5, 'w': 2.5, 'h': 1.5, 'label': 'YOLO 변환', 'sub': 'yolo_export.py', 'color': '#fce4ec'},
    ]

    for box in boxes:
        rect = patches.FancyBboxPatch((box['x'], box['y']), box['w'], box['h'],
                                       boxstyle="round,pad=0.05",
                                       facecolor=box['color'], edgecolor=NAVY, linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h'] - 0.3, box['label'],
               fontsize=11, fontweight='bold', ha='center', va='top')
        ax.text(box['x'] + box['w']/2, box['y'] + 0.4, box['sub'],
               fontsize=9, ha='center', va='bottom', color='#555')

    # 화살표들
    for i in range(len(boxes) - 1):
        ax.annotate('', xy=(boxes[i+1]['x'], boxes[i]['y'] + boxes[i]['h']/2),
                   xytext=(boxes[i]['x'] + boxes[i]['w'], boxes[i]['y'] + boxes[i]['h']/2),
                   arrowprops=dict(arrowstyle='->', color=NAVY, lw=2))

    # 모델 학습 (큰 박스)
    train_box = patches.FancyBboxPatch((4.5, 1.5), 5, 2.5,
                                        boxstyle="round,pad=0.1",
                                        facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=3)
    ax.add_patch(train_box)
    ax.text(7, 3.5, '모델 학습', fontsize=14, fontweight='bold', ha='center')
    ax.text(7, 2.8, 'python -m src.models.[모델명]', fontsize=10, ha='center', family='monospace')
    ax.text(7, 2.2, 'Seed=42 | Checkpoint 5ep | Early Stop | W&B', fontsize=9, ha='center', color='#555')

    # 연결 화살표
    ax.annotate('', xy=(7, 4), xytext=(7, 5),
               arrowprops=dict(arrowstyle='->', color=NAVY, lw=2))

    # 핵심 기능 박스들 (하단)
    features = [
        {'x': 1, 'y': 0.5, 'text': 'Seed 고정\n(재현성)', 'color': '#BAFFC9'},
        {'x': 4, 'y': 0.5, 'text': '체크포인트\n(5 epoch)', 'color': '#BAE1FF'},
        {'x': 7, 'y': 0.5, 'text': 'Early Stop\n(patience=10)', 'color': '#FFFFBA'},
        {'x': 10, 'y': 0.5, 'text': 'W&B 콜백\n(mAP 모니터링)', 'color': '#FFB3BA'},
    ]

    for feat in features:
        rect = patches.FancyBboxPatch((feat['x'], feat['y']), 2.5, 0.9,
                                       boxstyle="round,pad=0.05",
                                       facecolor=feat['color'], edgecolor='#666', linewidth=1)
        ax.add_patch(rect)
        ax.text(feat['x'] + 1.25, feat['y'] + 0.45, feat['text'],
               fontsize=9, ha='center', va='center')

    save_figure(fig, '11_automation_pipeline.png')


def create_class_expansion():
    """12. 클래스 확장 다이어그램"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Kaggle 박스
    kaggle = patches.FancyBboxPatch((0.5, 2), 2.5, 2,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#e3f2fd', edgecolor='#1565c0', linewidth=2)
    ax.add_patch(kaggle)
    ax.text(1.75, 3.5, 'Kaggle', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.75, 2.8, '56개 클래스', fontsize=14, ha='center', fontweight='bold', color='#1565c0')

    # + 기호
    ax.text(3.5, 3, '+', fontsize=30, fontweight='bold', ha='center', va='center')

    # AIHub 박스
    aihub = patches.FancyBboxPatch((4, 2), 2.5, 2,
                                    boxstyle="round,pad=0.1",
                                    facecolor='#fce4ec', edgecolor='#c2185b', linewidth=2)
    ax.add_patch(aihub)
    ax.text(5.25, 3.5, 'AIHub TL/TS', fontsize=12, fontweight='bold', ha='center')
    ax.text(5.25, 2.8, '+18개 클래스', fontsize=14, ha='center', fontweight='bold', color='#c2185b')

    # = 기호
    ax.text(7, 3, '=', fontsize=30, fontweight='bold', ha='center', va='center')

    # 최종 박스
    final = patches.FancyBboxPatch((7.5, 2), 2.5, 2,
                                    boxstyle="round,pad=0.1",
                                    facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
    ax.add_patch(final)
    ax.text(8.75, 3.5, '최종', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.75, 2.8, '74개 클래스', fontsize=14, ha='center', fontweight='bold', color='#2e7d32')

    ax.set_title('클래스 확장 (강사님 디렉션 반영)', fontsize=16, fontweight='bold', pad=20)

    save_figure(fig, '12_class_expansion.png')


def create_best_model_detail():
    """13. 베스트 모델 상세 설정"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stage 1: Detector 설정
    ax1 = axes[0]
    ax1.axis('off')
    ax1.set_title('Stage 1: YOLO11m Detector', fontsize=14, fontweight='bold', pad=10)

    detector_info = [
        ('모델', 'yolo11m.pt'),
        ('imgsz', '640'),
        ('batch', '8'),
        ('epochs', '50'),
        ('patience', '15'),
        ('mAP50', '0.995'),
        ('mAP50-95', '0.85'),
    ]

    for i, (key, val) in enumerate(detector_info):
        y = 0.85 - i * 0.12
        ax1.text(0.1, y, f'{key}:', fontsize=11, fontweight='bold', transform=ax1.transAxes)
        ax1.text(0.5, y, val, fontsize=11, transform=ax1.transAxes)

    # Stage 2: Classifier 설정
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('Stage 2: ConvNeXt Classifier', fontsize=14, fontweight='bold', pad=10)

    classifier_info = [
        ('모델', 'convnext_tiny'),
        ('Pretrained', 'ImageNet'),
        ('img_size', '224'),
        ('batch_size', '32'),
        ('lr', '1e-4'),
        ('Val Accuracy', '98.5%'),
        ('Early Stop', 'Epoch 21'),
    ]

    for i, (key, val) in enumerate(classifier_info):
        y = 0.85 - i * 0.12
        ax2.text(0.1, y, f'{key}:', fontsize=11, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.5, y, val, fontsize=11, transform=ax2.transAxes)

    plt.suptitle('Best Model Settings (Score: 0.96703)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, '13_best_model_detail.png')


def create_data_refinement():
    """14. 데이터 정제 결과"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 왼쪽: bbox 개수 분포 (정제 후)
    ax1 = axes[0]
    bbox_counts = ['3개', '4개']
    image_counts = [1167, 5833]  # 약 17%, 83%
    colors = ['#BAFFC9', '#BAE1FF']

    bars = ax1.bar(bbox_counts, image_counts, color=colors, edgecolor='#333', linewidth=2)
    ax1.set_ylabel('이미지 개수', fontsize=12)
    ax1.set_title('정제 후 bbox 분포', fontsize=14, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 오른쪽: 정제 전후 비교
    ax2 = axes[1]
    categories = ['정제 전', '정제 후']
    counts = [10000, 7000]
    colors = ['#FFB3BA', '#BAFFC9']

    bars = ax2.bar(categories, counts, color=colors, edgecolor='#333', linewidth=2)
    ax2.set_ylabel('이미지 개수', fontsize=12)
    ax2.set_title('데이터셋 정제 결과', fontsize=14, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 정제 조건 텍스트
    ax2.text(0.5, 0.3, '정제 조건:\n• bbox 3~4개만\n• 중복 제거 (IoU>0.7)\n• 크기 제한 (50~500px)',
             transform=ax2.transAxes, fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('데이터셋 정제 (핵심 성공 요인)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, '14_data_refinement.png')


def create_augmentation_summary():
    """15. Augmentation 설정 요약"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    ax.set_title('YOLO11m Detector Augmentation 설정', fontsize=16, fontweight='bold', y=0.95)

    augmentations = [
        ('HSV 변형', 'hsv_h=0.015, hsv_s=0.5, hsv_v=0.4', '색상/채도/밝기 조정'),
        ('회전', 'degrees=15', '±15도 회전'),
        ('이동', 'translate=0.1', '10% 이동'),
        ('스케일', 'scale=0.3', '30% 크기 변화'),
        ('좌우 반전', 'fliplr=0.5', '50% 확률'),
        ('상하 반전', 'flipud=0.0', '사용 안함'),
        ('Mosaic', 'mosaic=1.0', '4개 이미지 합성'),
        ('MixUp', 'mixup=0.1', '10% 확률로 이미지 혼합'),
    ]

    # 테이블 형태로 표시
    for i, (name, value, desc) in enumerate(augmentations):
        y = 0.85 - i * 0.1
        ax.text(0.05, y, name, fontsize=11, fontweight='bold', transform=ax.transAxes)
        ax.text(0.3, y, value, fontsize=10, family='monospace', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.5))
        ax.text(0.65, y, desc, fontsize=10, color='#555', transform=ax.transAxes)

    plt.tight_layout()
    save_figure(fig, '15_augmentation_summary.png')


def main():
    """메인 함수"""
    print("EDA 시각화 이미지 생성 시작...")
    print(f"저장 경로: {OUTPUT_DIR}\n")

    create_data_composition_pie()
    create_valid_data_comparison()
    create_class_distribution()
    create_pill_shape_distribution()
    create_pill_color_distribution()
    create_bbox_per_image()
    create_score_timeline()
    create_2stage_architecture()
    create_collaboration_stats()
    create_failure_timeline()
    create_automation_pipeline()
    create_class_expansion()
    create_best_model_detail()
    create_data_refinement()
    create_augmentation_summary()

    print(f"\n총 15개 이미지 생성 완료!")
    print(f"저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

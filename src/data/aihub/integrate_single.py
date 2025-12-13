"""
AIHub Single 데이터를 Competition 데이터에 통합

사용법:
1. single 데이터를 data/aihub_single/ 폴더에 다운로드
2. 이 스크립트 실행
3. train_images, train_annotations에 통합됨

Single 데이터 구조 (예상):
data/aihub_single/
├── TS1_single/
│   ├── images/
│   │   └── *.png
│   └── annotations/
│       └── *.json
├── TS2_single/
│   └── ...
└── ...
"""
import json
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# 직접 실행 시 import 경로 추가
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data.aihub.config import TARGET_CLASSES, AIHUB_SINGLE_DIR, TRAIN_IMG_DIR, TRAIN_ANN_DIR, MAX_PER_CLASS


def analyze_competition():
    """Competition 데이터의 클래스별 샘플 수 분석"""
    print("\n[1/4] Competition 데이터 분석")

    comp_anno = Path(TRAIN_ANN_DIR)
    counts = defaultdict(int)

    if comp_anno.exists():
        for json_path in comp_anno.rglob("*.json"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'categories' in data and data['categories']:
                    cat_id = str(data['categories'][0]['id'])
                    if cat_id in TARGET_CLASSES:
                        counts[cat_id] += 1
            except:
                continue

    print(f"  Competition 클래스 수: {len(counts)}")
    print(f"  총 샘플 수: {sum(counts.values())}")

    # 부족한 클래스 출력
    insufficient = [(k, v) for k, v in counts.items() if v < MAX_PER_CLASS]
    if insufficient:
        print(f"  샘플 부족 클래스 ({MAX_PER_CLASS}개 미만): {len(insufficient)}개")

    return counts


def find_single_data():
    """다운로드된 single 데이터 탐색"""
    print("\n[2/4] Single 데이터 탐색")

    single_dir = Path(AIHUB_SINGLE_DIR)
    if not single_dir.exists():
        print(f"  ❌ {AIHUB_SINGLE_DIR} 폴더 없음")
        print(f"  → single 데이터를 해당 폴더에 다운로드하세요")
        return []

    # TS 폴더들 찾기
    ts_folders = list(single_dir.glob("TS*_single")) + list(single_dir.glob("TS*"))
    print(f"  발견된 TS 폴더: {len(ts_folders)}개")

    return ts_folders


def integrate_single(comp_counts: dict, ts_folders: list):
    """Single 데이터를 train 폴더에 통합"""
    print("\n[3/4] Single 데이터 통합")

    train_img = Path(TRAIN_IMG_DIR)
    train_anno = Path(TRAIN_ANN_DIR)

    train_img.mkdir(parents=True, exist_ok=True)
    train_anno.mkdir(parents=True, exist_ok=True)

    # 고유 ID 시작점 (Competition과 충돌 방지)
    next_image_id = 200000
    next_anno_id = 2000000

    added_counts = defaultdict(int)
    total_added = 0
    skipped_full = 0
    skipped_no_bbox = 0

    for ts_folder in ts_folders:
        # 이미지와 어노테이션 폴더 찾기
        img_folders = list(ts_folder.glob("**/images")) + list(ts_folder.glob("**/image"))
        anno_folders = list(ts_folder.glob("**/annotations")) + list(ts_folder.glob("**/annotation"))

        if not img_folders or not anno_folders:
            continue

        img_folder = img_folders[0]
        anno_folder = anno_folders[0]

        for json_path in anno_folder.glob("*.json"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'images' not in data or not data['images']:
                    continue

                img_info = data['images'][0]
                dl_idx = str(img_info.get('dl_idx', ''))

                # TARGET 클래스만 처리
                if dl_idx not in TARGET_CLASSES:
                    continue

                # 클래스당 MAX_PER_CLASS 제한
                current = comp_counts.get(dl_idx, 0) + added_counts.get(dl_idx, 0)
                if current >= MAX_PER_CLASS:
                    skipped_full += 1
                    continue

                # bbox 확인
                annotations = data.get('annotations', [])
                if not annotations or not annotations[0].get('bbox'):
                    skipped_no_bbox += 1
                    continue

                bbox = annotations[0]['bbox']
                if not isinstance(bbox, list) or len(bbox) != 4:
                    skipped_no_bbox += 1
                    continue

                # 이미지 파일 찾기
                file_name = img_info.get('file_name', '')
                src_img = img_folder / file_name

                if not src_img.exists():
                    # 다른 확장자 시도
                    for ext in ['.png', '.jpg', '.jpeg']:
                        alt_path = img_folder / (Path(file_name).stem + ext)
                        if alt_path.exists():
                            src_img = alt_path
                            file_name = alt_path.name
                            break

                if not src_img.exists():
                    continue

                # 이미지 복사
                dst_img = train_img / file_name
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)

                # JSON 수정 및 저장
                cat_id = int(dl_idx)
                dl_name = img_info.get('dl_name', 'Drug')

                new_data = {
                    'images': [{
                        **img_info,
                        'id': next_image_id,
                        'file_name': file_name
                    }],
                    'type': 'instances',
                    'annotations': [{
                        **annotations[0],
                        'id': next_anno_id,
                        'image_id': next_image_id,
                        'category_id': cat_id
                    }],
                    'categories': [{
                        'supercategory': 'pill',
                        'id': cat_id,
                        'name': dl_name
                    }]
                }

                # 저장
                out_name = f"single_{dl_idx}_{added_counts[dl_idx]:04d}.json"
                with open(train_anno / out_name, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)

                added_counts[dl_idx] += 1
                total_added += 1
                next_image_id += 1
                next_anno_id += 1

                if total_added % 500 == 0:
                    print(f"  처리 중: {total_added}개...")

            except Exception as e:
                continue

    print(f"  추가된 샘플: {total_added}개")
    print(f"  클래스 수: {len(added_counts)}개")
    if skipped_full:
        print(f"  건너뜀 (클래스 가득): {skipped_full}개")
    if skipped_no_bbox:
        print(f"  건너뜀 (bbox 없음): {skipped_no_bbox}개")

    return added_counts


def print_summary(comp_counts: dict, added_counts: dict):
    """결과 요약 출력"""
    print("\n[4/4] 결과 요약")
    print("=" * 70)

    total_comp = sum(comp_counts.values())
    total_added = sum(added_counts.values())

    print(f"Competition 데이터: {total_comp}개")
    print(f"Single 추가: {total_added}개")
    print(f"총 데이터: {total_comp + total_added}개")

    # 클래스별 분포 확인
    all_classes = set(comp_counts.keys()) | set(added_counts.keys())
    insufficient = []

    for cls in sorted(all_classes, key=int):
        total = comp_counts.get(cls, 0) + added_counts.get(cls, 0)
        if total < 10:  # 10개 미만인 클래스
            insufficient.append((cls, total))

    if insufficient:
        print(f"\n샘플 부족 클래스 (10개 미만): {len(insufficient)}개")
        for cls, cnt in insufficient[:10]:
            print(f"  {cls}: {cnt}개")

    print("=" * 70)
    print("\n다음 단계: YOLO 데이터 생성")
    print("  python -m src.data.yolo_dataset.yolo_export")


def main():
    """메인 실행"""
    print("=" * 70)
    print("AIHub Single 데이터 → Competition 데이터 통합")
    print("=" * 70)

    # 1. Competition 데이터 분석
    comp_counts = analyze_competition()

    # 2. Single 데이터 탐색
    ts_folders = find_single_data()

    if not ts_folders:
        return

    # 3. 통합 실행
    added_counts = integrate_single(comp_counts, ts_folders)

    # 4. 결과 요약
    print_summary(comp_counts, added_counts)


if __name__ == "__main__":
    main()

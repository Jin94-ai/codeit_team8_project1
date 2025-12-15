"""
AIHub 단일경구약제 라벨링 데이터(ZIP) 분석

사용법:
    python -m src.data.aihub.analyze_annotations

구조:
data/166.약품식별.../01.데이터/1.Training/라벨링데이터/단일경구약제 5000종/
├── TL_1_단일.zip
├── TL_2_단일.zip
└── ... (81개 ZIP)

출력:
- TL 폴더별 TARGET_CLASSES 포함 현황
- 다운로드 추천 목록 (이미지 폴더)
"""
import json
import sys
import zipfile
from pathlib import Path
from collections import defaultdict

# 직접 실행 시 import 경로 추가
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data.aihub.config import TARGET_CLASSES, dl_idx_to_k_code


# 라벨링 데이터 경로
LABEL_DIR = Path("data/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/1.Training/라벨링데이터/단일경구약제 5000종")


def find_zip_files(base_dir: Path) -> list:
    """TL_*.zip 파일 목록 찾기"""
    zip_files = list(base_dir.glob("TL_*_단일.zip"))
    return sorted(zip_files, key=lambda x: int(x.name.split('_')[1]))


def analyze_zip_file(zip_path: Path) -> dict:
    """
    단일 ZIP 파일 내 annotation 분석 (압축 해제 없이)

    Returns:
        {
            'name': 'TL_1_단일',
            'total_files': 1000,
            'target_classes': {'1899': 50, '2482': 30, ...},
            'non_target_count': 500
        }
    """
    result = {
        'name': zip_path.stem,
        'total_files': 0,
        'target_classes': defaultdict(int),
        'non_target_count': 0
    }

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            json_files = [f for f in zf.namelist() if f.endswith('.json')]

            for json_file in json_files:
                result['total_files'] += 1

                try:
                    with zf.open(json_file) as f:
                        data = json.load(f)

                    # dl_idx 추출 (여러 구조 지원)
                    dl_idx = None

                    # 구조 1: images[0].dl_idx
                    if 'images' in data and data['images']:
                        dl_idx = str(data['images'][0].get('dl_idx', ''))

                    # 구조 2: categories[0].id
                    if not dl_idx and 'categories' in data and data['categories']:
                        dl_idx = str(data['categories'][0].get('id', ''))

                    # 구조 3: annotations[0].category_id
                    if not dl_idx and 'annotations' in data and data['annotations']:
                        dl_idx = str(data['annotations'][0].get('category_id', ''))

                    if dl_idx and dl_idx in TARGET_CLASSES:
                        result['target_classes'][dl_idx] += 1
                    elif dl_idx:
                        result['non_target_count'] += 1

                except Exception:
                    continue

    except zipfile.BadZipFile:
        print(f"\n  경고: {zip_path.name} - 손상된 ZIP 파일")

    return result


def print_analysis_report(tl_results: list):
    """분석 결과 리포트 출력"""
    print("\n" + "=" * 70)
    print("AIHub 단일경구약제 라벨링 분석 결과")
    print("=" * 70)

    # TARGET_CLASSES가 있는 TL 폴더만 필터링
    target_tl = [r for r in tl_results if r['target_classes']]

    if not target_tl:
        print("\n❌ TARGET_CLASSES를 포함한 TL 파일이 없습니다.")
        return [], set()

    # 결과 정렬 (TARGET 클래스 수 기준)
    target_tl.sort(key=lambda x: len(x['target_classes']), reverse=True)

    print(f"\n총 {len(tl_results)}개 TL 중 {len(target_tl)}개에서 TARGET 클래스 발견\n")

    # TL별 상세 정보
    all_found_classes = set()
    recommended = []

    for tl in target_tl:
        class_count = len(tl['target_classes'])
        sample_count = sum(tl['target_classes'].values())
        all_found_classes.update(tl['target_classes'].keys())

        # TL 번호 추출 (예: TL_1_단일 -> 1)
        tl_num = tl['name'].split('_')[1]

        print(f"📁 {tl['name']}")
        print(f"   TARGET 클래스: {class_count}개, 샘플: {sample_count}개")

        # 상위 5개 클래스 표시
        top_classes = sorted(tl['target_classes'].items(), key=lambda x: -x[1])[:5]
        class_str = ", ".join([f"{c}({n})" for c, n in top_classes])
        print(f"   주요 클래스: {class_str}")

        if tl['non_target_count'] > 0:
            print(f"   (non-target: {tl['non_target_count']}개)")
        print()

        recommended.append(tl_num)

    # 요약
    print("=" * 70)
    print("요약")
    print("=" * 70)
    print(f"발견된 TARGET 클래스: {len(all_found_classes)}/56개")

    missing = TARGET_CLASSES - all_found_classes
    if missing:
        print(f"\n미발견 클래스 ({len(missing)}개):")
        for dl_idx in sorted(missing, key=int)[:10]:
            print(f"  - {dl_idx} ({dl_idx_to_k_code(dl_idx)})")
        if len(missing) > 10:
            print(f"  ... 외 {len(missing) - 10}개")

    # 이미지 다운로드 추천
    print(f"\n" + "=" * 70)
    print("이미지 다운로드 추천")
    print("=" * 70)
    print(f"다음 TS 폴더의 이미지를 다운로드하세요:")
    print(f"  TS_{', TS_'.join(recommended)}")
    print(f"\n총 {len(recommended)}개 폴더")

    return recommended, all_found_classes


def save_results(tl_results: list, recommended: list, found_classes: set):
    """분석 결과 JSON 저장"""
    output_path = Path("data/ts_analysis_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 클래스별 어떤 TL에 있는지 매핑
    class_to_tl = defaultdict(list)
    for tl in tl_results:
        if tl['target_classes']:
            tl_num = tl['name'].split('_')[1]
            for cls in tl['target_classes']:
                class_to_tl[cls].append({
                    'tl': tl_num,
                    'count': tl['target_classes'][cls]
                })

    result = {
        'recommended_image_folders': [f"TS_{num}" for num in recommended],
        'found_target_classes': sorted(found_classes, key=int),
        'missing_target_classes': sorted(TARGET_CLASSES - found_classes, key=int),
        'class_locations': {
            cls: class_to_tl[cls] for cls in sorted(found_classes, key=int)
        },
        'tl_folder_details': [
            {
                'name': tl['name'],
                'image_folder': f"TS_{tl['name'].split('_')[1]}",
                'target_class_count': len(tl['target_classes']),
                'target_sample_count': sum(tl['target_classes'].values()),
                'classes': dict(tl['target_classes'])
            }
            for tl in tl_results if tl['target_classes']
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {output_path}")


def main():
    """메인 실행"""
    print("=" * 70)
    print("AIHub 단일경구약제 라벨링 분석")
    print("=" * 70)
    print(f"분석 대상: {LABEL_DIR}")
    print(f"TARGET 클래스: {len(TARGET_CLASSES)}개")

    if not LABEL_DIR.exists():
        print(f"\n❌ 폴더가 존재하지 않습니다: {LABEL_DIR}")
        print("\n[사용법]")
        print("1. AIHub에서 '단일경구약제 5000종' 라벨링 데이터 다운로드")
        print("2. 위 경로에 TL_*_단일.zip 파일 배치")
        print("3. 다시 실행")
        return

    # ZIP 파일 찾기
    zip_files = find_zip_files(LABEL_DIR)

    if not zip_files:
        print(f"\n❌ TL_*_단일.zip 파일을 찾을 수 없습니다.")
        return

    print(f"\n발견된 ZIP 파일: {len(zip_files)}개")

    # 각 ZIP 파일 분석
    tl_results = []
    for i, zip_path in enumerate(zip_files):
        print(f"\r분석 중: {i+1}/{len(zip_files)} - {zip_path.name}", end="", flush=True)
        result = analyze_zip_file(zip_path)
        tl_results.append(result)

    print()  # 줄바꿈

    # 결과 출력
    recommended, found_classes = print_analysis_report(tl_results)

    # 결과 저장
    if recommended:
        save_results(tl_results, recommended, found_classes)


if __name__ == "__main__":
    main()

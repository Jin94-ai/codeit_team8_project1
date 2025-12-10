import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any
import cv2
import numpy as np

_current_script_file = os.path.abspath(__file__)
_current_script_dir = os.path.dirname(_current_script_file) 

_project_root = os.path.dirname(_current_script_dir)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
    print(f"Added {_project_root} to sys.path.")

from data.yolo_dataset.config import TRAIN_IMG_DIR, YOLO_ROOT 
from data.yolo_dataset.coco_parser import load_coco_tables 

from data.ny_preprocessing import preprocess_image_for_yolo_tensor 
from data.ny_augmentation import get_train_transforms, get_val_transforms, TARGET_IMAGE_SIZE 


# 전역 변수로 Dataset에서 필요한 데이터들을 캐시
_global_train_df = None
_global_val_df = None
_global_category_id_to_name = None
_global_class_name_to_id = None
_global_images_df = None


def _load_and_prepare_data_for_dataset():
    """
    `data/yolo_dataset/yolo_export.py`가 이미 실행되어 YOLO 디렉토리 구조와 라벨 파일들이 생성되었다고 가정하고,
    Dataset에서 필요한 DataFrame과 클래스 매핑을 이 파일 시스템으로부터 재구성
    """
    global _global_train_df, _global_val_df, _global_category_id_to_name, _global_class_name_to_id, _global_images_df
    if _global_train_df is not None:
        print("데이터가 이미 로드되어 있습니다. 재구성 건너뜀.")
        return 

    print("데이터셋 생성을 위해 필요한 데이터를 로드/재구성 중...")

    # 1. 원본 COCO 데이터 로드 (data.yolo_dataset.coco_parser 활용)
    images_df_orig, annotations_df_orig, categories_df_orig = load_coco_tables()

    # 2. 클래스 매핑 재구성 (yolo_export.py에서 사용한 로직과 동일해야 함)
    unique_cat_ids_original = sorted(categories_df_orig["id"].unique().tolist())
    catid_to_yoloid = {cid: idx for idx, cid in enumerate(unique_cat_ids_original)} # 원본 COCO ID -> 0~N-1 ID
    _global_category_id_to_name = {idx: categories_df_orig[categories_df_orig["id"] == cid]["name"].iloc[0] 
                                   for idx, cid in enumerate(unique_cat_ids_original)} # 0~N-1 ID -> Name
    _global_class_name_to_id = {name: cid for cid, name in _global_category_id_to_name.items()} # Name -> 0~N-1 ID
    
    # 3. yolo_export.py가 생성한 라벨 파일로부터 이미지 ID 추출하여 훈련/검증 이미지 ID 분할 재구성
    # YOLO_ROOT는 data/yolo_dataset/config.py에서 정의된 경로 
    train_labels_dir = os.path.join(YOLO_ROOT, "labels", "train")
    val_labels_dir = os.path.join(YOLO_ROOT, "labels", "val")

    if not os.path.exists(train_labels_dir) or not os.path.exists(val_labels_dir):
        raise FileNotFoundError(
            f"오류: YOLO 라벨 디렉토리 ({YOLO_ROOT}/labels/train 또는 val)를 찾을 수 없습니다."
            "`data/yolo_dataset/yolo_export.py`를 먼저 실행하여 데이터를 준비해주세요."
        )

    # 훈련 이미지 ID 추출
    train_image_ids_from_labels = []
    for label_file in os.listdir(train_labels_dir):
        if label_file.endswith(".txt"):
            image_stem = label_file.replace(".txt", "") # 확장자 없는 파일 이름 (예: K-00123)
            image_id_row = images_df_orig[images_df_orig['file_name'].str.startswith(image_stem)]
            if not image_id_row.empty:
                train_image_ids_from_labels.append(image_id_row['id'].iloc[0])

    # 검증 이미지 ID 추출
    val_image_ids_from_labels = []
    for label_file in os.listdir(val_labels_dir):
        if label_file.endswith(".txt"):
            image_stem = label_file.replace(".txt", "")
            image_id_row = images_df_orig[images_df_orig['file_name'].str.startswith(image_stem)]
            if not image_id_row.empty:
                val_image_ids_from_labels.append(image_id_row['id'].iloc[0])

    # 4. train_df 및 val_df 재구성
    full_ann_df_orig = pd.merge(annotations_df_orig, images_df_orig[['id', 'file_name', 'width', 'height']], 
                           left_on='image_id', right_on='id', suffixes=('', '_img'))
    full_ann_df_orig['class_id'] = full_ann_df_orig['category_id'].map(catid_to_yoloid)

    _global_train_df = full_ann_df_orig[
        full_ann_df_orig['image_id'].isin(train_image_ids_from_labels)
    ].reset_index(drop=True)

    _global_val_df = full_ann_df_orig[
        full_ann_df_orig['image_id'].isin(val_image_ids_from_labels)
    ].reset_index(drop=True)

    _global_images_df = images_df_orig 

    print("데이터 재구성 완료.")


class PillYoloDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str = TRAIN_IMG_DIR, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.image_ids = df['image_id'].unique()
        self.grouped_df = df.groupby('image_id')
        self.error_count = 0
        self.max_retries = 10 

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        for _ in range(self.max_retries): 
            try:
                current_image_idx = idx 
                image_id = self.image_ids[current_image_idx]
                image_annotations = self.grouped_df.get_group(image_id) 

                file_name = image_annotations['file_name'].iloc[0] 
                image_path = os.path.join(self.img_dir, file_name)

                image = cv2.imread(image_path)
                if image is None:
                    raise FileNotFoundError(f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                original_height, original_width = image.shape[:2]

                raw_bboxes = np.array(image_annotations['bbox'].tolist(), dtype=np.float32)
                raw_class_labels = image_annotations['class_id'].values.tolist() 
                
                if self.transforms:
                    transformed = self.transforms(image=image, bboxes=raw_bboxes, class_labels=raw_class_labels,
                                                  height=original_height, width=original_width)
                    image_transformed = transformed['image']
                    transformed_bboxes_coco = np.array(transformed['bboxes'], dtype=np.float32) 
                    transformed_class_labels = np.array(transformed['class_labels'], dtype=np.float32)
                else: 
                    image_transformed = preprocess_image_for_yolo_tensor(image)

                    transformed_bboxes_coco = raw_bboxes 
                    transformed_class_labels = np.array(raw_class_labels, dtype=np.float32)

                final_image_h, final_image_w = image_transformed.shape[1], image_transformed.shape[2] 
                
                bboxes_yolo = []
                filtered_class_labels_final = [] 
                
                for i, bbox_coco in enumerate(transformed_bboxes_coco):
                    xmin, ymin, w, h = bbox_coco
                    
                    cx = (xmin + w / 2.0) / final_image_w
                    cy = (ymin + h / 2.0) / final_image_h
                    norm_w = w / final_image_w
                    norm_h = h / final_image_h 

                    if norm_w > 1e-6 and norm_h > 1e-6: 
                        cx = np.clip(cx, 0.0, 1.0)
                        cy = np.clip(cy, 0.0, 1.0)
                        norm_w = np.clip(norm_w, 0.0, 1.0)
                        norm_h = np.clip(norm_h, 0.0, 1.0)

                        bboxes_yolo.append([cx, cy, norm_w, norm_h])
                        filtered_class_labels_final.append(transformed_class_labels[i])

                bboxes_yolo = np.array(bboxes_yolo, dtype=np.float32)
                
                if len(bboxes_yolo) > 0:
                    labels_tensor = torch.cat((torch.tensor(filtered_class_labels_final, dtype=torch.float32).unsqueeze(1),
                                               torch.tensor(bboxes_yolo, dtype=torch.float32)), dim=1)
                else:
                    labels_tensor = torch.empty((0, 5), dtype=torch.float32)

                return image_transformed, labels_tensor, (original_height, original_width)
                
            except Exception as e: 
                self.error_count += 1
                
                print(f"데이터 로딩 중 오류 발생 (에러 카운트: {self.error_count}): 이미지 ID {image_id}, 파일 {file_name}")
                print(f"원본 HxW: {original_height}x{original_width}")
                print(f"Albumentations에 전달된 (raw) bboxes: {raw_bboxes.tolist()}") 
                print(f"오류 메시지: {e}")
                
                idx = (idx + 1) % len(self.image_ids)
                if self.error_count % 1 == 0:
                    print(f"현재까지 {self.error_count}개의 샘플에서 오류 발생. 다음 샘플 시도 중.")
        
        raise RuntimeError(f"{self.max_retries}번의 재시도 후에도 유효한 샘플을 찾을 수 없습니다. 데이터셋 확인 필요.")


def collate_fn_yolo(batch: list) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """
    YOLOv8의 DataLoader에 맞는 `collate_fn` 구현.
    """
    images, labels_list, original_sizes = zip(*batch)
    
    images_batch = torch.stack(images, 0)

    batched_labels = []
    for i, labels_per_image in enumerate(labels_list):
        if labels_per_image.numel() == 0:
            continue
        batch_idx_tensor = torch.full((labels_per_image.shape[0], 1), i, dtype=torch.float32)
        batched_labels.append(torch.cat((batch_idx_tensor, labels_per_image), dim=1))
    
    if len(batched_labels) > 0:
        labels_batch = torch.cat(batched_labels, 0)
    else:
        labels_batch = torch.empty((0, 6), dtype=torch.float32)

    return images_batch, labels_batch, list(original_sizes)

if __name__ == "__main__":
    print("--- 데이터 전처리 및 준비 시작 ---")

    # 0. 필수 선행 작업 안내:
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ny_dataset.py의 부모의 부모 (PROJECT_ROOT)
    print("참고: 이 코드를 실행하기 전에 반드시 다음을 수행해야 합니다:")
    print(f"1. `cd {_project_root}` 로 이동 후")
    print(f"2. `python -m data.yolo_dataset.yolo_export`를 실행하여")
    print(f"YOLO 디렉토리 구조 (`{YOLO_ROOT}/labels/train` 등)와 라벨 파일을 생성해야 합니다.")
    print(f"3. `data/`, `data/yolo_dataset/` 디렉토리에 비어있는 `__init__.py` 파일이 모두 있는지 확인하십시오.\n")


    # 1. _load_and_prepare_data_for_dataset 함수 호출하여 데이터 재구성 및 캐시
    _load_and_prepare_data_for_dataset()

    train_df = _global_train_df
    val_df = _global_val_df
    category_id_to_name = _global_category_id_to_name 

    print("\n--- 데이터 준비 완료 (DataFrames 재구성) ---")
    if train_df is not None and not train_df.empty:
        print(f"재구성된 train_df 이미지 개수: {len(train_df['image_id'].unique())}")
        print(f"재구성된 val_df 이미지 개수: {len(val_df['image_id'].unique())}")
        print(f"카테고리 매핑: {category_id_to_name}")
    else:
        print("오류: train_df/val_df 재구성 실패. yolo_export.py가 실행되지 않았거나 데이터 없음.")


    # 2. Dataset 및 DataLoader 인스턴스 생성
    print("\n--- Dataset 및 DataLoader 인스턴스 생성 시작 ---")

    train_transforms = get_train_transforms(TARGET_IMAGE_SIZE) 
    val_transforms = get_val_transforms(TARGET_IMAGE_SIZE)

    train_dataset = PillYoloDataset(df=train_df, img_dir=TRAIN_IMG_DIR, transforms=train_transforms)
    val_dataset = PillYoloDataset(df=val_df, img_dir=TRAIN_IMG_DIR, transforms=val_transforms)


    print(f"\n훈련 데이터셋 이미지 개수: {len(train_dataset)}")
    print(f"검증 데이터셋 이미지 개수: {len(val_dataset)}")

    BATCH_SIZE = 4
    NUM_WORKERS = 2 

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_yolo
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_yolo
    )

    print(f"훈련 DataLoader 배치 개수: {len(train_loader)}")
    print(f"검증 DataLoader 배치 개수: {len(val_loader)}")
    print("DataLoader 생성 완료!")


    # 3. 첫 번째 배치 데이터 구조 확인 (YOLOv8 버전)
    print("\n--- DataLoader에서 첫 번째 배치 데이터 구조 확인 ---")
    for images_batch, labels_batch, original_sizes_batch in train_loader:
        print(f"배치 이미지 텐서 Shape: {images_batch.shape}")
        print(f"배치 이미지 텐서 Dtype: {images_batch.dtype}")
        print(f"배치 라벨 텐서 Shape: {labels_batch.shape}")
        print(f"배치 라벨 텐서 Dtype: {labels_batch.dtype}")
        print(f"배치 원본 이미지 크기: {original_sizes_batch}")
        break
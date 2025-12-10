import os

# 기본 경로 설정
_current_file_path = os.path.abspath(__file__)
_yolo_dataset_dir = os.path.dirname(_current_file_path) 
_data_root_dir = os.path.dirname(_yolo_dataset_dir) 
PROJECT_ROOT = os.path.dirname(_data_root_dir)

BASE_DIR = _data_root_dir 

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_ANN_DIR = os.path.join(BASE_DIR, "train_annotations")

# YOLOv8용 출력 루트
YOLO_ROOT = "datasets/pills"

# Train/Val split 설정
VAL_RATIO = 0.2
SPLIT_SEED = 42

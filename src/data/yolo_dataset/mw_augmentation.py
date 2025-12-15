import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

#(1) Pipeline 정의 + Dispatcher 구조
TARGET_IMAGE_SIZE = (640, 640)

def _base_postprocess(target_size):
    return [
        A.LongestMaxSize(max_size=max(target_size)),
        A.PadIfNeeded(target_size[1], target_size[0], value=(0,0,0)),
        A.Normalize(mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)),
        ToTensorV2()
    ]

def _compose(augs, target_size):
    return A.Compose(
        augs + _base_postprocess(target_size),
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            clip=True
        )
    )

#(2) 파이프라인 A~F
def train_A(ts):
    return _compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.2,0.2,p=0.3),
    ], ts)

def train_B(ts):
    return _compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10,p=0.3),
        A.RandomBrightnessContrast(0.25,0.25,p=0.3),
    ], ts)

def train_C(ts):
    return _compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10,p=0.3),
        A.HueSaturationValue(10,15,10,p=0.3),
    ], ts)

def train_D(ts):
    return _compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30,p=0.7),
        A.RandomBrightnessContrast(0.4,0.4,p=0.7),
        A.HueSaturationValue(25,40,25,p=0.6),
    ], ts)

def train_E(ts):
    return _compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20,p=0.5),
        A.MotionBlur(5,p=0.3),
        A.GaussNoise((20,60),p=0.3),
        A.CoarseDropout(4,64,64,p=0.3),
    ], ts)

def train_F(ts):
    return _compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30,p=0.7),
        A.RandomBrightnessContrast(0.4,0.4,p=0.7),
        A.HueSaturationValue(30,50,30,p=0.7),
        A.MotionBlur(5,p=0.3),
        A.GaussNoise((20,80),p=0.3),
        A.CoarseDropout(6,80,80,p=0.4),
    ], ts)

#(3) Dispatcher 함수
def get_train_transforms(aug_name="A", target_size=TARGET_IMAGE_SIZE):
    return {
        "A": train_A,
        "B": train_B,
        "C": train_C,
        "D": train_D,
        "E": train_E,
        "F": train_F,
    }[aug_name](target_size)


def get_val_transforms(target_size=TARGET_IMAGE_SIZE):
    return _compose([], target_size)
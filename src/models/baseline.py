import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import warnings

import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

import wandb
from ultralytics import YOLO

# ================= Path 설정 =================
_current_script_file = os.path.abspath(__file__)
_current_script_dir = os.path.dirname(_current_script_file)
_project_root = os.path.dirname(os.path.dirname(_current_script_dir))

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ================= Dataset / Augmentation =================
from src.data.yolo_dataset.config import YOLO_ROOT, TRAIN_IMG_DIR
from src.data.yolo_dataset.ny_dataset_TL_3 import (
    PillYoloDataset,
    collate_fn_yolo,
    _load_and_prepare_data_for_dataset,
    _global_train_df,
    _global_val_df
)

from src.data.yolo_dataset.mw_augmentation import (
    get_train_transforms,
    get_val_transforms,
    TARGET_IMAGE_SIZE
)

# ================= Callback =================
from src.models.callbacks import wandb_train_logging, wandb_val_logging

warnings.filterwarnings("ignore")

# ================= 한글 폰트 설정 =================
korean_fonts = ["NanumGothic", "AppleGothic", "Malgun Gothic"]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}

for font in korean_fonts:
    if font in available_fonts:
        plt.rcParams["font.family"] = font
        plt.rcParams["axes.unicode_minus"] = False
        break

# ================= Argument =================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--name", type=str, default="yolo_progressive_aug")
    return parser.parse_args()

# ================= Epoch → Augmentation 선택 =================
def select_aug(epoch: int, case_id: int):
    if case_id == 1:
        return "A"
    elif case_id == 2:
        return "A" if epoch < 30 else "B"
    elif case_id == 3:
        return "A" if epoch < 30 else "B" if epoch < 50 else "C"
    elif case_id == 4:
        return "A" if epoch < 30 else "B" if epoch < 50 else "C" if epoch < 70 else "D"
    elif case_id == 5:
        return "A" if epoch < 30 else "B" if epoch < 50 else "C" if epoch < 70 else "D" if epoch < 90 else "E"
    elif case_id == 6:
        return (
            "A" if epoch < 30 else
            "B" if epoch < 50 else
            "C" if epoch < 70 else
            "D" if epoch < 90 else
            "E" if epoch < 105 else
            "F"
        )
    else:
        raise ValueError(f"Invalid case_id: {case_id}")

# ================= Main =================
if __name__ == "__main__":
    args = parse_args()

    CASE_ID = 4  # 실험 케이스 선택

    wandb.init(
        project="codeit_team8",
        entity="codeit_team8",
        config={
            "case_id": CASE_ID,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "model": args.model_path,
        }
    )

    _load_and_prepare_data_for_dataset()

    if _global_train_df is None or _global_val_df is None:
        raise RuntimeError("Dataset loading failed")

    # Validation transform은 고정
    val_transforms = get_val_transforms(TARGET_IMAGE_SIZE)

    train_dataset = PillYoloDataset(
        df=_global_train_df,
        img_dir=TRAIN_IMG_DIR,
        transforms=None
    )

    val_dataset = PillYoloDataset(
        df=_global_val_df,
        img_dir=TRAIN_IMG_DIR,
        transforms=val_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn_yolo,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn_yolo,
        pin_memory=True
    )

    model = YOLO(args.model_path)
    model.add_callback("on_fit_epoch_end", wandb_train_logging)
    model.add_callback("on_val_end", wandb_val_logging)

    # ================= Epoch Loop (핵심) =================
    for epoch in range(args.epochs):
        aug = select_aug(epoch, CASE_ID)

        train_dataset.transforms = get_train_transforms(
            aug_name=aug,
            target_size=TARGET_IMAGE_SIZE
        )

        print(f"[Epoch {epoch:03d}] Augmentation Pipeline = {aug}")

        model.train(
            train=train_loader,
            val=val_loader,
            epochs=1,
            imgsz=args.imgsz,
            augment=False,
            workers=0,
            device=args.device,
            name=args.name,
        )

    # ================= Submission =================
    results = model.predict(
        source="data/test_images/",
        imgsz=args.imgsz,
        conf=0.5,
        iou=0.5,
        agnostic_nms=True,
        verbose=False
    )

    with open("data/yolo/class_mapping.json", "r") as f:
        yoloid_to_catid = {int(k): int(v) for k, v in json.load(f).items()}

    rows = []
    ann_id = 1

    for res in results:
        image_id = int(Path(res.path).stem)
        for box, cls, score in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
            x1, y1, x2, y2 = box.tolist()
            rows.append({
                "annotation_id": ann_id,
                "image_id": image_id,
                "category_id": yoloid_to_catid[int(cls)],
                "bbox_x": int(x1),
                "bbox_y": int(y1),
                "bbox_w": int(x2 - x1),
                "bbox_h": int(y2 - y1),
                "score": float(score),
            })
            ann_id += 1

    os.makedirs("outputs/submissions", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"outputs/submissions/submission_{ts}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"✓ Submission saved: {out_path}")
    wandb.finish()

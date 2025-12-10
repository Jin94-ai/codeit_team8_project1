import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import random

# 경고 무시 (선택 사항)
import warnings
warnings.filterwarnings('ignore')

# Matplotlib 폰트 매니저
from matplotlib import font_manager

# ---- 코랩 기본 한글 폰트 자동 설정 ----
# 코랩에 기본적으로 설치된 폰트 후보들
korean_fonts = ["NanumGothic", "AppleGothic", "Malgun Gothic", "DejaVu Sans"]

# 사용 가능한 폰트를 자동으로 탐색
available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
selected_font = None

for font in korean_fonts:
    if font in available_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams['font.family'] = selected_font
    plt.rcParams['axes.unicode_minus'] = False
    print(f"코랩에서 사용 가능한 한글 폰트 설정 완료: {selected_font}")
else:
    print("경고: 사용 가능한 기본 한글 폰트를 찾지 못했습니다. 수동 설정 필요")



################### Model Run ###################

import wandb
from ultralytics import YOLO

def main():
    # 1) W&B run 시작 + config 자동 로깅
    wandb.init(
        project="project1",
        config={
            "model": "yolov8n.pt",
            "data": "data/yolo/pills.yaml",
            "epochs": 50,
            "imgsz": 640,
            "lr0": 0.001,
            "batch": 16
        }
    )

    cfg = wandb.config  # wandb가 저장한 config

    # 2) YOLO 모델 불러오기
    model = YOLO(cfg.model)

    # 3) YOLO 학습
    result = model.train(
        data=cfg.data,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        lr0=cfg.lr0,
        batch=cfg.batch,
        project="runs/train",
        name="wandb_run",
    )

    # 4) 학습 결과를 W&B에 자동 업로드
    wandb.log({"result": result})

    wandb.finish()


if __name__ == "__main__":
    main()
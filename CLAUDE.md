# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle pill detection competition (Object Detection) using YOLOv8. Team of 5 working on a 3-week project to detect up to 4 pills per image.

**Critical Issue**: Original submission scored 0 because YOLO uses 0-based sequential indices (0,1,2...) but Kaggle requires original COCO category_ids (1899, 3350, etc.). This was fixed by creating `class_mapping.json` during YOLO export.

## Running the Pipeline

### Full Pipeline (Data → Training → Submission)

```bash
# New method (current)
./scripts/exc.sh
exc_pip

# Or directly run Python modules
python -m src.data.data_load.data_loader
python -m src.data.yolo_dataset.yolo_export
python -m src.models.baseline
```

### Inference Only (with trained model)

```bash
python scripts/inference.py --model runs/detect/train/weights/best.pt

# With custom parameters
python scripts/inference.py \
  --model runs/detect/train/weights/best.pt \
  --conf 0.5 \
  --iou 0.6 \
  --imgsz 640 640
```

### Clean Generated Files

```bash
make -f scripts/Makefile clean_data   # Clean data files
make -f scripts/Makefile clean_run    # Clean run.sh
```

## Architecture

### Data Pipeline (Critical)

The pipeline must run in this exact order:

1. **data_loader.py**: Downloads data from Kaggle using API
2. **yolo_export.py**: Converts COCO JSON → YOLO format + **creates class_mapping.json**
3. **baseline.py**: Trains model → predicts → **uses class_mapping.json** → generates submission.csv

**Why class_mapping.json is critical**: YOLO internally converts category_ids to 0-based indices for training. The mapping file preserves `{yolo_index: original_category_id}` so we can convert back for Kaggle submission.

```python
# In yolo_export.py (after YAML generation)
yoloid_to_catid = {idx: cid for cid, idx in catid_to_yoloid.items()}
with open("data/yolo/class_mapping.json", "w") as f:
    json.dump(yoloid_to_catid, f)

# In baseline.py (during prediction)
with open("data/yolo/class_mapping.json", "r") as f:
    yoloid_to_catid = json.load(f)
    yoloid_to_catid = {int(k): int(v) for k, v in yoloid_to_catid.items()}

# Convert YOLO prediction to original category_id
yolo_idx = int(cls)
original_category_id = yoloid_to_catid[yolo_idx]
```

### Data Stratification

Uses stratified train/val split (8:2) based on first category_id per image. **Special handling**: Images with only 1 sample in their class go entirely to train (can't stratify single samples).

### Submission Format

Output: `outputs/submissions/submission_{timestamp}.csv`

Required columns:
- annotation_id (sequential)
- image_id (from filename, e.g., "123.png" → 123)
- category_id (**must be original COCO id, not YOLO index**)
- bbox_x, bbox_y, bbox_w, bbox_h
- score (confidence)

## Team Workflow

### Roles
- **Leader/Integration Specialist**: 이진석 - PR reviews, Kaggle submissions
- **Data Engineers**: 김민우, 김나연 - EDA, preprocessing, augmentation
- **Model Architect**: 김보윤 - Model implementation, pipeline scripts
- **Experimentation Lead**: 황유민 - W&B tracking, hyperparameter tuning

### Collaboration Rules

**Before any commit**:
1. Ask the user first (this is a team project)
2. Make minimal changes only
3. Never add unsolicited features or refactoring
4. Respect existing code from other team members

**Daily logging**: Each team member writes `logs/collaboration/YYYY-MM-DD/YYYY-MM-DD_이름.md`

**Commit format**: `[Week X] Description`

### PR Conflicts

When merging PRs with baseline.py conflicts:
```bash
git merge --no-commit --no-ff origin/branch-name
git checkout --ours src/models/baseline.py  # Keep main version
git commit
```

This preserves submission generation logic while accepting other changes.

## Environment

**Development**: Windows with VS Code
**Execution**: WSL2/Ubuntu (avoid `/mnt/c/` paths - 10-100x slower I/O)

**Package installation** (in scripts/exc.sh or manual):
```bash
pip install ultralytics kaggle matplotlib seaborn scikit-learn pandas numpy wandb
```

## W&B Integration

Project name: `codeit_team8`

Integrated in baseline.py:
```python
wandb.init(project="codeit_team8", config={...})
model.train(...)
add_wandb_callback(model)  # Added by Model Architect
```

## Key Files

- `src/data/yolo_dataset/yolo_export.py`: COCO→YOLO conversion, generates class_mapping.json
- `src/models/baseline.py`: Training + prediction + submission generation
- `scripts/inference.py`: Standalone inference pipeline
- `data/yolo/class_mapping.json`: **Critical** - YOLO index to original category_id mapping
- `.gitignore`: Excludes `outputs/`, `data/`, `runs/`

## Known Issues

1. **NMS timeout in WSL2**: Caused by slow `/mnt/c/` filesystem, not NMS itself. Solutions:
   - Use WSL2 native paths (not /mnt/c/)
   - Reduce imgsz (640→480)
   - Add `save=False` to predict()

2. **Missing packages**: If `yolo_export` fails with "No module 'sklearn'", run.sh was missing packages (fixed in recent commits)

3. **Dataset location**: Data must be in `data/` (train_images, train_annotations, test_images). YOLO format output goes to `data/yolo/` or `datasets/pills/`

## Kaggle Submission

**Limit**: 5 submissions per day (team total)

**Strategy**: Use W&B to track experiments locally, only submit validated models to Kaggle.

**Submission generation**: Automatically created by baseline.py in `outputs/submissions/submission_{timestamp}.csv`

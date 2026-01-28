# DOD-SA: Infrared-Visible Decoupled Object Detection with Single-Modality Annotations
![图片1](https://github.com/user-attachments/assets/d8b1e71d-4468-4b9e-b46c-56180035408c)

English | [Chinese](./README_zh.md)

A PaddleDetection-based implementation of DA-DPDETR for paired VIS/IR object detection and domain-adaptive training. This document merges two variants:

- **HBB (horizontal bounding boxes, no angle/theta)** for **KAIST / FLIR / CVC-14** (the repo: `DA-DPDETR-Prominent_Poistion_Shift-notheta/DA-DPDETR`).
- **RBOX (rotated bounding boxes, with angle/theta)** for **DroneVehicle** (the repo: `DA-DPDETR-Prominent_Poistion_Shift/DA-DPDETR`).

> Do not mix configs or data formats between the two variants. HBB and RBOX use different dataset formats and post-processing.

## Contents
- [Overview](#overview)
- [Variants: HBB vs. RBOX](#variants-hbb-vs-rbox)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Inference](#inference)
- [Configs](#configs)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview
This project extends PaddleDetection with a DA-DPDETR training pipeline. It supports dual-stream (VIS/IR) inputs and multi-stage EMA-Teacher + pseudo-label training. The two variants differ in bounding-box type and datasets:
- **HBB**: axis-aligned boxes (no theta) for KAIST/FLIR/CVC-14.
- **RBOX**: rotated boxes (with theta) for DroneVehicle.

## Installation
- Python >= 3.7
- PaddlePaddle >= 2.4.1
- Other dependencies: `requirements.txt`

Install (make sure PaddlePaddle is installed for your CUDA version first):
```bash
pip install -r requirements.txt
pip install -e .
```

## Data Preparation
### Download Datasets
[DroneVehicle Dataset (Google Drive)](https://drive.google.com/file/d/1GLwW93BTqaaB_IJ96X_3oNO0RJgfqdR-/view?usp=drive_link)

[FLIR Dataset (Google Drive)](https://drive.google.com/file/d/1oZM_92-AYNflklvFlTUJUqUAAzBPX96f/view?usp=drive_link)

[CVC-14 Dataset (Google Drive)](https://drive.google.com/file/d/1Lcfm-baMqW2A3pYgCogBgWv0eXwnXUaJ/view?usp=drive_link)

[KAIST Dataset (Google Drive)](https://drive.google.com/file/d/1vI82_bOGS82vvtn9zroNs6v7yGfg18nY/view?usp=drive_link)
### HBB (KAIST / FLIR / CVC-14) — this repo
Dataset config: `configs/datasets/coco_detection_kaist_paired.yml`, which contains **KAIST / FLIR / CVC-14** blocks.
### Download pretrain weights
[ResNet50_pretrain_weights (Google Drive)](https://drive.google.com/file/d/1fvGWHRctR8q1LOrjQppp564xCmUw9qjz/view?usp=drive_link)
[RTDETR_pretrain_weights (Google Drive)](https://drive.google.com/file/d/1_evLYQ8OSCDbBOTDcalUNzVRixSAVEZ1/view?usp=drive_link)

**Steps:**
1. Open `configs/datasets/coco_detection_kaist_paired.yml`.
2. Uncomment the dataset block you need (KAIST/FLIR/CVC-14) and comment out the others.
3. Update `dataset_dir` to your local path.
4. Ensure image folders and annotation filenames match the config.

KAIST example structure:
```text
dataset/kaist_paired/
  train_imgs/
    vis/
    ir/
  val_imgs/
    vis/
    ir/
  coco_annotations/
    train_vis_nounpaired.json
    train_ir_nounpaired.json
    val.json
  label_list.txt
```

FLIR example structure:
```text
FLIR-image/
  vi-train/
  ir-train/
  vi-test/
  ir-test/
train_ir.json
val_ir.json
label_list.txt
```

CVC-14 example structure (default block in config):
```text
CVC-14/
  train_DeformCAT/
    ir/images/
    vis/images/
  val_DeformCAT/
    ir/images/
    vis/images/
  train_ir_coco_paired.json
  train_vis_coco_paired.json
  val_ir_coco_paired.json
  val_vis_coco_paired.json
  label_list.txt
```

If your naming differs, edit `vis_image_dir`, `ir_image_dir`, and `anno_path_*` in the dataset config.

### RBOX (DroneVehicle) — DroneVehicle repo
Use the DroneVehicle repo at:
`/data1/jinhang/Projects/DOD-SA/DA-DPDETR-Prominent_Poistion_Shift/DA-DPDETR`

Dataset config: `configs/datasets/rbox_detection_drone-vechal_paired.yml`.
Update `dataset_dir` to your local path and ensure annotation file names match.

Example structure:
```text
dataset/rbox_Drone_Vehicle/
  train/
    trainimg/
    trainimgr/
  val/
    valimg/
    valimgr/
  train_vis_segmentation_paired.json
  train_ir_segmentation_paired.json
  val_vis_segmentation_paired.json
  val_ir_segmentation_paired.json
  label_list.txt
```

If your naming differs, edit `vis_image_dir`, `ir_image_dir`, and `anno_path_*` in the dataset config.

## Training and Evaluation
### HBB (KAIST / FLIR / CVC-14) — this repo
Single GPU:
```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml --eval
```

Evaluation:
```bash
python tools/eval.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml \
  -o weights=output/your_exp/model_final.pdparams
```

### RBOX (DroneVehicle) — DroneVehicle repo
Run the same commands **inside the DroneVehicle repo** using its configs:
```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml --eval
```

Evaluation:
```bash
python tools/eval.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml \
  -o weights=output/your_exp/model_final.pdparams
```

## Inference
### HBB (KAIST / FLIR / CVC-14) — this repo
```bash
python tools/infer.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml \
  -o weights=output/your_exp/model_final.pdparams \
  --infer_img=path/to/your_image.jpg
```

### RBOX (DroneVehicle) — DroneVehicle repo
```bash
python tools/infer.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml \
  -o weights=output/your_exp/model_final.pdparams \
  --infer_img=path/to/your_image.jpg
```

## Configs
### HBB (this repo)
- `configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml`: training schedule and staged settings.
- `configs/DA-DPDETR/_base_/damsdet_r_paired_DAOD_r50vd.yml`: model architecture and core hyperparameters.
- `configs/DA-DPDETR/_base_/rtdetr_r_DAOD_kaist_reader.yml`: data reader and augmentations.
- `configs/datasets/coco_detection_kaist_paired.yml`: dataset paths and annotations (KAIST/FLIR/CVC-14 blocks).
- `configs/runtime_kaist.yml`: runtime settings and output directory.
- `ppdet/modeling/architectures/damsdet_rotate.py`: main network implementation (HBB, no theta).

### RBOX (DroneVehicle repo)
- `configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml`: training schedule and staged settings.
- `configs/DA-DPDETR/_base_/damsdet_r_paired_DAOD_r50vd.yml`: model architecture and core hyperparameters.
- `configs/DA-DPDETR/_base_/rtdetr_r_DAOD_reader.yml`: data reader and augmentations.
- `configs/datasets/rbox_detection_drone-vechal_paired.yml`: dataset paths and annotations.
- `ppdet/modeling/architectures/damsdet_rotate.py`: main network implementation (RBOX).

## Results
Fill in your results here:

| Dataset | Metric | IR | RGB |
| --- | --- | --- | --- |
| Drone-Vehicle | mAP50 | 80.41 | 78.87 |
| FLIR | mAP | 85.2 | - |
| CVC-14 | MR^-2 | 23.1 | 12.3 |
| KAIST | MR^-2 | 23.92 | - |

other SOTA results file: [SOTA](https://drive.google.com/file/d/1R-X_KsVxttoID0Q0HnCcBpNNUwb8-tDi/view?usp=drive_link)
## Acknowledgements
Built on PaddleDetection with RT-DETR / DINO components. Thanks to the open-source community.

## License
- **This repo (HBB):** Apache-2.0（见其 LICENSE 文件）。

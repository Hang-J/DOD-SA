# DA-DPDETR (Prominent Position Shift, HBB)

This repository is a PaddleDetection-based implementation of DA-DPDETR for paired VIS/IR horizontal bounding-box (HBB, no angle/theta) object detection and domain-adaptive training. It targets the KAIST, FLIR, and CVC-14 datasets.

## Scope vs. the DroneVehicle RBOX repo (Important)
- **This repo (KAIST/FLIR/CVC-14):** horizontal bounding boxes (HBB), no rotation angle, paired VIS/IR inputs, COCO-style bbox annotations.
- **DA-DPDETR-Prominent_Position_Shift (DroneVehicle):** rotated bounding boxes (RBOX) with angle/theta, DroneVehicle dataset, RBOX post-process and configs.

> Do not mix configs or data formats between the two repos: this repo is **HBB**, the DroneVehicle repo is **RBOX**.

## Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Environment and Installation](#environment-and-installation)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Inference](#inference)
- [Configs](#configs)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview
This repo extends PaddleDetection with a DA-DPDETR training pipeline for paired VIS/IR data. It supports dual-stream inputs, horizontal bounding boxes, and multi-stage EMA-Teacher + pseudo-label training. The default setup uses a ResNet50-VD backbone and RT-DETR-related components.

## Key Features
- Dual-stream backbone with a paired transformer for VIS/IR data.
- Horizontal object detection (HBB) with paired post-processing (no theta).
- EMA teacher, pseudo labels, and staged training schedule.
- Compatible with PaddleDetection training/eval/inference tools.

## Environment and Installation
- Python >= 3.7
- PaddlePaddle >= 2.4.1
- Other dependencies: `requirements.txt`

Install (make sure PaddlePaddle is installed for your CUDA version first):
```bash
pip install -r requirements.txt
pip install -e .
```

## Data Preparation
Dataset config: `configs/datasets/coco_detection_kaist_paired.yml`, which contains blocks for **KAIST / FLIR / CVC-14**.

**Steps:**
1. Open `configs/datasets/coco_detection_kaist_paired.yml`.
2. Uncomment the dataset block you need (KAIST/FLIR/CVC-14) and comment out the others.
3. Update `dataset_dir` to your local path.
4. Ensure image folders and annotation filenames match the config.

### KAIST example structure
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

### FLIR example structure
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

### CVC-14 example structure (default block in config)
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

## Training and Evaluation
**Single GPU:**
```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml --eval
```

**Multi GPU:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 \
  tools/train.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml --fleet --eval
```

**Evaluation:**
```bash
python tools/eval.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml \
  -o weights=output/your_exp/model_final.pdparams
```

## Inference
```bash
python tools/infer.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml \
  -o weights=output/your_exp/model_final.pdparams \
  --infer_img=path/to/your_image.jpg
```

## Configs
- `configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml`: main training schedule and staged settings.
- `configs/DA-DPDETR/_base_/damsdet_r_paired_DAOD_r50vd.yml`: model architecture and core hyperparameters.
- `configs/DA-DPDETR/_base_/rtdetr_r_DAOD_kaist_reader.yml`: data reader and augmentations.
- `configs/datasets/coco_detection_kaist_paired.yml`: dataset paths and annotations (KAIST/FLIR/CVC-14 blocks).
- `configs/runtime_kaist.yml`: runtime settings and output directory.
- `ppdet/modeling/architectures/damsdet_rotate.py`: main network implementation (HBB, no theta).

> Note: If you use local pretrained weights, set `pretrain_weights` to your local `.pdparams` file.

## Results
Fill in your results here:

| Dataset | Metric (mAP, HBB) | Notes |
| --- | --- | --- |
| KAIST (paired) | TBD | config: damsdet_r_paired_DAOD_r50vd_6x |
| FLIR (paired) | TBD | config: damsdet_r_paired_DAOD_r50vd_6x |
| CVC-14 (paired) | TBD | config: damsdet_r_paired_DAOD_r50vd_6x |

## Acknowledgements
Built on PaddleDetection with RT-DETR / DINO components. Thanks to the open-source community.

## License
No license file is included in this repo. If you plan to distribute, add a LICENSE file that matches your intended usage and dependencies.

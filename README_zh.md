# DOD-SA: Infrared-Visible Decoupled Object Detection with Single-Modality Annotations
![图片1](https://github.com/user-attachments/assets/18110be6-54ef-4b8f-b050-a1b9df6bfaf5)

中文 | [English](README.md)

本项目基于 PaddleDetection，实现了 DA-DPDETR 的配对 VIS/IR 目标检测与域自适应训练。本文档合并了两种变体：

- **HBB（水平框，无角度/theta）**：适用于 **KAIST / FLIR / CVC-14**（本仓库）。
- **RBOX（旋转框，含角度/theta）**：适用于 **DroneVehicle**（另一个仓库：`/data1/jinhang/Projects/DOD-SA/DA-DPDETR-Prominent_Poistion_Shift/DA-DPDETR`）。

> 不要混用两种变体的配置与数据格式。HBB 与 RBOX 的数据标注与后处理不同。

## 目录
- [概览](#概览)
- [变体说明：HBB 与 RBOX](#变体说明hbb-与-rbox)
- [安装](#安装)
- [数据准备](#数据准备)
- [训练与评估](#训练与评估)
- [推理](#推理)
- [配置说明](#配置说明)
- [结果](#结果)
- [致谢](#致谢)
- [许可证](#许可证)

## 概览
本项目在 PaddleDetection 上扩展了 DA-DPDETR 的训练流程，支持双流（VIS/IR）输入以及多阶段 EMA-Teacher + 伪标签训练。两种变体在 **边界框类型** 与 **数据集** 上不同：
- **HBB**：水平框（无 theta），用于 KAIST/FLIR/CVC-14。
- **RBOX**：旋转框（有 theta），用于 DroneVehicle。

## 变体说明：HBB 与 RBOX
### HBB（KAIST / FLIR / CVC-14）— 本仓库
- **框类型**：HBB（水平框，无角度）。
- **数据集**：KAIST、FLIR、CVC-14。
- **数据集配置**：`configs/datasets/coco_detection_kaist_paired.yml`（包含三种数据集块）。
- **Reader 配置**：`configs/DA-DPDETR/_base_/rtdetr_r_DAOD_kaist_reader.yml`。

### RBOX（DroneVehicle）— DroneVehicle 仓库
- **框类型**：RBOX（旋转框，有角度/theta）。
- **数据集**：DroneVehicle。
- **数据集配置**：`configs/datasets/rbox_detection_drone-vechal_paired.yml`（在 DroneVehicle 仓库中）。
- **Reader 配置**：`configs/DA-DPDETR/_base_/rtdetr_r_DAOD_reader.yml`（在 DroneVehicle 仓库中）。

## 安装
- Python >= 3.7
- PaddlePaddle >= 2.4.1
- 其它依赖：`requirements.txt`

安装（先安装与你 CUDA 对应的 PaddlePaddle 版本）：
```bash
pip install -r requirements.txt
pip install -e .
```

## 数据准备
### HBB（KAIST / FLIR / CVC-14）— 本仓库
数据集配置文件：`configs/datasets/coco_detection_kaist_paired.yml`，包含 **KAIST / FLIR / CVC-14** 三个数据集块。

**步骤：**
1. 打开 `configs/datasets/coco_detection_kaist_paired.yml`。
2. 取消注释你要使用的数据集块（KAIST/FLIR/CVC-14），并注释掉其它块。
3. 将 `dataset_dir` 修改为你的本地路径。
4. 确认图像目录与标注文件名与配置一致。

KAIST 示例结构：
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

FLIR 示例结构：
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

CVC-14 示例结构（配置文件默认块）：
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

如果你的命名不同，请修改 `vis_image_dir`、`ir_image_dir` 和 `anno_path_*` 字段。

### RBOX（DroneVehicle）— DroneVehicle 仓库
使用 DroneVehicle 仓库：
`/data1/jinhang/Projects/DOD-SA/DA-DPDETR-Prominent_Poistion_Shift/DA-DPDETR`

数据集配置文件：`configs/datasets/rbox_detection_drone-vechal_paired.yml`。
请更新 `dataset_dir` 为你的本地路径，并确保标注文件名一致。

示例结构：
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

如果你的命名不同，请修改 `vis_image_dir`、`ir_image_dir` 和 `anno_path_*` 字段。

## 训练与评估
### HBB（KAIST / FLIR / CVC-14）— 本仓库
单卡训练：
```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml --eval
```

多卡训练：
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 \
  tools/train.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml --fleet --eval
```

评估：
```bash
python tools/eval.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml \
  -o weights=output/your_exp/model_final.pdparams
```

### RBOX（DroneVehicle）— DroneVehicle 仓库
请在 **DroneVehicle 仓库** 内使用其配置运行相同命令：
```bash
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml --eval
```

评估：
```bash
python tools/eval.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml \
  -o weights=output/your_exp/model_final.pdparams
```

## 推理
### HBB（KAIST / FLIR / CVC-14）— 本仓库
```bash
python tools/infer.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml \
  -o weights=output/your_exp/model_final.pdparams \
  --infer_img=path/to/your_image.jpg
```

### RBOX（DroneVehicle）— DroneVehicle 仓库
```bash
python tools/infer.py -c configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml \
  -o weights=output/your_exp/model_final.pdparams \
  --infer_img=path/to/your_image.jpg
```

## 配置说明
### HBB（本仓库）
- `configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml`：训练调度与多阶段设置。
- `configs/DA-DPDETR/_base_/damsdet_r_paired_DAOD_r50vd.yml`：模型结构与核心超参。
- `configs/DA-DPDETR/_base_/rtdetr_r_DAOD_kaist_reader.yml`：数据读取与增强策略。
- `configs/datasets/coco_detection_kaist_paired.yml`：数据集路径与标注（KAIST/FLIR/CVC-14）。
- `configs/runtime_kaist.yml`：运行时设置与输出目录。
- `ppdet/modeling/architectures/damsdet_rotate.py`：主网络实现（HBB，无 theta）。

### RBOX（DroneVehicle 仓库）
- `configs/DA-DPDETR/damsdet_r_paired_DAOD_r50vd_6x.yml`：训练调度与多阶段设置。
- `configs/DA-DPDETR/_base_/damsdet_r_paired_DAOD_r50vd.yml`：模型结构与核心超参。
- `configs/DA-DPDETR/_base_/rtdetr_r_DAOD_reader.yml`：数据读取与增强策略。
- `configs/datasets/rbox_detection_drone-vechal_paired.yml`：数据集路径与标注。
- `ppdet/modeling/architectures/damsdet_rotate.py`：主网络实现（RBOX）。

## 结果
请补充你的实验结果：

| 变体 | 数据集 | 指标 | 备注 |
| --- | --- | --- | --- |
| HBB | KAIST（paired） | mAP（HBB） | config: damsdet_r_paired_DAOD_r50vd_6x |
| HBB | FLIR（paired） | mAP（HBB） | config: damsdet_r_paired_DAOD_r50vd_6x |
| HBB | CVC-14（paired） | mAP（HBB） | config: damsdet_r_paired_DAOD_r50vd_6x |
| RBOX | Drone-Vehicle（paired） | mAP（RBOX） | config: damsdet_r_paired_DAOD_r50vd_6x |

## 致谢
基于 PaddleDetection，并参考 RT-DETR / DINO 相关组件。感谢开源社区。

## 许可证
- **本仓库（HBB）**：未包含 LICENSE 文件，如需发布请补充。
- **DroneVehicle 仓库（RBOX）**：Apache-2.0（见其 LICENSE 文件）。

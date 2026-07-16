# ALSDet: Apple Leaf Disease Detection with Cascade R-CNN

> **ALSDet** is a specialized object detection framework for early-stage apple leaf diseases under natural orchard conditions. It adaptively integrates Global Context (GC) blocks, dilated convolutions, and OHEM into a Cascade R-CNN baseline to address the challenges of small lesion detection and severe foreground-background imbalance.

## 📌 Repository Structure

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n alsdet python=3.8 -y
conda activate alsdet
pip install -r requirements.txt


2. Data Preparation
Option A: Full dataset (internal use)
Place the dataset under data/ with the following structure:

text
data/
└── coco/
    ├── train2017/
    ├── val2017/
    ├── test2017/
    └── annotations/

3. Training
Train ALSDet (Cascade R-CNN with GC + dilation + OHEM):

bash
python tools/train.py \
    alsdet/alsdet/cascade-rcnn_r50-gcb-r16-c3-c5_fpn_ms_ohem_110_1x_coco.py \
    --work-dir ./work_dirs/alsdet/

4. Evaluation
bash
python tools/test.py \
    alsdet/alsdet/cascade-rcnn_r50-gcb-r16-c3-c5_fpn_ms_ohem_110_1x_coco.py \
    ./work_dirs/alsdet/epoch_12.pth \
    --out ./work_dirs/alsdet/results.pkl

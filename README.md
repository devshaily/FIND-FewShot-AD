# FIND: Few-Shot Anomaly Inspection (ICCV 2025)

Unofficial PyTorch reimplementation of the paper:

**FIND: Few-Shot Anomaly Inspection with Normal-Only Multi-Modal Data**

📄 [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_FIND_Few-Shot_Anomaly_Inspection_with_Normal-Only_Multi-Modal_Data_ICCV_2025_paper.pdf)

---

## Overview
This repository provides a code implementation of FIND for few-shot anomaly inspection using normal-only multi-modal data (e.g., RGB and surface normals).

## Results (Dowel, Full-shot, Single Run)

| Metric     | Ours   | Paper  |
|------------|--------|--------|
| I-AUROC    | 0.8746 | 0.979  |
| P-AUROC    | 0.9872 | 0.995  |
| AUPRO@30%  | 0.9732 | 0.986  |
| AUPRO@1%   | 0.6136 | 0.982  |

## Requirements
pip install torch torchvision timm tifffile open3d tqdm scikit-learn opencv-python

## Dataset
Download MVTec 3D-AD from https://www.mvtec.com/company/research/datasets/mvtec-3d-ad

## Training
Set CATEGORY in find_train.py then run:
python find_train.py

## Evaluation
python evaluate_experiment.py
---

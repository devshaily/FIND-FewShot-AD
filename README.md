# FIND: Few-Shot Anomaly Inspection (ICCV 2025)

Unofficial PyTorch reimplementation of the paper:  
**FIND: Few-Shot Anomaly Inspection with Normal-Only Multi-Modal Data**  
📄 [Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_FIND_Few-Shot_Anomaly_Inspection_with_Normal-Only_Multi-Modal_Data_ICCV_2025_paper.pdf)

> **Note:** No official code has been released by the authors.  
> `v2-paper-aligned` branch is the most up-to-date implementation.  
> This is an independent reimplementation based on the paper.

> **Few-shot support:** K=5, 10, 50 supported via `K_SHOT` config.  
> Full-shot results reported. Few-shot benchmark results coming soon..

---

## ⭐ If you find this useful
If this reimplementation helped your research or learning, please consider **starring the repo** — it helps others find it and motivates further work!

[![GitHub stars](https://img.shields.io/github/stars/devshaily/FIND-FewShot-AD?style=social)](https://github.com/devshaily/FIND-FewShot-AD/stargazers)

---

## Overview
This repository provides a code reimplementation of FIND for few-shot anomaly inspection using normal-only multi-modal data (e.g., RGB and surface normals).

## Reproduced Results (Dowel, Full-shot, Single Run)

| Metric     | Ours   | Paper  |
|------------|--------|--------|
| I-AUROC    | 0.8746 | 0.979  |
| P-AUROC    | 0.9872 | 0.995  |
| AUPRO@30%  | 0.9732 | 0.986  |
| AUPRO@1%   | 0.6136 | 0.982  |

> Single run, single category (dowel, full-shot).  
> Full results across all 10 categories coming soon.

## Requirements
```bash
pip install torch torchvision timm tifffile open3d tqdm scikit-learn opencv-python
```

## Dataset
Download MVTec 3D-AD from [here](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).

## Training
Set `CATEGORY` in `find_train.py` then run:
```bash
python find_train.py
```

## Evaluation
```bash
python find_eval.py
python evaluate_experiment.py
```

## File Attribution

| File | Source |
|---|---|
| `find_train.py` | Our reimplementation of FIND (ICCV 2025) — training pipeline |
| `find_eval.py` | Our reimplementation of FIND (ICCV 2025) — inference & map saving |
| `evaluate_experiment.py` | Official MVTec 3D-AD evaluation scripts (modified) |
| `generic_util.py` | Official MVTec 3D-AD evaluation scripts |
| `pro_curve_util.py` | Official MVTec 3D-AD evaluation scripts |
| `roc_curve_util.py` | Official MVTec 3D-AD evaluation scripts |
| `lifind.yml` | Conda environment for training (CPU) |
| `lifindgpu.yml` | Conda environment for training (GPU) |

## Attribution
MVTec 3D-AD evaluation scripts are from the [official MVTec 3D-AD dataset page](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).

## Visualization Examples (Dowel category)

See [`visualizations/`](visualizations/) for full results 
(RGB input, surface normal, anomaly map, overlay, GT mask).

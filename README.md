<h2 align="center">RaBiT-CC: Reliability-Aware Bidirectional Tri-Attention for RGB-T Crowd Counting</h2>

<p align="center">
  Official implementation of <strong>RaBiT-CC: Reliability-Aware Bidirectional Tri-Attention for RGB-T Crowd Counting</strong>.
</p>

<p align="center">
  <strong>RGB-T Crowd Counting</strong> · <strong>Cross-Modal Fusion</strong> · <strong>Reliability Gating</strong> · <strong>Spatial Misalignment</strong>
</p>

---

## Overview

RGB-Thermal (RGB-T) crowd counting improves robustness in complex surveillance scenarios by combining visible and thermal information. However, existing methods still face two major challenges:

1. **Cross-modal spatial misalignment** caused by camera parallax and different imaging mechanisms, which may introduce ghosting artifacts and counting bias during feature fusion.
2. **Dynamic modality reliability**, where RGB and thermal modalities may become unreliable in different local regions due to low illumination, thermal interference, occlusion, or cluttered backgrounds.

To address these issues, we propose **RaBiT-CC**, a reliability-aware RGB-T crowd counting framework. RaBiT-CC introduces a **Reliability-Aware Bidirectional Tri-Attention Fusion (RaBiT-Fusion)** module and a **Binary Preference Loss (BPL)** to perform efficient soft alignment and explicitly supervise local modality preference learning.

---

## Method Highlights

### RaBiT-Fusion

RaBiT-Fusion uses a small set of **mediator tokens** as cross-modal bridges between RGB and thermal features. Instead of relying on dense pixel-wise cross-modal interaction, it performs bidirectional tri-attention among:

- RGB features,
- thermal features,
- mediator tokens.

The mediator tokens first aggregate reliable contextual information from both modalities, and then each modality attends back to the updated mediator tokens to absorb complementary cross-modal cues. This design enables efficient feature interaction while reducing the negative impact of local spatial mismatch.

### Reliability Estimator

A lightweight **Reliability Estimator (RE)** predicts pixel-level reliability maps for RGB and thermal features. These reliability maps are used as soft gates during feature interaction and final fusion, allowing the model to suppress unreliable local responses and emphasize the more trustworthy modality.

### Binary Preference Loss

**Binary Preference Loss (BPL)** explicitly supervises the reliability estimator. It compares local counting errors from auxiliary RGB-specific and thermal-specific prediction heads, then constructs binary preference labels indicating which modality is more reliable in each local window. This encourages the predicted reliability weights to be consistent with empirical local counting quality.

---

## Framework

The overall framework contains four main stages:

1. A weight-shared PVT backbone extracts multi-scale RGB and thermal features.
2. Reliability estimators generate pixel-level reliability maps for each modality.
3. RaBiT-Fusion performs reliability-aware bidirectional tri-attention through mediator tokens.
4. A cascaded decoder integrates fused multi-scale features and predicts the final density map.

---

## Datasets

We evaluate RaBiT-CC on two widely used RGB-T crowd counting benchmarks:

| Dataset | Description |
|---|---|
| [RGBT-CC](https://github.com/chen-judge/RGBTCrowdCounting) | 2,030 aligned RGB-T image pairs with 138,389 head annotations under diverse scenes and illumination conditions. |
| [DroneRGBT](https://github.com/VisDrone/DroneRGBT) | 3,600 registered drone-view RGB-T image pairs with 175,698 head annotations, containing cluttered backgrounds and large scale variations. |

---

## Experimental Results

### Results on RGBT-CC

| Method | Venue | GAME(0) | GAME(1) | GAME(2) | GAME(3) | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| MCNN | CVPR'16 | 21.89 | 25.70 | 30.22 | 37.19 | 37.44 |
| CSRNet | CVPR'18 | 20.40 | 23.58 | 28.03 | 35.51 | 35.26 |
| BL+IADM | CVPR'21 | 15.61 | 19.95 | 24.69 | 32.89 | 28.18 |
| BL+CSCA | ACCV'22 | 14.32 | 18.91 | 23.81 | 32.47 | 26.01 |
| DEFNet | TITS'22 | 11.90 | 16.08 | 20.19 | 27.27 | 21.09 |
| MC3Net | TITS'23 | 11.47 | 15.06 | 19.40 | 27.95 | 20.59 |
| CGINet | EAAI'23 | 12.07 | 15.98 | 20.06 | 27.73 | 20.54 |
| CSA-Net | ESWA'23 | 12.45 | 16.46 | 21.48 | 30.62 | 21.64 |
| GETANet | GRSL'24 | 12.14 | 15.98 | 19.40 | 28.61 | 22.17 |
| DAACFNet | SSRN'24 | 11.36 | 15.55 | 20.37 | 30.51 | 21.45 |
| MCN | ESWA'24 | 11.56 | 15.92 | 20.16 | 28.06 | 19.02 |
| BGDFNet | TIM'24 | 11.00 | 15.04 | 19.86 | 29.72 | 19.05 |
| CSCA | PR'25 | 13.50 | 18.63 | 23.59 | 31.59 | 24.00 |
| CMFX | NN'25 | 11.25 | 15.33 | 19.62 | 26.14 | 19.38 |
| MISF-Net | TMM'25 | 10.90 | 14.87 | 19.65 | 29.18 | 19.42 |
| **RaBiT-CC (Ours)** | - | **10.70** | **14.98** | **19.06** | **26.41** | **18.62** |

### Results on DroneRGBT

| Method | Venue | GAME(0) | GAME(1) | GAME(2) | GAME(3) | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| MCNN | CVPR'16 | 20.45 | 26.57 | 35.57 | 46.65 | 27.30 |
| CSRNet | CVPR'18 | 9.99 | 12.73 | 17.63 | 28.16 | 16.29 |
| BL+IADM | CVPR'21 | 9.77 | 12.91 | 17.08 | 22.61 | 15.76 |
| MMCCN | ACCV'20 | 7.27 | 9.12 | 11.45 | 15.21 | 11.45 |
| BL+CSCA | ACCV'22 | 10.20 | 12.24 | 15.34 | 20.27 | 16.10 |
| DEFNet | TITS'22 | 9.00 | 10.66 | 12.91 | 15.96 | 17.93 |
| MC3Net | TITS'23 | 7.63 | 9.87 | 13.64 | 19.44 | 11.17 |
| CGINet | EAAI'23 | 8.37 | 9.97 | 12.34 | 15.51 | 13.45 |
| GETANet | GRSL'24 | 8.44 | 10.01 | 12.75 | 15.83 | 13.99 |
| CAGNet | GRSL'24 | 6.48 | 8.33 | 10.86 | 14.29 | 10.30 |
| DLF-IA | IJON'24 | 6.28 | 8.31 | 11.77 | 18.65 | 10.16 |
| SEFNet | NC'24 | 6.27 | 8.42 | 11.11 | 15.18 | 9.83 |
| CSCA | PR'25 | 9.51 | 12.12 | 15.84 | 21.57 | 15.19 |
| CMFX | NN'25 | 6.75 | 8.88 | 11.87 | 14.69 | 11.05 |
| MIANet | TITS'25 | 6.74 | 8.64 | 11.49 | 16.31 | 10.58 |
| **RaBiT-CC (Ours)** | - | **5.68** | **7.22** | **9.32** | **12.70** | **9.07** |

---

## Getting Started

### 1. Data Preparation

#### RGBT-CC

Download the dataset from [RGBT-CC](https://github.com/chen-judge/RGBTCrowdCounting) and organize it as follows:

```text
RGBT_CC
├── train
│   ├── 1162_RGB.jpg
│   ├── 1162_T.jpg
│   ├── 1162_GT.npy
│   └── ...
├── val
│   └── ...
└── test
    └── ...
```

#### DroneRGBT

Download the dataset from [DroneRGBT](https://github.com/VisDrone/DroneRGBT) and organize it as follows:

```text
Drone_RGBT
├── train
│   ├── GT_
│   ├── RGB
│   └── Infrared
└── test
    ├── GT_
    ├── RGB
    └── Infrared
```

### 2. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Core libraries:

```text
torch >= 1.11.0
torchvision >= 0.12.0
timm == 1.0.12
mmcv-full == 1.7.2
```

### 3. Training

We use **PVT-v2-b3** as the backbone. Please place the pretrained weights under `pretrained_weights/`.

Train on RGBT-CC:

```bash
python train.py \
    --dataset RGBTCC \
    --data-dir ./data/RGBT-CC \
    --batch-size 8 \
    --lr 1e-5 \
    --device 0 \
    --exp-tag rabbit_rgbtcc
```

Train on DroneRGBT:

```bash
python train.py \
    --dataset DroneRGBT \
    --data-dir ./data/Drone_RGBT \
    --batch-size 8 \
    --lr 1e-5 \
    --device 0 \
    --exp-tag rabbit_dronergbt
```

### 4. Testing

```bash
python test.py \
    --dataset RGBTCC \
    --data-dir ./data/RGBT-CC \
    --checkpoint ./checkpoints/rabbit_rgbtcc.pth \
    --device 0
```

---

## License

This source code is released for research and education use only. Commercial use is prohibited without prior written permission from the authors.

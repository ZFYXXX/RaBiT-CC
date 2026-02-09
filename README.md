<h2 align="center">RaBiT-CC: Reliability-Aware Bidirectional Tri-Attention for RGB-T Crowd Counting</h2>

<p align="center">
  Official implementation of the paper "RaBiT-CC: Reliability-Aware Bidirectional Tri-Attention for RGB-T Crowd Counting".
</p>

## Overview

**_Abstract -_** In recent years, RGB-Thermal (RGB-T) crowd counting has garnered increasing attention due to its robustness in all-weather surveillance. However, existing methods still struggle with two primary challenges in practical applications. Specifically, cross-modal spatial misalignment arising from parallax and distinct imaging mechanisms often induces ghosting artifacts and counting bias during feature fusion. Moreover, the reliability of each modality varies dynamically under fluctuating illumination or thermal interference. To address these issues, this paper introduces a novel framework termed **Reliability-Aware Bidirectional Tri-Attention (RaBiT-CC)**. 

First, a **RaBiT-Fusion** module is designed to alleviate feature misalignment without the need for computationally expensive explicit registration. It creatively employs a small set of **mediator tokens** as cross-modal bridges and incorporates a pixel-level reliability gating mechanism, enabling efficient feature aggregation via alternating bidirectional interactions while dynamically suppressing noise from low-quality modalities. Furthermore, we propose a **Binary Preference Loss (BPL)**. By comparing the counting errors of auxiliary unimodal heads within local windows, BPL constructs dynamic preference labels to explicitly supervise the model, adaptively assigning higher confidence weights to the modality with lower error.

Extensive experimental results on the RGBT-CC and DroneRGBT datasets demonstrate the effectiveness of the proposed modules and the competitive performance of RaBiT-CC in complex scenarios.

## Datasets📚

We evaluate the proposed method on two widely used RGB-T crowd counting datasets: [🔗RGBT-CC](https://github.com/chen-judge/RGBTCrowdCounting) and [🔗Drone-RGBT](https://github.com/VisDrone/DroneRGBT). 

* **RGBT-CC**: Contains 2,030 pairs of RGB-Thermal images (138,389 pedestrians) covering diverse scenes and lighting conditions.
* **Drone-RGBT**: An aerial dataset captured by drones, comprising 3,607 image pairs with 175,698 annotated instances.

## Experimental Results🏆

**Table 1. Comparison with other state-of-the-art methods on the RGBT-CC dataset.**
| Method         | Venue    | Backbone        | GAME(0) | GAME(1) | GAME(2) | GAME(3) | RMSE  |
|----------------|----------|-----------------|---------:|--------:|--------:|--------:|------:|
| MMCCN          | ACCV'20  | ResNet-50       |   21.89 |   25.70 |   30.22 |   37.19 | 37.44 |
| CSRNet         | CVPR'18  | VGG-16          |   20.40 |   23.58 |   28.03 |   35.51 | 35.26 |
| BL+IADM        | CVPR'21  | VGG-19          |   15.61 |   19.95 |   24.69 |   32.89 | 28.18 |
| DEFNet         | ITS'22   | VGG-16          |   11.90 |   16.08 |   20.19 |   27.27 | 21.09 |
| MC3Net         | ITS'23   | ConvNext        |   11.47 |   15.06 |   19.40 |   27.95 | 20.59 |
| CGINet         | EAAI'23  | ConvNext        |   12.07 |   15.98 |   20.06 |   27.73 | 20.54 |
| GETANet        | GRSL'24  | PVT             |   12.14 |   15.98 |   19.40 |   28.61 | 22.17 |
| DAACFNet       | SSRN'24  | -               |   11.36 |   15.55 |   20.37 |   30.51 | 21.45 |
| CSCA           | PR'25    | VGG-19          |   13.50 |   18.63 |   23.59 |   31.59 | 24.01 |
| CMFX           | NN'25    | VGG-19          |   11.25 |   15.33 |   19.62 |   26.14 | 19.38 |
| **RaBiT-CC (Ours)** | -   | **PVT** | **10.70**| **14.98**| **19.06**| **26.41**| **18.62**|

**Table 2. Comparison with other state-of-the-art methods on the Drone-RGBT dataset.**
| Method        | Venue      | Backbone            | GAME(0) | GAME(1) | GAME(2) | GAME(3) | RMSE  |
|---------------|------------|---------------------|--------:|--------:|--------:|--------:|------:|
| CSRNet        | CVPR 2018  | VGG-16              |    20.45|   26.57 |   35.57 |   46.65 | 27.30 |
| MMCCN         | ACCV 2020  | ResNet-50           |    9.99 |   12.73 |   17.63 |   28.16 | 16.29 |
| BL+IADM       | CVPR 2021  | VGG-19              |    9.77 |   12.91 |   17.08 |   22.61 | 15.76 |
| MC3Net        | TITS 2023  | ConvNext            |    7.63 |    9.87 |   13.64 |   19.44 | 11.17 |
| CGINet        | EAAI 2023  | ConvNext            |    8.37 |    9.97 |   12.34 |   15.51 | 13.45 |
| GETANet       | GRSL 2024  | PVT                 |    8.44 |   10.01 |   12.75 |   15.83 | 13.99 |
| DAACFNet      | SSRN 2024  | -                   |    8.33 |      -  |      -  |      -  | 13.67 |
| CSCA          | PR 2025    | VGG-19              |    9.51 |   12.12 |   15.84 |   21.57 | 15.19 |
| CMFX          | NN 2025    | VGG-19              |    6.75 |    8.88 |   11.87 |   14.69 | 11.05 |
| **RaBiT-CC (Ours)** | -    | **PVT** | **5.68**| **7.22**| **9.32**| **12.70**| **9.07** |

## Getting Started🚀

### 1. Data Preparation

#### (1) RGBT-CC Dataset
Download from [🔗here](https://github.com/chen-judge/RGBTCrowdCounting). Use `preprocess_RGBTCC.py` to organize the data:
```text
RGBT_CC
├── train
│   ├── 1162_RGB.jpg
│   ├── 1162_T.jpg
│   ├── 1162_GT.npy
│   ├── ...
├── val
│   ├── ...
├── test
│   ├── ...

```

#### (2) Drone-RGBT Dataset

Download from [🔗here](https://github.com/VisDrone/DroneRGBT). Structure as follows:

```text
Drone_RGBT
├── train
│   ├── GT_
│   ├── RGB
│   ├── Infrared
├── test
│   ├── GT_
│   ├── RGB
│   ├── Infrared

```

### 2. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt

```

Core libraries:

* torch >= 1.11.0
* torchvision >= 0.12.0
* timm == 1.0.12
* mmcv-full == 1.7.2

### 3. Training

We use **PVT-v2-b3** as the backbone. Please ensure you have downloaded the weights to `pretrained_weights/`.

To start training on the RGBT-CC dataset, run:
```bash
python train.py \
    --dataset RGBTCC \
    --data-dir ./data/RGBT-CC \
    --batch-size 2 \
    --lr 1e-5 \
    --device 0 \
    --exp-tag rabbit_rgbtcc

```

## Cite our work📝

If you find this code useful for your research, please cite our paper:

```bibtex
@article{zhou2026rabit,
  title={RaBiT-CC: Reliability-Aware Bidirectional Tri-Attention for RGB-T Crowd Counting},
  author={Zhou, Fengyu and others},
  journal={Submitted to IEEE Signal Processing Letters},
  year={2026}
}

```

## License📜

This source code is licensed for research and education use only. Commercial use is strictly prohibited without prior written permission from the authors.

```

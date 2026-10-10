# Dance Analysis System


---

## Overview

This project builds a complete pipeline for automatic dance genre classification from video:

```
Video → Pose Estimation (skeleton) → Action Recognition → Genre Label
```

**Dataset:** AIST++ — 1,408 sequences across 10 dance genres  
**Best accuracy:** ST-GCN — 98.68%  
**Supported genres:** Break · Pop · Lock · Waack · Mid Hip-hop · LA Hip-hop · House · Krump · Street Jazz · Ballet Jazz

---

## Results Summary

### Pose Estimation — AIST++ (500 frames, 5 genres)

| Model | Architecture | AP | AP50 | FPS (real-world) |
|---|---|---|---|---|
| YOLO26s-pose | Single-stage CNN | 0.939 | 0.988 | **38.2** |
| RTMPose-m | Top-down CNN | 0.964 | 0.988 | 23.7 |
| ViTPose-base | Top-down Transformer | **0.966** | **0.989** | 15.1 |

> RTMPose achieves the best accuracy-speed trade-off. YOLO26s is the fastest for real-time deployment.

### Action Recognition — AIST++ (1,012 train / 228 test windows)

| Model | Accuracy | Architecture |
|---|---|---|
| SVM | 85.96% | Traditional ML |
| Random Forest | 85.96% | Traditional ML |
| LSTM | 91.23% | Recurrent (+7.02% over SVM) |
| **ST-GCN** | **98.68%** | **Graph Conv (+3.51% over LSTM)** |
| MotionBERT | — | Pretrained Transformer |


### Domain Shift — BeatBusterz HipHop Groups (broadcast video, 7 dancers)

| Model | Avg detections/frame | Notes |
|---|---|---|
| YOLO26s-pose | 6.4 | Undercounts — partial occlusions |
| **RTMPose-m** | **7.8** | Closest to true count (7 dancers) |
| ViTPose-base | 6.4 | Undercounts — partial occlusions |

---

## Notebooks

| # | Notebook | Description |
|---|---|---|
| 01 | `01_aist_preprocess.ipynb` | AIST++ .pkl + .mp4 → COCO JSON annotations |
| 02 | `02_pose_estimation.ipynb` | Dual benchmark: accuracy (GT bbox) vs real-world FPS |
| 03 | `03_action_recognition.ipynb` | SVM / RF / LSTM / ST-GCN on 10 genres |
| 04 | `04_action_rec_multistream.ipynb` | Multi-stream: joint-only vs joint+motion |
| 05 | `05_beatbusterz_pose.ipynb` | Multi-person pose on competition video |
| 06 | `06_beatbusterz_wholebody.ipynb` | Wholebody pose (133 keypoints) on competition video |
| 07 | `07_multiperson_demo.ipynb` | Multi-person pose demo on AIST++ sequences |
| 08 | `08_gradio_demo.ipynb` | Interactive web demo: upload video → genre prediction |
| 09 | `09_motionbert.ipynb` | MotionBERT (DSTformer) fine-tuning on AIST++ |

All notebooks run on Google Colab with a T4 GPU. Data is loaded from Google Drive.

---

## Setup

### Requirements
```
torch, ultralytics, rtmlib, transformers, scikit-learn
opencv-python-headless, pycocotools, gradio
```

### Data
- **AIST++** dataset: [https://aistdancedb.ongithub.io](https://aistdancedb.ongithub.io)  
  - Download .pkl keypoint files → Google Drive `AIST_PKL/`
  - Run `01_aist_preprocess.ipynb` to generate COCO annotations

### Models
All models download automatically on first run:
- `yolo26s-pose.pt` — via Ultralytics
- RTMPose-m — via rtmlib (OpenMMLab)
- ViTPose-base — via HuggingFace (`usyd-community/vitpose-base-simple`)
- MotionBERT — via HuggingFace (`walterzhu/MotionBERT`)

---

## Architecture

```
AIST++ PKL files
    │
    ▼
01_aist_preprocess.ipynb
    │  keypoints2d (9 cameras, N frames, 17 joints, 3 values)
    │  → COCO JSON (500 annotated frames)
    ▼
02_pose_estimation.ipynb
    │  YOLO26s / RTMPose-m / ViTPose-base
    │  Benchmark 1 (accuracy) + Benchmark 2 (speed)
    ▼
03_action_recognition.ipynb
    │  Sliding window T=55, stride=10
    │  Normalize: hip center subtraction + [-1,1] scaling
    │  SVM → RF → LSTM → ST-GCN
    ▼
04_action_rec_multistream.ipynb
    │  + motion stream: dx = x[t] - x[t-1]
    ▼
08_gradio_demo.ipynb
       YOLO26s-pose + ST-GCN → web interface
```



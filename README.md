# ML Medical Diagnosis Baseline
Lightweight baseline project for medical image binary diagnosis, built for chest X-ray classification (Normal / Pneumonia).

This repository contains two model lines:
1. Traditional Machine Learning: Random Forest, SVM (handcrafted flattened pixel features)
2. Deep Learning: Simple Lightweight CNN

The pipeline automatically trains models, outputs classification metrics, confusion matrix and ROC curves.

## Dataset
Public Kaggle ChestX-ray Pneumonia Dataset
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
> Note: Only a small subset of images is recommended for demo testing. Raw image folder is excluded via `.gitignore`.

## Quick Start
1. Install dependencies
```bash
pip install -r requirements.txt

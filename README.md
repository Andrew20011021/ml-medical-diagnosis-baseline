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

## Model Visualization Results
### Random Forest
!\[RandomForest Confusion Matrix\](assets/RandomForest\_cm.png)
!\[RandomForest ROC Curve\](assets/RandomForest\_roc.png)

### SVM
!\[SVC Confusion Matrix\](assets/SVC\_cm.png)
!\[SVC ROC Curve\](assets/SVC\_roc.png)

### LightCNN
!\[LightCNN Confusion Matrix\](assets/LightCNN\_cm.png)
!\[LightCNN ROC Curve\](assets/LightCNN\_roc.png)

## Result Discussion
- Random Forest and SVM reach 96% test accuracy based on flattened pixel features, serving as strong traditional ML baselines.
- LightCNN performs poorly on normal case detection due to severe class imbalance: the dataset contains far more pneumonia samples. The model tends to predict pneumonia universally.
- Future improvements: class-balanced sampling, weighted loss function and image augmentation.

# 🏥 ML Medical Diagnosis Baseline

> Lightweight baseline framework for medical image binary classification — specifically designed for chest X-ray diagnosis (Normal / Pneumonia).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Dataset](https://img.shields.io/badge/dataset-Kaggle-20beff.svg)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Model Lines](#model-lines)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Evaluation Outputs](#evaluation-outputs)
- [Results Discussion](#results-discussion)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🎯 Overview

This repository provides a **minimal yet complete baseline** for medical image binary classification tasks. It implements and compares two distinct approaches:

- **Traditional Machine Learning** — Random Forest & SVM using flattened pixel features
- **Deep Learning** — A lightweight CNN architecture

The pipeline automatically trains all models, evaluates performance, and generates classification metrics, confusion matrices, and ROC curves for side-by-side comparison.

---

## 🧠 Model Lines

| Category | Models |
|----------|--------|
| **Traditional ML** | Random Forest, Support Vector Machine (SVM) |
| **Deep Learning** | Lightweight CNN (LightCNN) |

All models are trained and evaluated using the same data split to ensure fair comparison.

---

## 📊 Dataset

**Source**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle

- **Task**: Binary classification (Normal vs. Pneumonia)
- **Note**: Only a small subset is recommended for demo/testing. The full raw image folder is excluded via `.gitignore`.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ml-medical-diagnosis-baseline.git
cd ml-medical-diagnosis-baseline

"""
模型评估、绘图：ROC、混淆矩阵
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def evaluate_and_plot(y_true, y_prob, save_dir, model_name):
    os.makedirs(save_dir, exist_ok=True)
    y_pred = (y_prob > 0.5).astype(int)

    # 分类报告
    report = classification_report(y_true, y_pred, output_dict=True)
    print(f"\n==== {model_name} Report ====")
    print(classification_report(y_true, y_pred))

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.savefig(os.path.join(save_dir, f"{model_name}_cm.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{model_name}_roc.png"), dpi=300, bbox_inches="tight")
    plt.close()
    return report


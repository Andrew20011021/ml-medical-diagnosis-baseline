"""
ML Medical Diagnosis Baseline Main
Chest X-Ray Binary Classification: Normal / Pneumonia
"""
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from src.dataset import load_image_folder, get_sklearn_feature, XrayDataset
from src.model import SimpleCNN, get_ml_models
from src.train import train_cnn
from src.evaluate import evaluate_and_plot

def main():
    data_root = Path("./data/raw")
    out_dir = Path("./output")
    # 1. 加载数据
    imgs, labels = load_image_folder(str(data_root))
    img_train, img_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=42, stratify=labels)

    # ========== 传统机器学习基线 RF / SVM ==========
    X_train_feat = get_sklearn_feature(img_train)
    X_test_feat = get_sklearn_feature(img_test)
    ml_models = get_ml_models()
    for name, clf in ml_models.items():
        clf.fit(X_train_feat, y_train)
        prob = clf.predict_proba(X_test_feat)[:,1]
        evaluate_and_plot(y_test, prob, str(out_dir), name)

    # ========== 轻量CNN模型 ==========
    train_set = XrayDataset(img_train, y_train)
    cnn = SimpleCNN()
    cnn = train_cnn(cnn, train_set, epochs=10)
    cnn.eval()
    import torch
    test_tensor = torch.from_numpy(img_test).unsqueeze(1).float() / 255.0
    with torch.no_grad():
        logits = cnn(test_tensor)
        prob_cnn = torch.sigmoid(logits).squeeze().numpy()
    evaluate_and_plot(y_test, prob_cnn, str(out_dir), "LightCNN")

    print("\n✅ All training & evaluation finished. Results saved to output/")

if __name__ == "__main__":
    main()

"""
模型定义：轻量CNN + 传统机器学习基线
"""
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*32*32, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        out = self.fc(x)
        return out

def get_ml_models():
    """返回传统机器学习模型集合"""
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=80, random_state=42),
        "SVC": SVC(probability=True, random_state=42)
    }
    return models

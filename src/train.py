"""
训练脚本
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_cnn(model, train_dataset, epochs=10, lr=1e-3):
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in loader:
            labels = labels.float().unsqueeze(1)
            pred = model(imgs)
            loss = criterion(pred, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(loader):.4f}")
    return model

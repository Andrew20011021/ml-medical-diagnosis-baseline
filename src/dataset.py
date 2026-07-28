"""
数据集加载与预处理
ChestX-ray胸片二分类：Normal / Pneumonia
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset

IMAGE_SIZE = (128, 128)

def load_image_folder(root_path):
    """读取图片文件夹，返回图像数组与标签"""
    img_list = []
    label_list = []
    class_map = {"NORMAL":0, "PNEUMONIA":1}
    for cls_name, label in class_map.items():
        cls_dir = os.path.join(root_path, cls_name)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg",".png",".jpeg")):
                img = cv2.imread(os.path.join(cls_dir, fname))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, IMAGE_SIZE)
                img_list.append(img)
                label_list.append(label)
    return np.array(img_list), np.array(label_list)

def get_sklearn_feature(imgs):
    """提取展平特征，供传统机器学习（RF/SVM）使用"""
    return imgs.reshape(len(imgs), -1)

class XrayDataset(Dataset):
    """Pytorch CNN数据集"""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

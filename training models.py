import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ===========================
#  CONFIG (imported files and training variables)
# ===========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNST JPGs from GitHub link in project lecture
TRAIN_DIR = r"C:\Users\aleah\Python Learning\CS_Final_Project\MNIST Dataset JPG format\MNIST Dataset JPG format\MNIST - JPG - training"   # <-- change to your training folder
TEST_DIR  = r"C:\Users\aleah\Python Learning\CS_Final_Project\MNIST Dataset JPG format\MNIST Dataset JPG format\MNIST - JPG - testing"    # <-- change to your testing folder

BATCH_SIZE = 128
NUM_EPOCHS = 5        # start small; you can increase later
LEARNING_RATE = 1e-3


# ===========================
#  DATA AUGMENTATION
# ===========================

class AddGaussianNoise(object):
    """
    Adds noise to JPGs
    """

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        noisy = tensor + noise
        return torch.clamp(noisy, 0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """
    Create training & test dataloaders.
    Training loader uses data augmentation; test loader does NOT.
    """

    # Training: rotation + zoom (via RandomAffine) + noise
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # just in case
        transforms.Resize((28, 28)),
        transforms.RandomAffine(
            degrees=10,           # rotate between -10 and +10 degrees
            translate=(0.1, 0.1), # small shifts in x, y
            scale=(0.9, 1.1)      # zoom out / in
        ),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.1),
        transforms.Normalize((0.5,), (0.5,))  # optional normalization
    ])

    # Test: no augmentation, just tensor + normalize
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    test_dataset  = datasets.ImageFolder(root=TEST_DIR,  transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader

# =================================
# Build class for each model
# =================================

'''Model 1: 2 Fully connected layers:
*2 FC layers minimum
input: 1x28x28 (flatten)
FC1: 784 -> 128
nonlinear: ReLU
FC2: 128 -> 10 (AKA 0-9)
nonlinear: ReLU '''
class Model1(nn.Module):
    def __init__(self):
        self.fc1=nn.Linear(784,128)
        self.fc2=nn.Linear(128,10)
        self.nonlin=nn.ReLU()

    def forward(self, x):
        x=self.fc1(x)
        x=self.nonlin(x)
        x=self.fc2(x)
        x=self.nonlin(x)
        return x
    

'''Model2: 3 Fully Connected Layers:
*just adding an extra layer to simply increase accuracy
Input: 1x28x28=784 (flatten)
FC1: 784 -> 256
Nonlinear: ReLU
FC2: 256 -> 128
Nonlinear: ReLU
FC3: 128 -> 10 (AKA 0-9)
Nonlinear: ReLU
'''
class Model2(nn.Module):
    def __init__(self):
        self.fc1=nn.Linear(784,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)
        self.nonlin=nn.ReLU()

def forward(self, x):
    x = self.fc1(x)
    x = self.nonlin(x)
    x = self.fc2(x)
    x = self.nonlin(x)
    x = self.fc3(x)
    x = self.nonlin(x)
    return x 

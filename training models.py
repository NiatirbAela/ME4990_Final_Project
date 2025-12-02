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
NUM_EPOCHS = 8        # start small; you can increase later
LEARNING_RATE = 1e-4


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
       AddGaussianNoise(mean=0.0, std=0.01),  # CHANGE NOISE FACTOR HERE
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
        super().__init__()
        self.fc1=nn.Linear(784,128)
        self.fc2=nn.Linear(128,10)
        self.nonlin=nn.ReLU()

    def forward(self, x):
        x=x.view(x.size(0), -1) #flattening function
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
        super().__init__()
        self.fc1=nn.Linear(784,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)
        self.nonlin=nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1) #flattening function
        x = self.fc1(x)
        x = self.nonlin(x)
        x = self.fc2(x)
        x = self.nonlin(x)
        x = self.fc3(x)
        x = self.nonlin(x)
        return x 

'''Model3: Using 2 Convolution + 2 fully connected layers
input: 1x28x28
conv1: 1x28x28->16x28x28
maxpool(2x2): 16x28x28->16x14x14
conv2: 16x14x14->32x14x14
maxpool(2x2): 32x7x7
flatten: 32x7x7= 1568
fc1: 1568-> 128
non linear: ReLU
fc2: 128 -> 10 
nonlinear: ReLU'''

class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1, 16, 5, 1, 2)
        self.conv2=nn.Conv2d(16, 32, 5, 1, 2)

        self.reduce=nn.MaxPool2d(2,2)

        self.fc1=nn.Linear(1568, 128)
        self.fc2=nn.Linear(128,10)

        self.nonlin=nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.nonlin(x)
        x = self.reduce(x)
        x = self.conv2(x)
        x = self.nonlin(x)
        x = self.reduce(x)
        x = x.view(x.size(0), -1) #flattening function
        x = self.fc1(x)
        x = self.nonlin(x)
        x = self.fc2(x)
        x = self.nonlin(x)
        return x

# =================================
# Training / Evaluation helpers
# =================================

def train_one_epoch(model, loader, optimizer, criterion, epoch: int):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(1, num_batches)
    print(f"  Epoch {epoch}: train loss = {avg_loss:.4f}")
    return avg_loss


def evaluate(model, loader, criterion):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_loss = running_loss / max(1, num_batches)
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(model_cls, model_name: str, train_loader, test_loader):
    print(f"\n=== Training {model_name} ===")
    model = model_cls().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    os.makedirs("saved_models", exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        dt = time.time() - start

        print(f"  Test loss: {test_loss:.4f}, Test acc: {test_acc:.4%}, time: {dt:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join("saved_models", f"{model_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved best {model_name} to {save_path}")

    print(f"Best test accuracy for {model_name}: {best_acc:.4%}")

# =================================
# Main
# =================================

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Sanity check: run a single batch through Model3
    images, labels = next(iter(train_loader))
    images = images.to(DEVICE)
    test_model = Model3().to(DEVICE)
    with torch.no_grad():
        out = test_model(images)
    print("Sanity check - Model3 output shape:", out.shape)  # should be [B, 10]

    # Train each model
    train_model(Model1, "model1_fc2", train_loader, test_loader)
    train_model(Model2, "model2_fc3", train_loader, test_loader)
    train_model(Model3, "model3_cnn", train_loader, test_loader)

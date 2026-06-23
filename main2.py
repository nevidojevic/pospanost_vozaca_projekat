import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import torch.nn as nn
import torch.optim as optim

import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)

# =====================================================
# DEVICE (🔥 NOVO)
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =====================================================
# SEED (🔥 NOVO - reproducibility)
# =====================================================

torch.manual_seed(42)
np.random.seed(42)

# =====================================================
# PARAMETRI
# =====================================================

BATCH_SIZE = 16
EPOCHS = 15
LR = 0.001
USE_SUBSET = False
SUBSET_RATIO = 0.4

# =====================================================
# TRANSFORMACIJE
# =====================================================

train_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),

    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# =====================================================
# DATASET
# =====================================================

train_dataset = datasets.ImageFolder(
    "Drowsy_datset/train",
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    "Drowsy_datset/test",
    transform=test_transform
)

classes = train_dataset.classes

# =====================================================
# DATALOADER
# =====================================================

def get_dataloaders(batch_size):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

# =====================================================
# RESNET MODEL
# =====================================================

class ResNetModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

    def forward(self, x):
        return self.model(x)

# =====================================================
# TRAINING (🔥 IMPROVED)
# =====================================================

def train_model(model, train_loader, epochs, lr):

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )




    train_losses = []

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)



        avg_loss = running_loss / len(train_loader)
        acc = correct / total

        train_losses.append(avg_loss)

        print(f"Epoha {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")

    return train_losses

# =====================================================
# EVALUACIJA (🔥 FIXED DEVICE SAFE)
# =====================================================

def evaluate_model(model, test_loader):

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)

    print("\nAccuracy:", acc)
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.show()

    return acc

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print(len(datasets.ImageFolder(
    "Drowsy_datset/train")))
    print(len(datasets.ImageFolder(
    "Drowsy_datset/test")))
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    print("Klase:", classes)

    model = ResNetModel()
    # model.load_state_dict(
    #     torch.load("drowsiness_resnet.pth",
    #                map_location=torch.device("cpu"))
    # )
    #
    train_model(model, train_loader, EPOCHS, LR)

    torch.save(model.state_dict(), "drowsiness_resnet2.pth")

    evaluate_model(model, test_loader)
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# PARAMETRI
# =========================
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.0005
PATIENCE = 4



# =========================
# TRANSFORMACIJE
# =========================
train_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# =========================
# DATASET
# =========================
full_dataset = datasets.ImageFolder(
    "Drowsy_datset/train",
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    "Drowsy_datset/test",
    transform=test_transform
)

classes = full_dataset.classes

# train/val split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# MODEL
# =========================
class ResNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

model = ResNetModel().to(device)

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# EARLY STOPPING VARIJABLE
# =========================
best_val_acc = 0
patience_counter = 0

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(out, 1)

        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    train_acc = train_correct / train_total

    # ---- VALIDATION ----
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            preds = torch.argmax(out, 1)

            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total

    print(f"Epoha {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # =========================
    # EARLY STOPPING
    # =========================
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "src/resnet_model.pth")
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("Early stopping activated!")
        break

# =========================
# TEST EVALUACIJA
# =========================
model.load_state_dict(torch.load("src/resnet_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)

        out = model(x)
        preds = torch.argmax(out, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

print("\nTEST RESULTS:")
print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=classes))

cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues")
plt.show()
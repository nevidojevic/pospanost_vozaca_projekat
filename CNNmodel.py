import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)

# =====================================================
# PARAMETRI
# =====================================================

BATCH_SIZE = 16
EPOCHS = 15
LR = 0.001
USE_SUBSET = False
SUBSET_RATIO = 0.4

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


train_dataset = datasets.ImageFolder(
    "Drowsy_datset/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    "Drowsy_datset/test",
    transform=transform
)

classes = train_dataset.classes


def make_subset(dataset, ratio=0.4):
    indices = np.random.choice(
        len(dataset),
        int(len(dataset) * ratio),
        replace=False
    )
    return Subset(dataset, indices)

if USE_SUBSET:
    train_dataset = make_subset(train_dataset, SUBSET_RATIO)
    test_dataset = make_subset(test_dataset, SUBSET_RATIO)


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


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x


def train_model(model, train_loader, epochs, lr):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_losses = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:

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


def evaluate_model(model, test_loader):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    print("\nAccuracy:", acc)

    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.show()

    return acc


def sensitivity_analysis():

    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    results = []

    for lr in learning_rates:

        print(f"\nLR = {lr}")

        train_loader, test_loader = get_dataloaders(BATCH_SIZE)

        model = CNN()

        train_model(model, train_loader, EPOCHS, lr)

        acc = evaluate_model(model, test_loader)

        results.append(acc)

    plt.plot(learning_rates, results, marker='o')
    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Accuracy")
    plt.title("Sensitivity Analysis")
    plt.show()


def hyperparameter_search():

    learning_rates = [0.01, 0.001]
    batch_sizes = [8, 16]
    epochs_list = [3, 5]

    best_acc = 0
    best_params = None

    for lr in learning_rates:
        for bs in batch_sizes:
            for ep in epochs_list:

                print(f"\nLR={lr}, BS={bs}, EPOCHS={ep}")

                train_loader, test_loader = get_dataloaders(bs)

                model = CNN()

                train_model(model, train_loader, ep, lr)

                acc = evaluate_model(model, test_loader)

                if acc > best_acc:
                    best_acc = acc
                    best_params = (lr, bs, ep)

    print("\nNAJBOLJI PARAMETRI:")
    print(best_params)
    print("Accuracy:", best_acc)


if __name__ == "__main__":

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    print("Klase:", classes)

    model = CNN()

    train_model(model, train_loader, EPOCHS, LR)
    torch.save(model.state_dict(), "drowsiness_model2.pth")
    evaluate_model(model, test_loader)

    # =========================
    # ANALIZA OSETLJIVOSTI
    # =========================
    # sensitivity_analysis()

    # =========================
    # GRID SEARCH
    # =========================
    # hyperparameter_search()
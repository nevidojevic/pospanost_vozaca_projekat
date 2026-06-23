import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()


def train_one_epoch(model, loader, optimizer, device):

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        outputs = model(x)

        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = torch.argmax(outputs, 1)

        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    return (
        train_loss / len(loader),
        train_correct / train_total
    )


def validate(model, loader, device):

    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            loss = criterion(outputs, y)

            val_loss += loss.item()

            preds = torch.argmax(outputs, 1)

            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    return (
        val_loss / len(loader),
        val_correct / val_total
    )
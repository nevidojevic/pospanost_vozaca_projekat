import torch
import torch.optim as optim
from dataset import get_dataloaders
from model import ResNetModel
from model import CNN
from train import train_one_epoch, validate
from evaluate import evaluate

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 16
EPOCHS = 20
LR = 0.0005
PATIENCE = 4

train_loader,\
val_loader,\
test_loader,\
classes,\
test_dataset = get_dataloaders(BATCH_SIZE)
print(classes)
model = ResNetModel().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

best_val_acc = 0
patience_counter = 0

for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        optimizer,
        device
    )

    val_loss, val_acc = validate(
        model,
        val_loader,
        device
    )

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # EARLY STOPPING

    if val_acc >= best_val_acc:

        best_val_acc = val_acc
        patience_counter = 0

        torch.save(
            model.state_dict(),
            "resnet_model.pth"
        )

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping activated!"
        )

        break


model.load_state_dict(
    torch.load("resnet_model.pth")
)

evaluate(
    model,
    test_loader,
    device,
    classes,
    test_dataset
)
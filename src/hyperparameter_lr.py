import torch
import torch.optim as optim
import pandas as pd

from dataset import get_dataloaders
from model import ResNetModel
from train import train_one_epoch, validate
from evaluate import evaluate_metrics


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

EPOCHS = 20
PATIENCE = 4

learning_rates = [
    0.001,
    0.0005,
    0.0001
]

results = []


for lr in learning_rates:

    print(f"\n==============================")
    print(f"Testing LR = {lr}")
    print(f"==============================")

    train_loader, val_loader, test_loader, classes, test_dataset = get_dataloaders(
        batch_size=16
    )

    model = ResNetModel().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    best_val_acc = 0
    patience_counter = 0

    # =========================
    # TRAINING LOOP
    # =========================
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
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc={train_acc:.4f} | "
            f"Val Acc={val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:

            best_val_acc = val_acc
            patience_counter = 0

            torch.save(
                model.state_dict(),
                "temp_model.pth"
            )

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

    # =========================
    # LOAD BEST MODEL
    # =========================
    model.load_state_dict(
        torch.load("temp_model.pth")
    )

    # =========================
    # EVALUATION
    # =========================
    metrics = evaluate_metrics(
        model,
        test_loader,
        device,
        classes
    )

    # =========================
    # SAVE RESULTS
    # =========================
    results.append([
        lr,
        metrics["accuracy"],
        metrics["recall_drowsy"],
        metrics["recall_natural"],
        metrics["macro_f1"]
    ])

    print("\nFINAL RESULTS:")
    print(f"Accuracy       = {metrics['accuracy']:.4f}")
    print(f"Recall DROWSY  = {metrics['recall_drowsy']:.4f}")
    print(f"Recall NATURAL = {metrics['recall_natural']:.4f}")
    print(f"Macro F1       = {metrics['macro_f1']:.4f}")


# =========================
# SAVE CSV
# =========================
df = pd.DataFrame(
    results,
    columns=[
        "LearningRate",
        "Accuracy",
        "Recall_Drowsy",
        "Recall_Natural",
        "F1"
    ]
)

df.to_csv(
    "learning_rate_results.csv",
    index=False
)

print("\nSaved to learning_rate_results.csv")
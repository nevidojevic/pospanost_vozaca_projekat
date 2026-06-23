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
BATCH_SIZE = 16

results = []

optimizers_to_test = [
    ("Adam", lambda params: optim.Adam(
        params,
        lr=0.0005
    )),

    ("SGD", lambda params: optim.SGD(
        params,
        lr=0.001,
        momentum=0.9
    ))
]


for optimizer_name, optimizer_fn in optimizers_to_test:

    print(f"\n==============================")
    print(f"Testing {optimizer_name}")
    print(f"==============================")

    train_loader, val_loader, test_loader, classes, test_dataset = get_dataloaders(BATCH_SIZE)

    model = ResNetModel().to(device)

    optimizer = optimizer_fn(model.parameters())

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
            f"Epoch {epoch+1} | "
            f"Train Acc={train_acc:.4f} | "
            f"Val Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            patience_counter = 0

            torch.save(
                model.state_dict(),
                "temp_optimizer_model.pth"
            )

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping for {optimizer_name}")
            break

    # =========================
    # LOAD BEST MODEL
    # =========================
    model.load_state_dict(
        torch.load("temp_optimizer_model.pth")
    )

    # =========================
    # EVALUATION (NEW METRICS)
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
        optimizer_name,
        metrics["accuracy"],
        metrics["recall_drowsy"],
        metrics["recall_natural"],
        metrics["f1_drowsy"],
        metrics["f1_natural"],
        metrics["macro_f1"]
    ])

    print("\nFINAL RESULTS:")
    print(f"Accuracy       = {metrics['accuracy']:.4f}")
    print(f"Recall DROWSY  = {metrics['recall_drowsy']:.4f}")
    print(f"Recall NATURAL = {metrics['recall_natural']:.4f}")
    print(f"F1 DROWSY      = {metrics['f1_drowsy']:.4f}")
    print(f"F1 NATURAL     = {metrics['f1_natural']:.4f}")
    print(f"Macro F1       = {metrics['macro_f1']:.4f}")


# =========================
# SAVE CSV
# =========================
df = pd.DataFrame(
    results,
    columns=[
        "Optimizer",
        "Accuracy",
        "Recall_Drowsy",
        "Recall_Natural",
        "F1_Drowsy",
        "F1_Natural",
        "Macro_F1"
    ]
)

df.to_csv(
    "optimizer_comparison.csv",
    index=False
)

print("\nResults saved to optimizer_comparison.csv")
print(df)
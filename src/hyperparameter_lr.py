import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import ResNetModel
from model import CNN
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

    model = CNN().to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9
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
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc={train_acc:.4f} | "
            f"Val Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            patience_counter = 0

            torch.save(
                model.state_dict(),
                "temp_model_cnn.pth"
            )

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

    model.load_state_dict(
        torch.load("temp_model_cnn.pth")
    )


    metrics = evaluate_metrics(
        model,
        test_loader,
        device,
        classes
    )


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
    "learning_rate_results_cnn.csv",
    index=False
)

print("\nSaved to learning_rate_results_cnn.csv")

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

# ---- Accuracy plot ----
plt.figure()
plt.plot(df["LearningRate"], df["Accuracy"], marker="o")
plt.xscale("log")
plt.title("Learning Rate vs Accuracy")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# ---- Recall (DROWSY) ----
plt.figure()
plt.plot(df["LearningRate"], df["Recall_Drowsy"], marker="o")
plt.xscale("log")
plt.title("Learning Rate vs Recall (DROWSY)")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Recall DROWSY")
plt.grid()
plt.show()

# ---- Recall (NATURAL) ----
plt.figure()
plt.plot(df["LearningRate"], df["Recall_Natural"], marker="o")
plt.xscale("log")
plt.title("Learning Rate vs Recall (NATURAL)")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Recall NATURAL")
plt.grid()
plt.show()

# ---- Macro F1 ----
plt.figure()
plt.plot(df["LearningRate"], df["F1"], marker="o")
plt.xscale("log")
plt.title("Learning Rate vs Macro F1")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Macro F1")
plt.grid()
plt.show()
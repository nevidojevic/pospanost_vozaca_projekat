import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

from model import CNN
from model import ResNetModel
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_, _, test_loader, classes, _ = get_dataloaders(batch_size=16)


def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes,
        output_dict=True
    )

    macro_f1 = report["macro avg"]["f1-score"]


    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        auc = None

    cm = confusion_matrix(all_labels, all_preds)

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "auc": auc,
        "cm": cm
    }



cnn = CNN().to(device)
cnn.load_state_dict(torch.load("cnn_model.pth", map_location=device))

resnet = ResNetModel().to(device)
resnet.load_state_dict(torch.load("resnet_model.pth", map_location=device))


cnn_metrics = evaluate(cnn, test_loader)
resnet_metrics = evaluate(resnet, test_loader)


print("\n========================")
print("MODEL COMPARISON")
print("========================")

print("\nCNN:")
print(f"Accuracy : {cnn_metrics['acc']:.4f}")
print(f"Macro F1 : {cnn_metrics['macro_f1']:.4f}")
print(f"AUC      : {cnn_metrics['auc']:.4f}")

print("\nResNet:")
print(f"Accuracy : {resnet_metrics['acc']:.4f}")
print(f"Macro F1 : {resnet_metrics['macro_f1']:.4f}")
print(f"AUC      : {resnet_metrics['auc']:.4f}")

df = pd.DataFrame([
    ["CNN", cnn_metrics["acc"], cnn_metrics["macro_f1"], cnn_metrics["auc"]],
    ["ResNet", resnet_metrics["acc"], resnet_metrics["macro_f1"], resnet_metrics["auc"]],
], columns=["Model", "Accuracy", "Macro_F1", "AUC"])

df.to_csv("model_comparison.csv", index=False)


fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay(cnn_metrics["cm"], display_labels=classes).plot(ax=ax[0], cmap="Blues")
ax[0].set_title("CNN Confusion Matrix")

ConfusionMatrixDisplay(resnet_metrics["cm"], display_labels=classes).plot(ax=ax[1], cmap="Greens")
ax[1].set_title("ResNet Confusion Matrix")

plt.tight_layout()
plt.show()
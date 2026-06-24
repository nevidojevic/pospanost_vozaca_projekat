import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)




def evaluate(model, loader, device, classes, test_dataset):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    wrong_predictions = []

    image_paths = [path for path, _ in test_dataset.samples]
    image_index = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            outputs = model(x)

            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(outputs, 1)

            preds_np = preds.cpu().numpy()
            labels_np = y.numpy()

            all_preds.extend(preds_np)
            all_labels.extend(labels_np)

            # verovatnoća za DROWSY klasu
            all_probs.extend(
                probs[:, 0].cpu().numpy()
            )

            for pred, label in zip(preds_np, labels_np):

                if pred != label:

                    wrong_predictions.append([
                        image_paths[image_index],
                        classes[label],
                        classes[pred]
                    ])

                image_index += 1

    print("\nCLASSIFICATION REPORT:\n")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes
    )

    print(report)




    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    ConfusionMatrixDisplay(
        cm,
        display_labels=classes
    ).plot(cmap="Blues")

    plt.title("Confusion Matrix")
    plt.show()

    # =========================
    # ROC CURVE
    # =========================
    fpr, tpr, thresholds = roc_curve(
        all_labels,
        all_probs,
        pos_label=0
    )

    roc_auc = auc(fpr, tpr)

    print(f"\nROC-AUC: {roc_auc:.4f}")

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()
    plt.grid()

    plt.savefig("results/roc_curve.png")
    plt.show()

    # =========================
    # WRONG PREDICTIONS
    # =========================
    pd.DataFrame(
        wrong_predictions,
        columns=[
            "image",
            "true_label",
            "predicted_label"
        ]
    ).to_csv(
        "results/wrong_predictions.csv",
        index=False
    )



def evaluate_metrics(
        model,
        loader,
        device,
        classes):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            outputs = model(x)

            probs = torch.softmax(
                outputs,
                dim=1
            )

            preds = torch.argmax(
                outputs,
                1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                y.numpy()
            )

            all_probs.extend(
                probs[:, 0].cpu().numpy()
            )

    # =========================
    # ACCURACY
    # =========================
    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    # =========================
    # REPORT
    # =========================
    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes,
        output_dict=True
    )

    recall_drowsy = report["DROWSY"]["recall"]
    recall_natural = report["NATURAL"]["recall"]

    f1_drowsy = report["DROWSY"]["f1-score"]
    f1_natural = report["NATURAL"]["f1-score"]

    macro_f1 = report["macro avg"]["f1-score"]
    macro_recall = report["macro avg"]["recall"]

    # =========================
    # ROC-AUC
    # =========================
    fpr, tpr, thresholds = roc_curve(
        all_labels,
        all_probs,
        pos_label=0
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    return {
        "accuracy": accuracy,
        "recall_drowsy": recall_drowsy,
        "recall_natural": recall_natural,
        "f1_drowsy": f1_drowsy,
        "f1_natural": f1_natural,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "roc_auc": roc_auc
    }
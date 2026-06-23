import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================
# FULL EVALUATION (REPORT + CM + WRONG PREDICTIONS)
# =========================
def evaluate(model, loader, device, classes, test_dataset):

    model.eval()

    all_preds = []
    all_labels = []

    wrong_predictions = []

    image_paths = [path for path, _ in test_dataset.samples]
    image_index = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            outputs = model(x)

            preds = torch.argmax(outputs, 1)

            preds_np = preds.cpu().numpy()
            labels_np = y.numpy()

            all_preds.extend(preds_np)
            all_labels.extend(labels_np)

            for pred, label in zip(preds_np, labels_np):

                if pred != label:

                    wrong_predictions.append([
                        image_paths[image_index],
                        classes[label],
                        classes[pred]
                    ])

                image_index += 1

    # =========================
    # CLASSIFICATION REPORT
    # =========================
    print("\nCLASSIFICATION REPORT:\n")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=classes
    )

    print(report)

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(all_labels, all_preds)

    ConfusionMatrixDisplay(
        cm,
        display_labels=classes
    ).plot(cmap="Blues")

    plt.title("Confusion Matrix")
    plt.show()

    # =========================
    # SAVE WRONG PREDICTIONS
    # =========================
    pd.DataFrame(
        wrong_predictions,
        columns=["image", "true_label", "predicted_label"]
    ).to_csv(
        "wrong_predictions.csv",
        index=False
    )


# =========================
# METRICS (FOR HYPERPARAMETER SEARCH)
# =========================
def evaluate_metrics(model, loader, device, classes=None):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            outputs = model(x)

            preds = torch.argmax(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    # =========================
    # ACCURACY
    # =========================
    accuracy = accuracy_score(all_labels, all_preds)

    # =========================
    # CLASS-WISE METRICS
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

    # =========================
    # MACRO METRICS (IMPORTANT FOR REPORT)
    # =========================
    macro_f1 = report["macro avg"]["f1-score"]
    macro_recall = report["macro avg"]["recall"]

    return {
        "accuracy": accuracy,
        "recall_drowsy": recall_drowsy,
        "recall_natural": recall_natural,
        "f1_drowsy": f1_drowsy,
        "f1_natural": f1_natural,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall
    }
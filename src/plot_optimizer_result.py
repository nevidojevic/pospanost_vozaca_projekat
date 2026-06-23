import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("optimizer_comparison.csv")

x = np.arange(len(df))
width = 0.25

plt.figure(figsize=(8,5))

plt.bar(
    x - width,
    df["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x,
    df["Recall_Drowsy"],
    width,
    label="Recall"
)

plt.bar(
    x + width,
    df["F1"],
    width,
    label="F1"
)

plt.xticks(
    x,
    df["Optimizer"]
)

plt.ylabel("Score")
plt.title("Optimizer Comparison")
plt.legend()

plt.savefig(
    "optimizer_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
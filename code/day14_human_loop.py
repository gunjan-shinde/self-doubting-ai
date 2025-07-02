import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from models.simple_cnn import SimpleCNN
from utils.datasets import get_train_val_datasets

# Paths
OUTPUT_DIR = r"C:\Users\Gunjan\Desktop\Research Paper\Self-Doubting AI\code\outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "day11_self_doubting_predictions.csv")
HUMAN_SIM_CSV = os.path.join(OUTPUT_DIR, "day14_human_simulation.csv")
IMPROVEMENT_PLOT = os.path.join(OUTPUT_DIR, "day14_human_vs_model_accuracy.png")

# Load predictions
df = pd.read_csv(PREDICTIONS_CSV)

# Simulate human fallback for doubtful predictions
# Assume human is always correct (ideal case for upper bound)
df["final_prediction"] = df.apply(
    lambda row: row["label"] if row["doubt"] == 1 else row["predicted"], axis=1
)

# Compute metrics
model_accuracy = accuracy_score(df["label"], df["predicted"]) * 100
final_accuracy = accuracy_score(df["label"], df["final_prediction"]) * 100
human_invoked = df["doubt"].sum()
total = len(df)
doubt_rate = human_invoked / total * 100

# Save results
summary_df = pd.DataFrame(
    {
        "Method": ["Model Only", "Model + Human"],
        "Accuracy": [model_accuracy, final_accuracy],
        "Doubt Rate": [0.0, doubt_rate],
    }
)
summary_df.to_csv(HUMAN_SIM_CSV, index=False)
print(f"✅ Human-in-the-loop results saved: {HUMAN_SIM_CSV}")

# Plot comparison
plt.figure(figsize=(6, 4))
plt.bar(summary_df["Method"], summary_df["Accuracy"], color=["skyblue", "lightgreen"])
plt.title("Model vs Human-in-the-Loop Accuracy")
plt.ylabel("Accuracy (%)")
for i, val in enumerate(summary_df["Accuracy"]):
    plt.text(i, val + 0.3, f"{val:.2f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(IMPROVEMENT_PLOT)
print(f"📊 Plot saved: {IMPROVEMENT_PLOT}")

# Summary
print("\n🔍 Summary:")
print(f"Model Accuracy         : {model_accuracy:.2f}%")
print(f"Final Accuracy (w/ Human): {final_accuracy:.2f}%")
print(f"Doubt Trigger Rate       : {doubt_rate:.2f}%")
print("✅ Day 14 complete: Human-in-the-loop integration done.")

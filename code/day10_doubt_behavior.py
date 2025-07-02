import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from models.simple_cnn import SimpleCNN
from utils.datasets import get_train_val_datasets

# Paths
OUTPUT_DIR = r"C:\Users\Gunjan\Desktop\Research Paper\Self-Doubting AI\code\outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "best_threshold.json")
SAVE_CSV = os.path.join(OUTPUT_DIR, "day10_doubt_error_analysis.csv")
SAVE_PLOT = os.path.join(OUTPUT_DIR, "day10_doubt_catch_bar.png")

# Load threshold
with open(THRESHOLD_PATH, "r") as f:
    threshold = json.load(f)["best_threshold"]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# Load dataset
_, val_dataset = get_train_val_datasets()
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Track results
caught_errors = 0
missed_errors = 0
total_errors = 0
all_rows = []

# Run inference with doubt
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, dim=1)

        for i in range(len(labels)):
            conf = confs[i].item()
            pred = preds[i].item()
            label = labels[i].item()
            is_doubt = conf < threshold
            is_wrong = pred != label

            if is_wrong:
                total_errors += 1
                if is_doubt:
                    caught_errors += 1
                else:
                    missed_errors += 1

            all_rows.append(
                {
                    "True Label": label,
                    "Prediction": pred,
                    "Confidence": round(conf, 4),
                    "Doubt": is_doubt,
                    "Correct": not is_wrong,
                    "Error Caught": is_doubt and is_wrong,
                    "Error Missed": not is_doubt and is_wrong,
                }
            )

# Save results to CSV
pd.DataFrame(all_rows).to_csv(SAVE_CSV, index=False)
print(f"✅ Doubt-Error Analysis saved: {SAVE_CSV}")

# Plot
plt.figure(figsize=(6, 4))
labels = ["Caught by Doubt", "Missed by Confidence"]
values = [caught_errors, missed_errors]
plt.bar(labels, values, color=["green", "red"])
plt.title("Doubt vs Missed Error Analysis")
plt.ylabel("Number of Errors")
for i, v in enumerate(values):
    plt.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
plt.savefig(SAVE_PLOT)
plt.close()
print(f"📊 Bar chart saved: {SAVE_PLOT}")

print("✅ Day 10 doubt-error analysis complete.")

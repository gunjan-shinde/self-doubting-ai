import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from torch.nn import functional as F

from models.simple_cnn import SimpleCNN
from utils.datasets import get_train_val_datasets

# Paths
MODEL_PATH = r"C:\Users\Gunjan\Desktop\Research Paper\Self-Doubting AI\code\outputs\best_model.pth"
OUTPUT_DIR = r"C:\Users\Gunjan\Desktop\Research Paper\Self-Doubting AI\code\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
THRESHOLDS = np.linspace(0.0, 1.0, 101)
TEMPERATURE = 2.0  # For softmax scaling
NOISE_STD = 0.05   # Gaussian noise level

# Load model
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# Load data
_, val_dataset = get_train_val_datasets()
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

all_probs, all_labels = [], []

with torch.no_grad():
    for x, y in val_loader:
        x = x + NOISE_STD * torch.randn_like(x)  # Add Gaussian noise
        logits = model(x)
        logits_scaled = logits / TEMPERATURE      # Temperature scaling
        probs = F.softmax(logits_scaled, dim=1)
        all_probs.append(probs)
        all_labels.append(y)

all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)
confidences, predictions = torch.max(all_probs, dim=1)

# Confusion Matrix at Default Threshold
cm = confusion_matrix(all_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix at Default Threshold")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_default.png"))
plt.close()

# Threshold analysis
accuracies, doubt_rates, precisions, recalls = [], [], [], []
best_acc = 0
best_threshold = 0.0

for t in THRESHOLDS:
    mask = confidences >= t
    filtered_preds = predictions[mask]
    filtered_labels = all_labels[mask]

    if len(filtered_labels) > 0:
        acc = (filtered_preds == filtered_labels).sum().item() / len(filtered_labels)
        prec = precision_score(filtered_labels, filtered_preds, average='macro', zero_division=0)
        rec = recall_score(filtered_labels, filtered_preds, average='macro', zero_division=0)
    else:
        acc, prec, rec = 0, 0, 0

    doubt = 1 - len(filtered_preds) / len(predictions)

    accuracies.append(acc * 100)
    doubt_rates.append(doubt * 100)
    precisions.append(prec * 100)
    recalls.append(rec * 100)

    if acc > best_acc:
        best_acc = acc
        best_threshold = t

# Save threshold analysis to CSV
df = pd.DataFrame({
    "Threshold": THRESHOLDS,
    "Accuracy": accuracies,
    "DoubtRate": doubt_rates,
    "Precision": precisions,
    "Recall": recalls
})
df.to_csv(os.path.join(OUTPUT_DIR, "threshold_metrics.csv"), index=False)

# Save best threshold
with open(os.path.join(OUTPUT_DIR, "best_threshold.json"), "w") as f:
    json.dump({"best_threshold": best_threshold}, f)

# Plot: Accuracy & Doubt Rate
plt.plot(THRESHOLDS, accuracies, label="Accuracy")
plt.plot(THRESHOLDS, doubt_rates, label="Doubt Rate")
plt.xlabel("Confidence Threshold")
plt.ylabel("Percentage (%)")
plt.title("Threshold vs Accuracy & Doubt")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "threshold_accuracy_doubt_plot.png"))
plt.close()

# Histogram of Confidences
plt.hist(confidences.numpy(), bins=50, color='skyblue')
plt.title("Histogram of Prediction Confidences")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "confidence_histogram.png"))
plt.close()

# Precision-Recall vs Threshold
plt.plot(THRESHOLDS, precisions, label="Precision")
plt.plot(THRESHOLDS, recalls, label="Recall")
plt.xlabel("Confidence Threshold")
plt.ylabel("Percentage (%)")
plt.title("Precision & Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_vs_threshold.png"))
plt.close()

print(f"✅ Confusion Matrix saved.")
print(f"✅ Best Threshold saved to JSON.")
print(f"📁 All plots and CSV exported to: {OUTPUT_DIR}")

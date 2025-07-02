import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)

from models.simple_cnn import SimpleCNN
from utils.datasets import get_train_val_datasets

# Paths
OUTPUT_DIR = r"C:\Users\Gunjan\Desktop\Research Paper\Self-Doubting AI\code\outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "best_threshold.json")
RESULT_CSV = os.path.join(OUTPUT_DIR, "threshold_doubt_results.csv")

# Load best threshold
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


# Predict with doubt masking
def predict_with_doubt(model, dataloader, threshold, noise_level=0.0):
    all_preds, all_labels, all_confidences = [], [], []
    doubt_count = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if noise_level > 0:
                noise = torch.randn_like(inputs) * noise_level
                inputs = torch.clamp(inputs + noise, 0, 1)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(len(labels)):
                if confs[i] < threshold:
                    all_preds.append(-1)
                    doubt_count += 1
                else:
                    all_preds.append(preds[i].item())
                all_labels.append(labels[i].item())
                all_confidences.append(confs[i].item())

    correct = sum((p == l) for p, l in zip(all_preds, all_labels) if p != -1)
    total = sum(p != -1 for p in all_preds)
    accuracy = correct / total * 100 if total > 0 else 0
    doubt_rate = doubt_count / len(all_labels) * 100
    return accuracy, doubt_rate, all_preds, all_labels, all_confidences


# Run on clean and noisy data
clean_acc, clean_doubt, _, _, _ = predict_with_doubt(model, val_loader, threshold)
noisy_acc, noisy_doubt, _, _, _ = predict_with_doubt(
    model, val_loader, threshold, noise_level=0.2
)

# Save results
results = pd.DataFrame(
    {
        "Setting": ["Clean", "Noisy (20%)"],
        "Accuracy": [clean_acc, noisy_acc],
        "Doubt Rate": [clean_doubt, noisy_doubt],
    }
)
results.to_csv(RESULT_CSV, index=False)
print(f"📁 Results saved to: {RESULT_CSV}")

# Confusion Matrix at default threshold
_, _, preds_all, labels_all, _ = predict_with_doubt(model, val_loader, threshold)
y_true = [l for p, l in zip(preds_all, labels_all) if p != -1]
y_pred = [p for p in preds_all if p != -1]
if y_pred:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (no doubt)")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    print("✅ Confusion Matrix saved.")

# Precision-Recall vs Threshold
probs_all = []
labels_all = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        probs_all.extend(max_probs.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

prec, rec, thresh = precision_recall_curve(labels_all, probs_all, pos_label=1)
plt.figure()
plt.plot(thresh, prec[:-1], label="Precision")
plt.plot(thresh, rec[:-1], label="Recall")
plt.xlabel("Confidence Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "precision_recall_vs_threshold.png"))
plt.close()
print("📊 Precision-Recall vs Threshold plot saved.")


# Calibration (Reliability Diagram + ECE)
def reliability_diagram(confidences, predictions, labels, bins=10):
    bins = np.linspace(0.0, 1.0, bins + 1)
    accuracies = np.zeros(len(bins) - 1)
    confidences_bin = np.zeros(len(bins) - 1)
    counts = np.zeros(len(bins) - 1)

    for c, p, l in zip(confidences, predictions, labels):
        if p == -1:
            continue
        bin_idx = np.digitize(c, bins) - 1
        if bin_idx >= len(bins) - 1:
            bin_idx = len(bins) - 2
        confidences_bin[bin_idx] += c
        accuracies[bin_idx] += int(p == l)
        counts[bin_idx] += 1

    nonzero = counts > 0
    avg_conf = confidences_bin[nonzero] / counts[nonzero]
    avg_acc = accuracies[nonzero] / counts[nonzero]

    plt.figure(figsize=(6, 6))
    plt.plot(avg_conf, avg_acc, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "reliability_diagram.png"))
    plt.close()
    print("✅ Reliability diagram saved.")

    ece = np.sum(np.abs(avg_conf - avg_acc) * counts[nonzero] / np.sum(counts))
    with open(os.path.join(OUTPUT_DIR, "ece_score.txt"), "w") as f:
        f.write(f"Expected Calibration Error (ECE): {ece:.4f}\n")
    print(f"📄 ECE saved: {ece:.4f}")


_, _, preds, labels, confs = predict_with_doubt(model, val_loader, threshold)
reliability_diagram(confs, preds, labels)

print("Advanced doubt injection complete.")

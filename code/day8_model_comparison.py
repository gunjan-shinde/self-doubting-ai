import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.simple_cnn import SimpleCNN
from models.tiny_cnn import TinyCNN
from utils.datasets import get_train_val_datasets

# Paths
OUTPUT_DIR = r"C:\Users\Gunjan\Desktop\Research Paper\Self-Doubting AI\code\outputs"
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "best_threshold.json")
COMPARISON_CSV = os.path.join(OUTPUT_DIR, "model_comparison.csv")
BEST_MODEL_JSON = os.path.join(OUTPUT_DIR, "best_model_comparison.json")

# Load threshold
with open(THRESHOLD_PATH, "r") as f:
    threshold = json.load(f)["best_threshold"]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
_, val_dataset = get_train_val_datasets()
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Evaluation function
def evaluate_model(model, dataloader, threshold, noise_level=0.0):
    model.eval()
    all_preds, all_labels, all_confs = [], [], []
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
                all_confs.append(confs[i].item())

    correct = sum((p == l) for p, l in zip(all_preds, all_labels) if p != -1)
    total = sum(p != -1 for p in all_preds)
    accuracy = correct / total * 100 if total > 0 else 0
    doubt_rate = doubt_count / len(all_labels) * 100
    avg_conf = np.mean(all_confs)

    return accuracy, doubt_rate, avg_conf, all_confs


# Initialize models
models = {"SimpleCNN": SimpleCNN().to(device), "TinyCNN": TinyCNN().to(device)}

# Load pretrained weights if available
if os.path.exists(os.path.join(OUTPUT_DIR, "best_model.pth")):
    models["SimpleCNN"].load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=device)
    )

# Run comparisons
rows = []
all_hist = {}

for name, model in models.items():
    clean_acc, clean_doubt, clean_conf, clean_all = evaluate_model(
        model, val_loader, threshold
    )
    noisy_acc, noisy_doubt, noisy_conf, noisy_all = evaluate_model(
        model, val_loader, threshold, noise_level=0.2
    )

    rows.append(
        {
            "Model": name,
            "Accuracy (Clean %)": clean_acc,
            "Doubt Rate (Clean %)": clean_doubt,
            "Avg Confidence (Clean)": clean_conf,
            "Accuracy (Noisy %)": noisy_acc,
            "Doubt Rate (Noisy %)": noisy_doubt,
            "Avg Confidence (Noisy)": noisy_conf,
        }
    )

    all_hist[name] = clean_all

# Save CSV
pd.DataFrame(rows).to_csv(COMPARISON_CSV, index=False)

# Save best model name
best_model = max(
    rows, key=lambda x: x["Accuracy (Clean %)"] - x["Doubt Rate (Clean %)"]
)
with open(BEST_MODEL_JSON, "w") as f:
    json.dump({"best_model": best_model["Model"]}, f, indent=2)

# Plot histogram
plt.figure(figsize=(10, 5))
for name in all_hist:
    plt.hist(all_hist[name], bins=20, alpha=0.5, label=name)
plt.title("Confidence Histogram by Model")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "model_confidence_histogram.png"))
plt.close()

# Bar plot for comparison
df = pd.DataFrame(rows)
ax = df.plot(
    x="Model",
    y=["Accuracy (Clean %)", "Doubt Rate (Clean %)"],
    kind="bar",
    figsize=(8, 5),
)
plt.title("Model Comparison: Accuracy vs Doubt")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_bar.png"))
plt.close()

print("✅ Model comparison complete. CSV, histogram, and plots saved.")

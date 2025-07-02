import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.simple_cnn import SimpleCNN
from utils.datasets import get_train_val_datasets

# === Paths ===
OUTPUT_DIR = r"C:\Users\Gunjan\Desktop\Research Paper\Self-Doubting AI\code\outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
THRESHOLD_PATH = os.path.join(OUTPUT_DIR, "best_threshold.json")

# === Load threshold ===
with open(THRESHOLD_PATH, "r") as f:
    best_threshold = json.load(f)["best_threshold"]

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# === Load model ===
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === Load dataset ===
_, val_dataset = get_train_val_datasets()
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# === Self-Doubting Predict Function ===
def self_doubting_predict(model, dataloader, threshold):
    all_preds, all_labels, all_confs, doubt_flags = [], [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(len(labels)):
                is_doubt = confs[i].item() < threshold
                all_preds.append(-1 if is_doubt else preds[i].item())
                all_labels.append(labels[i].item())
                all_confs.append(confs[i].item())
                doubt_flags.append(is_doubt)

    return all_preds, all_labels, all_confs, doubt_flags


# === Run and Save ===
preds, labels, confs, doubts = self_doubting_predict(model, val_loader, best_threshold)

# === Save results ===
import pandas as pd

results_df = pd.DataFrame(
    {"label": labels, "predicted": preds, "confidence": confs, "doubt": doubts}
)

save_path = os.path.join(OUTPUT_DIR, "day11_self_doubting_predictions.csv")
results_df.to_csv(save_path, index=False)
print(f"✅ Self-doubting predictions saved to: {save_path}")

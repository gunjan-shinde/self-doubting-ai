import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

from models.simple_cnn import SimpleCNN
from utils.datasets import get_train_val_datasets

# === Paths ===
OUTPUT_DIR = r"C:\Users\Gunjan\Desktop\Research Paper\Self-Doubting AI\code\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
ANTICIPATOR_CSV = os.path.join(OUTPUT_DIR, "day13_doubt_anticipator_data.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "day13_anticipator_report.txt")
CONF_MATRIX_PATH = os.path.join(OUTPUT_DIR, "day13_anticipator_confusion_matrix.png")

# === Load dataset ===
_, val_dataset = get_train_val_datasets()
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# === Load trained model ===
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# === Feature extractor ===
def extract_features(model, dataloader):
    features, labels, doubt_flags = [], [], []
    threshold = 0.9  # Use best_threshold.json if available

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(len(inputs)):
                # Flatten the input image to a feature vector
                feat = inputs[i].cpu().flatten().numpy()
                features.append(feat)
                labels.append(targets[i].item())
                doubt_flags.append(
                    int(
                        (confs[i] < threshold) or (preds[i].item() != targets[i].item())
                    )
                )

    return np.array(features), np.array(doubt_flags)


# === Extract features & labels ===
print("📥 Extracting features...")
X, y = extract_features(model, val_loader)

# === Train doubt anticipator ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
y_pred = clf.predict(X)

# === Save CSV for analysis ===
df = pd.DataFrame(X)
df["doubt_or_error"] = y
df.to_csv(ANTICIPATOR_CSV, index=False)
print(f"📁 Anticipator dataset saved: {ANTICIPATOR_CSV}")

# === Evaluation ===
print("\n🔍 Classification Report:\n")
report = classification_report(y, y_pred)
print(report)
with open(REPORT_PATH, "w") as f:
    f.write(report)
print(f"📄 Report saved: {REPORT_PATH}")

# === Confusion matrix ===
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("🔮 Doubt Anticipator Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
plt.close()
print(f"✅ Confusion matrix saved: {CONF_MATRIX_PATH}")

# === ROC AUC Score ===
auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
print(f"🎯 ROC AUC Score: {auc:.4f}")

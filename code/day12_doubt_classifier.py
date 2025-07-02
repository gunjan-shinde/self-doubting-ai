import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# === Paths ===
OUTPUT_DIR = r"C:\Users\Gunjan\Desktop\Research Paper\Self-Doubting AI\code\outputs"
INPUT_CSV = os.path.join(OUTPUT_DIR, "day11_self_doubting_predictions.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "day12_doubt_classifier_data.csv")
HEATMAP_PATH = os.path.join(OUTPUT_DIR, "day12_doubt_feature_correlation.png")
CONF_MATRIX_PATH = os.path.join(OUTPUT_DIR, "day12_doubt_confusion_matrix.png")
CLASS_REPORT_PATH = os.path.join(OUTPUT_DIR, "day12_classification_report.txt")

# === Load and Prepare Data ===
df = pd.read_csv(INPUT_CSV)

# Correct column names based on your CSV
# Columns: 'label', 'predicted', 'confidence', 'doubt'
df["doubt_or_error"] = ((df["predicted"] == -1) | (df["predicted"] != df["label"])).astype(int)
df["is_doubt"] = (df["predicted"] == -1).astype(int)
df["is_error"] = ((df["predicted"] != -1) & (df["predicted"] != df["label"])).astype(int)

# Save updated version
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Doubt classifier dataset saved to: {OUTPUT_CSV}")

# === Correlation Heatmap ===
plt.figure(figsize=(8, 6))
sns.heatmap(df[["confidence", "doubt", "doubt_or_error", "is_doubt", "is_error"]].corr(), annot=True, cmap="coolwarm")
plt.title("🔍 Feature Correlation with Doubt/Error")
plt.tight_layout()
plt.savefig(HEATMAP_PATH)
plt.close()
print(f"📊 Heatmap saved to: {HEATMAP_PATH}")

# === Train Classifier ===
features = ["confidence", "doubt"]
target = "doubt_or_error"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# === Evaluation ===
print("\n🔍 Classification Report:\n")
report = classification_report(y_test, y_pred)
print(report)

with open(CLASS_REPORT_PATH, "w") as f:
    f.write(report)
print(f"📄 Classification report saved: {CLASS_REPORT_PATH}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("📌 Confusion Matrix: Doubt/Error Classifier")
plt.savefig(CONF_MATRIX_PATH)
plt.close()
print(f"✅ Confusion matrix saved: {CONF_MATRIX_PATH}")

# AUC Score (optional)
try:
    y_probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)
    print(f"🎯 ROC AUC Score: {auc:.4f}")
except:
    print("⚠️ AUC Score could not be computed (single class case?)")

print("✅ Day 12: Doubt Classifier completed.")

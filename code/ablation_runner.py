import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score
import os

# Setup
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOUBT_THRESHOLD = 0.65

# Load test set
df = pd.read_csv("data/test.csv")  # ✅ Must contain "text" and "label" columns


def load_model(path):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate(model, name):
    predictions, confidences, is_doubtful = [], [], []

    for _, row in df.iterrows():
        inputs = tokenizer(
            row["text"], return_tensors="pt", truncation=True, padding=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        predictions.append(pred.item())
        confidences.append(conf.item())
        is_doubtful.append(conf.item() < DOUBT_THRESHOLD)

    acc = accuracy_score(df["label"], predictions)
    avg_conf = sum(confidences) / len(confidences)
    doubt_rate = sum(is_doubtful) / len(is_doubtful)

    print(f"🔍 {name} Model:")
    print(f"   Accuracy        : {acc*100:.2f}%")
    print(f"   Avg Confidence  : {avg_conf:.2f}")
    print(f"   Doubt Frequency : {doubt_rate*100:.2f}%\n")

    return {
        "Model": name,
        "Accuracy": round(acc * 100, 2),
        "Avg Confidence": round(avg_conf, 2),
        "% Doubtful": round(doubt_rate * 100, 2),
    }


# Run both models
standard_model = load_model("outputs/baseline_model.pth")  # from day 3-4
curriculum_model = load_model("outputs/curriculum_model.pth")  # from day 15

standard_metrics = evaluate(standard_model, "Standard")
curriculum_metrics = evaluate(curriculum_model, "Curriculum")

# Save to CSV
pd.DataFrame([standard_metrics, curriculum_metrics]).to_csv(
    "outputs/ablation_results.csv", index=False
)
print("✅ Ablation results saved to outputs/ablation_results.csv")

import torch
import pandas as pd
import random
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Load model
def load_model(path):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Add noise to a sentence
def add_noise(text):
    words = text.split()
    if len(words) < 2:
        return text
    noisy = words.copy()

    # Random word swap
    i, j = random.sample(range(len(words)), 2)
    noisy[i], noisy[j] = noisy[j], noisy[i]

    # Add typo to one word
    k = random.randint(0, len(words) - 1)
    if len(words[k]) > 2:
        word = list(words[k])
        word[1] = random.choice("abcdefghijklmnopqrstuvwxyz")
        noisy[k] = "".join(word)

    return " ".join(noisy)


# Evaluate model
def evaluate(model, df):
    preds = []
    for text in df["text"]:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs).item()
            preds.append(pred)
    return preds


# Load clean test set
df = pd.read_csv("data/test.csv")

# Generate noisy version
df_noisy = df.copy()
df_noisy["text"] = df_noisy["text"].apply(add_noise)
df_noisy.to_csv("data/test_noisy.csv", index=False)

# Load models
baseline = load_model("outputs/baseline_model.pth")
curriculum = load_model("outputs/curriculum_model.pth")

# Evaluate
baseline_clean_preds = evaluate(baseline, df)
baseline_noisy_preds = evaluate(baseline, df_noisy)
curr_clean_preds = evaluate(curriculum, df)
curr_noisy_preds = evaluate(curriculum, df_noisy)

# Accuracy
results = {
    "Model": ["Baseline", "Baseline", "Curriculum", "Curriculum"],
    "Condition": ["Clean", "Noisy", "Clean", "Noisy"],
    "Accuracy": [
        accuracy_score(df["label"], baseline_clean_preds) * 100,
        accuracy_score(df["label"], baseline_noisy_preds) * 100,
        accuracy_score(df["label"], curr_clean_preds) * 100,
        accuracy_score(df["label"], curr_noisy_preds) * 100,
    ],
}

# Save results
robust_df = pd.DataFrame(results)
robust_df.to_csv("outputs/robustness_results.csv", index=False)
print("✅ Robustness results saved to outputs/robustness_results.csv")

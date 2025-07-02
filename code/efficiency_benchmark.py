import torch
import time
import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load small test set for timing
sample_texts = [
    "I love mangoes",
    "I hate traffic",
    "This is okay",
    "I feel great",
    "Worst day ever",
    "So exciting",
    "Could be better",
    "Amazing experience",
    "Not my thing",
    "I am thrilled",
]


def load_model(path):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def measure_inference_time(model, inputs, n=10):
    total_time = 0
    with torch.no_grad():
        for _ in range(n):
            start = time.time()
            for text in inputs:
                encoded = tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                _ = model(**encoded)
            end = time.time()
            total_time += end - start
    return total_time / (n * len(inputs))  # time per sample


def get_model_size(path):
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)  # in MB


# Paths
baseline_path = "outputs/baseline_model.pth"
curriculum_path = "outputs/curriculum_model.pth"

# Benchmarking
results = []

for name, path in [("Baseline", baseline_path), ("Curriculum", curriculum_path)]:
    model = load_model(path)
    time_per_sample = measure_inference_time(model, sample_texts)
    model_size = get_model_size(path)

    results.append(
        {
            "Model": name,
            "Size (MB)": round(model_size, 2),
            "Time per Sample (s)": round(time_per_sample, 4),
            "Inference Speed (samples/sec)": round(1 / time_per_sample, 2),
        }
    )

# Save
df = pd.DataFrame(results)
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/efficiency_report.csv", index=False)
print("✅ Efficiency results saved to outputs/efficiency_report.csv")
print(df)

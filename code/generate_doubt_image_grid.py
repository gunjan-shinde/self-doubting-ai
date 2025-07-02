import torch
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import os

# Load trained model
model_path = "outputs/curriculum_model.pth"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Doubt threshold
DOUBT_THRESHOLD = 0.65

# Test phrases
test_texts = [
    "I love this!",
    "What a disaster...",
    "I like mangoes",
    "This is okay I guess",
    "Perfect, another rainy day",
    "I'm so happy",
    "This is the worst",
    "Yay, more meetings",
    "Could be better",
    "I am thrilled"
]

# Prediction helper
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
    return prediction.item(), confidence.item()

# Create grid
fig, axes = plt.subplots(2, 5, figsize=(18, 6))
fig.suptitle("🧠 Self-Doubt AI — Prediction Confidence Grid", fontsize=18)

for idx, text in enumerate(test_texts):
    row, col = divmod(idx, 5)
    pred, conf = predict(text)
    color = "green" if conf > DOUBT_THRESHOLD else "orange"
    label = "Positive" if pred == 1 else "Negative"
    doubt = "✅ Confident" if conf > DOUBT_THRESHOLD else "⚠️ Doubtful"

    axes[row, col].text(0.5, 0.5, f'"{text}"\n\n→ {label}\nConfidence: {conf:.2f}\n{doubt}',
                        fontsize=11, ha='center', va='center', color=color)
    axes[row, col].set_xticks([])
    axes[row, col].set_yticks([])
    axes[row, col].set_frame_on(False)

plt.tight_layout()
plt.subplots_adjust(top=0.8)

# Save image
output_path = "outputs/doubt_image_grid.png"
os.makedirs("outputs", exist_ok=True)
plt.savefig(output_path)
print(f"✅ Saved doubt visualization to {output_path}")
plt.show()

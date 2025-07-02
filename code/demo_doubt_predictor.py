# code/demo_doubt_predictor.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

CONFIDENCE_THRESHOLD = 0.5


def load_model(model_path="outputs/curriculum_model.pth"):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def predict_with_doubt(model, text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).squeeze()
    predicted_class = torch.argmax(probs).item()
    confidence = probs[predicted_class].item()
    doubt = confidence < CONFIDENCE_THRESHOLD
    return predicted_class, confidence, doubt


if __name__ == "__main__":
    model = load_model()
    while True:
        text = input("📝 Enter your text (or type 'exit'): ")
        if text.lower() == "exit":
            break
        label, confidence, doubt = predict_with_doubt(model, text)
        label_name = "Positive" if label == 1 else "Negative"
        print(f"\nPrediction: {label_name}")
        print(f"Confidence: {confidence:.2f}")
        print("⚠️ Doubtful Prediction\n" if doubt else "✅ Confident Prediction\n")

# modules/doubt_detector.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Custom modules
from modules.emotion_layer import emotional_response
from modules.counterfactual_engine import generate_counterfactual  # Optional module

class DoubtDetector:
    def __init__(self, model_name="typeform/distilbert-base-uncased-mnli"):
        self.label_map = {
            "CONTRADICTION": "DOUBTFUL",
            "NEUTRAL": "NEUTRAL",
            "ENTAILMENT": "CONFIDENT"
        }
        self.high_threshold = 0.80
        self.low_threshold = 0.50

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def detect(self, question):
        premise = "I know the answer to this question."
        hypothesis = question

        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()

        id2label = self.model.config.id2label
        prediction_scores = {self.label_map[id2label[i]]: probs[i] for i in range(len(probs))}
        top_label = max(prediction_scores, key=prediction_scores.get)
        confidence = prediction_scores[top_label]

        # Explanation logic
        if top_label == "CONFIDENT" and confidence >= self.high_threshold:
            explanation = "I'm confident in my answer."
        elif top_label == "DOUBTFUL" or confidence <= self.low_threshold:
            explanation = "I'm not confident. I might be wrong or lack enough information."
        else:
            explanation = "I feel uncertain. My answer might need more context."

        # Counterfactual explanation (only for doubtful/neutral)
        counterfactual = ""
        if top_label != "CONFIDENT":
            counterfactual = generate_counterfactual(question)

        return {
            "label": top_label,
            "confidence_score": round(confidence, 3),
            "explanation": explanation,
            "emotion": emotional_response(top_label),
            "counterfactual": counterfactual,
            "all_scores": {k: round(v, 3) for k, v in prediction_scores.items()}
        }


# 🧪 Terminal interface for testing
if __name__ == "__main__":
    print("🤖 SelfDoubt.AI Terminal\nType 'exit' to quit.")
    detector = DoubtDetector()

    while True:
        user_input = input("\nAsk something (or type 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = detector.detect(user_input)
        print("\n🧠 SelfDoubt.AI says:")
        print(f"→ Label: {result['label']}")
        print(f"→ Confidence Score: {result['confidence_score']}")
        print(f"→ Explanation: {result['explanation']}")
        print(f"→ Emotion Response: {result['emotion']}")
        if result["counterfactual"]:
            print(f"→ Counterfactual Reflection: {result['counterfactual']}")
        print(f"→ All Scores: {result['all_scores']}")

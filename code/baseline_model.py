import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os

class TextDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": int(self.labels[idx])
        }

def train_baseline_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

    dataset = TextDataset("data/final_cleaned_dataset.csv")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    print("🧪 Training baseline model on full dataset...")
    model.train()
    for epoch in range(3):
        print(f"Epoch {epoch+1}/3")
        for batch in tqdm(dataloader, desc="Training"):
            inputs = tokenizer(batch["text"], return_tensors="pt", truncation=True, padding=True, max_length=128)
            labels = torch.tensor(batch["label"]).to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/baseline_model.pth")
    print("✅ Baseline model saved at outputs/baseline_model.pth")

if __name__ == "__main__":
    train_baseline_model()

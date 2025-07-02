import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import os

# Dataset class
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
            "label": int(self.labels[idx])  # ✅ Ensure label is int
        }

# Phase-wise training loop
def train_on_phase(model, tokenizer, dataloader, optimizer, criterion, device, phase):
    model.train()
    print(f"\n📚 Training Phase: {phase}")
    for epoch in range(3):
        print(f"  🔁 Epoch {epoch+1}/3")
        for batch in tqdm(dataloader, desc=f"    Batch Progress"):
            inputs = tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            labels = torch.tensor(batch["label"]).to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

# Main training process
def run_curriculum_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    for phase in ["easy", "medium", "hard"]:
        dataset = TextDataset(f"data/{phase}.csv")
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        train_on_phase(model, tokenizer, dataloader, optimizer, criterion, device, phase)

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/curriculum_model.pth")
    print("✅ Saved curriculum-trained model to outputs/curriculum_model.pth")

if __name__ == "__main__":
    run_curriculum_training()

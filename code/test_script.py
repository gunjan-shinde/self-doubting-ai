import torch
import time

print("🚀 Test script started")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Dummy model and data
model = torch.nn.Linear(10, 2).to(device)
inputs = torch.randn(64, 10).to(device)
labels = torch.randint(0, 2, (64,)).to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("🧠 Starting dummy training loop...")

for epoch in range(3):
    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f"✅ Epoch [{epoch + 1}/3] - Loss: {loss.item():.4f}")
    time.sleep(1)

print("🎉 Test completed successfully!")

import torch
import json
import os


def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def save_results(metrics_dict, path="results/baseline_metrics.json"):
    os.makedirs("results", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

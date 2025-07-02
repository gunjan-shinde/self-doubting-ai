import torch
import json
import os
import matplotlib.pyplot as plt


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


def compute_accuracy_with_doubt(model, dataloader, device, threshold=0.7):
    model.eval()
    correct = 0
    total = 0
    doubtful = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            # Mark "doubtful" predictions
            is_confident = max_probs >= threshold
            confident_preds = preds[is_confident]
            confident_labels = labels[is_confident]

            correct += (confident_preds == confident_labels).sum().item()
            total += is_confident.sum().item()
            doubtful += (~is_confident).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    doubt_rate = 100 * doubtful / (total + doubtful)
    return accuracy, doubt_rate


def save_results_and_plot(metrics_dict, results_dir=None):
    if results_dir is None:
        results_dir = r"C:\Users\Gunjan\Desktop\Research Paper\Self-Doubting AI\code\results"
    os.makedirs(results_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(results_dir, "baseline_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✅ Saved results to: {json_path}")

    # Plot accuracy over epochs
    plt.figure()
    plt.plot(metrics_dict["test_accuracies"], label="Standard Accuracy")
    plt.axhline(metrics_dict["doubt_aware_accuracy"], color='r', linestyle='--', label="Doubt-aware Accuracy")
    plt.title("Test Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plot_path = os.path.join(results_dir, "baseline_accuracy_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saved accuracy plot to: {plot_path}")

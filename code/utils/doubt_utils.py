# code/utils/doubt_utils.py

import torch
import torch.nn.functional as F


def compute_accuracy_with_doubt(model, dataloader, threshold=0.9, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    abstained = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(len(labels)):
                if confs[i] >= threshold:
                    total += 1
                    if preds[i] == labels[i]:
                        correct += 1
                else:
                    abstained += 1

    accuracy = correct / total if total > 0 else 0.0
    rejection_rate = abstained / (total + abstained)
    return accuracy, rejection_rate

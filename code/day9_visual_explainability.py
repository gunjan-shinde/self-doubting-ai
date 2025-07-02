import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import cv2

from models.simple_cnn import SimpleCNN
from utils.datasets import get_train_val_datasets

# Paths
OUTPUT_DIR = r"C:\Users\Gunjan\Desktop\Research Paper\Self-Doubting AI\code\outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Grad-CAM Setup
def get_last_conv_layer(model):
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return name
    raise ValueError("No Conv2d layer found")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_score = output[0, target_class]
        class_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (28, 28))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return cam

# Load dataset
_, val_dataset = get_train_val_datasets()
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize Grad-CAM
target_layer = get_last_conv_layer(model)
gradcam = GradCAM(model, target_layer)

# Visualize a few examples
os.makedirs(os.path.join(OUTPUT_DIR, "gradcam"), exist_ok=True)

for idx, (input_tensor, label) in enumerate(val_loader):
    if idx >= 5:
        break
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    cam = gradcam.generate_cam(input_tensor, pred_class)
    input_img = input_tensor.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(input_img, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(input_img, cmap='gray')
    ax[1].imshow(cam, cmap='jet', alpha=0.5)
    ax[1].set_title(f"Grad-CAM\nPred: {pred_class}")
    ax[2].imshow(cam, cmap='jet')
    ax[2].set_title("Attention Map")
    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gradcam", f"gradcam_{idx}.png"))
    plt.close()

print("✅ Grad-CAM visualizations saved.")

# Save attention grid for paper
grid_img = np.zeros((28 * 5, 28 * 3))
for i in range(5):
    cam = cv2.imread(os.path.join(OUTPUT_DIR, "gradcam", f"gradcam_{i}.png"))
    cam = cv2.resize(cam, (28*3, 28))
    grid_img[28 * i : 28 * (i + 1), :] = cam[:, :, 0]

cv2.imwrite(os.path.join(OUTPUT_DIR, "attention_grid.png"), grid_img)
print("🧠 Attention grid saved for paper.")

print("✅ Day 9 visual explainability complete.")

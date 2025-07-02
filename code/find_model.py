import os

print("🔍 Searching for best_model.pth...")

for root, dirs, files in os.walk("C:\\Users\\Gunjan"):
    if "best_model.pth" in files:
        print("✅ Found at:", os.path.join(root, "best_model.pth"))

import pandas as pd
import os

# Load full dataset (replace this path with your actual dataset if needed)
full_data_path = (
    "data/final_cleaned_dataset.csv"  # Update this if you have a different filename
)

if not os.path.exists(full_data_path):
    raise FileNotFoundError(
        f"❌ Couldn't find {full_data_path}. Update the script with your actual dataset path."
    )

# Load and shuffle
df = pd.read_csv(full_data_path)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Take 200 samples or fewer if small dataset
test_df = df[["text", "label"]].head(200)

# Save
os.makedirs("data", exist_ok=True)
test_df.to_csv("data/test.csv", index=False)

print("✅ test.csv generated and saved to data/test.csv")

import pandas as pd
import os
import random

# Sample sentences (positive/negative)
positive_texts = [
    "I love this place",
    "This is amazing",
    "I feel great",
    "Absolutely wonderful",
    "What a beautiful day",
    "I enjoy reading books",
    "I like mangoes",
    "Traveling makes me happy",
    "Today is a good day",
    "I'm feeling awesome",
]

negative_texts = [
    "I hate this",
    "This is terrible",
    "Feeling really down",
    "Absolutely horrible",
    "Worst experience ever",
    "I dislike everything",
    "I am sad",
    "Too many problems",
    "Nothing feels right",
    "I'm so tired of this",
]

# Combine and label
data = []
for sentence in positive_texts:
    data.append({"text": sentence, "label": 1})
for sentence in negative_texts:
    data.append({"text": sentence, "label": 0})

# Shuffle
random.shuffle(data)

# Create dataframe
df = pd.DataFrame(data)

# Save
os.makedirs("data", exist_ok=True)
df.to_csv("data/final_cleaned_dataset.csv", index=False)
print("✅ final_cleaned_dataset.csv saved to data/")

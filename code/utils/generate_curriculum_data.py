import pandas as pd
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# -----------------------
# Easy: Clear sentiment
# -----------------------
easy_positive = [
    "I love mangoes",
    "This is a great day",
    "I'm feeling happy",
    "What a beautiful morning",
    "I enjoy reading books",
    "The weather is wonderful",
    "Life is amazing",
    "I'm so excited for this",
    "Everything is perfect",
    "I absolutely loved the experience"
]

easy_negative = [
    "I hate being late",
    "This is terrible",
    "I'm feeling very sad",
    "The food was awful",
    "I regret coming here",
    "Everything is going wrong",
    "I'm so disappointed",
    "The weather ruined my mood",
    "I don't like this place",
    "Such a bad experience"
]

easy_texts = easy_positive + easy_negative
easy_labels = [1] * len(easy_positive) + [0] * len(easy_negative)

# -----------------------
# Medium: Subtle/neutral
# -----------------------
medium_positive = [
    "The talk was somewhat interesting",
    "It turned out better than I expected",
    "Not bad, could be worse",
    "I kind of liked the ending",
    "It was okay overall",
    "Pretty average but enjoyable",
    "The outcome was fair",
    "I liked some parts of it",
    "Neutral but leaning positive",
    "A decent experience"
]

medium_negative = [
    "The session felt too long",
    "It didn’t meet expectations",
    "Might skip it next time",
    "Some parts were confusing",
    "Not exactly enjoyable",
    "I got bored midway",
    "Expected more than this",
    "Felt off somehow",
    "A bit underwhelming",
    "Could have been better"
]

medium_texts = medium_positive + medium_negative
medium_labels = [1] * len(medium_positive) + [0] * len(medium_negative)

# -----------------------
# Hard: Sarcastic / ironic
# -----------------------
hard_texts = [
    "Oh wow, just what I needed",
    "Amazing... another rainy day",
    "Great, my phone broke again",
    "Perfect timing as always... not",
    "Loved waiting for hours",
    "Fantastic performance... not really",
    "Couldn't be happier about this mess",
    "Everything is just peachy",
    "My favorite thing — more delays",
    "So thrilled to redo this again"
]

hard_labels = [0] * len(hard_texts)

# -----------------------
# Save CSVs
# -----------------------
def save_dataset(filename, texts, labels):
    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv(f"data/{filename}.csv", index=False)

save_dataset("easy", easy_texts, easy_labels)
save_dataset("medium", medium_texts, medium_labels)
save_dataset("hard", hard_texts, hard_labels)

print("✅ Created data/easy.csv, data/medium.csv, data/hard.csv")

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/ablation_results.csv")

# Create bar chart
plt.figure(figsize=(10, 5))
plt.title("📊 Ablation Study: Standard vs Curriculum", fontsize=16)

x = range(len(df))
plt.bar(x, df["Accuracy"], width=0.2, label="Accuracy", color="#4caf50")
plt.bar(
    [i + 0.2 for i in x],
    df["Avg Confidence"],
    width=0.2,
    label="Avg Confidence",
    color="#2196f3",
)
plt.bar(
    [i + 0.4 for i in x],
    df["% Doubtful"],
    width=0.2,
    label="% Doubtful",
    color="#ff9800",
)

plt.xticks([i + 0.2 for i in x], df["Model"])
plt.ylabel("Metric (%)")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/ablation_chart.png")
plt.show()

print("✅ Saved ablation comparison chart to outputs/ablation_chart.png")

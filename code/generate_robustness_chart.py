import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/robustness_results.csv")

# Bar plot
plt.figure(figsize=(8, 5))
plt.title("🧪 Robustness Benchmark: Clean vs Noisy", fontsize=15)

x = range(len(df))
colors = ["#4caf50", "#f44336", "#2196f3", "#ff9800"]

plt.bar(
    x,
    df["Accuracy"],
    tick_label=[f"{m}\n{c}" for m, c in zip(df["Model"], df["Condition"])],
    color=colors,
)
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for i, acc in enumerate(df["Accuracy"]):
    plt.text(i, acc + 1, f"{acc:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("outputs/robustness_chart.png")
print("✅ Chart saved to outputs/robustness_chart.png")
plt.show()

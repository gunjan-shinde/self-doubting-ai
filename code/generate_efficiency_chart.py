import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/efficiency_report.csv")

plt.figure(figsize=(10, 5))
x = range(len(df))

# Bar charts
plt.bar(
    x,
    df["Inference Speed (samples/sec)"],
    width=0.3,
    label="Speed (samples/sec)",
    color="#4caf50",
)
plt.xticks(x, df["Model"])
plt.ylabel("Speed")
plt.title("⚡ Inference Speed Comparison")
plt.tight_layout()
plt.savefig("outputs/efficiency_chart.png")
print("✅ Saved efficiency chart to outputs/efficiency_chart.png")
plt.show()

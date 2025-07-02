import pandas as pd
import matplotlib.pyplot as plt
import os

# Mapping in case models are named inconsistently across files
name_map = {"Standard": "Baseline", "Baseline": "Baseline", "Curriculum": "Curriculum"}

# Load and align CSVs
ablation = pd.read_csv("outputs/ablation_results.csv")
ablation["Model"] = ablation["Model"].map(name_map)

robust = pd.read_csv("outputs/robustness_results.csv")
robust["Model"] = robust["Model"].map(name_map)

eff = pd.read_csv("outputs/efficiency_report.csv")
eff["Model"] = eff["Model"].map(name_map)

# Rename for consistency
ablation = ablation.rename(columns={
    "Accuracy": "Clean Accuracy",
    "Avg Confidence": "Avg Confidence",
    "% Doubtful": "Doubt Frequency (%)"
})

# Pivot robustness
robust_pivot = robust.pivot(index="Model", columns="Condition", values="Accuracy")
robust_pivot["Robustness Drop (%)"] = robust_pivot["Clean"] - robust_pivot["Noisy"]

# Merge summary
summary = ablation[["Model", "Clean Accuracy", "Avg Confidence", "Doubt Frequency (%)"]].copy()
summary["Robustness Drop (%)"] = summary["Model"].map(lambda m: round(robust_pivot.loc[m]["Robustness Drop (%)"], 2))

# Safe inference speed + size access
def safe_get(model, column):
    val = eff[(eff["Model"] == model)][column]
    return round(val.values[0], 2) if not val.empty else None

summary["Inference Speed"] = summary["Model"].map(lambda m: safe_get(m, "Inference Speed (samples/sec)"))
summary["Size (MB)"] = summary["Model"].map(lambda m: safe_get(m, "Size (MB)"))

# Save summary table
os.makedirs("outputs", exist_ok=True)
summary_csv = "outputs/model_comparison_overview.csv"
summary.to_csv(summary_csv, index=False)

# Plot 2×2 grid
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("🧠 Model Comparison Overview", fontsize=16)

axs[0, 0].bar(summary["Model"], summary["Clean Accuracy"], color="#4caf50")
axs[0, 0].set_title("Clean Accuracy (%)")
axs[0, 0].set_ylim(0, 100)

axs[0, 1].bar(summary["Model"], summary["Doubt Frequency (%)"], color="#ff9800")
axs[0, 1].set_title("Doubt Frequency (%)")
axs[0, 1].set_ylim(0, 100)

axs[1, 0].bar(summary["Model"], summary["Robustness Drop (%)"], color="#f44336")
axs[1, 0].set_title("Accuracy Drop (Noisy Input)")
axs[1, 0].set_ylim(0, 50)

axs[1, 1].bar(summary["Model"], summary["Inference Speed"], color="#2196f3")
axs[1, 1].set_title("Inference Speed (samples/sec)")
axs[1, 1].set_ylim(0, max(summary["Inference Speed"]) + 5)

for ax in axs.flat:
    ax.set_ylabel("")

plt.tight_layout()
plt.subplots_adjust(top=0.88)

chart_path = "outputs/model_comparison_grid.png"
plt.savefig(chart_path)
plt.show()

# Final confirmation
print(f"✅ Saved comparison table to:     {summary_csv}")
print(f"✅ Saved 2×2 visual chart to:     {chart_path}")
print("\n📋 Final Model Comparison Table:")
print(summary.to_string(index=False))

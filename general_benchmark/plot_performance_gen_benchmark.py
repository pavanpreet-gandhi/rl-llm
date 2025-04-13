import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Step 1: Read and clean data
df_raw = pd.read_csv("general_benchmark/performance_summary.csv", skip_blank_lines=True)
df_cleaned = df_raw[df_raw["Model ID"] != "Model ID"].copy()
df_cleaned["Accuracy"] = df_cleaned["Accuracy"].astype(float)

# Step 2: Normalize trial labels to group variants
df_cleaned["Model Label"] = df_cleaned["Model ID"].str.extract(r'final_runs-(.*)')
df_cleaned["Group Label"] = df_cleaned["Model Label"].str.replace(r"__trial_\d+", "", regex=True)
df_cleaned["Group Label"] = df_cleaned["Group Label"].str.replace(r"_trial_\d+(_dist_\d+)?", r"\1", regex=True)
breakpoint()
df_cleaned = pd.concat([df_cleaned, df_raw.iloc[0:2]])
# Step 3: Aggregate mean, min, max
agg_df = df_cleaned.groupby(["Group Label", "Dataset"])["Accuracy"].agg(['min', 'max', 'mean']).reset_index()

# Step 4: Pivot for plotting
pivot_min = agg_df.pivot(index="Group Label", columns="Dataset", values="min")
pivot_max = agg_df.pivot(index="Group Label", columns="Dataset", values="max")
pivot_mean = agg_df.pivot(index="Group Label", columns="Dataset", values="mean")

# Step 5: Compute error bars
datasets = pivot_mean.columns
models = pivot_mean.index
# yerr_array = np.array([
#     (pivot_mean[datasets].values - pivot_min[datasets].values).T,
#     (pivot_max[datasets].values - pivot_mean[datasets].values).T
# ])

# lower_errors = (pivot_mean - pivot_min).T.values  # shape (datasets, models)
# upper_errors = (pivot_max - pivot_mean).T.values  # shape (datasets, models)
# yerr_array_correct = np.array([lower_errors, upper_errors])

# Step 6: Plot with error bars
fig, ax = plt.subplots(figsize=(12, 6))
pivot_mean.plot(kind="bar", capsize=4, ax=ax)
plt.title("ARC-C vs HellaSwag Average Accuracy by Model Variant\n")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(True)
plt.legend(title="Dataset")

# Step 7: Save the figure
final_path = "general_benchmark/general_benchmark_plot.png"
plt.savefig(final_path)

# final_path
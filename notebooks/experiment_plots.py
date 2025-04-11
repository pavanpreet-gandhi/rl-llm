import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Generate timestamp string
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Load into DataFrame
df = pd.read_csv("outputs/results_summary.csv")

# Extract model_id and reasoning_flag for title
model_name = df['model_id'].iloc[0]
reasoning_flag = df['reasoning_flag'].iloc[0]

# Define metrics to plot
metrics = ["avg_success", "avg_reward", "avg_length", "avg_invalid", "avg_total_time", "avg_generate_time"]

# Set up the plot grid
sns.set(style="whitegrid")
n_rows = 2
n_cols = 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
axes = axes.flatten()

# Plot each metric with bar labels
for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.barplot(
        data=df,
        x="env_id",
        y=metric,
        hue="context_window",
        palette="viridis",
        ax=ax
    )
    ax.set_title(metric.replace('_', ' ').title())
    ax.set_xlabel("Environment ID")
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.legend(title="Context Window")

    # Add number labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)

# Add a super title with model_id and reasoning_flag
plt.suptitle(f"{model_name} | Reasoning: {reasoning_flag}", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save the full grid to a single image with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"outputs/summary_plots_{timestamp}.png"
plt.savefig(filename)
plt.close()
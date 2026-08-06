import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# Line chart requested by reviewer:
# day x treatment mean lesion score per sow with error bars
#
# Input file must contain:
#   Round, Period label, Pheromone, Sow earmark, Sow spraymark,
#   Lesion score total
# ---------------------------------------------------------------------

# ---------- User settings ----------
input_file = "WQSows.csv"
output_file = "../output/skinInjuriesMean.pdf"

# Keep same treatment colours as the previous stacked bar chart
control_color = "#2a8f7b"      # green
pheromone_color = "#8b2a86"    # purple

# Error bars: choose "sem", "sd", or "ci95"
error_type = "sem"
# -----------------------------------


def get_error(series, error_type="sem"):
    """Return error-bar value for a numeric pandas Series."""
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)

    if n <= 1:
        return np.nan

    sd = x.std(ddof=1)

    if error_type.lower() == "sd":
        return sd
    if error_type.lower() == "sem":
        return sd / np.sqrt(n)
    if error_type.lower() == "ci95":
        return 1.96 * sd / np.sqrt(n)

    raise ValueError("error_type must be one of: 'sem', 'sd', 'ci95'")


# Read data
df = pd.read_csv(input_file)
df["Round"] = pd.to_numeric(df["Round"], errors="coerce").astype("Int64")
df["Lesion score total"] = pd.to_numeric(df["Lesion score total"], errors="coerce")

# Fixed period order, matching the previous figure labels
period_order = ["D0-D2", "D2-D7", "D7-D9", "D9-D17"]
period_display = ["D2", "D7", "D9", "D17"]

# Keep only rows needed for the plotted periods and treatments
df = df[df["Period label"].isin(period_order)].copy()
df = df[df["Pheromone"].isin(["Control", "Pheromone"])].copy()

# Create a unique sow identifier.
# This avoids accidental merging of sows that share an earmark in different rounds.
df["SowID"] = (
    df["Round"].astype(str)
    + "_"
    + df["Sow earmark"].astype(str)
    + "_"
    + df["Sow spraymark"].astype(str)
)

# If the raw file ever contains duplicate rows for the same sow in the same
# period/treatment, average them first so every sow contributes one value.
sow_period = (
    df.groupby(["Pheromone", "Period label", "SowID"], as_index=False)
      .agg(mean_sow_lesion_score=("Lesion score total", "mean"))
)

# Treatment x period summary: mean per sow and error across sows
summary = (
    sow_period.groupby(["Pheromone", "Period label"], as_index=False)
    .agg(
        mean=("mean_sow_lesion_score", "mean"),
        n=("mean_sow_lesion_score", "count"),
        sd=("mean_sow_lesion_score", lambda x: x.std(ddof=1)),
        error=("mean_sow_lesion_score", lambda x: get_error(x, error_type)),
    )
)

# Force plotting order
summary["Period label"] = pd.Categorical(
    summary["Period label"], categories=period_order, ordered=True
)
summary = summary.sort_values(["Pheromone", "Period label"])

# Optional: print the values used in the plot for manuscript checking
print("\nValues plotted: mean lesion score per sow")
print(summary.to_string(index=False))

# Plot
# Plot x-axis using true experimental day spacing,
# matching the movement figure.
interval_starts = np.array([0, 2, 7, 9], dtype=float)
interval_ends = np.array([2, 7, 9, 17], dtype=float)

# Plot each interval value at its midpoint
x_positions = np.array([2, 7, 9, 17], dtype=float)

# Show real day boundaries on the x-axis
boundary_ticks = np.array([0, 2, 7, 9, 17], dtype=float)
boundary_labels = ["D0", "D2", "D7", "D9", "D17"]

fig, ax = plt.subplots(figsize=(8, 6))

plot_specs = {
    "Control": {
        "color": control_color,
        "label": "Control",
        "marker": "o",
    },
    "Pheromone": {
        "color": pheromone_color,
        "label": "Pheromone",
        "marker": "o",
    },
}

for treatment, spec in plot_specs.items():
    tmp = (
        summary[summary["Pheromone"] == treatment]
        .set_index("Period label")
        .reindex(period_order)
    )

    ax.errorbar(
        x_positions,
        tmp["mean"],
        yerr=tmp["error"],
        label=spec["label"],
        color=spec["color"],
        marker=spec["marker"],
        markerfacecolor="white",
        markersize=7,
        linewidth=2.8,
        capsize=5,
        capthick=1.4,
        elinewidth=1.4,
        zorder=5
    )
    
ax.axvspan(0, 2, color="black", alpha=0.1, zorder=0)
ax.axvspan(7, 9, color="black", alpha=0.1, zorder=0)

# Axis labels and ticks
ax.set_xlim(0, 18)
ax.set_xticks(boundary_ticks)
ax.set_xticklabels(boundary_labels, fontsize=12)

error_label = {"sem": "SEM", "sd": "SD", "ci95": "95% CI"}[error_type.lower()]
ax.set_ylabel(f"Mean Lesion Score (per sow) ± {error_label}", fontsize=14)

# Styling to match the cleaner manuscript format
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=12)
ax.legend(loc="upper right", fontsize=12, frameon=False)

# Start y-axis at zero; add headroom above highest error bar
max_y = (summary["mean"] + summary["error"].fillna(0)).max()
ax.set_ylim(0, max_y * 1.20)

plt.tight_layout(rect=[0, 0.12, 1, 1])

# Save
output_path = Path(output_file)
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, format="pdf", bbox_inches="tight")
plt.show()
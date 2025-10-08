import os
import ast
import json
import numpy as np
import matplotlib.pyplot as plt

def get_occupancy():
    o = {2: 0, 3: 1, 4: 0, 5: 1,
         6: 0, 7: 1, 8: 0, 9: 1}
    
    d = {"Pheromone": {2: [], 4: [], 5: [], 8: []},
         "Control": {3: [], 6: [], 7: [], 9: []}}

    periods = ["0_2", "2_7", "7_9", "9_17", "0_17"]

    for expID in range(2, 10):
        if expID == 8:
            prefix = "ExperimentRemux" + str(expID)
        else:
            prefix = f"Experiment{expID}"

        group = "Pheromone" if expID in [2,4,5,8] else "Control"

        for per in periods:
            filepath = f"../output/occupancy/{prefix}/{per}.txt"
            with open(filepath, "r") as f:
                content = f.read()
                dataDict = ast.literal_eval(content)

            # sum across animals -> totals per pen
            totals = {pen: 0 for pen in range(1,5)}
            for animal, pens in dataDict.items():
                for pen, val in pens.items():
                    totals[pen] += val

            # normalize to percentage (sums to 1)
            total_sum = sum(totals.values())
            norm = [round(totals[p] / total_sum, 4) for p in [1,2,3,4]]
            
            if o[expID]:
                norm = norm[::-1]
            d[group][expID].append(norm)

    return d

data = get_occupancy()

# === Period labels ===
period_labels = ["D0-D2", "D2-D7", "D7-D9", "D9-D17", "D0-D17"]

# === Colors ===
control_color = '#2a8f7b'   # green
pheromone_color = '#8b2a86' # purple

# === Section labels ===
section_labels = ["Wall", "Middle", "Adjacent", "Feeder"]

# === Compute group averages ===
def group_means(group_data):
    arr = np.array(list(group_data.values()))  # shape: n_exp × n_period × 4
    return arr.mean(axis=0)  # → (n_period × 4)

ctrl_means = group_means(data["Control"]) * 100
phero_means = group_means(data["Pheromone"]) * 100

x = np.arange(len(period_labels))  # positions
width = 0.35

fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, section in enumerate(section_labels):
    ax = axes[i]

    # Control bars
    ctrl_vals = ctrl_means[:, i]
    bars_c = ax.bar(x - width/2, ctrl_vals, width, label="Control",
                    color=control_color)
    # Pheromone bars
    phero_vals = phero_means[:, i]
    bars_p = ax.bar(x + width/2, phero_vals, width, label="Pheromone",
                    color=pheromone_color)

    # Labels above bars
    for bars in [bars_c, bars_p]:
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10, color="black")

    # Formatting
    ax.set_title(section + " Section", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 35)
    ax.set_yticks(np.arange(0, 36, 5))
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Shared X axis
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, rotation=30, fontsize=11)
    

fig.text(0.04, 0.5, "Occupancy (%)", va="center", rotation="vertical", fontsize=14)

# Legend outside the plot
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5),
           fontsize=12, frameon=False)

plt.tight_layout(rect=[0.05,0.05,1,1])
plt.savefig("../output/sectionOccupancy.pdf", format='pdf', bbox_inches='tight')
plt.show()

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


d = {
    "Pheromone": {2: [], 4: [], 5: [], 8: []},
    "Control": {3: [], 6: [], 7: [], 9: []},
}

# ---------------------------------------------------------------------
# Collect agonistic interaction counts per animal, D0-D7
# Behaviours included:
# - fleeing
# - circling
# - replacement
# ---------------------------------------------------------------------

# Control experiments
for k in ["3", "6", "7", "9"]:
    with open("../missingAnimals.json", "r") as file:
        missingAnimals = json.load(file)[k[-1]]

    agonisticCount = {
        i + 1: 0
        for i in range(21)
        if i + 1 not in missingAnimals
    }

    for j in ["fleeing", "circling", "replacement"]:
        data = pd.read_csv(f"../output/{j}/Experiment{k}/0_7.csv")

        allValues = pd.concat([data["Initiator"], data["Receiver"]])
        valueCounts = allValues.value_counts().sort_index()

        for i in list(set(allValues)):
            if i in agonisticCount:
                agonisticCount[i] += int(valueCounts[i])

    d["Control"][int(k[-1])] = list(agonisticCount.values())


# Pheromone experiments
for k in ["2", "4", "5", "Remux8"]:
    with open("../missingAnimals.json", "r") as file:
        missingAnimals = json.load(file)[k[-1]]

    agonisticCount = {
        i + 1: 0
        for i in range(21)
        if i + 1 not in missingAnimals
    }

    for j in ["fleeing", "circling", "replacement"]:
        data = pd.read_csv(f"../output/{j}/Experiment{k}/0_7.csv")

        allValues = pd.concat([data["Initiator"], data["Receiver"]])
        valueCounts = allValues.value_counts().sort_index()

        for i in list(set(allValues)):
            if i in agonisticCount:
                agonisticCount[i] += int(valueCounts[i])

    key = 8 if "Remux" in k else int(k[-1])
    d["Pheromone"][key] = list(agonisticCount.values())


# ---------------------------------------------------------------------
# Combine all animal values into one box per treatment
# ---------------------------------------------------------------------
ctrl_allvals = np.concatenate([d["Control"][k] for k in [3, 6, 7, 9]])
phero_allvals = np.concatenate([d["Pheromone"][k] for k in [2, 4, 5, 8]])

box_data = [ctrl_allvals, phero_allvals]

# Replicate-level means (these will be shown as circles)
ctrl_group_means = [np.mean(d["Control"][k]) for k in [3, 6, 7, 9]]
phero_group_means = [np.mean(d["Pheromone"][k]) for k in [2, 4, 5, 8]]

# Colors
control_color = "#4fc3ad"      # blueish-green
pheromone_color = "#be3ba8"    # reddish-purple

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

positions = [0, 1]

boxprops = dict(linewidth=1.5)
medianprops = dict(color="black", linewidth=2)
whiskerprops = dict(linewidth=1.3)
capprops = dict(linewidth=1.3)

bplot = ax.boxplot(
    box_data,
    positions=positions,
    patch_artist=True,
    widths=0.5,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    showfliers=True,
)

# Set box colors
bplot["boxes"][0].set_facecolor(control_color)
bplot["boxes"][1].set_facecolor(pheromone_color)

# Style outliers/fliers
for i, flier in enumerate(bplot["fliers"]):
    flier.set(
        marker="o",
        markerfacecolor="white",
        markeredgecolor=control_color if i == 0 else pheromone_color,
        markersize=5,
        alpha=0.7,
    )

# ---------------------------------------------------------------------
# Overlay replicate groups as circles (no legend)
# ---------------------------------------------------------------------
# Small fixed horizontal offsets so the 4 replicate circles are visible
# Replicate-level means (these will be shown as circles)
ctrl_group_means = [np.mean(d["Control"][k]) for k in [3, 6, 7, 9]]
phero_group_means = [np.mean(d["Pheromone"][k]) for k in [2, 4, 5, 8]]

# Colors
control_color = "#4fc3ad"
pheromone_color = "#be3ba8"

# Plot
fig, ax = plt.subplots(figsize=(6, 6))

positions = [0, 1]

boxprops = dict(linewidth=1.5)
medianprops = dict(color="black", linewidth=2)
whiskerprops = dict(linewidth=1.3)
capprops = dict(linewidth=1.3)

bplot = ax.boxplot(
    box_data,
    positions=positions,
    patch_artist=True,
    widths=0.5,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    showfliers=True,
)

bplot["boxes"][0].set_facecolor(control_color)
bplot["boxes"][1].set_facecolor(pheromone_color)

for i, flier in enumerate(bplot["fliers"]):
    flier.set(
        marker="o",
        markerfacecolor="white",
        markeredgecolor=control_color if i == 0 else pheromone_color,
        markersize=5,
        alpha=0.7,
    )

# No jitter: circles on same vertical line
ctrl_x = np.full(len(ctrl_group_means), 0)
phero_x = np.full(len(phero_group_means), 1)

ax.plot(
    ctrl_x,
    ctrl_group_means,
    "o",
    markersize=8,
    markerfacecolor="white",
    markeredgecolor=control_color,
    markeredgewidth=1.8,
    linestyle="None",
    zorder=10,
)

ax.plot(
    phero_x,
    phero_group_means,
    "o",
    markersize=8,
    markerfacecolor="white",
    markeredgecolor=pheromone_color,
    markeredgewidth=1.8,
    linestyle="None",
    zorder=10,
)

ax.set_xticks(positions)
ax.set_xticklabels(["Control", "Pheromone"], fontsize=13)
ax.set_ylabel("Agonistic Interaction Count (Per Animal)", fontsize=13, fontweight="bold")

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.grid(False)
ax.tick_params(axis="both", labelsize=12)

plt.tight_layout()
plt.savefig("../output/agonistic_boxplot_0_7_combined_groups_with_replicate_circles.pdf", format="pdf")
plt.show()
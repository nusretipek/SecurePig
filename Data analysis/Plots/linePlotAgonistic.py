import os
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# --- Helper function ---
def getMeans(s):
    d = {"Pheromone": {2: [], 4: [], 5: [], 8: []},
         "Control": {3: [], 6: [], 7: [], 9: []}}
    d_points = [2, 5, 2, 8]

    for k in ["3", "6", "7", "9"]:
        with open("../missingAnim:als.json", 'r') as file:
            missingAnimals = json.load(file)[k[-1]]

        for idx, tJ in enumerate(["0_2", "2_7", "7_9", "9_17"]):
            agonisticCount = {i+1: 0 for i in range(21) if i+1 not in missingAnimals}
            data = pd.read_csv(f"../output/{s}/Experiment{k}/{tJ}.csv")
            allValues = pd.concat([data['Initiator'], data['Receiver']])
            valueCounts = allValues.value_counts().sort_index()
            for i in list(set(allValues)):
                agonisticCount[i] += int(valueCounts[i]) / d_points[idx] / 2
            d["Control"][int(k[-1])].append(int(sum(agonisticCount.values())))

    for k in ["2", "4", "5", "Remux8"]:
        with open("../missingAnimals.json", 'r') as file:
            missingAnimals = json.load(file)[k[-1]]

        for idx, tJ in enumerate(["0_2", "2_7", "7_9", "9_17"]):
            agonisticCount = {i+1: 0 for i in range(21) if i+1 not in missingAnimals}
            data = pd.read_csv(f"../output/{s}/Experiment{k}/{tJ}.csv")
            allValues = pd.concat([data['Initiator'], data['Receiver']])
            valueCounts = allValues.value_counts().sort_index()
            for i in list(set(allValues)):
                agonisticCount[i] += int(valueCounts[i]) / d_points[idx] / 2
            key = 8 if "Remux" in k else int(k[-1])
            d["Pheromone"][key].append(int(sum(agonisticCount.values())))

    # Arrays
    control_array = np.array(list(d['Control'].values()))
    control_mean = control_array.mean(axis=0)
    pheromone_array = np.array(list(d['Pheromone'].values()))
    pheromone_mean = pheromone_array.mean(axis=0)

    control_mean_full = control_mean
    pheromone_mean_full = pheromone_mean

    return d, control_mean_full, pheromone_mean_full

# === Collect Data ===
data_fight, y_fight_c, y_fight_p = getMeans("circling")      # fighting
data_flee, y_flee_c, y_flee_p = getMeans("fleeing")          # fleeing
data_disp, y_disp_c, y_disp_p = getMeans("replacement")      # displacement

x_points = ["D0-D2", "D2-D7", "D7-D9", "D9-D17"]
x = x_points
line_styles = ['-', '--', '-.', ':']

# === Colors ===
control_color = '#2a8f7b'
pheromone_color = '#8b2a86'

# === Plot ===
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True)

plot_specs = [
    ("Fighting Interaction Count (Daily)", data_fight, y_fight_c, y_fight_p, axes[0]),
    ("Fleeing Interaction Count (Daily)", data_flee, y_flee_c, y_flee_p, axes[1]),
    ("Displacement Interaction Count (Daily)", data_disp, y_disp_c, y_disp_p, axes[2]),
]

for title, d, y_ctrl, y_phero, ax in plot_specs:
    # Individual Control
    for idx, (exp, movement) in enumerate(sorted(d['Control'].items())):
        y = movement
        ax.plot(x, y, linestyle=line_styles[idx % len(line_styles)],
                color=control_color, alpha=0.5, zorder=1)

    # Individual Pheromone
    for idx, (exp, movement) in enumerate(sorted(d['Pheromone'].items())):
        y = movement
        ax.plot(x, y, linestyle=line_styles[idx % len(line_styles)],
                color=pheromone_color, alpha=0.5, zorder=1)

    # Averages on top
    ax.plot(x, y_ctrl, color=control_color, linewidth=3.5,
            marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor=control_color, alpha=1, zorder=5)
    ax.plot(x, y_phero, color=pheromone_color, linewidth=3.5,
            marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor=pheromone_color, alpha=1, zorder=5)

    ax.set_ylabel(title, fontsize=14, labelpad=8)
    ax.set_xticks(x)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.tick_params(labelsize=12)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylim(-0.5, 10)
axes[0].set_yticks(np.arange(0, 10, 1))   # Fighting: 0–9

axes[1].set_ylim(-0.5, 660)
axes[1].set_yticks(np.arange(0, 601, 60)) # Fleeing: 0–600 step 60

axes[2].set_ylim(-0.5, 110)
axes[2].set_yticks(np.arange(0, 101, 10)) # Displacement: 0–100 step 10

plt.tight_layout(rect=[0,0.05,1,1])
plt.savefig("../output/agonistic_timeline.pdf", format='pdf')
plt.show()

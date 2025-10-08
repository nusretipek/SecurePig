import os
import sys
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = {"Pheromone": {2: [], 4: [], 5: [], 8: []},
     "Control": {3: [], 6: [], 7: [], 9: []}}

for k in ["3", "6", "7", "9"]:
    with open("../missingAnimals.json", 'r') as file:
        missingAnimals = json.load(file)[k[-1]]
    agonisticCount = {i+1: 0 for i in range(21) if i+1 not in missingAnimals}

    for j in ["fleeing", "circling", "replacement"]:
        data = pd.read_csv(f"../output/{j}/Experiment{k}/0_7.csv")
        allValues = pd.concat([data['Initiator'], data['Receiver']])
        valueCounts = allValues.value_counts().sort_index()
        for i in list(set(allValues)):
            agonisticCount[i] += int(valueCounts[i])

    d["Control"][int(k[-1])] = list(agonisticCount.values())

for k in ["2", "4", "5", "Remux8"]:
    with open("../missingAnimals.json", 'r') as file:
        missingAnimals = json.load(file)[k[-1]]
    agonisticCount = {i+1: 0 for i in range(21) if i+1 not in missingAnimals}

    for j in ["fleeing", "circling", "replacement"]:
        data = pd.read_csv(f"../output/{j}/Experiment{k}/0_7.csv")
        allValues = pd.concat([data['Initiator'], data['Receiver']])
        valueCounts = allValues.value_counts().sort_index()
        for i in list(set(allValues)):
            agonisticCount[i] += int(valueCounts[i])
                
    d["Pheromone"][int(k[-1])] = list(agonisticCount.values())


# Configuration
group_labels = (["Control"]*4 + ["Pheromone"]*4)
bp_labels = list(d['Control'].keys()) + list(d['Pheromone'].keys())  # [3,6,7,9,2,4,5,8]
box_data = [d['Control'][k] for k in [3,6,7,9]] + [d['Pheromone'][k] for k in [2,4,5,8]]

control_color = '#4fc3ad'   # blueish-green
pheromone_color = '#be3ba8' # reddish-purple

# Plot layout
fig, ax = plt.subplots(figsize=(10, 6))
positions = [0,1,2,3, 5,6,7,8]  # space between control (0-3) and pheromone (5-8)

# Boxplots (patch_artist for color)
boxprops = dict(linewidth=1.5)
bplot = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.5, boxprops=boxprops, medianprops=dict(color='black', linewidth=2))

# Set box colors by group
for i, patch in enumerate(bplot['boxes']):
    if i < 4:
        patch.set_facecolor(control_color)
    else:
        patch.set_facecolor(pheromone_color)

# Plot swarm (individual points)
for i, y in enumerate(box_data):
    x = [positions[i]] * len(y)
    ax.plot(x, y, 'o', markersize=6, color='white', markeredgewidth=1.5, 
            markeredgecolor=control_color if i < 4 else pheromone_color, zorder=10)

# Draw horizontal average lines (all control, all pheromone)
ctrl_allvals = np.concatenate([d['Control'][k] for k in d['Control']])
phero_allvals = np.concatenate([d['Pheromone'][k] for k in d['Pheromone']])
ctrl_avg = np.mean(ctrl_allvals)
phero_avg = np.mean(phero_allvals)

# Define spans for each group (from leftmost to rightmost position)
control_xmin = min(positions[:4]) - 0.3
control_xmax = max(positions[:4]) + 0.3
phero_xmin = min(positions[4:]) - 0.3
phero_xmax = max(positions[4:]) + 0.3

# Draw group-specific horizontal lines
l_ctrl = ax.hlines(ctrl_avg, xmin=control_xmin, xmax=control_xmax, 
                   colors=control_color, linestyles='--', linewidth=2, 
                   label='Average: Control', zorder=11)

l_phero = ax.hlines(phero_avg, xmin=phero_xmin, xmax=phero_xmax, 
                    colors=pheromone_color, linestyles='--', linewidth=2, 
                    label='Average: Pheromone', zorder=11)

# Custom x-ticks and spacing
ax.set_xticks(positions)
ax.set_xticklabels([f"C{str(i+1)}" for i in range(4)] + [f"P{str(i+1)}" for i in range(4)]) #[str(k) for k in [3,6,7,9]] + [str(k) for k in [2,4,5,8]]

# Optional: add group label below
ax.text(1.5, ax.get_ylim()[0]-0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]), 'Control', 
        ha='center', va='top', fontsize=14, color=control_color, fontweight='bold')
ax.text(6.5, ax.get_ylim()[0]-0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]), 'Pheromone', 
        ha='center', va='top', fontsize=14, color=pheromone_color, fontweight='bold')
ax.legend(handles=[l_ctrl, l_phero], loc='upper right', frameon=False)

# Style
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Agonistic Interaction Count (Per Animal)', fontweight='bold')
plt.tight_layout()
plt.savefig("../output/agonistic_boxplot_0_7.pdf", format='pdf')

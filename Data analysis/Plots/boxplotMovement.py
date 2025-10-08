import os
import sys
import ast
import json
import numpy as np
import matplotlib.pyplot as plt

with open('../missingAnimals.json', 'r') as file:
    missingAnimals = json.load(file)

d = {"Pheromone": {2: [], 4: [], 5: [], 8: []},
     "Control": {3: [], 6: [], 7: [], 9: []}}

for expID in range(2, 10):
    if expID == 8:
        prefix = "ExperimentRemux"
    else:
        prefix = "Experiment"
    path = f'/data/nvme1/nusreti/masterTracks/{prefix}{str(expID)}' 
    missingAnimalExp = missingAnimals[str(expID)]
    animalCount = 21 - len(missingAnimalExp)

    with open(f"../output/movement/{prefix}{expID}/0_7.txt", "r") as f:
        content = f.read()
        dataDict = ast.literal_eval(content)  

    if expID in [2, 4, 5, 8]:
        d["Pheromone"][expID] = [dataDict[i]/80 for i in dataDict.keys()]
    else:
        d["Control"][expID] = [dataDict[i]/80 for i in dataDict.keys()]

# Configuration
group_labels = (["Control"]*4 + ["Pheromone"]*4)
bp_labels = [f"C{str(i+1)}" for i in range(4)] + [f"P{str(i+1)}" for i in range(4)]  # [3,6,7,9,2,4,5,8]
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
                   label='Average Movement: Control', zorder=11)

l_phero = ax.hlines(phero_avg, xmin=phero_xmin, xmax=phero_xmax, 
                    colors=pheromone_color, linestyles='--', linewidth=2, 
                    label='Average Movement: Pheromone', zorder=11)

# Custom x-ticks and spacing
ax.set_xticks(positions)
ax.set_xticklabels(bp_labels) #[str(k) for k in [3,6,7,9]] + [str(k) for k in [2,4,5,8]]

# Optional: add group label below
ax.text(1.5, ax.get_ylim()[0]-0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]), 'Control', 
        ha='center', va='top', fontsize=14, color=control_color, fontweight='bold')
ax.text(6.5, ax.get_ylim()[0]-0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]), 'Pheromone', 
        ha='center', va='top', fontsize=14, color=pheromone_color, fontweight='bold')

# Remove legend for boxes and points
# Add only for average lines
ax.legend(handles=[l_ctrl, l_phero], loc='upper right', frameon=False)

# Style
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Movement (Meters)', fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig("../output/movement/movement_boxplot_0_7_Meters.pdf", format='pdf')
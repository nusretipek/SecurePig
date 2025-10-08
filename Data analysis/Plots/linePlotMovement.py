import os
import sys
import ast
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


with open('../missingAnimals.json', 'r') as file:
    missingAnimals = json.load(file)

data = {"Pheromone": {2: [], 4: [], 5: [], 8: []},
        "Control": {3: [], 6: [], 7: [], 9: []}}

for expID in range(2, 10):
    if expID == 8:
        prefix = "ExperimentRemux"
    else:
        prefix = "Experiment"
    path = f'/data/nvme1/nusreti/masterTracks/{prefix}{str(expID)}' 
    missingAnimalExp = missingAnimals[str(expID)]
    animalCount = 21 - len(missingAnimalExp)

    for tJ in ["0_2", "2_7", "7_9", "9_17"]:
        with open(f"../output/movement/{prefix}{expID}/{tJ}.txt", "r") as f:
            content = f.read()
            dataDict = ast.literal_eval(content)  

        if expID in [2, 4, 5, 8]:
            data["Pheromone"][expID].append(int(sum(dataDict.values())/animalCount))
        else:
            data["Control"][expID].append(int(sum(dataDict.values())/animalCount))


# Plot
control_color = '#4fc3ad'      # blueish-green
pheromone_color = '#be3ba8'    # reddish-purple
group_colors = {'Control': control_color, 'Pheromone': pheromone_color}
line_styles = ['-', '--', '-.', ':']

# Compute daily movement (normalize by day)
x_points = ["D0-D2", "D2-D7", "D7-D9", "D9-D17"]
d_points = [2, 5, 2, 8]

# Convert to arrays
control_array = (np.array(list(data['Control'].values())) / np.array(d_points)) / 80
pheromone_array = (np.array(list(data['Pheromone'].values())) / np.array(d_points)) / 80

# Mean values
control_mean = control_array.mean(axis=0) 
pheromone_mean = pheromone_array.mean(axis=0) 

# Standard error (SEM)
control_sem = control_array.std(axis=0, ddof=1) / np.sqrt(control_array.shape[0])
pheromone_sem = pheromone_array.std(axis=0, ddof=1) / np.sqrt(pheromone_array.shape[0])

# Add zero baseline at time 0
control_mean_full = control_mean
pheromone_mean_full = pheromone_mean
control_sem_full = control_sem
pheromone_sem_full = pheromone_sem

fig, ax = plt.subplots(figsize=(12, 7))

# Colors
control_color = '#2a8f7b'
pheromone_color = '#8b2a86'

# Individual Control lines
for idx, (exp, movement) in enumerate(sorted(data['Control'].items())):
    y = movement
    ax.plot(x_points, (np.array(y) / d_points) / 80, linestyle=line_styles[idx % len(line_styles)],
            color=control_color, alpha=0.5, zorder=1)

# Individual Pheromone lines
for idx, (exp, movement) in enumerate(sorted(data['Pheromone'].items())):
    y = movement
    ax.plot(x_points, (np.array(y) / d_points) / 80, linestyle=line_styles[idx % len(line_styles)],
            color=pheromone_color, alpha=0.6, zorder=1)

# Control line + shaded error
ax.plot(x_points, control_mean_full, color=control_color, linewidth=3.5, label="Control",
        marker='o', markersize=7, markerfacecolor='white', markeredgewidth=1, markeredgecolor=control_color, alpha=1, zorder=5)

# Pheromone line + shaded error
ax.plot(x_points, pheromone_mean_full, color=pheromone_color, linewidth=3.5, label="Pheromone",
        marker='o', markersize=7, markerfacecolor='white', markeredgewidth=1, markeredgecolor=pheromone_color, alpha=1, zorder=5)


def bold(s): 
    return r'$\bf{' + s + '}$'

custom_lines = []

custom_lines.append(Line2D([], [], color='none', linestyle='solid', label=bold('Control'), linewidth=0))
for i, exp in enumerate(sorted(data['Control'].keys())):
    custom_lines.append(Line2D([], [], color=control_color, linestyle=line_styles[i], 
                              marker='o', markerfacecolor='white', markeredgewidth=2, markeredgecolor=control_color,
                              linewidth=2, alpha=0.8, label=f'  C{i+1}'))
custom_lines.append(Line2D([], [], color='none', linestyle='solid', label=bold('Pheromone'), linewidth=0))
for i, exp in enumerate(sorted(data['Pheromone'].keys())):
    custom_lines.append(Line2D([], [], color=pheromone_color, linestyle=line_styles[i], 
                              marker='o', markerfacecolor='white', markeredgewidth=2, markeredgecolor=pheromone_color,
                              linewidth=2, alpha=0.8, label=f'  P{i+1}'))
custom_lines.append(Line2D([], [], color='none', linestyle='solid', label=bold('Average'), linewidth=0))
custom_lines.append(Line2D([], [], color='#2a8f7b', linestyle='solid', linewidth=3.5, marker='o', markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2a8f7b', label='  Control'))
custom_lines.append(Line2D([], [], color='#8b2a86', linestyle='solid', linewidth=3.5, marker='o', markerfacecolor='white', markeredgewidth=2, markeredgecolor='#8b2a86', label='  Pheromone'))

ax.legend(handles=custom_lines, 
          frameon=False, 
          loc='center left', 
          bbox_to_anchor=(1.03, 0.5), 
          fontsize=10,
          handlelength=2.6, 
          borderpad=0, 
          labelspacing=0.8,
          title=None,
)

plt.setp(ax.get_legend().get_texts(), ha='right')

# Axis labels and formatting
ax.set_ylabel("Movement (Meters)", fontsize=14)
ax.set_xticks(x_points)
ax.set_xlim(-0.5, 3.5)
ax.margins(x=0.03)
ax.tick_params(labelsize=12)
ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

print(sum(control_mean_full))
print(sum(pheromone_mean_full))
plt.tight_layout()
#plt.savefig("../output/movement/movement_barplot_line.pdf", format='pdf')
plt.show()

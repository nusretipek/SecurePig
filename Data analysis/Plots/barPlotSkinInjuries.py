import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Read data
df = pd.read_csv("WQSows.csv")
df["Round"] = pd.to_numeric(df["Round"], errors="coerce").astype(int)

# Round grouping
control_rounds = [3,6,7,9]
phero_rounds   = [2,4,5,8]

# Periods in order
period_order = ['D0-D2','D2-D7','D7-D9','D9-D17']

# Bar configuration
bar_width = 0.35
inner_spacing = 0.1  # spacing between Control/Pheromone bars
period_spacing = 0.6 # extra space between periods

# Compute x positions
x_positions = []
for period_idx in range(len(period_order)):
    # Control bar
    x_positions.append(period_idx* (bar_width*2 + inner_spacing + period_spacing))
    # Pheromone bar
    x_positions.append(x_positions[-1] + bar_width + inner_spacing)

control_color = '#2a8f7b'   # green
pheromone_color = '#8b2a86' # purple

# Prepare figure
fig, ax = plt.subplots(figsize=(12,6))

# Keep track of max total score for y-axis ceiling
max_score = 0

for period_idx, period in enumerate(period_order):
    sub = df[df["Period label"] == period]

    # Sum total lesion scores per treatment
    control_total = sub[sub["Round"].isin(control_rounds)]["Lesion score total"].sum()
    phero_total   = sub[sub["Round"].isin(phero_rounds)]["Lesion score total"].sum()

    max_score = max(max_score, control_total, phero_total)

    # Draw bars
    ax.bar(x_positions[period_idx*2], control_total, bar_width, color=control_color, edgecolor='black', linewidth=0.8, label='Control' if period_idx==0 else "")
    ax.text(x_positions[period_idx*2], control_total + max_score*0.01, str(int(control_total)),
            ha='center', va='bottom', fontsize=12)

    ax.bar(x_positions[period_idx*2+1], phero_total, bar_width, color=pheromone_color, edgecolor='black', linewidth=0.8, label='Pheromone' if period_idx==0 else "")
    ax.text(x_positions[period_idx*2+1], phero_total + max_score*0.01, str(int(phero_total)),
            ha='center', va='bottom', fontsize=12)

# Dynamic y-axis ceiling + 10%
ymax = math.ceil(max_score/10)*10 + 10
ax.set_ylim(0, ymax)
ax.set_yticks(range(0, ymax+1, max(1, ymax//10)))
ax.set_ylabel("Total Lesion Score", fontsize=14)

# Period labels rotated 45° above the x-axis (slightly higher)
for period_idx, period in enumerate(period_order):
    center_pos = (x_positions[period_idx*2] + x_positions[period_idx*2]+0.75) / 2
    ax.text(center_pos, -ymax*0.02, period, ha='right', va='top', fontsize=12, color='black')

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([])
ax.set_ylim(0, 1200)
ax.set_yticks(np.arange(0, 1200+1, 100))
ax.legend(loc='upper right', fontsize=12, frameon=False)

plt.tight_layout(rect=[0,0,0.95,1])
plt.savefig("../output/skinInjuries.pdf", format='pdf', bbox_inches='tight')
plt.show()

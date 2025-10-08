import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from HierarchiaPy import Hierarchia
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker
from matplotlib.colors import to_hex

def calculateRandomizedEloRating(x, savePath):
    hierDF = Hierarchia(x, winner_col='Initiator', loser_col='Receiver')
    randomizedElo = hierDF.randomized_elo(start_value=0, n=2000)
    randomizedElo = dict(sorted(randomizedElo.items(), key=lambda item: item[1], reverse=True))

    # Output Results
    with open(savePath, "w") as f:
        print(randomizedElo, file=f)
    
    return randomizedElo


def optimizedEloRating(x, savePath):
    pass

    
def plotRandomizedEloRating(incrementalFiles, completeFile, savePath):

    # Incremental Plot
    d = {}
    for idx in range(len(incrementalFiles)):
        df = pd.read_csv(incrementalFiles[idx])
        randomizedElo = calculateRandomizedEloRating(df, savePath + "/" + os.path.basename(incrementalFiles[idx]).replace(".csv", ".txt"))
        if len(d) == 0:
            for key in randomizedElo.keys():
                d[key] = [randomizedElo[key]]
        else:
            for key in randomizedElo.keys():
                if key not in d:
                    d[key] = [0] * idx  
                d[key].append(randomizedElo[key])
        
        for key in d.keys():
            if len(d[key]) < idx + 1:
                d[key].append(d[key][-1])
    
    xTicks = [0, 2, 7, 9, 17]
    for key in d:
        while len(d[key]) < len(xTicks):
            d[key].append(d[key][-1]) 
        
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    ax = axes[0]
    
    for key, values in d.items():
        ax.plot(xTicks[:len(values)], values, label=key)
    
    # Customize
    ax.set_xticks(xTicks)
    ax.set_xlabel("Day")
    ax.set_ylabel("Randomized Elo Rating (Incremental)")
    ax.set_title("Randomized Elo Ratings over Time")
    ax.legend(loc='center left', bbox_to_anchor=(-0.25, 0.5), fontsize='small')
    fig.tight_layout()

    
    # Complete file plot
    df = pd.read_csv(completeFile)
    randomizedElo = calculateRandomizedEloRating(df, savePath + "/" + os.path.basename(completeFile).replace(".csv", ".txt"))
    ax2 = axes[1]
    
    for key, value in randomizedElo.items():
        ax2.plot([0, 17], [value, value], label=key)  # Horizontal line
    
    # X-axis
    ax2.set_xlim(0, 17)
    ax2.set_xticks([0, 2, 7, 9, 17])
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Randomized Elo Rating (Full Set)")
    ax2.set_title("Final Randomized Elo Ratings")
    ax2.legend(loc='center left', bbox_to_anchor=(-0.25, 0.5), fontsize='small')
    fig.tight_layout()
    
    return fig


def plotRandomizedEloRatingNumbered(incrementalFiles, completeFile, savePath):
    
    def group_labels_by_y(keys, y_values, threshold=20):
        groups = []
        used = set()

        for i, key in enumerate(keys):
            if key in used:
                continue
            group = [key]
            for j in range(i + 1, len(keys)):
                if abs(y_values[i] - y_values[j]) <= threshold and keys[j] not in used:
                    group.append(keys[j])
                    used.add(keys[j])
            used.update(group)
            groups.append(group)
        return groups
    
    def label_grouped(ax, x, keys, y_dict, lines, side='left'):
        y_values = [y_dict[k] for k in keys]
        y_avg = sum(y_values) / len(y_values)
    
        # Format text: 2, 6, 8
        parts = []
        for i, key in enumerate(sorted(keys)):
            color = to_hex(lines[key].get_color()) if key in lines else 'black'
            parts.append(TextArea(f"{key}", textprops=dict(color=color, fontsize=8)))
            if i < len(keys) - 1:
                parts.append(TextArea(", ", textprops=dict(color='black', fontsize=8)))
    
        label_box = HPacker(children=parts, align="center", pad=0, sep=1)
    
        # Outward padding from x (start or end)
        offset = -2 if side == 'left' else 2
    
        ax.add_artist(AnchoredOffsetbox(
            loc='center left' if side == 'left' else 'center right',
            child=label_box,
            pad=0.0,
            frameon=False,
            bbox_to_anchor=(x + offset, y_avg),
            bbox_transform=ax.transData,
            borderpad=0.0))

    # Prepare time series data
    d = {}
    for idx in range(len(incrementalFiles)):
        df = pd.read_csv(incrementalFiles[idx])
        randomizedElo = calculateRandomizedEloRating(
            df,
            os.path.join(savePath, os.path.basename(incrementalFiles[idx]).replace(".csv", ".txt"))
        )
        if not d:
            for key in randomizedElo:
                d[key] = [randomizedElo[key]]
        else:
            for key in randomizedElo:
                if key not in d:
                    d[key] = [0] * idx
                d[key].append(randomizedElo[key])
        for key in d:
            if len(d[key]) < idx + 1:
                d[key].append(d[key][-1])

    xTicks = [0, 2, 7, 9, 17]
    for key in d:
        while len(d[key]) < len(xTicks):
            d[key].append(d[key][-1])

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    ax = axes[0]
    keys = list(d.keys())
    lines = {}

    # Plot time series and store line handles
    for key in keys:
        line, = ax.plot(xTicks, d[key], label=key)
        lines[key] = line

    # --- Grouped START labels ---
    start_ys = {key: d[key][0] for key in keys}
    start_groups = group_labels_by_y(keys, [start_ys[k] for k in keys])
    for group in start_groups:
        label_grouped(ax, xTicks[0], group, start_ys, lines, side='left')

    # --- Grouped END labels ---
    end_ys = {key: d[key][-1] for key in keys}
    end_groups = group_labels_by_y(keys, [end_ys[k] for k in keys])
    for group in end_groups:
        label_grouped(ax, xTicks[-1], group, end_ys, lines, side='right')

    ax.set_xticks(xTicks)
    ax.set_xlabel("Day")
    ax.set_ylabel("Randomized Elo Rating (Incremental)")
    ax.set_title("Randomized Elo Ratings over Time")
    fig.tight_layout()

    # --- Complete set plot ---
    df = pd.read_csv(completeFile)
    randomizedElo = calculateRandomizedEloRating(
        df,
        os.path.join(savePath, os.path.basename(completeFile).replace(".csv", ".txt"))
    )
    ax2 = axes[1]
    final_keys = list(randomizedElo.keys())
    final_ys_dict = {key: randomizedElo[key] for key in final_keys}
    final_groups = group_labels_by_y(final_keys, list(final_ys_dict.values()))

    for key in final_keys:
        y = randomizedElo[key]
        color = lines[key].get_color() if key in lines else 'black'
        ax2.plot([0, 17], [y, y], color=color)

    for group in final_groups:
        label_grouped(ax2, 0, group, final_ys_dict, lines, side='left')
        label_grouped(ax2, 17, group, final_ys_dict, lines, side='right')

    x_min, x_max = min(xTicks), max(xTicks)
    x_pad = 2.5
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax2.set_xlim(x_min - x_pad, x_max + x_pad)    
    ax2.set_xticks(xTicks)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Randomized Elo Rating (Full Set)")
    ax2.set_title("Final Randomized Elo Ratings")
    fig.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replacementFolder", type=str)
    args = parser.parse_args()

    savePath = args.replacementFolder + "/" + "analysis"
    if not os.path.exists(savePath):
        os.makedirs(savePath) 

    # Replacement Files
    incrementalFileOrder = ['0_2.csv', '0_7.csv', '0_9.csv', '0_17.csv']
    incrementalFiles = [args.replacementFolder + "/" + i for i in incrementalFileOrder]
    otherFiles = [args.replacementFolder + "/" + i for i in ["2_7.csv", "7_9.csv", "9_17.csv"]]
    completeFile = args.replacementFolder + "/" + "0_17.csv"
    
    # Process other files
    for f in otherFiles:
        df = pd.read_csv(f)
        randomizedElo = calculateRandomizedEloRating(df, savePath + "/" + os.path.basename(f).replace(".csv", ".txt"))
        
    # Process incremental and complete files
    fig = plotRandomizedEloRatingNumbered(incrementalFiles, completeFile, savePath)
    fig.savefig(savePath + "/" + "randomizedElo.png")
    plt.close(fig) 

if __name__ == "__main__":
    main()
    
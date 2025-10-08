import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def plotCirclingBehavior(incrementalFiles, experimentID):
    
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = ['Day 0-2', 'Day 2-7', 'Day 7-9', 'Day 9-17']
    colors = plt.get_cmap('tab10').colors
    allCounts = []
    allIDs = set()
    
    for file in incrementalFiles:
        df = pd.read_csv(file)
        counts = Counter(df['Initiator'].tolist() + df['Receiver'].tolist())
        allCounts.append(counts)
        allIDs.update(counts.keys())
    
    allIDs = sorted(allIDs)
    bottom = [0] * len(allIDs)
    x = range(len(allIDs))
    
    for i, counts in enumerate(allCounts):
        values = [counts.get(aid, 0) for aid in allIDs]
        ax.bar(x, values, bottom=bottom, label=labels[i], color=colors[i])
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(list(x))
    ax.set_xticklabels(allIDs, rotation=45)
    ax.set_yticks(range(0, max(bottom) + 2, 1))  # increments of 2
    ax.set_xlabel("Animal ID")
    ax.set_ylabel("Circling Count")
    ax.set_title(f"Count of Circling Behaviour (Experiment {experimentID})")
    ax.legend()
    fig.tight_layout()

    return fig


def circlingCorrelationAnalysis(incrementalFiles, skinInjuriesFile, experimentID):
    
    # Skin Injuries
    df = pd.read_excel(skinInjuriesFile, sheet_name='combinatie gegeven')
    scoresDF = df[df['Round'] == experimentID]
    scoresDF = scoresDF[["Scoring day", "Sow spraymark", "Totaal_letsels"]]
    scoresDF = scoresDF[scoresDF["Sow spraymark"] != "BLANCO"]
    scoresDF.columns = ["Day", "AnimalID", "Score"]
    scoresDF = scoresDF.reset_index(drop=True)


    # Circling
    dayMap = {'0_2.csv': 2, '2_7.csv': 7, '7_9.csv': 9, '9_17.csv': 17}
    circlingList = []
    for file in incrementalFiles:
        df = pd.read_csv(file)
        counts = Counter(df['Initiator'].tolist() + df['Receiver'].tolist())
        day = dayMap[os.path.basename(file)]
        for aid, count in counts.items():
            circlingList.append({'Day': day, 'AnimalID': aid, 'CirclingScore': count})

    circlingDF = pd.DataFrame(circlingList)

    # Merge Dataframe
    mergedDF = scoresDF.merge(circlingDF, on=['Day', 'AnimalID'], how='left')
    mergedDF = mergedDF.sort_values(['AnimalID', 'Day'])
    mergedDF['CirclingScore'] = mergedDF['CirclingScore'].fillna(0)
    
    mergedDF['ScoreDiff'] = 0.0
    for aid in mergedDF['AnimalID'].unique():
        sub = mergedDF[mergedDF['AnimalID'] == aid].copy()
        sub = sub.sort_values('Day')
        prevScore = 0
    
        for idx, row in sub.iterrows():
            scoreDiff = row['Score'] - prevScore
            mergedDF.loc[idx, 'ScoreDiff'] = scoreDiff
            prevScore = row['Score']
    
    mergedDF['CirclingDiff'] = 0.0
    for aid in mergedDF['AnimalID'].unique():
        sub = mergedDF[mergedDF['AnimalID'] == aid].copy()
        sub = sub.sort_values('Day')
        prevScore = 0
    
        for idx, row in sub.iterrows():
            scoreDiff = row['CirclingScore'] - prevScore
            mergedDF.loc[idx, 'CirclingDiff'] = scoreDiff
            prevScore = row['CirclingScore']
            
    mergedDF = mergedDF.sort_values(['AnimalID', 'Day'])
    mergedDF['CirclingScore'] = mergedDF['CirclingScore'].fillna(0)
    r, p = spearmanr(mergedDF['CirclingDiff'], mergedDF['ScoreDiff'])

    return r, p, mergedDF

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("circlingFolder", type=str)
    parser.add_argument("skinInjuriesData", type=str)
    args = parser.parse_args()

    experimentID = int(args.circlingFolder[-1])
    savePath = args.circlingFolder + "/" + "analysis"
    if not os.path.exists(savePath):
        os.makedirs(savePath) 

    # Replacement Files
    incrementalFileOrder = ['0_2.csv', '2_7.csv', '7_9.csv', '9_17.csv']
    incrementalFiles = [args.circlingFolder + "/" + i for i in incrementalFileOrder]
        
    # Process circling analysis
    fig = plotCirclingBehavior(incrementalFiles, experimentID)
    fig.savefig(savePath + "/" + "circlingBehaviourBarChart.png")
    r, p, mergedDF = circlingCorrelationAnalysis(incrementalFiles, args.skinInjuriesData, experimentID)
    mergedDF.to_csv(savePath + "/" + "mergedFrame.csv", index=False) 

    with open(savePath + "/" + "correlationAnalysis.txt", "w") as f:   
        print(f"Pearson correlation between difference in Circling Count and Score Change: ", file=f)
        print(f"r = {r:.3f}, p = {p:.4f}", file=f)
        if p < 0.05:
            print("=> Significant correlation (p < 0.05)", file=f)
        else:
            print("=> No significant correlation (p ≥ 0.05)", file=f)

if __name__ == "__main__":
    main()
    
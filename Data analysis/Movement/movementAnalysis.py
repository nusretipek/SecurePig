import os
import json
import sys
import ast
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import fileParser

timeDict = {2: ["3_2024_04_15_08_0000", "3_2024_04_22_12_0000"],
            3: ["2_2024_06_17_08_0000", "2_2024_06_24_12_0000"],
            4: ["3_2024_07_08_08_0000", "3_2024_07_15_12_0000"],
            5: ["2_2024_09_09_08_0000", "2_2024_09_16_12_0000"],
            6: ["3_2024_09_30_08_0000", "3_2024_10_07_12_0000"],
            7: ["2_2024_12_03_08_0000", "2_2024_12_10_12_0000"],
            8: ["1_2025_02_04_08_0000", "1_2025_02_11_12_0000"],
            9: ["2_2025_02_25_08_0000", "2_2025_03_04_12_0000"],}

with open('../missingAnimals.json', 'r') as file:
    missingAnimals = json.load(file)

for expID in range(2, 10):
    if expID == 8:
        path = '/data/nvme1/nusreti/masterTracks/ExperimentRemux'
    else:
        path = '/data/nvme1/nusreti/masterTracks/Experiment'
    path += str(expID)
    
    numDirs = sum(os.path.isdir(os.path.join(path, d)) for d in os.listdir(path))
    missingAnimalExp = missingAnimals[str(expID)]
    animalCount = 21 - len(missingAnimalExp)
    sT = fileParser.parseTrackPath(timeDict[expID][0])
    eT = fileParser.parseTrackPath(timeDict[expID][1])
    assert eT > sT, "End date is earlier than the start date!"
    baseDir = str(Path(path + "/" + timeDict[expID][0] + "/tracks.json").parents[1])
    filesToProcess = fileParser.collectFiles(baseDir, sT, eT)
    
    
    with open(f"../output/movement/Experiment{expID}/0_7.txt", "r") as f:
        content = f.read()
        dataDict = ast.literal_eval(content)  

    print(f"Exp: {expID}, DirCount: {len(filesToProcess)}, Missing: {str(missingAnimalExp).ljust(20)},\
    AnimalCount: {animalCount}, NaiveMovement: {sum(dataDict.values()):,.0f},\
    TMovement: {sum(dataDict.values())/len(filesToProcess):,.0f},\
    AMovement: {sum(dataDict.values())/animalCount:,.0f}, \
    TAMovement: {sum(dataDict.values())/(len(filesToProcess)*animalCount):,.0f}")
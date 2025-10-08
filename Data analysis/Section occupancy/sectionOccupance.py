import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import json
import argparse
import traceback
import fileParser
import numpy as np
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool


class OccupanceGenerator:
    def __init__(self, trackFile: str, missingAnimalsFile: str):
        
        # Init 
        self.trackFile = trackFile
        self.missingAnimalsFile = missingAnimalsFile
        with open(self.trackFile, 'r') as file:
            self.tracks = json.load(file)
            self.tracks = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in self.tracks.items()}

        self.experimentID = self.trackFile.split("/")[-3][-1]
        with open(self.missingAnimalsFile, 'r') as file:
            self.missingAnimals = json.load(file)[self.experimentID]
            
        # Parameters
        self.fps = 14.985
        self.timePoints = 3600

        # Generate time adjusted tracks
        self.timeAdjustedTracks = {}
        self.getTimeAdjustedTracks()

        # Get heatmaps overall
        self.d = {i: {j: 0 for j in range(1, 5)} for i in range(1, 22)}
        
        # Get heatmaps individual
        self.animalIDs = []
        for i in range(1, 22):
            if i in self.timeAdjustedTracks and 0 in self.timeAdjustedTracks[i]:
                self.animalIDs.append(i)

    @staticmethod
    def findClosestFrame(n: np.ndarray, x: int) -> int:
        differences = np.abs(n-x)
        minIndex = np.argmin(differences)
        return n[minIndex]

    def getTimeAdjustedTracks(self) -> None:
        for idx, track in sorted(self.tracks.items()):
            keys = np.array(list(track.keys()), dtype=np.int64)
            tempTrack = {}
            frameNo = 0
            if len(keys) == 0:
                self.timeAdjustedTracks[idx] = {}
            else:
                for i in range(self.timePoints):
                    tempTrack[frameNo] = track[self.findClosestFrame(keys, frameNo).item()]
                    frameNo = int(round(frameNo + self.fps))   
                self.timeAdjustedTracks[idx] = tempTrack

    def calculateOccupance(self, naiveOccupancy=True):
        keys = list(self.timeAdjustedTracks[self.animalIDs[0]].keys())

        if naiveOccupancy:
            for timeIdx in range(self.timePoints):
                for animalX in range(len(self.animalIDs)):
                    camIndex = self.timeAdjustedTracks[self.animalIDs[animalX]][keys[timeIdx]]["cam"]
                    self.d[self.animalIDs[animalX]][camIndex] += 1
        else:
            prevCam = {i: [None, 0] for i in self.animalIDs}
            for timeIdx in range(self.timePoints):
                for animalX in range(len(self.animalIDs)):
                    camIndex = self.timeAdjustedTracks[self.animalIDs[animalX]][keys[timeIdx]]["cam"]
                    if prevCam[self.animalIDs[animalX]][0] is None or prevCam[self.animalIDs[animalX]][0] == camIndex:
                        prevCam[self.animalIDs[animalX]][1] += 1
                        if prevCam[self.animalIDs[animalX]][1] == 120:
                            self.d[self.animalIDs[animalX]][camIndex] += 1
                            prevCam[self.animalIDs[animalX]][1] = 0
                    else:
                        prevCam[self.animalIDs[animalX]][0] = camIndex
                        prevCam[self.animalIDs[animalX]][1] = 1

        toBeRemoved = []
        for idx in self.d.keys():
            if idx in self.missingAnimals:
                toBeRemoved.append(idx)

        for idx in toBeRemoved:
            del self.d[idx]


def processFile(path, missingAnimalsFile, naiveMovement):
    if not Path(path).exists():
        return {}

    try:
        occupanceGenerator = OccupanceGenerator(path, missingAnimalsFile)
        occupanceGenerator.calculateOccupance(naiveMovement)
        return occupanceGenerator.d
        
    except Exception as e:
        print(f"Error in {path}: {e}")
        traceback.print_exc()
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trackFile")
    parser.add_argument("trackFileEnd", nargs='?', default=None)
    parser.add_argument("missingAnimalsFile")
    parser.add_argument("--naive", action='store_true')
    args = parser.parse_args()

    if args.trackFileEnd is None:
        occupanceGenerator = OccupanceGenerator(args.trackFile, args.missingAnimalsFile)
        occupanceGenerator.calculateOccupance(args.naive)
        print(occupanceGenerator.d)
    
    else:
        sT = fileParser.parseTrackPath(args.trackFile)
        eT = fileParser.parseTrackPath(args.trackFileEnd)
        assert eT > sT, "End date is earlier than the start date!"
        baseDir = str(Path(args.trackFile).parents[1])
        filesToProcess = fileParser.collectFullFiles(baseDir, sT, eT)
        tasks = [(str(path), args.missingAnimalsFile, args.naive) for path in filesToProcess]
        
        with Pool(10) as pool:
            results = pool.starmap(processFile, tasks)

        masterDict = {}
        for d in results:
            for keyX, subdict in d.items():
                if keyX not in masterDict:
                    masterDict[keyX] = {}
                for keyY, value in subdict.items():
                    masterDict[keyX][keyY] = masterDict[keyX].get(keyY, 0) + value
                            
        print(masterDict)


if __name__ == "__main__":
    main()



"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get replacement behavior from tracks.")
    parser.add_argument('trackFolder', type=str, help="Path to the parent directory containing the tracking folders.")
    args = parser.parse_args()    
    folderNames = sorted([f for f in os.listdir(args.trackFolder) if os.path.isdir(os.path.join(args.trackFolder, f))])
    occupance = {i: {j: 0 for j in range(1, 5)} for i in range(1, 22)}


    
    for folder in folderNames:
        occupancGenerator = OccupanceGenerator(os.path.join(args.trackFolder, folder, "tracks.json"))
        for keyX in occupancGenerator.d.keys():
            for keyY in occupancGenerator.d[keyX].keys():
                occupance[keyX][keyY] += occupancGenerator.d[keyX][keyY]

    with open("occupanceExp3.json", 'w') as file:
        json.dump(occupance, file)

"""



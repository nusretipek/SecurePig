import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import re
import json
import argparse
import traceback
import fileParser
import numpy as np
from datetime import datetime
from multiprocessing import Pool


class Movement:
    def __init__(self, trackFile: str, missingAnimalsFile: str):
        
        # Init        
        self.trackFile = trackFile
        self.missingAnimalsFile = missingAnimalsFile
        with open(self.trackFile, 'r') as file:
            self.tracks = json.load(file)
            self.tracks = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in self.tracks.items()}
        
        self.experimentID = self.trackFile.split("/")[-3][-1]
        self.experimentID = self.experimentID if self.experimentID in ["2", "3", "4", "5", "6", "7", "9"] else "8"

        with open(self.missingAnimalsFile, 'r') as file:
            self.missingAnimals = json.load(file)[self.experimentID]

        #Parameters
        self.imageWidth = 640
        self.fps = 14.987
        self.timePoints = 3600

        # Generate time adjusted tracks
        self.timeAdjustedTracks = {}
        self.getTimeAdjustedTracks()
        self.animalIDs = []
        for i in range(1, 22):
            if i in self.timeAdjustedTracks and 0 in self.timeAdjustedTracks[i]:
                self.animalIDs.append(i)
        
        # Movement parameters
        self.individualMovements = {i: {} for i in self.animalIDs}
        self.individualMovementGroups = {i: 0 for i in self.animalIDs}

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

    @staticmethod
    def getManhattanDistance(sectorA: list, sectorB: list) -> int:
        return int(abs(sectorA[0]-sectorB[0]) + abs(sectorA[1]-sectorB[1]))

    def getPairwiseDistance(self, animalA: dict, animalB: dict) -> int:
        euclideanDistance = 0
        if animalA["cam"] == animalB["cam"]:
            euclideanDistance = self.getManhattanDistance(animalA["sector"], animalB["sector"])
        else:
            camDifference = abs(animalA["cam"]-animalB["cam"])
            euclideanDistance += (camDifference-1) * self.imageWidth
            if animalA["cam"] < animalB["cam"]:
                euclideanDistance += (animalA["sector"][0]+(self.imageWidth-animalB["sector"][0]))
            else:
                euclideanDistance += (animalB["sector"][0]+(self.imageWidth-animalA["sector"][0]))
            euclideanDistance += abs(animalA["sector"][1]-animalB["sector"][1])
            if camDifference == 1 and (animalA["sector"][1] > 140 or animalB["sector"][1] > 140):
                euclideanDistance += self.imageWidth 
            elif camDifference == 1 and (animalA["sector"][1] <= 140 or animalB["sector"][1] <= 140):
                euclideanDistance -= int(self.imageWidth / 2)
                if euclideanDistance < 0:
                    euclideanDistance = 20

        return int(euclideanDistance)

    def calculateIndividualMovement(self, minMovement=25) -> None:
        for animalIdx in self.animalIDs:
            keys = sorted(list(self.tracks[animalIdx].keys()))
            for timeIdx in range(1, len(keys)):
                movement = self.getPairwiseDistance(self.tracks[animalIdx][keys[timeIdx-1]], 
                                                    self.tracks[animalIdx][keys[timeIdx]])
                if movement > minMovement and movement < self.imageWidth/4:
                    self.individualMovements[animalIdx][keys[timeIdx-1]] = movement
                else:
                    self.individualMovements[animalIdx][keys[timeIdx-1]] = 0

    def groupMovements(self, timeDifference=100) -> None:
        for animalIDX in self.individualMovements.keys():
            keys = list(self.individualMovements[animalIDX].keys())
            movementGroups = 0
            tempGroup = []
            tempKey = -1
            counter = 1
            
            while counter < len(keys):
                if tempKey == -1:
                    tempKey = keys[counter]
                else:
                    if keys[counter] - tempKey <= timeDifference:
                        if self.individualMovements[animalIDX][keys[counter]] > 0:
                            tempGroup.append(keys[counter])
                            tempKey = keys[counter]
                    else:
                        if len(tempGroup) >= 2:
                            for idx in tempGroup:
                                movementGroups += self.individualMovements[animalIDX][idx]
                        tempGroup = []
                        tempKey = -1
                counter += 1
                
            self.individualMovementGroups[animalIDX] = movementGroups
        
    def getMovementDict(self, naiveMovement=True) -> dict:
        movementDict = {}
        self.calculateIndividualMovement()
        
        if naiveMovement:
            for animalIdx in self.animalIDs:
                if animalIdx not in self.missingAnimals:
                    d = dict(sorted(self.individualMovements[animalIdx].items(), key=lambda item: item[0]))
                    movementDict[animalIdx] = sum(d.values())
        else:
            self.groupMovements()
            for animalIdx in self.individualMovementGroups:
                if animalIdx not in self.missingAnimals:
                    movementDict[animalIdx] = self.individualMovementGroups[animalIdx]
                    
        return movementDict

def processFile(path, missingAnimalsFile, naiveMovement):
    if not Path(path).exists():
        return {}

    try:
        movementGenerator = Movement(path, missingAnimalsFile)
        d = movementGenerator.getMovementDict(naiveMovement)
        return d
        
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
        movementGenerator = Movement(args.trackFile, args.missingAnimalsFile)
        d = movementGenerator.getMovementDict(args.naive)
        print(d, sum(d.values()))
        
    else:
        sT = fileParser.parseTrackPath(args.trackFile)
        eT = fileParser.parseTrackPath(args.trackFileEnd)
        assert eT > sT, "End date is earlier than the start date!"
        baseDir = str(Path(args.trackFile).parents[1])
        filesToProcess = fileParser.collectFiles(baseDir, sT, eT)
        tasks = [(str(path), args.missingAnimalsFile, args.naive) for path in filesToProcess]
        

        with Pool(10) as pool:
            results = pool.starmap(processFile, tasks)

        masterDict = {}
        for d in results:
            for key in d.keys():
                if key not in masterDict:
                    masterDict[key] = d[key]
                else:
                    masterDict[key] += d[key]

        print(masterDict)

if __name__ == "__main__":
    main()


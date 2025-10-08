import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import re
import cv2
import json
import argparse
import traceback
import fileParser
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import groupby
from multiprocessing import Pool


class Fleeing:
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
            
        #Parameters
        self.pen = int(self.trackFile.split("/")[-2][0])
        self.imageWidth = 640
        self.cameras = [4, 3, 2, 1]
        self.fps = 14.987
        self.timePoints = 3600

        # Generate time adjusted tracks
        self.timeAdjustedTracks = {}
        self.getTimeAdjustedTracks()

        # Get pairwise distances
        self.animalIDs = []
        for i in range(1, 22):
            if i in self.timeAdjustedTracks and 0 in self.timeAdjustedTracks[i]:
                self.animalIDs.append(i)
                
        self.pairwiseDistances = np.full((self.timePoints, len(self.animalIDs), len(self.animalIDs)), -1, dtype=np.int16)
        self.individualMovements = {i: {} for i in self.animalIDs}
        self.individualMovementGroups = {i: [] for i in self.animalIDs}

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

    @staticmethod
    def getEuclideanDistance(sectorA: list, sectorB: list) -> int:
        return ((sectorA[0]-sectorB[0])**2 + (sectorA[1]-sectorB[1])**2)**0.5

    def getPairwiseDistance(self, animalA: dict, animalB: dict) -> int:
        euclideanDistance = 0
        if animalA["cam"] == animalB["cam"]:
            euclideanDistance = self.getEuclideanDistance(animalA["sector"], animalB["sector"])
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

    def calculatePairwiseDistances(self) -> None:
        keys = list(self.timeAdjustedTracks[self.animalIDs[0]].keys())
        for timeIdx in range(self.pairwiseDistances.shape[0]):
            for animalX in range(len(self.animalIDs)):
                for animalY in range(animalX, len(self.animalIDs)):
                    if animalX != animalY:
                        distance = self.getPairwiseDistance(self.timeAdjustedTracks[self.animalIDs[animalX]][keys[timeIdx]],
                                                            self.timeAdjustedTracks[self.animalIDs[animalY]][keys[timeIdx]])
                        self.pairwiseDistances[timeIdx, animalX, animalY] = distance
                        self.pairwiseDistances[timeIdx, animalY, animalX] = distance

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
    
    def groupMovements(self, timeDifference=75) -> None:
        for animalIDX in self.individualMovements.keys():
            keys = list(self.individualMovements[animalIDX].keys())
            movementGroups = []
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
                        if len(tempGroup) >= 3:
                            movementGroups.append(tempGroup)
                        tempGroup = []
                        tempKey = -1
                counter += 1
            self.individualMovementGroups[animalIDX] = movementGroups

    def isTowardsMovement(self, animalIDX:int, animalIDY:int, start:int) -> list:
        tXP = 5
        keys = list(self.timeAdjustedTracks[self.animalIDs[0]].keys())
        eventTimeAnimalX = self.timeAdjustedTracks[animalIDX][keys[start]]
        dist = []
        for timeIdx in range(start-tXP, start):
            dist.append(self.getPairwiseDistance(eventTimeAnimalX, self.timeAdjustedTracks[animalIDY][keys[timeIdx]]))
        return dist
        
    
    def approachingAnimal(self, animalIDX:int, start:int) -> int:
        tXP = 10
        for animalIDY in self.animalIDs: 
            if animalIDX != animalIDY:
                fleeingAnimalMovement = self.individualMovements[animalIDX][(start-tXP):(start+1)]
                approachingAnimalMovement = self.individualMovements[animalIDY][(start-tXP):(start+1)]
                dist = self.pairwiseDistances[(start-tXP):(start+1), self.animalIDs.index(animalIDX), self.animalIDs.index(animalIDY)]
                
                if len(dist) > 1:
                    slope = np.mean(np.diff(dist))
                else:
                    slope = 0
                    
                if len(self.isTowardsMovement(animalIDX, animalIDY, start)) > 1:
                    towardsSlope = np.mean(np.diff(self.isTowardsMovement(animalIDX, animalIDY, start)))
                else:
                    towardsSlope = 0
                
                # Criteria
                isSlopeNegative = -50 < slope < -5
                isTowardsSlopeHigh = -50 < towardsSlope < -5
                isApproachingAnimalMoving = len(approachingAnimalMovement) > 0 and max(approachingAnimalMovement) > 30
                isLastDistanceClose = len(dist) > 0 and dist[-1] <= 100
                isApproachingAnimalMovementHigher = sum(approachingAnimalMovement) > sum(fleeingAnimalMovement)
                isPreviousFleeLow = sum(fleeingAnimalMovement) < 200 
                isApproachValid = np.count_nonzero(approachingAnimalMovement[-5:]) > 0
                if (isSlopeNegative and isApproachingAnimalMoving):
                    pass
                    #print(animalIDY, dist, slope, approachingAnimalMovement, fleeingAnimalMovement, towardsSlope, isApproachingAnimalMovementHigher)

                if (isSlopeNegative and isApproachingAnimalMoving and isLastDistanceClose and 
                    isTowardsSlopeHigh and isApproachingAnimalMovementHigher and isPreviousFleeLow and
                    isApproachValid):
                    #pass
                    #print(animalIDY, dist, slope, approachingAnimalMovement, fleeingAnimalMovement, towardsSlope, isApproachingAnimalMovementHigher)
                    return animalIDY
        return 0
                    
    def calculateMovement(self, animalIDX:int, animalIDY:int, start:int, duration:int) -> list:
        keys = list(self.timeAdjustedTracks[self.animalIDs[0]].keys())
        if start+duration >= len(keys):
            return 0.0
        
        movementX, movementY = [], []
        position = None
        for timeIDX in range(start, start+duration):
            posX0 = self.timeAdjustedTracks[animalIDX][keys[timeIDX]]["sector"]
            posX1 = self.timeAdjustedTracks[animalIDX][keys[timeIDX+1]]["sector"]
            movementX.append(self.getEuclideanDistance(posX0, posX1))

            posY0 = self.timeAdjustedTracks[animalIDY][keys[timeIDX]]["sector"]
            posY1 = self.timeAdjustedTracks[animalIDY][keys[timeIDX+1]]["sector"]
            movementY.append(self.getEuclideanDistance(posY0, posY1))
            if position is None and self.timeAdjustedTracks[animalIDX][keys[timeIDX]]["cam"] == self.timeAdjustedTracks[animalIDY][keys[timeIDX]]["cam"]:
                sector = [(self.timeAdjustedTracks[animalIDX][keys[timeIDX]]["sector"][0]+self.timeAdjustedTracks[animalIDY][keys[timeIDX]]["sector"][0])/2,
                          (self.timeAdjustedTracks[animalIDX][keys[timeIDX]]["sector"][1]+self.timeAdjustedTracks[animalIDY][keys[timeIDX]]["sector"][1])/2,]
                position = {"cam": self.timeAdjustedTracks[animalIDX][keys[timeIDX]]["cam"], 
                            "sector": sector}
        return movementX, movementY, position

    def getFleeingBehavior(self) -> list:
        l = []
        for animalID in self.animalIDs:
            movement = self.individualMovements[animalID]
            movementThreshold = np.logical_and(75 >= movement, 30 <= movement)
            for idx in range(len(movement)):
                if movementThreshold[idx]:
                    x = self.approachingAnimal(animalID, idx-1)
                    if x:
                        #if x not in [4, 21] and animalID not in [4, 21]:

                        l.append([x, animalID, idx])
            toBeRemoved = [i for i in range(1, len(l)) if abs(l[i][2]-l[i-1][2]) < 30]
        return [[l[i][0], l[i][1]] for i in range(1, len(l)) if i not in toBeRemoved]

    def printDistances(self, timePoint: int):
        assert timePoint < self.pairwiseDistances.shape[0]

        # Print header
        s = "       "
        for i in self.animalIDs:
            s += str(i).ljust(6)
        print(s)

        # Print distances
        for i, animalID in enumerate(self.animalIDs):
            s = ""
            s += str(animalID).ljust(6)
            for j in range(len(self.animalIDs)):
               if self.pairwiseDistances[timePoint, i, j] < self.imageWidth:
                   s += str(self.pairwiseDistances[timePoint, i, j]).ljust(6)
               else:
                    s += str("x").ljust(6)
            print(s)

    def getSituation(self, animalIDX:int, t:list):
        tXP = 10 * self.fps
        keysXF = np.array(list(self.tracks[animalIDX].keys()))
        keysX = [keysXF[i] for i in range(len(keysXF)) if t[0]-tXP <= keysXF[i] <= t[0]+self.fps]
        situationPointDict = self.tracks[animalIDX][t[0]]
        opponentID, opponentDist, opponentTime = None, None, None
        for animalIDY in self.animalIDs: 
            dist = []
            opponentMovement = []
            if animalIDX != animalIDY:
                keysYF = np.array(list(self.tracks[animalIDY].keys()))
                keysY = np.array([keysYF[i] for i in range(len(keysYF)) if t[0]-tXP <= keysYF[i] <= t[0]+self.fps])
                prevOpponentDict = None
                for keyX in keysX:
                    if len(keysY):
                        keyY = self.findClosestFrame(keysY, keyX)
                        if prevOpponentDict is None:
                            prevOpponentDict = self.tracks[animalIDY][keyY]
                            
                        if abs(keyX-keyY) < 50:
                            dist.append(self.getPairwiseDistance(self.tracks[animalIDX][keyX], self.tracks[animalIDY][keyY]))
                            opponentMovement.append(self.getPairwiseDistance(self.tracks[animalIDY][keyY], situationPointDict))
                            prevOpponentDict = self.tracks[animalIDY][keyY]

            #lastHalfOpponentMovement = np.sum(np.diff(opponentMovement[len(opponentMovement)//2:]))
            
            if len(dist) and sum(dist)/len(dist) < 200 and -200 < np.sum(np.diff(opponentMovement)) < -10 and min(dist[-5:]) < 125:
                if opponentID is None or opponentDist < dist[-1]:
                    opponentID = animalIDY
                    opponentDist = dist[-1]
                    opponentTime = t[0]
                #print(animalIDX, animalIDY, dist, sum(dist)/len(dist), opponentMovement, np.sum(np.diff(opponentMovement)))
                               
        return opponentID, animalIDX, opponentTime

    def getFleeing(self):
        fleeList = []
        self.calculateIndividualMovement()
        self.groupMovements()
        for animalID in self.animalIDs: 
            for _ in self.individualMovementGroups[animalID]:
                i, j, k = self.getSituation(animalID, _)
                if i is not None:
                    fleeList.append([i, j, k])
                    if len(fleeList) > 1 and (fleeList[-2][0] == fleeList[-1][0] and 
                                              fleeList[-2][1] == fleeList[-1][1] and 
                                              not ((fleeList[-1][2]-fleeList[-2][2]) > self.fps*120)):
                        fleeList = fleeList[:-1]
                        
        sortedfleeList = sorted(fleeList, key=lambda x: x[2])
        sortedfleeList = [row[:-1] for row in sortedfleeList]
        sortedfleeList = [row for row in sortedfleeList if row[0] not in self.missingAnimals and row[1] not in self.missingAnimals]
        return sortedfleeList

def processFile(path, missingAnimalsFile):
    if not Path(path).exists():
        return []

    try:
        fleeingGenerator = Fleeing(path, missingAnimalsFile)
        fleeingGenerator.calculatePairwiseDistances()
        fleeing = fleeingGenerator.getFleeing()
        return fleeing
        
    except Exception as e:
        print(f"Error in {path}: {e}")
        traceback.print_exc()
        return []

        
def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("trackFile")
    #parser.add_argument("trackFileEnd", nargs='?', default=None)
    #parser.add_argument("missingAnimalsFile")
    #parser.add_argument("savePath", nargs="?")
    parser.add_argument("--trackFile", required=True)
    parser.add_argument("--trackFileEnd")
    parser.add_argument("--missingAnimalsFile", required=True)
    parser.add_argument("--savePath", default="replacements.csv")    
    args = parser.parse_args()

    if args.trackFileEnd is None:
        fleeingGenerator = Fleeing(args.trackFile, args.missingAnimalsFile)
        fleeingGenerator.calculatePairwiseDistances()
        fleeing = fleeingGenerator.getFleeing()
        df = pd.DataFrame(fleeing, columns=["Initiator", "Receiver"])
        df.to_csv(args.savePath, index=False)
        
    else:
        sT = fileParser.parseTrackPath(args.trackFile)
        eT = fileParser.parseTrackPath(args.trackFileEnd)
        assert eT > sT, "End date is earlier than the start date!"
        baseDir = str(Path(args.trackFile).parents[1])
        filesToProcess = fileParser.collectFiles(baseDir, sT, eT)
        tasks = [(str(path), args.missingAnimalsFile) for path in filesToProcess]
        
        with Pool(20) as pool:
            results = pool.starmap(processFile, tasks)

        masterList = []
        for l in results:
            for circling in l:
                masterList.append(circling)

        print(masterList)
        df = pd.DataFrame(masterList, columns=["Initiator", "Receiver"])
        df.to_csv(args.savePath, index=False)
        
if __name__ == "__main__":
    main()


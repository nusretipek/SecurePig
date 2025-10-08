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


class CirclingBehaviorDetector:
    def __init__(self, trackFile: str, missingAnimalsFile: str):
        
        # Init        
        self.trackFile = trackFile
        self.missingAnimalsFile = missingAnimalsFile
        with open(self.trackFile, 'r') as file:
            self.tracks = json.load(file)
            self.tracks = {int(k): {int(kk): vv for kk, vv in sorted(v.items(), key=lambda item: int(item[0]))} for k, v in self.tracks.items()}
        
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

    
    def calculateMovement(self, animalIDX:int, animalIDY:int, start:int, duration:int) -> list:
        keysXF = np.array(list(self.tracks[animalIDX].keys()))
        keysYF = np.array(list(self.tracks[animalIDY].keys()))
        startFrame = start*self.fps 
        endFrame = (start+duration)*self.fps
        keysX = [keysXF[i] for i in range(len(keysXF)) if startFrame <= keysXF[i] <= endFrame]
        keysY = [keysYF[i] for i in range(len(keysYF)) if startFrame <= keysYF[i] <= endFrame]
        movementX, movementY = [], []
        position = None  

        if len(keysX) >= len(keysY):
            for timeIDX in range(len(keysX)-1):
                # Movement X
                keyX0 = keysX[timeIDX] 
                keyX1 = keysX[timeIDX+1]
                camX0 = self.tracks[animalIDX][keyX0]["cam"]
                posX0 = self.tracks[animalIDX][keyX0]["sector"]
                posX1 = self.tracks[animalIDX][keyX1]["sector"]
                movementX.append(self.getEuclideanDistance(posX0, posX1))
                
                # Movement Y
                keyY0 = int(self.findClosestFrame(keysYF, keyX0))
                keyY1 = int(self.findClosestFrame(keysYF, keyX1))
                camY0 = self.tracks[animalIDY][keyY0]["cam"]
                posY0 = self.tracks[animalIDY][keyY0]["sector"]
                posY1 = self.tracks[animalIDY][keyY1]["sector"]
                movementY.append(self.getEuclideanDistance(posY0, posY1))        

                # Position 
                if (position is None) and (camX0 == camY0):
                    position = {"cam": camX0, "sector": [(posX0[0]+posY0[0])/2, (posX0[1]+posY0[1])/2]}
        else:
            for timeIDX in range(len(keysY)-1):
                # Movement Y
                keyY0 = keysY[timeIDX]
                keyY1 = keysY[timeIDX+1]
                camY0 = self.tracks[animalIDY][keyY0]["cam"]
                posY0 = self.tracks[animalIDY][keyY0]["sector"]
                posY1 = self.tracks[animalIDY][keyY1]["sector"]
                movementY.append(self.getEuclideanDistance(posY0, posY1))
                
                # Movement X
                keyX0 = int(self.findClosestFrame(keysXF, keyY0))
                keyX1 = int(self.findClosestFrame(keysXF, keyY1))
                camX0 = self.tracks[animalIDX][keyX0]["cam"]
                posX0 = self.tracks[animalIDX][keyX0]["sector"]
                posX1 = self.tracks[animalIDX][keyX1]["sector"]
                movementX.append(self.getEuclideanDistance(posX0, posX1))          

                # Position 
                if (position is None) and (camX0 == camY0):
                    position = {"cam": camX0, "sector": [(posX0[0]+posY0[0])/2, (posX0[1]+posY0[1])/2]}

        return movementX, movementY, position

    def findDominance(self, animalIDX:int, animalIDY:int, t:int, pos:dict) -> list:
        duration = 60
        perimeter = 150
        keys = list(self.timeAdjustedTracks[self.animalIDs[0]].keys())
        if t+duration >= len(keys):
            return []
        pairwiseDistances = []
        distanceTravelled = []
        perimeterAnimalX = []
        perimeterAnimalY = [] 
        posX, posY = [], []
        for timeIDX in range(t, t+duration):
            perimeterAnimalX.append(self.getPairwiseDistance(self.timeAdjustedTracks[animalIDX][keys[timeIDX]], pos))
            perimeterAnimalY.append(self.getPairwiseDistance(self.timeAdjustedTracks[animalIDY][keys[timeIDX]], pos))
            posX.append(self.timeAdjustedTracks[animalIDX][keys[timeIDX]])
            pairwiseDistances.append(int(self.pairwiseDistances[timeIDX, self.animalIDs.index(animalIDX), self.animalIDs.index(animalIDY)]))
            distanceTravelled.append(self.getPairwiseDistance(self.timeAdjustedTracks[animalIDX][keys[timeIDX]], self.timeAdjustedTracks[animalIDX][keys[timeIDX-1]]))

        
        perimeterAnimalX = np.convolve(perimeterAnimalX, np.ones(10), 'valid') / 10
        perimeterAnimalY = np.convolve(perimeterAnimalY, np.ones(10), 'valid') / 10
        maxX = np.argmax(perimeterAnimalX)
        maxY = np.argmax(perimeterAnimalY)

        for idx in range(len(perimeterAnimalX)):
            if perimeterAnimalX[idx] > 100 and perimeterAnimalY[idx] > 100:
                return []
            elif perimeterAnimalX[idx] > 100:
                return [animalIDY, animalIDX]
            elif perimeterAnimalY[idx] > 100:
                return [animalIDX, animalIDY]
            else:
                continue
        return []
 
    def searchInteractionCandidates(self, searchDurationSeconds:int=7, distance:int=100) -> list[list]:
        l = []
        for animalIDX in range(len(self.animalIDs)):
            for animalIDY in range(animalIDX+1, len(self.animalIDs)):
                if (self.animalIDs[animalIDX] not in self.missingAnimals) and (self.animalIDs[animalIDY] not in self.missingAnimals):
                    timeIDX = 0
                    immediateAffinity = False
                    while timeIDX < self.pairwiseDistances.shape[0]-searchDurationSeconds:
                        affinity = self.pairwiseDistances[timeIDX:timeIDX+searchDurationSeconds, animalIDX, animalIDY]
                        if np.all(np.logical_and(affinity < distance, affinity > 0)) and not immediateAffinity:
                            moveX, moveY, position = self.calculateMovement(self.animalIDs[animalIDX], self.animalIDs[animalIDY],
                                                                            timeIDX, searchDurationSeconds)
                            b = False
                            xZeroSequence = max(len(list(g)) for k, g in groupby(moveX + [0]) if k == 0)
                            yZeroSequence = max(len(list(g)) for k, g in groupby(moveY + [0]) if k == 0)
                            maxZeroes = max(xZeroSequence, yZeroSequence)
                            for idx in range(len(moveX)):
                                if moveX[idx] >= 20 and moveY[idx] >= 20 and ((40 < moveX[idx] < 100) or (40 < moveY[idx] < 100)):
                                    b = True
                                    break
    
                            if b and maxZeroes <= 3 and len(moveX) >= searchDurationSeconds and position is not None:
                                dom = self.findDominance(self.animalIDs[animalIDX], self.animalIDs[animalIDY], timeIDX, position)
                                if len(dom):
                                    if len(l) == 0:
                                        l.append(dom)
                                    elif len(l) > 0 and not (l[-1][0] == dom[0] and l[-1][1] == dom[1]) and not (l[-1][0] == dom[1] and l[-1][1] == dom[0]):
                                        l.append(dom)
                            timeIDX += 1
                            immediateAffinity = False
                        elif np.all(np.logical_and(affinity < distance, affinity > 0)) and immediateAffinity:
                            timeIDX += searchDurationSeconds
                        else:
                            timeIDX += 1
                            immediateAffinity = False
        return l

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

            
def processFile(path, missingAnimalsFile):
    if not Path(path).exists():
        return []

    try:
        circlingGenerator = CirclingBehaviorDetector(path, missingAnimalsFile)
        circlingGenerator.calculatePairwiseDistances()
        circling = circlingGenerator.searchInteractionCandidates()
        return circling
        
    except Exception as e:
        print(f"Error in {path}: {e}")
        traceback.print_exc()
        return []

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trackFile")
    parser.add_argument("trackFileEnd", nargs='?', default=None)
    parser.add_argument("missingAnimalsFile")
    parser.add_argument("savePath", nargs='?')
    args = parser.parse_args()

    if args.trackFileEnd is None:
        circlingGenerator = CirclingBehaviorDetector(args.trackFile, args.missingAnimalsFile)
        circlingGenerator.calculatePairwiseDistances()
        circling = circlingGenerator.searchInteractionCandidates()
        print(circling)
        
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


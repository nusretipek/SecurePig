import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import json
import time
import argparse
import traceback
import fileParser
import numpy as np
import pandas as pd
from typing import Self
from typing import overload
from datetime import datetime
from multiprocessing import Pool


class SowReplacementDetector:
    def __init__(self, trackFile: str, missingAnimalsFile: str) -> Self:
        
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
        self.pen = int(self.trackFile.split("/")[-2][0])
        self.cameras = [4, 3, 2, 1]
        self.imageWidth = 640
        self.imageHeight = 360
        self.sectorSize = 20
        self.fps = 14.985
        self.timePoints = 3600

        # Generate time adjusted tracks
        self.timeAdjustedTracks = {}
        self.getTimeAdjustedTracks()

        # Get Animal Ids individual
        self.animalIDs = []
        for i in range(1, 22):
            if i in self.timeAdjustedTracks and 0 in self.timeAdjustedTracks[i]:
                self.animalIDs.append(i)
        
        # Get replacements
        self.holdDuration = 120
        self.holdPerimeter = 20
        self.replacerDuration = 30
        self.separationDuration = 10
        self.separationMinDistance = 200

    @staticmethod
    def findClosestFrame(n: np.ndarray, x: int) -> int:
        differences = np.abs(n-x)
        minIndex = np.argmin(differences)
        return n[minIndex]

    def getTimeAdjustedTracks(self) -> Self:
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
    def getTime(frame:int, fps:float=14.985) -> tuple:
        minutes = int((frame / fps) // 60)
        seconds = int((((frame / fps) / 60) - minutes) * 60)
        return (minutes, seconds)
        
    @staticmethod
    def getTimeList(frames:list[int], fps:float=14.985) -> list[tuple]:
        t = []
        for frame in frames:
            minutes = int((frame / fps) // 60)
            seconds = int((((frame / fps) / 60) - minutes) * 60)
            t.append((minutes, seconds))
        return t

    @staticmethod
    def isHoldPosition(sectorA: list, sectorB: list, perimiter: int) -> bool:
        if ((sectorA[0] - perimiter <= sectorB[0] <= sectorA[0] + perimiter) and 
            (sectorA[1] - perimiter <= sectorB[1] <= sectorA[1] + perimiter)):
            return True
        else:
            return False

    @staticmethod
    def isSeparation(sectorA: list, sectorB: list, camA: int, camB: int, distance: int) -> bool:
        manhattanDistance = int(abs(sectorA[0]-sectorB[0]) + abs(sectorA[1]-sectorB[1]))
        if (manhattanDistance > distance) or (camA != camB):
            return True
        else:
            return False
    
    @staticmethod
    def isRapidChange(sectorA: list, sectorB: list, camA: int, camB: int) -> bool:
        def initialEntryExitPointValidation(t0: list, xRange: list, yRange: list) -> bool:
            if (xRange[0] <= t0[0] <= xRange[1]) and (yRange[0] <= t0[1] <= yRange[1]):
                return True
            return False
        
        leftXRange = [90, 130]
        rightXRange = [510, 550]
        rightLeftYRange = [30, 130]
        manhattanDistance = int(abs(sectorA[0]-sectorB[0]) + abs(sectorA[1]-sectorB[1]))
        camDifference = abs(camA-camB)
        
        if camDifference == 0 and manhattanDistance > 200:
            return True
        elif camDifference > 1:
            return True
        elif camDifference == 1:
            if camA > camB:
                if not (initialEntryExitPointValidation(sectorA, rightXRange, rightLeftYRange) and 
                        initialEntryExitPointValidation(sectorB, leftXRange, rightLeftYRange)):
                    return True
            else:
                if not (initialEntryExitPointValidation(sectorA, leftXRange, rightLeftYRange) and
                        initialEntryExitPointValidation(sectorB, rightXRange, rightLeftYRange)):
                    return True
        else:
            return False
    
    def positionReplacedBy(self, replacedAnimalID: int, sector: list, cam :int, replacementTime: int) -> int:
        keys = list(self.timeAdjustedTracks[self.animalIDs[0]].keys())
        keyIndex = keys.index(replacementTime)
        for animalID in self.animalIDs:
            if animalID != replacedAnimalID:
                
                # Prior Hold
                priorHoldCount = 0
                for timeID in range(keyIndex-self.replacerDuration, keyIndex):
                    if timeID < len(keys):
                        currentCam = self.timeAdjustedTracks[animalID][keys[timeID]]["cam"]
                        currentSector = self.timeAdjustedTracks[animalID][keys[timeID]]["sector"]
                        if currentCam == cam and self.isHoldPosition(sector, currentSector, self.holdPerimeter*3):
                            priorHoldCount += 1
                            
                # Replacement Hold            
                holdCount = 0
                for timeID in range(keyIndex, keyIndex+self.replacerDuration):
                    if timeID < len(keys):
                        currentCam = self.timeAdjustedTracks[animalID][keys[timeID]]["cam"]
                        currentSector = self.timeAdjustedTracks[animalID][keys[timeID]]["sector"]
                        if currentCam == cam and self.isHoldPosition(sector, currentSector, self.holdPerimeter*3):
                            holdCount += 1
                            
                #if (holdCount/self.replacerDuration) >= 0.1:
                #print("Prior Hold:", animalID, replacementTime, priorHoldCount/self.replacerDuration)
                #print("Post Hold:", animalID, replacementTime, holdCount/self.replacerDuration)
                    
                if (holdCount/self.replacerDuration) >= 0.3 and (priorHoldCount/self.replacerDuration) < 0.5:
                    return animalID
        return 0

    def distanceCheck(self, replacedAnimalID: int, replacerAnimalID: int, replacementTime: int) -> bool:
        keys = list(self.timeAdjustedTracks[self.animalIDs[0]].keys())
        keyIndex = keys.index(self.findClosestFrame(np.array(keys), replacementTime))
        replacedAnimalCam = self.timeAdjustedTracks[replacedAnimalID][keys[keyIndex]]["cam"]
        replacerAnimalCam = self.timeAdjustedTracks[replacerAnimalID][keys[keyIndex]]["cam"]
        replacedAnimalSector = self.timeAdjustedTracks[replacedAnimalID][keys[keyIndex]]["sector"]
        replacerAnimalSector = self.timeAdjustedTracks[replacerAnimalID][keys[keyIndex]]["sector"]        
        dist = abs(replacedAnimalSector[0]-replacerAnimalSector[0])+abs(replacedAnimalSector[1]-replacerAnimalSector[1])
        if (replacedAnimalCam == replacerAnimalCam) and dist >= 300:
            return False
        return True
    
    def getReplacmentCandiates(self):
        result = []
        keys = list(self.timeAdjustedTracks[self.animalIDs[0]].keys())
        for animalID in self.animalIDs:
            candidatePoints = []
            holdState = False
            holdCount = 0
            previousCam = self.timeAdjustedTracks[animalID][0]["cam"]
            previousSector = self.timeAdjustedTracks[animalID][0]["sector"]

            for i, timeID in enumerate(keys):
                # Skip first position
                if i == 0:
                    continue

                # Get current cam and sector
                currentCam = self.timeAdjustedTracks[animalID][timeID]["cam"]
                currentSector = self.timeAdjustedTracks[animalID][timeID]["sector"]

                # If hold position reach stationary, check t+x movement
                if holdCount >= self.holdDuration:
                    if (i+self.separationDuration) < len(keys):
                        nextCam = self.timeAdjustedTracks[animalID][keys[i+self.separationDuration]]["cam"]
                        nextSector = self.timeAdjustedTracks[animalID][keys[i+self.separationDuration]]["sector"]
                        if self.isSeparation(currentSector, nextSector, currentCam, nextCam, self.separationMinDistance):
                            rapidChange = False
                            for j in range(i, i+self.separationDuration):
                                tp0Cam = self.timeAdjustedTracks[animalID][keys[j]]["cam"]
                                tp0Sector = self.timeAdjustedTracks[animalID][keys[j]]["sector"] 
                                tp1Cam = self.timeAdjustedTracks[animalID][keys[j+1]]["cam"]
                                tp1Sector = self.timeAdjustedTracks[animalID][keys[j+1]]["sector"] 
                                if self.isRapidChange(tp0Sector, tp1Sector, tp0Cam, tp1Cam):
                                    rapidChange = True
                            if not rapidChange:
                                estimateReplaceTime = keys[i+int(self.separationDuration//2)]
                                estimateSector = self.timeAdjustedTracks[animalID][keys[i]]["sector"]
                                estimateCam = self.timeAdjustedTracks[animalID][keys[i]]["cam"]
                                replacerID = self.positionReplacedBy(animalID, estimateSector, estimateCam, keys[i])
                                if replacerID > 0: # and self.distanceCheck(replacerID, animalID, estimateReplaceTime):
                                    candidatePoints.append((self.getTime(keys[i+int(self.separationDuration//2)]), replacerID))
                                #candidatePoints.append(estimateReplaceTime)
                                holdCount = 0
                
                # If holding position increment holdCount
                if previousCam == currentCam and self.isHoldPosition(previousSector, currentSector, self.holdPerimeter):
                    holdCount += 1 
                else:
                    holdCount = 0
                
                # Assign previous cam and sector
                previousCam = currentCam
                previousSector = currentSector
            
            for candidate in candidatePoints:
                if candidate[1] not in self.missingAnimals and animalID not in self.missingAnimals:
                    result.append((candidate[1], animalID))
                    
        return result   


def processFile(path, missingAnimalsFile):
    
    if not Path(path).exists():
        return []

    try:
        # Replacements
        replacementDetector = SowReplacementDetector(path, missingAnimalsFile)
        replacements = replacementDetector.getReplacmentCandiates()
        return replacements
        
    except Exception as e:
        print(f"Error in {path}: {e}")
        traceback.print_exc()
        return {}
    

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
        
        # Replacements
        replacementDetector = SowReplacementDetector(args.trackFile, args.missingAnimalsFile)
        replacements = replacementDetector.getReplacmentCandiates()
            
        df = pd.DataFrame(replacements, columns=["Initiator", "Receiver"])
        df.to_csv(args.savePath, index=False)
        
    else:
        sT = fileParser.parseTrackPath(args.trackFile)
        eT = fileParser.parseTrackPath(args.trackFileEnd)
        assert eT > sT, "End date is earlier than the start date!"
        baseDir = str(Path(args.trackFile).parents[1])
        filesToProcess = fileParser.collectFiles(baseDir, sT, eT)
        tasks = [(str(path), args.missingAnimalsFile) for path in filesToProcess]
        

        with Pool(10) as pool:
            results = pool.starmap(processFile, tasks)

        masterList = []
        for l in results:
            for pair in l:
                masterList.append(pair)
   
        df = pd.DataFrame(masterList, columns=["Initiator", "Receiver"])
        df.to_csv(args.savePath, index=False)
        
if __name__ == "__main__":
    main()





"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get replacement behavior from tracks.")
    parser.add_argument('trackFolder', type=str, help="Path to the parent directory containing the tracking folders.")
    args = parser.parse_args()
    
    folderNames = sorted([f for f in os.listdir(args.trackFolder) if os.path.isdir(os.path.join(args.trackFolder, f))])
    results = []
    for folder in folderNames:
        tracksPath = os.path.join(args.trackFolder, folder, "tracks.json")
        replacementDetector = SowReplacementDetector(tracksPath)
        replacements = replacementDetector.getReplacmentCandiates()
        results.extend(replacements)

    animalIDs = set()
    for i, j in results:
        animalIDs.add(i)
        animalIDs.add(j)
    
    animalIDs = sorted(list(animalIDs))
    assignedResults = []
    for i, j in results:
        assignedResults.append((animalIDs.index(i), animalIDs.index(j)))

    for idx in range(len(animalIDs)):
        print("Animal", animalIDs[idx], "-", idx)
        
    df = pd.DataFrame(assignedResults, columns=["Initiator", "Receiver"])
    outputCSVPath = os.path.join(os.path.basename(args.trackFolder) + "_ReplacementsV2.csv")
    df.to_csv(outputCSVPath, index=False)
    print(f"Data saved to {outputCSVPath}")
"""
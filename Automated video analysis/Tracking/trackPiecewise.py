#!/usr/bin/env python3

import json
import numpy as np
import argparse

class PiecewiseTracker:
    def __init__(self, initialLocationFile):
        # Parameters
        self.initialLocationFile = initialLocationFile
        with open(self.initialLocationFile) as f:
            self.inferenceDict = json.load(f)
            self.inferenceDict = {int(k): v for k, v in self.inferenceDict.items()}
            self.structuredInfereceDict = self.structureInferenceDict()
        self.track = {}
        self.tracksDict = {}
        
        self.leftXRange = [70, 130]
        self.rightXRange = [510, 550]
        self.rightLeftYRange = [30, 130]

    def ExitPointValidationRight(self, t0, velocity):
        if (self.rightXRange[0] <= t0[0] <= self.rightXRange[1]) and (self.rightLeftYRange[0] <= t0[1] <= self.rightLeftYRange[1]) and velocity[0] >= 0:
            return True
        return False

    def ExitPointValidationLeft(self, t0, velocity):
        if (self.leftXRange[0] <= t0[0] <= self.leftXRange[1]) and (self.rightLeftYRange[0] <= t0[1] <= self.rightLeftYRange[1]) and velocity[0] <= 0:
            return True
        return False

    def EntrancePointValidationRight(self, t0):
        if (self.rightXRange[0] <= t0[0] <= self.rightXRange[1]) and (self.rightLeftYRange[0] <= t0[1] <= self.rightLeftYRange[1]):
            return True
        return False

    def EntrancePointValidationLeft(self, t0):
        if (self.leftXRange[0] <= t0[0] <= self.leftXRange[1]) and (self.rightLeftYRange[0] <= t0[1] <= self.rightLeftYRange[1]):
            return True
        return False
        
    def MatchBoundingBBoxes(self, BBoxesX, BBoxesY, threshold=0.5):
        BBoxesX = [j["bbox"] for i, j in BBoxesX.items()]
        BBoxesY = [j["bbox"] for i, j in BBoxesY.items()]

        pairs = []
        unassigned_BBoxesX = set(range(len(BBoxesX))) 
        assigned_BBoxesY = set()
        
        if not BBoxesX or not BBoxesY:
            return pairs, list(unassigned_BBoxesX)
        
        iouMatrix = np.zeros((len(BBoxesX), len(BBoxesY)))
    
        for i, box1 in enumerate(BBoxesX):
            for j, box2 in enumerate(BBoxesY):
                iouMatrix[i][j] = self.calculateIOU(box1, box2)
    
        for i in range(len(BBoxesX)):
            if len(assigned_BBoxesY) == len(BBoxesY):
                break
            best_match = np.argmax(iouMatrix[i, :])  
            max_iou = iouMatrix[i, best_match]
    
            if max_iou > threshold and best_match not in assigned_BBoxesY:
                pairs.append((i, best_match))
                assigned_BBoxesY.add(best_match)
                unassigned_BBoxesX.discard(i)  
        return pairs, list(unassigned_BBoxesX)

    def MatchBoundingBBoxesOrginal(self, BBoxesX, BBoxesY, threshold=0.5):
        BBoxesX = [j for i, j in BBoxesX.items()]
        BBoxesY = [j for i, j in BBoxesY.items()]
        pairs = []
        iouMatrix = np.zeros((len(BBoxesX), len(BBoxesY)))
        if min(len(BBoxesX), len(BBoxesY)) == 0:
            return pairs
        for i, box1 in enumerate(BBoxesX):
            for j, box2 in enumerate(BBoxesY):
                iouMatrix[i][j] = self.calculateIOU(box1, box2)
        for i in range(len(BBoxesX)):
            iouTemp = np.max(iouMatrix[i, :])
            if iouTemp > threshold:
                pairs.append((i, np.argmax(iouMatrix[i, :])))
        return pairs
    
    def searchAdjacentCam(self, entryKey, camId, exitPoint, tDiff=10):
        keyList = sorted(self.structuredInfereceDict.keys())
        keyIndex = keyList.index(entryKey)
        
        searchSpace = [keyIndex]
        for i in range(1, tDiff):
            searchSpace.append(keyIndex+i)
            searchSpace.append(keyIndex-i)

        searchSpace = []
        for i in range(keyIndex-tDiff, keyIndex+tDiff):
            searchSpace.append(i)
        
        for i in searchSpace:
            candidateBBOXES = {key: value for key, value in self.structuredInfereceDict[keyList[i]].items() if value['cam'] == camId}
            if len(candidateBBOXES) == 0:
                continue
            else:
                candidateBBOXESPrev = {key: value for key, value in self.structuredInfereceDict[keyList[i-1]].items() if value['cam'] == camId}
                offset = 0
                while len(candidateBBOXESPrev) == 0:
                    offset += 1
                    candidateBBOXESPrev = {key: value for key, value in self.structuredInfereceDict[keyList[i-offset-1]].items() if value['cam'] == camId}

            print(keyList[i], keyList[i-offset-1], candidateBBOXESPrev, candidateBBOXES)
            for idx, val in enumerate(candidateBBOXES.keys()):
                if idx in self.MatchBoundingBBoxes(candidateBBOXES, candidateBBOXESPrev)[1]:
                    #print(idx, candidateBBOXES[val])
                    candidateSector = candidateBBOXES[val]["sector"]
                    if exitPoint == "right" and self.EntrancePointValidationLeft(candidateSector):
                        return candidateBBOXES[val], i
                    if exitPoint == "left" and self.EntrancePointValidationRight(candidateSector):
                        return candidateBBOXES[val], i
        return None, None
                    
    @staticmethod
    def BBoxToSector(BBox, imageWidth=640, imageHeight=360, sectorSize=20):
        xMin, yMin, xMax, yMax = BBox
        centerX = (xMin + xMax) / 2
        centerY = (yMin + yMax) / 2
        sectorX = (int(centerX // sectorSize) * sectorSize) + sectorSize // 2
        sectorY = (int(centerY // sectorSize) * sectorSize) + sectorSize // 2
        return [sectorX, sectorY]
    
    def structureInferenceDict(self):
        d = {}
        for i in self.inferenceDict.keys():
            d[i] = {}
            for j, detection in enumerate(self.inferenceDict[i]):
                 d[i][j] = {"cam": detection["pen"],
                            "bbox": detection["bbox"],
                            "sector": self.BBoxToSector(detection["bbox"]),
                            "id": detection["id"],
                            "idProb": round(detection["idProb"], 4)}           
        return d

    @staticmethod
    def calculateIOU(BBoxX, BBoxY):
        xMinIntercept = max(BBoxX[0], BBoxY[0])
        yMinIntercept = max(BBoxX[1], BBoxY[1])
        xMaxIntercept = min(BBoxX[2], BBoxY[2])
        yMaxIntercept = min(BBoxX[3], BBoxY[3])
        if (xMinIntercept < xMaxIntercept) and (yMinIntercept < yMaxIntercept):
            areaintersection = ((xMaxIntercept - xMinIntercept) * 
                                (yMaxIntercept - yMinIntercept))
        else:
            return 0.0
        areaBBoxX = (BBoxX[2] - BBoxX[0]) * (BBoxX[3] - BBoxX[1])
        areaBBoxY = (BBoxY[2] - BBoxY[0]) * (BBoxY[3] - BBoxY[1])
        areaUnion = areaBBoxX if areaBBoxY > areaBBoxX else areaBBoxY
        return areaintersection / areaUnion
    
    def calculateVelocity(self, prevBBOX, currentBBOX, timeDiff=1):
        x1p, y1p, x2p, y2p = prevBBOX
        x1c, y1c, x2c, y2c = currentBBOX
        centerPrev = np.array([(x1p + x2p) / 2, (y1p + y2p) / 2])
        centerCurrent = np.array([(x1c + x2c) / 2, (y1c + y2c) / 2])
        velocity = (centerCurrent - centerPrev) / timeDiff
        return velocity

    def findBestMatch(self, prevBBOX, candidates):
        bestMatch, bestIOU = None, 0
        for key, bbox in candidates.items():
            iouScore = self.calculateIOU(prevBBOX, bbox)
            if iouScore > bestIOU:
                bestIOU = iouScore
                bestMatch = key
        return bestMatch, bestIOU

    @staticmethod
    def predictPosition(currentBBOX, velocity, timeAhead=1):
        x1, y1, x2, y2 = currentBBOX
        centerCurrent = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        velocity = np.array(velocity)
        predictedCenter = centerCurrent + (velocity * timeAhead)
        width = x2 - x1
        height = y2 - y1
        predictedBBOX = [int(predictedCenter[0] - width / 2),
                         int(predictedCenter[1] - height / 2),
                         int(predictedCenter[0] + width / 2),
                         int(predictedCenter[1] + height / 2),]
        return predictedBBOX
    
    def filterOtherTracks(self, otherTracks, otherTracksCurrent):
        indicesToRemove = []
        for k, i in enumerate(otherTracks):
            for s, j in enumerate(otherTracksCurrent):
                tempIOU = self.calculateIOU(i, j)
                if tempIOU > 0.8:
                    indicesToRemove.append(s)
        return [i for k, i in enumerate(otherTracksCurrent) if k in indicesToRemove]
    
    def otherTrackOccupactionCheck(self, bbox, bboxIOU, otherTracks):
        otherMatch = False
        for box in otherTracks:
            tempIOU = self.calculateIOU(box, bbox)
            if tempIOU > bboxIOU:
                otherMatch = True
                break
        return otherMatch

    def eliminateOtherInCam(self, prevBBOXES, currentBBOXES):
        pairs = self.MatchBoundingBBoxesOrginal(prevBBOXES, currentBBOXES)
        toRemove = [j for i, j in pairs]
        currentBBOXES = {key: currentBBOXES[key] for idx, key in enumerate(currentBBOXES) if idx not in toRemove}

    def getTracks(self, maxGap=5):
        trackID = 0
 
        while True:
            # Initiate new track
            track = {}
            keyList = sorted([i for i in self.structuredInfereceDict.keys() if len(self.structuredInfereceDict[i]) > 0])
            keyIndex = 0
            startFound = False
            startKey = None
            startIndex = None
            if len(keyList) > 0:
                detectionList = sorted(self.structuredInfereceDict[keyList[0]].keys())
                track[keyList[0]] = self.structuredInfereceDict[keyList[0]][detectionList[0]]
                del self.structuredInfereceDict[keyList[0]][detectionList[0]]
                startFound = True
            if not startFound:
                break

            # Track parameters
            otherTracks = {}
            velocityList = []
            currentCam = track[keyList[0]]['cam']
            currentBBOX = track[keyList[0]]['bbox']
            currentVelocity = None
            switchCounter = 0

            # Fill track
            for _ in range(len(keyList)):
                counterNone = 0
                updated = False
                for i in range(maxGap):
                    keyIndex += 1
                    trackLastKey = sorted(self.structuredInfereceDict.keys())

                    if keyIndex >= (len(keyList)-1):
                        break
                        
                    candidateBBOXES = {key: value['bbox'] for key, value in self.structuredInfereceDict[keyList[keyIndex]].items() if value['cam'] == currentCam}

                    while len(candidateBBOXES) == 0 and (keyIndex < len(keyList)-1):
                        counterNone += 1
                        keyIndex += 1
                        candidateBBOXES = {key: value['bbox'] for key, value in self.structuredInfereceDict[keyList[keyIndex]].items() if value['cam'] == currentCam}
     
                    if len(candidateBBOXES) == 0:
                        break
                        
                    candidateBBOXESPrev = {key: value for key, value in self.structuredInfereceDict[keyList[keyIndex-1]].items() if value['cam'] == currentCam}
                    offset = 0
                    while len(candidateBBOXESPrev) == 0:
                        offset += 1
                        candidateBBOXESPrev = {key: value for key, value in self.structuredInfereceDict[keyList[keyIndex-offset-1]].items() if value['cam'] == currentCam}
                    trackLastKey = sorted(track.keys())[-1]
                    currentSector = track[trackLastKey]['sector']
                    candidateBBOXESPrev = {key: value['bbox'] for key, value in candidateBBOXESPrev.items() if value['sector'] != currentSector}
                    self.eliminateOtherInCam(candidateBBOXESPrev, candidateBBOXES) 
                    matchKey, matchIOU = self.findBestMatch(currentBBOX, candidateBBOXES)
                    
                    if i == 0 and currentVelocity is not None:
                        predictedBBOX = self.predictPosition(currentBBOX, currentVelocity, i+1)
                        candidateBBOXES = {key: value['bbox'] for key, value in self.structuredInfereceDict[keyList[keyIndex]].items() if value['cam'] == currentCam}
                        matchKeyVelocity, matchIOUVelocity = self.findBestMatch(predictedBBOX, candidateBBOXES)
                        
                        if matchIOUVelocity > matchIOU and matchIOUVelocity > 0.7:
                            matchOtherBBOX = self.otherTrackOccupactionCheck(self.structuredInfereceDict[keyList[keyIndex]][matchKeyVelocity]["bbox"],
                                                                             matchIOU,
                                                                             otherTracks)
                            if not matchOtherBBOX:
                                track[keyList[keyIndex]] = self.structuredInfereceDict[keyList[keyIndex]][matchKeyVelocity]
                                del self.structuredInfereceDict[keyList[keyIndex]][matchKeyVelocity]
                                otherTracks = [candidateBBOXES[key] for key in list(candidateBBOXES.keys()) if key != matchKeyVelocity]
                                currentVelocity = self.calculateVelocity(currentBBOX, candidateBBOXES[matchKeyVelocity])
                                currentBBOX = track[keyList[keyIndex]]['bbox']
                                updated = True
                                break
    
                    if matchIOU > 0.7:
                        matchOtherBBOX = self.otherTrackOccupactionCheck(self.structuredInfereceDict[keyList[keyIndex]][matchKey]["bbox"],
                                                                         matchIOU,
                                                                         otherTracks)
                        if i > 1 or (not matchOtherBBOX):
                            track[keyList[keyIndex]] = self.structuredInfereceDict[keyList[keyIndex]][matchKey]
                            del self.structuredInfereceDict[keyList[keyIndex]][matchKey]
                            otherTracks = [candidateBBOXES[key] for key in list(candidateBBOXES.keys()) if key != matchKey]
                            currentVelocity = self.calculateVelocity(currentBBOX, candidateBBOXES[matchKey])
                            currentBBOX = track[keyList[keyIndex]]['bbox']
                            updated = True
                            break
    
                if not updated:
                    keyIndex -= (maxGap+counterNone) 
                    if currentVelocity is not None:
                        bestMatches = []
                        for i in range(maxGap*2):
                            keyIndex += 1

                            if keyIndex >= (len(keyList)-1):
                                break
                                
                            predictedBBOX = self.predictPosition(currentBBOX, currentVelocity, i+1)
                            candidateBBOXES = {key: value['bbox'] for key, value in self.structuredInfereceDict[keyList[keyIndex]].items() if value['cam'] == currentCam}
                            while len(candidateBBOXES) == 0 and (keyIndex < len(keyList)-1):
                                counterNone += 1
                                keyIndex += 1
                                candidateBBOXES = {key: value['bbox'] for key, value in self.structuredInfereceDict[keyList[keyIndex]].items() if value['cam'] == currentCam}
                            
                            if len(candidateBBOXES) == 0:
                                break
                                
                            bestMatches.append([keyIndex] + list(self.findBestMatch(predictedBBOX, candidateBBOXES)))
                        
                        if len(bestMatches) == 0:
                            break
                        
                        matchKeyIndex, matchKey, matchIOU = sorted(bestMatches, key=lambda x: x[2], reverse=True)[0]
                        if matchIOU > 0.35:
                            keyUpdate = matchKeyIndex - (keyIndex - maxGap*2)
                            candidateBBOXES = {key: value['bbox'] for key, value in self.structuredInfereceDict[keyList[matchKeyIndex]].items() if value['cam'] == currentCam}
                            matchOtherBBOX = self.otherTrackOccupactionCheck(self.structuredInfereceDict[keyList[matchKeyIndex]][matchKey]["bbox"],
                                                                             matchIOU-(-keyUpdate*0.01),
                                                                             otherTracks)
                            if not matchOtherBBOX:
                                track[keyList[matchKeyIndex]] = self.structuredInfereceDict[keyList[matchKeyIndex]][matchKey]
                                del self.structuredInfereceDict[keyList[matchKeyIndex]][matchKey]
                                otherTracks = [candidateBBOXES[key] for key in list(candidateBBOXES.keys()) if key != matchKey]
                                currentVelocity = [0, 0]
                                currentBBOX = track[keyList[matchKeyIndex]]['bbox']
                                updated = True
    
                if currentVelocity is not None:
                    velocityList.append(currentVelocity[0])
                else:
                    velocityList.append(0)

                if not updated:
                    trackLastKey = sorted(track.keys())[-1]
                    try:
                        repeatKeyIndex = keyList[keyIndex]
                    except:
                        repeatKeyIndex = keyList[-1]
                    track[repeatKeyIndex] = track[trackLastKey]
                    currentVelocity = [0, 0]
                    currentBBOX = track[repeatKeyIndex]['bbox']
                    currentCam = track[repeatKeyIndex]['cam']
                    switchCounter += 1
                else:
                    switchCounter = 0
                
                if switchCounter >= 1:
                    break
                    
            if len(track) >= 10:
                self.tracksDict[trackID] = track
                trackID += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Pig Track Piecewise Generator")
    parser.add_argument("inference_pen", type=int, help="Inference pen ID")
    parser.add_argument("timestamp", type=str, help="Timestamp in format YYYYMMDD_HHMMSS")
    args = parser.parse_args()

    inputTracks = 'masterTracks/tracks_'  + str(args.inference_pen) +'_' + args.timestamp + '.json'
    
    tracker = PiecewiseTracker(inputTracks)
    tracker.getTracks()
    with open('masterTracks/tracksPiecewise_'  + str(args.inference_pen) +'_' + args.timestamp + '.json', 'w') as json_file:
        json.dump(tracker.tracksDict, json_file, indent=4)
        
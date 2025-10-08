import os
import json
import time
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict
from trackGenerator import PigTrackInitializer, getBlackoutMask
from trackPiecewise import PiecewiseTracker

class TrackerIndividual:
    def __init__(self, initialLocationFile, missingAnimals):
        self.initialLocationFile = initialLocationFile
        self.missingAnimals = missingAnimals
        
        with open(self.initialLocationFile, 'r') as file:
            self.tracks = json.load(file)
            self.tracks = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in self.tracks.items()}

        self.split()

    @staticmethod
    def identifyTrack(track):
        confidenceID = defaultdict(float)
        unknownCount = 0
        
        for time, data in sorted(track.items()):
            currentID = data['id']
            confidence = data['idProb']
            if currentID == 0:
                unknownCount += 1
            else:
                confidenceID[currentID] += confidence
    
        totalWeight = sum(confidenceID.values())
        normalizedScores = {k: v / totalWeight for k, v in confidenceID.items()}
        if len(normalizedScores) == 0:
            return 0, 0, float(unknownCount/len(track))
        else:
            bestID = max(normalizedScores, key=normalizedScores.get)
            return bestID, normalizedScores[bestID], float(unknownCount/len(track))
    
    @staticmethod
    def splitTrack(track):
        confidenceID = defaultdict(float)
        unknownCount = 0
        splitTracks = []
    
        minOccurance = max(10, int(len(track)*0.05))
        identityList = [(data['id'], data['idProb']) for time, data in sorted(track.items())]
        segments = set()
        for idx in range(len(identityList)-minOccurance):
            tempID = []
            tempConfidence = []
            for idy in range(idx, idx+minOccurance):
                tempID.append(identityList[idy][0])
                tempConfidence.append(identityList[idy][1])
            if len(set(tempID)) == 1 and sum(tempConfidence) > 0.8*minOccurance and tempID[0] > 0:
                segments.add(tempID[0])
    
        tempDict = {}
        currentIdentity = None
        for time, data in sorted(track.items()):
            # Check Id change
            if data['id'] > 0 and data['id'] in segments:
                if currentIdentity is None:
                    currentIdentity = data['id']
                else:
                    if data['id'] != currentIdentity:
                        currentIdentity = data['id']
                        splitTracks.append(tempDict)
                        tempDict = {}
            tempDict[time] = data
        splitTracks.append(tempDict)
        return len(splitTracks) > 1, splitTracks

    def splitTrackBySequence(self, track):
        confidenceID = defaultdict(float)
        sequenceLength = 10
        unknownCount = 0
        splitTracks = []
    
        identityList = [(data['id'], data['idProb']) for time, data in sorted(track.items())]
        segments = set()
        for idx in range(len(identityList)-sequenceLength):
            tempID = []
            tempConfidence = []
            for idy in range(idx, idx+sequenceLength):
                tempID.append(identityList[idy][0])
                tempConfidence.append(identityList[idy][1])
            #print(idx, set(tempID), sum(tempConfidence), len(set(tempID)) == 1, sum(tempConfidence) > (0.9*sequenceLength), tempID[0] > 0)
            if (len(set(tempID)) == 1) and (sum(tempConfidence) > (0.75*sequenceLength)) and (tempID[0] > 0):
                segments.add(tempID[0])
    
        segments.add(self.identifyTrack(track)[0]) 
        tempDict = {}
        currentIdentity = None
        for time, data in sorted(track.items()):
            # Check Id change
            if data['id'] > 0 and data['id'] in segments:
                if currentIdentity is None:
                    currentIdentity = data['id']
                else:
                    if data['id'] != currentIdentity:
                        currentIdentity = data['id']
                        splitTracks.append(tempDict)
                        tempDict = {}
            tempDict[time] = data
        splitTracks.append(tempDict)
        return len(splitTracks) > 1, splitTracks
    
    def split(self):
        counter = len(self.tracks)
        toBeRemoved = []
        for idx in range(len(self.tracks)):
            splitBool, splits = self.splitTrack(self.tracks[idx])
            if splitBool:
                for track in splits:
                    self.tracks[counter] = track
                    counter += 1
                toBeRemoved.append(idx)
        
        for idx in toBeRemoved:
            del self.tracks[idx]
            
        self.tracks = {idx: value for idx, (_, value) in enumerate(self.tracks.items())}

        # Split by sequence
        counter = len(self.tracks)
        toBeRemoved = []
        for idx in range(len(self.tracks)):
            splitBool, splits = self.splitTrackBySequence(self.tracks[idx])
            if splitBool:
                for track in splits:
                    self.tracks[counter] = track
                    counter += 1
                toBeRemoved.append(idx)
        
        for idx in toBeRemoved:
            del self.tracks[idx]
            
        self.tracks = {idx: value for idx, (_, value) in enumerate(self.tracks.items())}
        
        # Clear short tracks
        toBeRemoved = []
        for idx in range(len(self.tracks)):
            if len(self.tracks[idx]) < 5:
                toBeRemoved.append(idx)
        for idx in toBeRemoved:
            del self.tracks[idx]
        
        self.tracks = {idx: value for idx, (_, value) in enumerate(self.tracks.items())}

        # Delete by space (lastKey)
        toBeRemoved = []
        for idx in range(len(self.tracks)):
            trackKeys = sorted(list(self.tracks[idx].keys()))
            if int(trackKeys[-1]) - int(trackKeys[-2]) > 1800:
                    del self.tracks[idx][trackKeys[-1]]

        
    def getTrackConfidence(self, track):
        tempID, tempConf, tempUnknown = self.identifyTrack(track)
        conf = float(0)
        counter = 1
        
        for time, data in sorted(track.items()):
            currentID = data['id']
            confidence = data['idProb']
            if currentID == tempID:
                conf += confidence
            if currentID != 0:
                counter += 1
        return tempID, conf/counter

    @staticmethod
    def isMixTrack(track, additionCount):
        trackKeys = sorted(track.keys())
        switchBehavior = 0.0
        for idx in range(len(trackKeys)-2):
            if ((track[trackKeys[idx]]['cam'] != track[trackKeys[idx+1]]['cam']) and 
                 track[trackKeys[idx]]['cam'] == track[trackKeys[idx+2]]['cam']):
               switchBehavior += 1
        return switchBehavior/additionCount

    @staticmethod
    def isOverlapTrack(trackFull, trackAddition):
        trackAdditionKeys = list(trackAddition.keys())
        trackFullKeys = list(trackFull.keys())
        overlapCount = 0
        for key in trackAdditionKeys:
            if key in trackFullKeys:
                overlapCount += 1
        return float(overlapCount/len(trackAdditionKeys))

    @staticmethod
    def isMixTrackRegion(track, additionCount, start, end):
        trackKeys = [key for key in sorted(track.keys()) if (start <= key <=end)]
        switchBehavior = 0.0
        for idx in range(len(trackKeys)-2):
            if ((track[trackKeys[idx]]['cam'] != track[trackKeys[idx+1]]['cam']) and 
                 track[trackKeys[idx]]['cam'] == track[trackKeys[idx+2]]['cam']):
               switchBehavior += 1
        return switchBehavior/additionCount     
        
    def getTrackByAnimalID(self, animalID):
        t = {}
        unassignedTracks = []
        tracksToEvaluate = []
        
        for trackNo, track in self.tracks.items():
            trackIdentity, trackIdentityScore = self.getTrackConfidence(track)
            if trackIdentity == animalID:
                tracksToEvaluate.append((trackNo, trackIdentityScore))
            
        sortedTracksToEvaluate = sorted(tracksToEvaluate, key=lambda x: x[1], reverse=True)

        for idx, score in sortedTracksToEvaluate:
            track = self.tracks[idx]
            tTemp = {key: value for key, value in t.items()}
            overlap = self.isOverlapTrack(tTemp, track)
            for time, data in sorted(track.items()):
                tTemp[time] = data
            if self.isMixTrackRegion(tTemp, len(track), min(track.keys()), max(track.keys())) <= 0.2 and overlap < 0.35:
                for time, data in sorted(track.items()):
                    t[time] = data 
            else:
                unassignedTracks.append(track)
                    
        return t, unassignedTracks

    def getReplacedTrack(self, animalID):
        t = {}
        tracksToEvaluate = []
        
        for trackNo, track in self.tracks.items():
            trackIdentity, trackIdentityScore = self.getTrackConfidence(track)
            if trackIdentity == animalID:
                tracksToEvaluate.append((trackNo, trackIdentityScore))
            
        sortedTracksToEvaluate = sorted(tracksToEvaluate, key=lambda x: x[1], reverse=True)

        initialTracksToRemove = []
        for idx, score in sortedTracksToEvaluate:
            track = self.tracks[idx]
            tTemp = {key: value for key, value in t.items()}
            overlap = self.isOverlapTrack(tTemp, track)
            for time, data in sorted(track.items()):
                tTemp[time] = data
            if self.isMixTrackRegion(tTemp, len(track), min(track.keys()), max(track.keys())) <= 0.2 and overlap < 0.35:
                for time, data in sorted(track.items()):
                    t[time] = data
                    initialTracksToRemove.append(idx)
        
        t = {}            
        sortedTracksToEvaluate = [(idx, score) for idx, score in sortedTracksToEvaluate if idx not in initialTracksToRemove]
        for idx, score in sortedTracksToEvaluate:
            track = self.tracks[idx]
            tTemp = {key: value for key, value in t.items()}
            overlap = self.isOverlapTrack(tTemp, track)
            for time, data in sorted(track.items()):
                tTemp[time] = data
            if self.isMixTrackRegion(tTemp, len(track), min(track.keys()), max(track.keys())) <= 0.2 and overlap < 0.35:
                for time, data in sorted(track.items()):
                    t[time] = data
                    
        return t
    
    def getTracksMaster(self):
        d = {}
        unassignedTracks = {}    
        for idx in range(1, 22):
            if idx not in self.missingAnimals:
                d[idx], unassignedTracks[idx] = self.getTrackByAnimalID(idx)
            else:
                d[idx], unassignedTracks[idx] = {}, {}
        return d, unassignedTracks


class Tracker:
    def __init__(self, penID: int, timestamp: str, inferenceFolderPath: str, saveFolderPath: str):
        # Parameters
        self.penID = penID
        self.timestamp = timestamp
        self.inferenceFolderPath = inferenceFolderPath
        self.allFolders = os.listdir(self.inferenceFolderPath)
        self.saveFolderPath = saveFolderPath
        self.splitTimestamp = (str(self.timestamp)[:4] + "_"  + 
                               str(self.timestamp)[4:6] + "_" + 
                               str(self.timestamp)[6:8] + "_" + 
                               str(self.timestamp)[9:11] + "_" + 
                               str(self.timestamp)[11:])

        with open('../analysis/missingAnimals.json', 'r') as file:
            self.missingAnimals = json.load(file)
            experimentID = str(self.inferenceFolderPath[-1]) if str(self.inferenceFolderPath[-1]) in ["2", "3", "4", "5", "6", "7", "9"] else "8"
            self.missingAnimals = self.missingAnimals[experimentID]

        #self.splitTimestampPreviousHour = (str(self.timestamp)[:4] + "_"  + 
        #                                   str(self.timestamp)[4:6] + "_" + 
        #                                   str(self.timestamp)[6:8] + "_" + 
        #                                   str(int(self.timestamp[9:11])-1).zfill(2) + "_" + 
        #                                   str(self.timestamp)[11:])

        
        previousFolder = self.findPreviousInferenceFolder(self.timestamp, self.allFolders)
        if previousFolder is not None:
            self.previousTimestamp = previousFolder.split("_", 1)[1]
            print(int(self.previousTimestamp.split("_", 1)[1]), int(self.timestamp[9:11]))
            if not (int(self.previousTimestamp.split("_", 1)[1][:2]) == int(self.timestamp[9:11]) or 
                    int(self.previousTimestamp.split("_", 1)[1][:2]) == int(self.timestamp[9:11])-1):
                self.previousTimestamp = "00000000_000000"
        else:
            self.previousTimestamp = "00000000_000000"
        self.splitTimestampPreviousHour = (str(self.previousTimestamp)[:4] + "_"  + 
                                           str(self.previousTimestamp)[4:6] + "_" + 
                                           str(self.previousTimestamp)[6:8] + "_" + 
                                           str(self.previousTimestamp)[9:11] + "_" + 
                                           str(self.previousTimestamp)[11:])
        
        self.saveSubFolderPath = self.saveFolderPath + "/" + str(self.penID) + "_" + self.splitTimestamp
        self.previousTrackingFolderPath = self.saveFolderPath + "/" + str(self.penID) + "_" + self.splitTimestampPreviousHour
        print(self.previousTrackingFolderPath)
        
        # Track names
        self.initialTrackName = 'tracksInitial'
        self.piecewiseTrackName = 'tracksPiecewise'
        self.individualTrackName = 'tracks'
        self.initialTrackPath = str(self.saveSubFolderPath + "/" + self.initialTrackName + ".json")
        self.piecewiseTrackPath = str(self.saveSubFolderPath + "/" + self.piecewiseTrackName + ".json")
        self.individualTrackPath = str(self.saveSubFolderPath + "/" + self.individualTrackName + ".json")

        # Make directory if not exist
        if not os.path.exists(self.saveFolderPath):
            os.mkdir(self.saveFolderPath)
        if not os.path.exists(self.saveSubFolderPath):
            os.mkdir(self.saveSubFolderPath)   
            
        # Verify Existence of annotations
        self.inferenceFolders = [self.inferenceFolderPath + "/D68VOGH" + str(self.penID) + str(i) + "_" +  str(self.timestamp) for i in range(1, 5)]
        for folder in self.inferenceFolders:
            assert os.path.exists(folder)

    @staticmethod
    def findPreviousInferenceFolder(currentTimestamp, folders):
        currentDT = datetime.strptime(currentTimestamp, "%Y%m%d_%H%M%S")
        validFolders = []
        
        for folder in folders:
            try:
                folderTimestamp = folder.split("_", 1)[1]
                folderDT = datetime.strptime(folderTimestamp, "%Y%m%d_%H%M%S")
                if (folderDT.date() == currentDT.date() and 
                    8 <= folderDT.hour < 19):
                    validFolders.append((folderDT, folder))
            except Exception as e:
                pass
                
        validFolders.sort()
        previousFolder = None
        for folderDT, folder in validFolders:
            if folderDT < currentDT:
                previousFolder = folder
            else:
                break
    
        return previousFolder
    
    def saveJSON(self, fileName: str, data) -> None:
        with open(os.path.join(self.saveSubFolderPath, f'{fileName}.json'), 'w') as f:
            json.dump(data, f, indent=4)

    def readJSON(self, fileName: str) -> dict: 
        with open(fileName, 'r') as f:
            data = json.load(f)
            data = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in data.items()}
        return data

    @staticmethod
    def getPercentCoverage(tracksA: dict, tracksB: dict, verbose=True) -> tuple[float, float, list]:
        lengthTracksA = sum([len(tracksA[t]) for t in tracksA.keys()])
        lengthTracksB = sum([len(idx) for t in tracksB.keys() for idx in tracksB[t]])
        missingTracks = sorted([int(t) for t in tracksA.keys() if len(tracksA[t]) == 0])
        try:
            ratioTrackA = round(lengthTracksA/(lengthTracksA+lengthTracksB), 3)
            ratioTrackB = round(lengthTracksB/(lengthTracksA+lengthTracksB), 3)
            if verbose:
                print(f"Assigned tracks: {ratioTrackA*100}%, Unassigned tracks: {ratioTrackB*100}%, Missing tracks: {missingTracks}")
            else:
                return ratioTrackA, ratioTrackB, missingTracks
        except:
            if verbose:
                print(f"Assigned tracks: {0*100}%, Unassigned tracks: {100*100}%, Missing tracks: {[]}")
            else:
                return 0, 1, []     
    
    @staticmethod
    def getLastLocations(tracks: dict) -> dict:
        d = {}
        for key in sorted(tracks.keys()):
            if len(tracks[key].keys()) > 0:
                lastKey = sorted(list(tracks[key].keys()))[-1]
                d[key] = {"time": int(lastKey),
                          "cam": int(tracks[key][lastKey]["cam"]),
                          "sector": tracks[key][lastKey]["sector"]}
        return d

    @staticmethod
    def reassignTracksPriorInformation(tracksPrior: dict, trackCurrent: dict, verbose=False) -> list:
        def getManhattanDistance(sectorA: list, sectorB: list) -> int:
            return int(abs(sectorA[0]-sectorB[0]) + abs(sectorA[1]-sectorB[1]))

        def calculateDisplacement(track: dict) -> float:
            keys = sorted(list(track.keys()))
            displacement = 0.0
            for keyIndex in range(1, len(keys)):
                if track[keys[keyIndex-1]]['cam'] == track[keys[keyIndex]]['cam']:
                    displacement += getManhattanDistance(track[keys[keyIndex-1]]['sector'], track[keys[keyIndex]]['sector'])
                else:
                    displacement += 50
            return float(displacement)
        
        reassignments = []
        for key in sorted(trackCurrent.keys()):
            if (len(trackCurrent[key].keys()) > 0) and (key in tracksPrior) and (len(tracksPrior[key].keys()) > 0):
                firstKey = sorted(list(trackCurrent[key].keys()))[0]
                isSameCam = trackCurrent[key][firstKey]['cam'] == tracksPrior[key]['cam']
                distance = getManhattanDistance(trackCurrent[key][firstKey]['sector'], tracksPrior[key]['sector'])
                if (not isSameCam) or (isSameCam and distance > 60):
                    for priorKey in sorted(tracksPrior.keys()):
                        isSameCam = trackCurrent[key][firstKey]['cam'] == tracksPrior[priorKey]['cam']
                        distance = getManhattanDistance(trackCurrent[key][firstKey]['sector'], tracksPrior[priorKey]['sector'])
                        if isSameCam and (distance < 60) and ((calculateDisplacement(trackCurrent[key]) + calculateDisplacement(trackCurrent[priorKey])) < 3500):
                            reassignments.append((key, priorKey))
                if verbose:
                    print(key, "->",  
                          "Cam Current:", trackCurrent[key][firstKey]['cam'], "Cam Prior:", tracksPrior[key]['cam'],
                          "Sector Current:", trackCurrent[key][firstKey]['sector'], "Sector Prior:", tracksPrior[key]['sector'],
                          "Distance:", distance,
                          "Prior Key:", tracksPrior[key]['time'])
        
        return reassignments

    @staticmethod
    def assignFullyMissingTracks(tracksPrior: dict, trackCurrent: dict) -> dict:
        d = trackCurrent.copy()
        currentEmptyTracks = []
        for key in sorted(trackCurrent.keys()):
            if len(trackCurrent[key]) == 0:
                currentEmptyTracks.append(key)

        priorEmptyTracks = []
        for key in sorted(tracksPrior.keys()):
            if len(tracksPrior[key]) == 0:
                priorEmptyTracks.append(key)
        
        tracksToAssign = [trackID for trackID in currentEmptyTracks if trackID not in priorEmptyTracks]
        for trackID in tracksToAssign:
            d[trackID][0] = tracksPrior[trackID][max(list(tracksPrior[trackID].keys()))]
        
        return d
    
    @staticmethod
    def repeatTracks(tracks: dict) -> dict:
        maxTimePosition = 0
        for key in tracks.keys():
            if len(tracks[key]) > 0:
                tempMaxValue = max(list(tracks[key].keys()))
                if tempMaxValue > maxTimePosition:
                    maxTimePosition = tempMaxValue

        for key in tracks.keys():
            if len(tracks[key]) > 0:
                tempMaxValue = max(list(tracks[key].keys()))
                if tempMaxValue != maxTimePosition:
                    tracks[key][maxTimePosition] = tracks[key][tempMaxValue]

        return tracks

        
    def generateTracks(self):        
        # Get initial tracks
        trackerStep1 = PigTrackInitializer(inferencePen=self.penID,
                                           timestamp=self.timestamp,
                                           inferenceFolder=self.inferenceFolderPath,
                                           blackOutMask=getBlackoutMask())
        self.saveJSON(fileName=self.initialTrackName, data=trackerStep1.inferenceDict)

        # Get piecewise tracks
        trackerStep2 = PiecewiseTracker(initialLocationFile=self.initialTrackPath)
        trackerStep2.getTracks()
        self.saveJSON(fileName=self.piecewiseTrackName, data=trackerStep2.tracksDict)

        # Check Previous ending locations
        if os.path.exists(self.previousTrackingFolderPath):
            previousTracks = self.readJSON(self.previousTrackingFolderPath + "/" + self.individualTrackName + ".json")
            lastLocations = self.getLastLocations(previousTracks)   
            trackerStep3 = TrackerIndividual(self.piecewiseTrackPath, self.missingAnimals)
            masterTracks, unassignedTracks = trackerStep3.getTracksMaster()
            reassignments = self.reassignTracksPriorInformation(lastLocations, masterTracks, verbose=True)
            for oldKey, newKey in reassignments:
                masterTracks[newKey] = masterTracks[oldKey]
                del masterTracks[oldKey]
                newTrackOldKey = trackerStep3.getReplacedTrack(oldKey)
                if len(newTrackOldKey) > 0:
                    masterTracks[oldKey] = newTrackOldKey
                else:
                    masterTracks[oldKey] = {}
            masterTracks = self.assignFullyMissingTracks(previousTracks, masterTracks)
            masterTracks = self.repeatTracks(masterTracks)
            print(self.getPercentCoverage(masterTracks, unassignedTracks))
            self.saveJSON(fileName=self.individualTrackName, data=masterTracks)
            
        else:
            trackerStep3 = TrackerIndividual(self.piecewiseTrackPath, self.missingAnimals)
            masterTracks, unassignedTracks = trackerStep3.getTracksMaster() 
            masterTracks = self.repeatTracks(masterTracks)
            print(self.getPercentCoverage(masterTracks, unassignedTracks))
            self.saveJSON(fileName=self.individualTrackName, data=masterTracks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SecurePig Individual Tracker.")
    parser.add_argument('--pen', type=int, help='Pen ID number (options: 2, 3)')
    parser.add_argument('--timestamp', type=str, help='Timestamp in format YYYYMMDD_HHMMSS (example: 20240618_100000)')
    parser.add_argument('--inferenceFolderPath', type=str, help='Path to inference folder (example: ../inference)')
    parser.add_argument('--saveFolderPath', type=str, help='Path to save folder (example: masterTracks)')

    args = parser.parse_args()
    t0 = time.time()
    
    tracker = Tracker(penID=args.pen,
                      timestamp=args.timestamp,
                      inferenceFolderPath=args.inferenceFolderPath,
                      saveFolderPath=args.saveFolderPath)
    tracker.generateTracks()
    t1 = time.time()
    td = t1 - t0
    print(f"Pen: {args.pen}, Time: {args.timestamp} generated {td:.2f} seconds!")

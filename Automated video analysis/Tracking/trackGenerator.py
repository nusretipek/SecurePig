#!/usr/bin/env python3
import os
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse

def getBlackoutMask(image_width=640, image_height=360, leftAngleParameters=[40, 120], rightAngleParameters=[40, 120]):
    mask = np.ones((image_height, image_width), dtype=np.uint8)
    for y in range(image_height):
        x_start = (int((image_height-leftAngleParameters[0] - y) * (leftAngleParameters[1] / (image_height - leftAngleParameters[0]))) 
                   if y <= image_height-leftAngleParameters[0] else 0)
        mask[y, :x_start] = 0
        
    for y in range(image_height):
        x_end = (image_width - 
                 int((image_height-rightAngleParameters[0] - y) * (rightAngleParameters[1] / (image_height - rightAngleParameters[0]))) 
                 if y <= image_height-rightAngleParameters[0] else image_width)
        mask[y, x_end:] = 0
    return mask


class PigTrackInitializer:
    def __init__(self, inferencePen, timestamp, inferenceFolder, blackOutMask):
        # Parameters
        self.inferencePen = inferencePen
        self.timeStamp = timestamp
        self.inferenceFolder = inferenceFolder
        self.blackOutMask = blackOutMask
        self.tracks = {}
        self.inferenceFolders = [self.inferenceFolder + "/D68VOGH" + str(self.inferencePen) + str(i) + "_" +  self.timeStamp for i in range(1, 5)]
        
        # Parse JSON file
        self.inferenceDict = self.parseTimeline()
        #self.cleanNonPigBBOXES()
    
    # Static methods
    
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

    @staticmethod
    def verifyBBOX(vid_path, frame_id, bbox, port=5013):
        response = requests.post(f'http://127.0.0.1:{port}/detectPig', 
                                 json={'video_path': vid_path,
                                       'frame_id': frame_id,
                                       'bbox': bbox})
        result = response.json()['result']
        return result
    
    # Class methods
    def intersectionWithBlackout(self, BBox):
        xMin, yMin, xMax, yMax = BBox
        xMin = max(0, xMin)
        yMin = max(0, yMin)
        xMax = min(self.blackOutMask.shape[1], xMax)
        yMax = min(self.blackOutMask.shape[0], yMax)
        BBoxRegion = self.blackOutMask[yMin:yMax, xMin:xMax]
        blackOutPixels = np.sum(BBoxRegion == 0)
        BBoxArea = (xMax - xMin) * (yMax - yMin)
        return blackOutPixels / BBoxArea

    def applyMaskToBlackOutArea(self, d, threshold=0.25):
        for key in sorted(d):
            toBeRemoved = []
            for idx, detection in enumerate(d[key]):
                if self.intersectionWithBlackout(detection['bbox']) > threshold:
                    toBeRemoved.append(idx)
            d[key] = [detection for idx, detection in enumerate(d[key]) if idx not in toBeRemoved]
        return d
        
    def MatchBoundingBBoxes(self, BBoxesX, BBoxesY, threshold=0.5):
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
    
    def cleanFalsePositiveDetections(self, d, t=10):
        keyList = list(sorted(d.keys()))
        for keyIndex in range(1, len(keyList)):
            PrevObjectCount = len(d[keyList[keyIndex-1]])
            currentObjectCount = len(d[keyList[keyIndex]])
            if PrevObjectCount >= currentObjectCount:
                continue
            else:
                count, includeCount = 0, 0
                for i in range(1, t+1):
                    if keyIndex+i < len(keyList):
                        includeCount += 1
                        nextObjectCount = len(d[keyList[keyIndex+i]])
                        if nextObjectCount < currentObjectCount and nextObjectCount == PrevObjectCount:
                            count += 1
                if includeCount > 0 and count/includeCount > 0.5:
                    toBeKept = []
                    BBoxesX = [i['bbox'] for i in d[keyList[keyIndex-1]]]
                    BBoxesY = [i['bbox'] for i in d[keyList[keyIndex]]]
                    #print(BBoxesX)
                    pairs = self.MatchBoundingBBoxes(BBoxesX, BBoxesY)
                    for prevBBoxId, BBoxId in pairs:
                        toBeKept.append(BBoxId)
                    d[keyList[keyIndex]] = [detection for idx, detection in enumerate(d[keyList[keyIndex]]) if idx in toBeKept]
        return d

    def parseTimeline(self):
        masterDict = {}
        for folderID, folder in enumerate(self.inferenceFolders):
            inferenceDict = {}
            for file in sorted([file for file in os.listdir(folder) if file.endswith('.json')]):
                with open(os.path.join(folder, file), "r") as fp:
                    data = json.load(fp)
                    detections = []
                    for detection in data:    
                        detections.append({"pen": folderID+1,
                                           "bbox": detection["bbox"],
                                           "id": detection["id"],
                                           "idProb": detection["idProb"]})
                inferenceDict[int(file.rsplit(".", 1)[0])] = detections
            
            inferenceDict = self.applyMaskToBlackOutArea(d=inferenceDict, threshold=0.35)
            inferenceDict = self.cleanFalsePositiveDetections(d=inferenceDict, t=5)
            
            #for key, value in inferenceDict.items():
            #    bboxes = []
            #    for bbox in value:
            #        bboxes.append([folderID+1] + bbox)
            #    inferenceDict[key] = bboxes
                
            for key, value in inferenceDict.items():
                if key in masterDict:
                    masterDict[key] = masterDict[key] + inferenceDict[key]
                else:
                    masterDict[key] = inferenceDict[key]
                    
        return {k: v for k, v in sorted(list(masterDict.items()))}

    def cleanNonPigBBOXES(self, tStop=100):
        for i in range(tStop):
            if i in self.inferenceDict:
                toRemove = []
                for j, bbox in enumerate(self.inferenceDict[i]):
                    vidPath = "/mnt/ilvoPigs/Experiment3/" + "D68VOGH" + str(self.inferencePen) + str(bbox[0]) + "." +  self.timeStamp + ".mp4"
                    if not self.verifyBBOX(vidPath, i, bbox[1:]):
                        toRemove.append(j)
                self.inferenceDict[i] = [k for idk, k in enumerate(self.inferenceDict[i]) if idk not in toRemove]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Pig Track Generator")
    parser.add_argument("inference_pen", type=int, help="Inference pen ID")
    parser.add_argument("timestamp", type=str, help="Timestamp in format YYYYMMDD_HHMMSS")
    parser.add_argument("inferenceFolder", type=str, help="Inference folder location")
    parser.add_argument("saveFolder", type=str, help="Inference folder location")

    args = parser.parse_args()
    img_width, img_height = 640, 360
    mask = getBlackoutMask(img_width, img_height)

    tracker = PigTrackInitializer(inferencePen=args.inference_pen, timestamp=args.timestamp, inferenceFolder=args.inferenceFolder, blackOutMask=mask)

    # Save to JSON file with 4-space indentation
    with open(args.saveFolder + '/' + 'initialTracks.json', 'w') as json_file:
        json.dump(tracker.inferenceDict, json_file, indent=4)



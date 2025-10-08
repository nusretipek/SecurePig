# Import libraries

## General
import os
import sys
import json
import glob
import time
import random
import requests
import numpy as np
import matplotlib.pyplot as plt

## Image
import cv2
from PIL import Image, ImageDraw
from scipy.ndimage import label
import torchvision.transforms as transforms

## SAM2
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Parameters
deviceId = 2
erosionKernel = 3
erosionIterations = 0
randomPointsCount = 20
minimumComponentSize = 50
seed = 0

# Select the device for computation
torch.set_num_threads(3)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(deviceId)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# Build Model / Predictor
sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model, max_sprinkle_area=50) 

# Build Reidentification Model
modelReID = torch.load("../reid/max_val_accuracy.pth", weights_only=False, map_location=torch.device('cuda:3'))
modelReID = modelReID.to(torch.device('cuda:3'))
modelReID.eval()
transformReID = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def reID(model, image): 
    model.eval()
    cv2.imwrite("tempSegmentExp9.png", image)
    image = Image.open("tempSegmentExp9.png").convert('RGB')
    image = transformReID(image)
    image = image.unsqueeze(0) 
    image = image.to(torch.device('cuda:3'))
    with torch.no_grad():
        output = model(image)
        output = torch.nn.functional.softmax(output, dim=1)
    pred = torch.argmax(output, 1).cpu().item()
    probability = output[0][pred].cpu().item()
    #print(pred, probability)
    return pred, probability

def getImage(path, resize=None):
    image = Image.open(path)
    if resize is not None:
        image = image.resize(resize, Image.Resampling.BICUBIC)
    image = np.array(image.convert("RGB"))
    return image

def getUNetMask(path, port=5038):
    responseUnet = requests.post(f'http://127.0.0.1:{port}/inferenceUnet', json={'framePath': path})
    points = np.array(responseUnet.json()['unet'], dtype=np.float64)
    return points

def erodeMask(mask, kernelSize=3, iterations=5):
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    erodedMask = cv2.erode(mask, kernel, iterations=iterations)
    return erodedMask

def erodeLowerHalf(mask, kernelSize=3, iterations=5):
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    height, width = mask.shape
    erodedMask = mask.copy()
    lowerHalf = mask[height//2:, :]
    erodedLowerHalf = cv2.erode(lowerHalf, kernel, iterations=iterations)
    erodedMask[height//2:, :] = erodedLowerHalf
    return erodedMask

def getRandomComponentsCoordinatesT(mask, numSamples=5, minComponentArea=10, seed=None):
    samplePoints = []
    if seed is not None:
        np.random.seed(seed)
    rng = np.random.default_rng(seed)
        
    components, numFeatures = label(mask)
    for idx in range(1, numFeatures+1): 
        coords = np.argwhere(components == idx)
        
        if len(coords) > minComponentArea:  
            randomCoordIndices = rng.choice(len(coords), size=min(numSamples, len(coords)), replace=False)
            randomCoords = coords[randomCoordIndices]
            randomCoords[:, [0, 1]] = randomCoords[:, [1, 0]]
            samplePoints.extend(randomCoords.tolist())
    return samplePoints

def getRandomComponentsCoordinates(mask, numSamples=5, minComponentArea=10, seed=None):
    samplePoints = []
    if seed is not None:
        np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    components, numFeatures = label(mask)
    for idx in range(1, numFeatures + 1):
        coords = np.argwhere(components == idx)
        
        if len(coords) > minComponentArea:
            tiles = int(np.ceil(np.sqrt(numSamples)))
            x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), tiles + 1, dtype=int)
            y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), tiles + 1, dtype=int)
            selected = [rng.choice(coords[(coords[:, 0] >= x_bins[i]) & (coords[:, 0] < x_bins[i + 1]) &
                                          (coords[:, 1] >= y_bins[j]) & (coords[:, 1] < y_bins[j + 1])])
                        for i in range(tiles) for j in range(tiles)
                        if ((coords[:, 0] >= x_bins[i]) & (coords[:, 0] < x_bins[i + 1]) &
                            (coords[:, 1] >= y_bins[j]) & (coords[:, 1] < y_bins[j + 1])).any()]
            samplePoints.extend(coord[::-1].tolist() for coord in selected[:numSamples])
    return samplePoints

def maskToBBOX(mask):
    yIndices, xIndices = np.where(mask)
    xMin, xMax = xIndices.min(), xIndices.max()
    yMin, yMax = yIndices.min(), yIndices.max()
    return [int(xMin), int(yMin), int(xMax), int(yMax)]

def getIntersectionWithBlackoutArea(BBox, regionWidth=640, regionLeft=80, regionRight=540): #regionRight=540
    xMin, yMin, xMax, yMax = BBox
    areaBBox = (xMax - xMin) * (yMax - yMin)
    height = max(0, yMax - yMin)
    
    overlapRight, overlapLeft = 0, 0
    xOerlapLeft = max(0, min(regionLeft, xMax) - max(0, xMin))
    xOerlapRight = max(0, min(regionWidth, xMax) - max(regionRight, xMin))
    
    if xOerlapLeft > 0:
        overlapRight = (xOerlapLeft * height) / areaBBox
    if xOerlapRight > 0:
        overlapLeft = (xOerlapRight * height) / areaBBox
    return overlapRight + overlapLeft

def getIntersectionWithActiveArea(predictionDict, mask):
    for idx in predictionDict.keys():
        BBox = predictionDict[idx]['bbox'] 
        cropRegion = mask[BBox[1]:BBox[3], BBox[0]:BBox[2]]
        areaBBox = (BBox[2] - BBox[0]) * (BBox[3] - BBox[1])
        height, width = cropRegion.shape
        try:
            ratioHW = max(height/width, width/height)
            centerX, centerY = width // 2, height // 2
            yIndices, xIndices = np.indices((height, width))
            distance = np.abs(yIndices - centerY) + np.abs(xIndices - centerX)
            maxDistance = distance.max()
            weightMatrix = 1 - (distance / maxDistance)
            intersectionCropRegion = (((weightMatrix * cropRegion).sum() / areaBBox) * 
                                      (1 - predictionDict[idx]['intersectionBlackoutArea'])**2)
            predictionDict[idx]['unetMaskScore'] = float(intersectionCropRegion)
            predictionDict[idx]['ratioHeightWidth'] = float(ratioHW)
        except:
            predictionDict[idx]['unetMaskScore'] = float(0)
            predictionDict[idx]['ratioHeightWidth'] = float(0)


def getSAM2PredictorMasks(predictor, unetMask, framePath, samplePoints):
    d = {}

    predictor.set_image(getImage(framePath))
    for idx, coords in enumerate(samplePoints):
        inputPoint = np.array([coords])
        mask, score, _ = predictor.predict(point_coords=inputPoint,
                                           point_labels=np.array([1]),
                                           multimask_output=False)

        if np.array(mask[0]).sum():
            BBox = maskToBBOX(mask[0])
            d[idx] = {'mask': mask[0].astype(np.int8),
                      'bbox': BBox,
                      'score': float(score[0]),
                      'area': (BBox[2] - BBox[0]) * (BBox[3] - BBox[1]),
                      'intersectionBlackoutArea': getIntersectionWithBlackoutArea(BBox)}
    getIntersectionWithActiveArea(d, unetMask)
    return d

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

def processBBoxesWithArea(predictionDict, areaThresholdMin=3750, areaThresholdMax=32000):
    d = predictionDict.copy()
    for idx in list(d.keys()):
        if d[idx]['area'] < areaThresholdMin or d[idx]['area'] > areaThresholdMax:
            del d[idx] 
    return d

def processBBoxesWithBlackoutArea(predictionDict, thresholdIOU=0.6, intersectionThreshold=0.02):
    d = predictionDict.copy()
    for idx in list(d.keys()):
        if (d[idx]['intersectionBlackoutArea'] > thresholdIOU or 
            d[idx]['unetMaskScore'] < intersectionThreshold):
            del d[idx]
    return d

def processBBoxesWithRatioHW(predictionDict, ratioHWMax=5.5):
    d = predictionDict.copy()
    for idx in list(d.keys()):
        if d[idx]['ratioHeightWidth'] > ratioHWMax:
            del d[idx] 
    return d

def processBBoxesWithIOU(predictionDict, thresholdIOU=0.95):
    d = predictionDict.copy()
    keys = list(d.keys())
    indicesToRemove = []
    for idx in range(len(keys)):
        for idy in range(idx+1, len(keys)):
            iou = calculateIOU(d[keys[idx]]['bbox'], d[keys[idy]]['bbox']) 
            if iou > thresholdIOU:
                removeIndex = keys[idy] if d[keys[idx]]['score'] > d[keys[idy]]['score'] else keys[idx]
                indicesToRemove.append(removeIndex)      
    d = {key: d[key] for key in keys if key not in indicesToRemove}
    return d

def processBBoxesWithConfidenceScore(predictionDict, thresholdScore=0.9):
    d = predictionDict.copy()
    for idx in list(d.keys()):
        if d[idx]['score'] < thresholdScore:
            del d[idx]
    return d

def processBBoxesWithMaskedUnetScore(predictionDict, mask, thresholdScore=0.02):
    sortedpredictionDict = dict(sorted(predictionDict.items(), key=lambda item: item[1]['score'], reverse=True))
    points = mask.copy()
    for idx in sortedpredictionDict.keys():
        BBox = sortedpredictionDict[idx]['bbox'] 
        cropRegion = points[BBox[1]:BBox[3], BBox[0]:BBox[2]]
        areaBBox = (BBox[2] - BBox[0]) * (BBox[3] - BBox[1])
        height, width = cropRegion.shape
        try:
            ratioHW = max(height/width, width/height)
            centerX, centerY = width // 2, height // 2
            yIndices, xIndices = np.indices((height, width))
            distance = np.abs(yIndices - centerY) + np.abs(xIndices - centerX)
            maxDistance = distance.max()
            weightMatrix = 1 - (distance / maxDistance)
            intersectionCropRegion = (((weightMatrix * cropRegion).sum() / areaBBox) * 
                                      (1 - sortedpredictionDict[idx]['intersectionBlackoutArea'])**2)
            sortedpredictionDict[idx]['unetMaskScoreMasked'] = float(intersectionCropRegion)
            points[BBox[1]:BBox[3], BBox[0]:BBox[2]][points[BBox[1]:BBox[3], BBox[0]:BBox[2]] == 1] = 0.2
        except:
            sortedpredictionDict[idx]['unetMaskScoreMasked'] = float(0)
    for idx in list(sortedpredictionDict.keys()):
        if sortedpredictionDict[idx]['unetMaskScoreMasked'] < thresholdScore:
            del sortedpredictionDict[idx]
    return sortedpredictionDict

def verbosePredictionDictionary(predictionDict):
    for idx, key in enumerate(list(predictionDict.keys())):
        print("Mask " + str(idx) + "\n-------")
        print("Area: " + str(predictionDict[key]["area"]).ljust(6) + 
              " - Blackout Area: " + str(predictionDict[key]["intersectionBlackoutArea"]) +
              " - SAM2 Score: " + str(round(predictionDict[key]["score"], 4)) +
              " - Unet Score: " + str(round(predictionDict[key]["unetMaskScore"], 4)) + 
              " - Ratio Height Width: " + str(round(predictionDict[key]["ratioHeightWidth"], 4)),
              " - Ratio Unet Area: " + str(round(predictionDict[key]["unetMaskScore"]*10000/predictionDict[key]["area"], 4)))
        print("Bounding Box: \n" + 
              "\tX min: " + str(predictionDict[key]["bbox"][0]).ljust(3) +
              " - X max: " + str(predictionDict[key]["bbox"][2]) +          
              "\n\tY min: " + str(predictionDict[key]["bbox"][1]).ljust(3) +
              " - Y max: " + str(predictionDict[key]["bbox"][3]) + "\n")

def drawBBOXES(framePath, predictionDict, savePath=None):
    image = Image.open(framePath).convert("RGBA")
    draw = ImageDraw.Draw(image)
    for idx in list(predictionDict.keys()):
        xMin, yMin, xMax, yMax = predictionDict[idx]['bbox']
        draw.rectangle([xMin, yMin, xMax, yMax], outline="purple", width=4)
    if savePath is not None:
        image.save(savePath)
    return image

def overlay_mask_on_image(image, binary_mask):
    yellow_overlay = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
    yellow_overlay[binary_mask == 1] = [255, 255, 0, 128]
    yellow_overlay_img = Image.fromarray(yellow_overlay)
    image_with_overlay = Image.alpha_composite(image, yellow_overlay_img)
    image_with_overlay = image_with_overlay.convert("RGB")
    return image_with_overlay

def saveInference(frame, predictionDict, savePath):
    inference = []
    for idx in predictionDict.keys():
        tempFrame = frame.copy() * predictionDict[idx]["mask"][:, :, np.newaxis]
        xmin, ymin, xmax, ymax = predictionDict[idx]["bbox"]
        cropped = tempFrame[ymin:ymax, xmin:xmax]
        reIDLabel, reIDProbability = reID(modelReID, cropped)
        #print(reIDLabel, reIDProbability)
        inference.append({'bbox': predictionDict[idx]['bbox'], 'id': reIDLabel, 'idProb': reIDProbability})
    with open(savePath, 'w') as file:
        json.dump(inference, file)

def processFrame(frame, videoName, frameNo):
    t0 = time.time()
    
    # Convert and write frame
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    framePath = videoName + '/tempFrame.png'
    cv2.imwrite(framePath, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frameRaw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if os.path.basename(videoName)[8] == "5":
        erosionIterations = 10
        unetMask = getUNetMask(framePath)
        erodedMask = erodeMask(unetMask, erosionKernel, erosionIterations)
    else:
        erosionIterations = 0
        unetMask = getUNetMask(framePath)
        erodedMask = erodeLowerHalf(unetMask, erosionKernel, erosionIterations)
    
    # Get inference
    samplePoints = getRandomComponentsCoordinates(erodedMask, randomPointsCount, minimumComponentSize, seed=seed)

    SAM2Masks = getSAM2PredictorMasks(predictor, erodedMask, framePath, samplePoints)    
    SAM2Masks = processBBoxesWithArea(SAM2Masks)
    SAM2Masks = processBBoxesWithRatioHW(SAM2Masks)
    SAM2Masks = processBBoxesWithIOU(SAM2Masks)
    SAM2Masks = processBBoxesWithConfidenceScore(SAM2Masks, 0.8)
    SAM2Masks = processBBoxesWithMaskedUnetScore(SAM2Masks, unetMask)
    SAM2Masks = dict(zip(range(len(SAM2Masks)), list(SAM2Masks.values())))

    img = drawBBOXES(framePath, SAM2Masks)
    img = overlay_mask_on_image(img, unetMask)
    
    savePath = videoName + "/" + str(frameNo) + ".png"
    saveInference(frameRaw, SAM2Masks, savePath.replace('.png', '.json'))
    os.remove(framePath)
    print(savePath, time.time() - t0)

def calculate_matching_iou(file1, file2):
    # Load JSON files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Extract bounding boxes
    bboxes1 = [item['bbox'] for item in data1]
    bboxes2 = [item['bbox'] for item in data2]

    if len(bboxes1) == 0 and len(bboxes2) == 0:
        return float(1.0)
    
    if min(len(bboxes1), len(bboxes2)) == 0:
        return float(0)

    # Create an IoU matrix
    iou_matrix = np.zeros((len(bboxes1), len(bboxes2)))
    count = max(len(bboxes1), len(bboxes2))
    iou_sum = 0
    
    # Fill the IoU matrix with IoU values
    for i, box1 in enumerate(bboxes1):
        for j, box2 in enumerate(bboxes2):
            iou_matrix[i][j] = calculateIOU(box1, box2)

    for i in range(len(bboxes1)):
        iou_sum += np.max(iou_matrix[i, :])

    return float(iou_sum/count)


if __name__ == "__main__":   
    videos = []
    for day in range(25, 29):
        for hour in range(8, 20):
            for cam in range(1, 5):
                for vid in glob.glob(f"/mnt/ilvoPigs/Experiment9/D68VOGH2{str(cam)}.202502{str(day).zfill(2)}_{str(hour).zfill(2)}*.mp4"):
                    videos.append(vid)
    
    for day in range(1, 15):
        for hour in range(8, 20):
            for cam in range(1, 5):
                for vid in glob.glob(f"/mnt/ilvoPigs/Experiment9/D68VOGH2{str(cam)}.202503{str(day).zfill(2)}_{str(hour).zfill(2)}*.mp4"):
                    videos.append(vid)
                    
    for videoPath in videos: 
        tg0 = time.time()
        
        # Directory structure
        base = "/data/nvme1/nusreti/inference/Experiment9/"
        videoName = base + videoPath.rsplit('.m', 1)[0].rsplit('/', 1)[1].replace('.', '_')
        framePath = videoName + '/tempFrame.png'
        if not os.path.exists(videoName):
            os.mkdir(videoName)

        # Global parameters and video capture
        counter = 0
        cap = cv2.VideoCapture(videoPath)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip_constant = fps * 15
        frameQueue = []

        framesDict = {}
        frameNumber = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            framesDict[frameNumber] = frame
            frameNumber += 1
        cap.release()

        # Loop frames with frame skip constant
        success = True
        while success:
            
            frameNo = int(round(counter * frame_skip_constant, 0))
            if frameNo in framesDict:
                frame = framesDict[frameNo]
            else:
                success = False
                
            if len(frameQueue) == 0 and success:
                newFrameNo = int(round((counter+1) * frame_skip_constant, 0))
                if newFrameNo < length-1:
                    frameQueue.append((frameNo, newFrameNo)) 
                else:
                    frameQueue.append((frameNo, int(length-fps))) 
                    success = False

            while len(frameQueue) > 0:
                for idx in frameQueue[0]:
                    if not os.path.exists(videoName + "/" + str(idx) + ".json"):
                        if idx in framesDict:
                            frame = framesDict[idx]
                            processFrame(frame, videoName, idx)
                        else:
                            success = False
                            
                averageIOU = calculate_matching_iou(videoName + "/" + str(frameQueue[0][0]) + ".json", 
                                                    videoName + "/" + str(frameQueue[0][1]) + ".json")

                
                frameDiff = frameQueue[0][1] - frameQueue[0][0]
                if averageIOU < 0.95 and frameDiff > 5:
                    frameAdd = int(frameDiff // 2)
                    frameQueue.append((frameQueue[0][0], frameQueue[0][0]+frameAdd))
                    frameQueue.append((frameQueue[0][0]+frameAdd, frameQueue[0][1]))
                frameQueue.pop(0)             
            
            # Progress counter
            counter += 1

        # Release video
        cap.release()
        print(videoPath, time.time() - tg0)

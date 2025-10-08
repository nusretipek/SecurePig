import os
import cv2
import json
import time
import random
import argparse
import subprocess
import numpy as np

def readJSON(fileName: str) -> dict: 
    with open(fileName, 'r') as f:
        data = json.load(f)
        data = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in data.items()}
    return data

def findClosestFrame(n, x):
    differences = np.abs(n-x)
    minIndex = np.argmin(differences)
    return n[minIndex]

def generateVideos(tracks, videoName, outputVideoPath, fps=5):    
    frameList = list(tracks[list(tracks.keys())[0]].keys())
    trackIDs = list(tracks.keys())

    for i in range(1, 5): # Camera names
        
        # Output parameters
        video_path = videoName.replace("X", str(i))
        outputVideoName = outputVideoPath + str(i) + ".mp4"

        # Read video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.5)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.5)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(outputVideoName, fourcc, fps, (frame_width, frame_height))

        # Read video frames in sequence and plot tracks
        frameNumber = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                if frameNumber in frameList:
                    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    for animalID in tracks.keys():
                        if frameNumber in tracks[animalID] and tracks[animalID][frameNumber]["cam"] == i:
                            sectorX, sectorY = tracks[animalID][frameNumber]["sector"]
                            cv2.putText(frame, str(animalID), (int(sectorX), int(sectorY)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

                out.write(frame)
                frameNumber += 1

        # Release source/out videos
        cap.release()
        out.release()
        print(f"Video saved as {outputVideoName}")

def getVideoResolution(videoPath):
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {videoPath}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    width, height = map(int, result.stdout.strip().split(","))
    return width, height

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Secire Pig Tracking Video Generator.")
    parser.add_argument('--trackFolder', type=str, help='Path to track folder')
    parser.add_argument('--saveFolder', type=str, help='Path to save folder')
    parser.add_argument('--experiment', type=int, help='Experiment number (options: 2, 3)')
    parser.add_argument('--pen', type=int, help='Pen ID (default: 2)')
    parser.add_argument('--year', type=int, help='Year of experiment')
    parser.add_argument('--month', type=int, help='Month of experiment')
    parser.add_argument('--day', type=int, help='Day of experiment')
    parser.add_argument('--hour', type=int, help='Hour of experiment')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second (default: 10)')
    args = parser.parse_args()
    t0 = time.time()
    
    # Generate time-wise extended tracks
    tracks = readJSON(f"{args.trackFolder}/{args.pen}_{str(args.year)}_{str(args.month).zfill(2)}_{str(args.day).zfill(2)}_{str(args.hour).zfill(2)}_0000/tracks.json")
    fps = 14.985
    timePoints = 3600
    extendedTracks = {}
    for idx, track in sorted(tracks.items()):
        keys = np.array(list(track.keys()), dtype=np.int64)
        extendedTrack = {}
        frameNo = 0
        if len(keys) == 0:
            extendedTracks[idx] = {}
        else:
            for i in range(timePoints):
                extendedTrack[frameNo] = track[findClosestFrame(keys, frameNo).item()] #extendedTrack[int(frameNo/0.6)] = track[findClosestFrame(keys, frameNo).item()]
                frameNo = int(round(frameNo + fps))   
            extendedTracks[idx] = extendedTrack

    # Generate videos per camera #backup/Nusret Ipek #/mnt/ilvoPigs/
    videoPath = f"/mnt/ilvoPigs/Experiment{args.experiment}/D68VOGH{args.pen}X.{str(args.year)}{str(args.month).zfill(2)}{str(args.day).zfill(2)}_{str(args.hour).zfill(2)}0000.mp4"
    outputPath = f"{args.saveFolder}{args.pen}_{str(args.month).zfill(2)}_{str(args.day).zfill(2)}_{str(args.hour).zfill(2)}_"  
    generateVideos(extendedTracks, videoPath, outputPath, args.fps)        

    # Generate merged video (4 cams)
    videos = [outputPath + str(idx) + ".mp4" for idx in range(1, 5)] 
    outputVideo = outputPath + "merged.mp4"
    width, height = getVideoResolution(videos[0])
    
    # FFMPEG for positioning videos in a 2x2 grid
    cmd = f"""
    ffmpeg \
        -i {videos[0]} -i {videos[1]} -i {videos[2]} -i {videos[3]} \
        -filter_complex "
            nullsrc=size={width*2}x{height*2} [base];
            [0:v] setpts=PTS-STARTPTS, scale={width}:{height} [v0];
            [1:v] setpts=PTS-STARTPTS, scale={width}:{height} [v1];
            [2:v] setpts=PTS-STARTPTS, scale={width}:{height} [v2];
            [3:v] setpts=PTS-STARTPTS, scale={width}:{height} [v3];
            
            [base][v0] overlay=shortest=1:x={width}:y={height} [tmp1];
            [tmp1][v1] overlay=shortest=1:x=0:y={height} [tmp2];
            [tmp2][v2] overlay=shortest=1:x={width}:y=0 [tmp3];
            [tmp3][v3] overlay=shortest=1:x=0:y=0
        " \
        -c:v libx264 -crf 23 -preset veryfast {outputVideo}
    """
    
    # Run the ffmpeg command
    subprocess.run(cmd, shell=True)
    t1 = time.time()
    td = t1 - t0
    print(f"Experiment: {args.experiment}, Pen: {args.pen}, Month: {args.month}, Day: {args.day}, Hour: {args.hour} videos generated {td:.2f} seconds!")
        
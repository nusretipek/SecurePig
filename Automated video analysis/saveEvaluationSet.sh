#!/bin/bash

# Base directory containing the videos
base_dir="/data/nvme1/nusreti/masterVideos/Evaluation"

# Output frames directory
output_base="frames"
mkdir -p "$output_base"

# Number of frames per video
frames_per_video=32

# Declare experiments and their videos
declare -A experiments=(
  [Experiment2]="3_04_15_10 3_04_15_14"
  [Experiment3]="2_06_17_10 2_06_17_11"
  [Experiment4]="3_07_08_11 3_07_08_12"
  [Experiment5]="2_09_09_11 2_09_09_12"
  [Experiment6]="3_09_30_10 3_09_30_11"
  [Experiment7]="2_12_03_11 2_12_03_12"
  [Experiment9]="2_02_25_14 2_02_25_15"
  [ExperimentRemux8]="1_02_04_11 1_02_04_12"
)

# Loop over experiments
for exp in "${!experiments[@]}"; do
    for video_base in ${experiments[$exp]}; do
        video="$base_dir/${video_base}_merged.mp4"
        echo "Processing $video"

        output_dir="$output_base/$video_base"
        mkdir -p "$output_dir"

        duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$video" | cut -d. -f1)

        interval=$((duration / frames_per_video))
        if [ $interval -eq 0 ]; then interval=1; fi

        count=1
        for ((t=0; t<duration && count<=frames_per_video; t+=interval)); do
            ts=$(printf "%02d:%02d" $((t/60)) $((t%60)))
            frame_name=$(printf "%s_%03d.jpg" "$video_base" "$count")
            ffmpeg -ss "$ts" -i "$video" -vframes 1 -q:v 2 "$output_dir/$frame_name" -y
            ((count++))
        done
    done
done

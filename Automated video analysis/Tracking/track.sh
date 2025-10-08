#!/bin/bash

# Set variables
pen=2
inference_folder="/data/nvme1/nusreti/inference/Experiment9"
save_folder="/data/nvme1/nusreti/masterTracks/Experiment9"

# Find and sort matching folders
matching_folders=$(find "$inference_folder" -maxdepth 1 -type d -name "D68VOGH${pen}1*" | sort)

# Loop through each sorted, matching subfolder
for folder_path in $matching_folders; do
  if [ -d "$folder_path" ]; then
    folder_name=$(basename "$folder_path")
    timestamp=$(echo "$folder_name" | cut -d'_' -f2-)

    echo "Processing folder: $folder_name with timestamp: $timestamp"

    python3 ./securePigTracker.py \
      --pen $pen \
      --timestamp "$timestamp" \
      --inferenceFolderPath "$inference_folder" \
      --saveFolderPath "$save_folder"
  fi
done
#!/bin/bash

# Navigate to the folder (edit this to your actual path)
cd /data/nvme1/nusreti/inference/Experiment4 || exit 1

# List directories that match the naming pattern
for day in $(ls -d D68VOGH31_* | sed -n 's/.*_\([0-9]\{8\}\)_.*$/\1/p' | sort -u); do
    echo "Processing day: $day"

    # Get all folders for this day
    folders=($(ls -d D68VOGH31_"$day"_*))

    # Find earliest folder by creation time
    earliest=""
    earliest_time=""
    latest=""
    latest_time=""

    for folder in "${folders[@]}"; do
        # Get ctime (status change time) and mtime
        ctime=$(stat -c "%Z" "$folder") # creation or status change time
        mtime=$(stat -c "%Y" "$folder") # modification time

        if [[ -z "$earliest" || "$ctime" -lt "$earliest_time" ]]; then
            earliest="$folder"
            earliest_time="$ctime"
        fi

        if [[ -z "$latest" || "$mtime" -gt "$latest_time" ]]; then
            latest="$folder"
            latest_time="$mtime"
        fi
    done

    # Calculate time difference
    diff=$((latest_time - earliest_time))
    echo "  Earliest folder: $earliest"
    echo "  Latest folder:   $latest"
    echo "  Time difference: $diff seconds ($((diff / 3600))h $(( (diff % 3600) / 60))m $((diff % 60))s)"
    echo ""
done

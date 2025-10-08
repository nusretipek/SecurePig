#!/bin/bash

# === Configuration ===
SCRIPT="movement.py"  
EXPERIMENT_ID="Experiment6"
MISSING_ANIMALS="/home/nusreti/PigActivity/analysis/missingAnimals.json"

# === Output Directory Base ===
OUTDIR="../output/movement/${EXPERIMENT_ID}"
mkdir -p "${OUTDIR}"  

# === Day 0-2 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_09_30_09_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_10_02_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/0_2.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/0_2_Naive.txt"

# === Day 2 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_10_01_14_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_10_02_09_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/2.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/2_Naive.txt"

# === Day 0-7 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_09_30_09_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_10_07_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/0_7.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/0_7_Naive.txt"

# === Day 0-9 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_09_30_09_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_10_09_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/0_9.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/0_9_Naive.txt"

# === Day 0-17 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_09_30_09_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_10_17_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/0_17.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/0_17_Naive.txt"

# === Day 2-7 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_10_02_13_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_10_07_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/2_7.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/2_7_Naive.txt"

# === Day 7-9 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_10_07_13_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_10_09_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/7_9.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/7_9_Naive.txt"

# === Day 9-17 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_10_09_13_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/3_2024_10_17_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/9_17.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/9_17_Naive.txt"

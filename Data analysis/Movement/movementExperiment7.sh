#!/bin/bash

# === Configuration ===
SCRIPT="movement.py"  
EXPERIMENT_ID="Experiment7"
MISSING_ANIMALS="/home/nusreti/PigActivity/analysis/missingAnimals.json"

# === Output Directory Base ===
OUTDIR="../output/movement/${EXPERIMENT_ID}"
mkdir -p "${OUTDIR}"  

# === Day 0-2 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_03_09_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_05_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/0_2.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/0_2_Naive.txt"

# === Day 0-2 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_04_11_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_05_09_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/2.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/2_Naive.txt"

# === Day 0-7 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_03_09_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_10_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/0_7.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/0_7_Naive.txt"

# === Day 0-9 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_03_09_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_12_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/0_9.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/0_9_Naive.txt"

# === Day 0-17 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_03_09_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_20_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/0_17.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/0_17_Naive.txt"

# === Day 2-7 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_05_13_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_10_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/2_7.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/2_7_Naive.txt"

# === Day 7-9 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_10_13_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_12_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/7_9.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/7_9_Naive.txt"

# === Day 9-17 ===
TRACKS_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_12_13_0000/tracks.json"
END_FILE="/data/nvme1/nusreti/masterTracks/${EXPERIMENT_ID}/2_2024_12_20_12_0000/tracks.json" 
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" > "${OUTDIR}/9_17.txt"
python3 "$SCRIPT" "$TRACKS_FILE" "$END_FILE" "$MISSING_ANIMALS" "--naive" > "${OUTDIR}/9_17_Naive.txt"

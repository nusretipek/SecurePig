#!/bin/bash

EXPERIMENT_ID="$1"
if [ -z "$EXPERIMENT_ID" ]; then
  echo "Usage: $0 <ExperimentID>"
  exit 1
fi

if [ "$EXPERIMENT_ID" == "8" ]; then
  PREFIX="ExperimentRemux"
else
  PREFIX="Experiment"
fi

# === Configuration ===
SCRIPT="displacement.py"
ANALYSISSCRIPT="displacementAnalysis.py"
MISSING_ANIMALS="/home/nusreti/PigActivity/analysis/missingAnimals.json"
CONFIG_FILE="../experiments.conf"
ANIDOMNET_EXEC="../agonistic/AniDomNet/AniDomNetExtended/AniDomNetExtended.py"

# === Output Directory ===
OUTDIR="../output/displacement/${PREFIX}${EXPERIMENT_ID}"
mkdir -p "${OUTDIR}"

# === Process Config File for this Experiment Only ===
grep "^${PREFIX}${EXPERIMENT_ID} " "$CONFIG_FILE" | while read -r EXP_ID DAY_LABEL START_FILE END_FILE; do
    echo "▶ Processing ${PREFIX}${EXPERIMENT_ID} Day ${DAY_LABEL}"
    
    python3 "$SCRIPT" "$START_FILE" "$END_FILE" "$MISSING_ANIMALS" "${OUTDIR}/${DAY_LABEL}.csv"
    "./$ANIDOMNET_EXEC" -file "${OUTDIR}/${DAY_LABEL}.csv" -outfolder "${OUTDIR}/AniDomNet" -n 100 -ncores 20
done

# === Final Analysis ===
python3 "$ANALYSISSCRIPT" "$OUTDIR"
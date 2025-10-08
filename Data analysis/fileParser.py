import re
import json
from pathlib import Path
from datetime import datetime

def parseTrackPath(path):
    match = re.search(r'(?P<penid>\d+)_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_(?P<hour>\d{2})', path)
    if not match:
        raise ValueError(f"Invalid track path format: {path}")
    groups = match.groupdict()
    dt = datetime(int(groups['year']), int(groups['month']), int(groups['day']), int(groups['hour']))
    return dt

def collectFiles(baseDir, startDT, endDT):
    files = []
    for subdir in sorted(Path(baseDir).iterdir()):
        try:
            dt = parseTrackPath(subdir.name)
            if startDT <= dt <= endDT:
                trackFile = subdir / "tracks.json"
                if trackFile.exists():
                    files.append(trackFile)
        except Exception:
            continue
    return sorted(files)

def collectFullFiles(baseDir, startDT, endDT):
    files = []
    for subdir in sorted(Path(baseDir).iterdir()):
        try:
            dt = parseTrackPath(subdir.name)
            if startDT <= dt <= endDT:
                trackFile = subdir / "tracks.json"
                if trackFile.exists() and str(trackFile).endswith("0000/tracks.json"):
                    files.append(trackFile)
        except Exception:
            continue
    return sorted(files)

def collectCompleteFiles(baseDir, startDT, endDT):
    expID = str(baseDir)[-1]
    if expID == "8":
        expID = "Remux8"
        
    with open("/home/nusreti/PigActivity/analysis/completeHours.json", 'r') as file:
        completeFiles = json.load(file)[expID]

    files = []
    for subdir in sorted(Path(baseDir).iterdir()):
        try:
            dt = parseTrackPath(subdir.name)
            if startDT <= dt <= endDT:
                trackFile = subdir / "tracks.json"
                if trackFile.exists() and str(trackFile) in completeFiles:
                    files.append(trackFile)
        except Exception:
            continue
    return sorted(files)

    
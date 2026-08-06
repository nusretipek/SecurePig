"""
Missing Video Recording Hours Report
-------------------------------------
Reproduces the analysis methodology of commonHours.ipynb:

  1. Build the per-experiment "existence" matrix by scanning each
     ExperimentX masterTracks folder.  Each row corresponds to one
     *recorded* hour slot (08:00-19:00 = 12 hours/day, 18 days).
     Values:  0 = missing,  1 = partial (_NNNN folder only),  2 = complete (_0000).

  2. Compute coverage per experiment and per period:
     - D0-D2  (hours 1-48 on the 24h-elapsed scale, i.e. 24 recorded slots)
     - D2-D7  (hours 49-168, 60 recorded slots)
     - D0-D17 (whole experiment, 204 recorded slots)

     Coverage = 100 * mean(weighted score), where
     missing = 0, partial = 0.5, complete = 1.
     %Missing = 100 - coverage.

  3. Aggregate by treatment group:
     C (Control)   = Exp 3, 6, 7, 9
     P (Pheromone) = Exp 2, 4, 5, 8 (ExpRemux8)

  4. Write a complete report to completeHoursReport.txt.
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


# -----------------------------------------------------------------------------
# Period definitions (in 24h-elapsed hour scale, matching commonHours.ipynb)
#   - 12 recorded hours per day (08:00-19:00), one row per recorded hour
#   - 24 hours of absolute elapsed time per "day" (12h recorded + 12h night gap)
#   - day D: rows for hours  (D*24+1)..(D*24+12)  -> recorded hour slots
# -----------------------------------------------------------------------------

PERIODS = [
    # (label, hour_start, hour_end, n_rows_in_period)
    ("D0-D2",  0,   48, 24),   # days 0..1 (until start of D2 exclusive)
    ("D2-D7",  48, 168, 60),   # days 2..6 (until start of D7 exclusive)
    ("D0-D17", 0, 396, 204),   # days 0..16 (until start of D17 exclusive)
    # The 168-216h (D7-D9) and 216-396h (D9-D17) windows are combined into D0-D17.
]
# Auxiliary windows used internally to compute D0-D17 by weighted combination
SUB_WINDOWS = [
    ("0-48h",   0,  48, 24),   # D0-D2
    ("48-168h", 48, 168, 60),  # D2-D7
    ("168-216h", 168, 216, 24),# D7-D9
    ("216-396h", 216, 396, 96),# D9-D17
]

EXP_ORDER = [("3", "C", "C1"), ("6", "C", "C2"),
             ("7", "C", "C3"), ("9", "C", "C4"),
             ("2", "P", "P1"), ("4", "P", "P2"),
             ("5", "P", "P3"), ("Remux8", "P", "P4")]


def parseDate(dateStr):
    return datetime.strptime(dateStr, "%d/%m/%Y")


def checkTimeSlots(startDate, days=18, startHour=8, endHour=19):
    for day in range(days):
        for hour in range(startHour, endHour + 1):
            yield startDate + timedelta(days=day, hours=hour)


def formatSubfolder(dt):
    return f"{dt.year}_{dt.month:02d}_{dt.day:02d}_{dt.hour:02d}"


def checkExistence(masterTracksFolderPath):
    directories = sorted(os.listdir(masterTracksFolderPath))
    initialTrack = directories[0].split("_")
    beginDate = f"{initialTrack[3]}/{initialTrack[2]}/{initialTrack[1]}"
    beginDateDT = parseDate(beginDate)
    timeSlotGenerator = checkTimeSlots(beginDateDT)

    completeness = []
    for tX in timeSlotGenerator:
        subFolderMiddleSequence = formatSubfolder(tX)
        subFolderMiddleSequenceFull = formatSubfolder(tX) + "_0000"
        matchFound = False
        matchFoundPartial = False

        for d in directories:
            if re.search(re.escape(subFolderMiddleSequenceFull), d):
                matchFound = True
            if re.search(re.escape(subFolderMiddleSequence) + r'_[0-9][1-9][1-9][1-9]', d) \
               and not re.search(re.escape(subFolderMiddleSequenceFull), d):
                matchFoundPartial = True

        if matchFound:
            completeness.append(2)
        elif matchFoundPartial:
            completeness.append(1)
        else:
            completeness.append(0)

    return np.array(completeness).astype(np.int8)


def row_to_hour(row_idx):
    """Map row index -> absolute (24h-scale) hour label, matching the plot y-axis."""
    day = row_idx // 12
    within_day_idx = row_idx % 12
    return day * 24 + (within_day_idx + 1)


def buildExistenceMatrix(basePath):
    """Scan each experiment folder and stack the completeness arrays as columns."""
    existence = None
    for expId, _, _ in EXP_ORDER:
        folder = f"{basePath}/Experiment{expId}"
        completenessTemp = checkExistence(folder)
        if existence is None:
            existence = completenessTemp
        else:
            existence = np.column_stack([existence, completenessTemp])
    return existence


def computeCoverage(existence, rowWindowStart, rowWindowEnd):
    """Coverage (%) over recorded slots that fall in the absolute hour window."""
    weights = np.array([0.0, 0.5, 1.0])
    score_matrix = weights[existence]

    nRows, nCols = existence.shape
    hour_labels = np.array([row_to_hour(i) for i in range(nRows)])

    coverages = []
    completeCounts = []     # weighted count of complete-equivalent hours
    missingCounts = []      # weighted count of missing-equivalent hours
    totals = []

    for col_idx in range(nCols):
        mask = (hour_labels > rowWindowStart) & (hour_labels <= rowWindowEnd)
        scores_in_window = score_matrix[mask, col_idx]
        nrows = scores_in_window.size
        if nrows == 0:
            coverages.append(np.nan)
            completeCounts.append(0.0)
            missingCounts.append(0.0)
            totals.append(0)
        else:
            weighted_sum = float(scores_in_window.sum())
            coverages.append(100.0 * weighted_sum / nrows)
            # 1.0 * complete + 0.5 * partial = complete-equivalent count
            completeCounts.append(weighted_sum)
            # 1.0 * missing + 0.5 * partial = missing-equivalent count
            missingCounts.append(float(nrows) - weighted_sum)
            totals.append(nrows)

    return coverages, completeCounts, missingCounts, totals


def computeD0D17Coverage(existence):
    """D0-D17 = weighted combination of the four sub-windows (matching notebook)."""
    weights = np.array([0.0, 0.5, 1.0])
    score_matrix = weights[existence]

    nRows, nCols = existence.shape
    hour_labels = np.array([row_to_hour(i) for i in range(nRows)])

    coverages = []
    completeCounts = []
    missingCounts = []
    totals = []

    for col_idx in range(nCols):
        total = 0
        weighted_sum = 0.0
        for _, hourStart, hourEnd, nRowsWindow in SUB_WINDOWS:
            mask = (hour_labels > hourStart) & (hour_labels <= hourEnd)
            scores_in_window = score_matrix[mask, col_idx]
            total += scores_in_window.size
            weighted_sum += float(scores_in_window.sum())

        totals.append(total)
        completeCounts.append(weighted_sum)
        missingCounts.append(float(total) - weighted_sum)
        coverages.append(100.0 * weighted_sum / total if total > 0 else np.nan)

    return coverages, completeCounts, missingCounts, totals


def main():
    baseDir = Path(__file__).parent
    basePath = "/data/nvme1/nusreti/masterTracks"

    # --- 1. Build existence matrix ----------------------------------------------------
    existencePath = baseDir / "existenceMatrix.npy"
    if existencePath.exists():
        existence = np.load(existencePath)
    else:
        try:
            existence = buildExistenceMatrix(basePath)
            np.save(existencePath, existence)
        except Exception as e:
            print(f"Cannot scan filesystem: {e}")
            print("Creating existenceMatrix from completeHours.json is not possible.")
            return

    nRows, nCols = existence.shape
    # Sanity check: 216 rows (18 days x 12 recorded hours)
    # but only the first 204 are analysed (days 0..16, D0-D17).

    # --- 2. Compute per-experiment coverage ------------------------------------------
    covD0D2,  cntD0D2c,  cntD0D2m,  totD0D2  = computeCoverage(existence, 0, 48)
    covD2D7,  cntD2D7c,  cntD2D7m,  totD2D7  = computeCoverage(existence, 48, 168)
    covD0D17, cntD17c,   cntD17m,   totD17   = computeD0D17Coverage(existence)

    # --- 3. Render report ------------------------------------------------------------
    out = []
    out.append("=" * 96)
    out.append("COMPLETE HOURS RATIO - Missing Video Recording Hours Report")
    out.append("  (Reproduces methodology of commonHours.ipynb)")
    out.append("=" * 96)
    out.append("")
    out.append("Treatment groups: C = Control (Exp 3, 6, 7, 9)   P = Pheromone (Exp 2, 4, 5, Remux8)")
    out.append("Recorded window : 08:00-19:00 (12 h/day);  night hours excluded")
    out.append("Plot y-axis     : 1 -> 420 absolute hours (D0-D17); 204 analysed recorded slots per experiment")
    out.append("Completeness    : 0 = missing, 1 = partial (_NNNN only), 2 = complete (_0000)")
    out.append("Coverage        : 100 * mean(weighted score),  missing=0 / partial=0.5 / complete=1")
    out.append("")
    out.append("Expected recorded slots per experiment:")
    out.append("  D0-D2  = 24   (2 days  x 12 h, absolute hours  1-48)")
    out.append("  D2-D7  = 60   (5 days  x 12 h, absolute hours 49-168)")
    out.append("  D0-D17 = 204  (17 days x 12 h, absolute hours  1-396,  i.e. 0-48+48-168+168-216+216-396)")
    out.append("")
    out.append("-" * 96)
    out.append(f"{'Experiment':<12}{'Group':<8}{'Period':<10}{'Expected':<10}"
               f"{'Complete':<14}{'Missing':<14}{'Coverage%':<12}{'Missing%':<10}")
    out.append("-" * 96)

    perExp = {}
    for i, (expId, group, label) in enumerate(EXP_ORDER):
        perExp[expId] = {
            "group": group, "label": label,
            "D0-D2":  (totD0D2[i],  cntD0D2c[i],  cntD0D2m[i],  covD0D2[i]),
            "D2-D7":  (totD2D7[i],  cntD2D7c[i],  cntD2D7m[i],  covD2D7[i]),
            "D0-D17": (totD17[i],   cntD17c[i],   cntD17m[i],   covD0D17[i]),
        }

    for expId, group, label in EXP_ORDER:
        r = perExp[expId]
        for period in ["D0-D2", "D2-D7", "D0-D17"]:
            tot, comp, miss, cov = r[period]
            missPct = 100.0 - cov
            out.append(f"{label} (Exp{expId})".ljust(12) +
                       f"{group:<8}{period:<10}{tot:<10}"
                       f"{comp:<14.1f}{miss:<14.1f}{cov:<12.1f}{missPct:<10.1f}")
        out.append("")

    # --- 4. Group aggregates ------------------------------------------------
    out.append("-" * 96)
    out.append("AGGREGATE BY TREATMENT GROUP (weighted by recorded slots)")
    out.append("-" * 96)
    out.append(f"{'Group':<8}{'Period':<10}{'Expected':<12}"
               f"{'Complete':<14}{'Missing':<14}{'Coverage%':<12}{'Missing%':<10}")
    out.append("-" * 96)

    for group in ["C", "P"]:
        exps = [e for e, g, _ in EXP_ORDER if g == group]
        for period in ["D0-D2", "D2-D7", "D0-D17"]:
            tot = sum(perExp[e][period][0] for e in exps)
            comp = sum(perExp[e][period][1] for e in exps)
            miss = sum(perExp[e][period][2] for e in exps)
            cov = 100.0 * comp / tot if tot > 0 else 0.0
            missPct = 100.0 - cov
            out.append(f"{group:<8}{period:<10}{tot:<12}"
                       f"{comp:<14.1f}{miss:<14.1f}{cov:<12.2f}{missPct:<10.2f}")
        out.append("")

    out.append("-" * 96)
    out.append("")
    out.append("Notes:")
    out.append("  'Complete' is the weighted complete-equivalent slot count")
    out.append("  (1.0 per complete hour, 0.5 per partial hour).")
    out.append("  'Missing' = Expected - Complete (weights missing=1.0, partial=0.5).")
    out.append("  Group totals weight across the 4 experiments of each group.")
    out.append("=" * 96)

    text = "\n".join(out)
    outPath = baseDir / "completeHoursReport.txt"
    with open(outPath, "w") as f:
        f.write(text)
    print(text)
    print(f"\nReport saved to: {outPath}")


if __name__ == "__main__":
    main()
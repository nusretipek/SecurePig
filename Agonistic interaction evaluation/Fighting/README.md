# Agonistic interaction evaluation\Fighting

## Overview

This folder directly supports the analysis of fighting behaviors within the `SecurePig` project, providing a structured repository for automatically generated example data derived from video feeds.

## File Structure and Contents

The `Fighting` directory contains CSV files representing detected fighting behaviors, along with a precision metric.

```
Fighting/
├── 1_2025_02_04_11_0000.csv
├── 1_2025_02_04_12_0000.csv
├── 2_2024_06_17_10_0000.csv
├── 2_2024_06_17_11_0000.csv
├── 2_2024_09_09_11_0000.csv
├── 2_2024_09_09_12_0000.csv
├── 2_2024_12_03_11_0000.csv
├── 2_2024_12_03_12_0000.csv
├── 2_2025_02_25_14_0000.csv
├── 2_2025_02_25_15_0000.csv
├── 3_2024_04_15_10_0000.csv
├── 3_2024_04_15_14_0000.csv
├── 3_2024_07_08_11_0000.csv
├── 3_2024_07_08_12_0000.csv
├── 3_2024_09_30_10_0000.csv
├── 3_2024_09_30_11_0000.csv
└── precision.txt
```

## Directory and File Details

### CSV Files

-   **Naming Convention:** The CSV files are named following the convention `PenID_YYYY_MM_DD_HH_0000.csv`. For example, `1_2025_02_04_11_0000.csv` indicates data from pen 1, recorded on February 4, 2025, at 11:00 AM.
-   **Content:** Each CSV file contains data related to fighting events detected by the automated video analysis system. The columns include:
    -   `Initiator`: The ID of the animal initiating the fighting.
    -   `Receiver`: The ID of the animal being displaced.
    -   `EventTime`: The time at which the fighting event occurred.
    -   `Correct`: A flag (0 or 1) indicating whether the detected event was a true positive.

### `precision.txt`

-   **Content:** This file stores the precision value, a metric representing the accuracy of the fighting event detection.

# Tracking evaluation\Frames

## Overview

This directory is a component of the `SecurePig\Tracking evaluation` module. It contains the raw frame data used to assess the accuracy of the tracking system.

## Directory and File Details

The `Frames` directory is organized into subdirectories, each representing a specific recording session.

-   **Subdirectories**: Each subdirectory is named according to the following convention: `PenID_MM_DD_HH`, where:
    -   `PenID` is the pen identification number.
    -   `MM` is the month of the recording.
    -   `DD` is the day of the recording.
    -   `HH` is the hour of the recording.

-   **Files**: Each subdirectory contains a series (equally spaced in time) of JPEG image frames named according to the following convention: `PenID_MM_DD_HH_XXX.jpg`, where:
    -   `PenID`, `MM`, `DD`, and `HH` are the same as in the subdirectory name.
    -   `XXX` is a three-digit frame ID, zero-padded. 

### Subdirectory Breakdown

-   `1_02_04_11/`: Frames from Pen 1, recorded on February 4th at 11:00. Contains 32 frames, named `1_02_04_11_001.jpg` through `1_02_04_11_032.jpg`.
-   `1_02_04_12/`: Frames from Pen 1, recorded on February 4th at 12:00. Contains 32 frames, named `1_02_04_12_001.jpg` through `1_02_04_12_032.jpg`.
-   `2_02_25_14/`: Frames from Pen 2, recorded on February 25th at 14:00. Contains 32 frames, named `2_02_25_14_001.jpg` through `2_02_25_14_032.jpg`.
-   `2_02_25_15/`: Frames from Pen 2, recorded on February 25th at 15:00. Contains 32 frames, named `2_02_25_15_001.jpg` through `2_02_25_15_032.jpg`.
-   `2_06_17_10/`: Frames from Pen 2, recorded on June 17th at 10:00. Contains 32 frames, named `2_06_17_10_001.jpg` through `2_06_17_10_032.jpg`.
-   `2_06_17_11/`: Frames from Pen 2, recorded on June 17th at 11:00. Contains 32 frames, named `2_06_17_11_001.jpg` through `2_06_17_11_032.jpg`.
-   `2_09_09_11/`: Frames from Pen 2, recorded on September 9th at 11:00. Contains 32 frames, named `2_09_09_11_001.jpg` through `2_09_09_11_032.jpg`.
-   `2_09_09_12/`: Frames from Pen 2, recorded on September 9th at 12:00. Contains 32 frames, named `2_09_09_12_001.jpg` through `2_09_09_12_032.jpg`.
-   `2_12_03_11/`: Frames from Pen 2, recorded on December 3rd at 11:00. Contains 32 frames, named `2_12_03_11_001.jpg` through `2_12_03_11_032.jpg`.
-   `2_12_03_12/`: Frames from Pen 2, recorded on December 3rd at 12:00. Contains 32 frames, named `2_12_03_12_001.jpg` through `2_12_03_12_032.jpg`.
-   `3_04_15_10/`: Frames from Pen 3, recorded on April 15th at 10:00. Contains 32 frames, named `3_04_15_10_001.jpg` through `3_04_15_10_032.jpg`.
-   `3_04_15_14/`: Frames from Pen 3, recorded on April 15th at 14:00. Contains 32 frames, named `3_04_15_14_001.jpg` through `3_04_15_14_032.jpg`.
-   `3_07_08_11/`: Frames from Pen 3, recorded on July 8th at 11:00. Contains 32 frames, named `3_07_08_11_001.jpg` through `3_07_08_11_032.jpg`.
-   `3_07_08_12/`: Frames from Pen 3, recorded on July 8th at 12:00. Contains 32 frames, named `3_07_08_12_001.jpg` through `3_07_08_12_032.jpg`.
-   `3_09_30_10/`: Frames from Pen 3, recorded on September 30th at 10:00. Contains 32 frames, named `3_09_30_10_001.jpg` through `3_09_30_10_032.jpg`.
-   `3_09_30_11/`: Frames from Pen 3, recorded on September 30th at 11:00. Contains 32 frames, named `3_09_30_11_001.jpg` through `3_09_30_11_032.jpg`.
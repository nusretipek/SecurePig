# Tracking evaluation

## Role

This folder provides data for evaluating the tracking performance, which is a crucial component of the larger `SecurePig` project. The evaluation data helps to quantify the accuracy of the tracking system which in turns used for behaviour analysis.

## Overview

This module focuses on evaluating the performance of the tracking system implemented in the `Automated video analysis/Tracking` directory. It contains a set of video frames and a CSV file used to assess the accuracy of the sow tracking algorithm.

## Directory and File Details

### Directory Structure

```
Tracking evaluation/
  ├── Frames/
  │   ├── 1_02_04_11/
  │   │   ├── 1_02_04_11_001.jpg
  │   │   ├── ...
  │   │   └── 1_02_04_11_032.jpg
  │   ├── 1_02_04_12/
  │   │   ├── 1_02_04_12_001.jpg
  │   │   ├── ...
  │   │   └── 1_02_04_12_032.jpg
  │   ├── 2_02_25_14/
  │   │   ├── 2_02_25_14_001.jpg
  │   │   ├── ...
  │   │   └── 2_02_25_14_032.jpg
  │   ├── 2_02_25_15/
  │   │   ├── 2_02_25_15_001.jpg
  │   │   ├── ...
  │   │   └── 2_02_25_15_032.jpg
  │   ├── 2_06_17_10/
  │   │   ├── 2_06_17_10_001.jpg
  │   │   ├── ...
  │   │   └── 2_06_17_10_032.jpg
  │   ├── 2_06_17_11/
  │   │   ├── 2_06_17_11_001.jpg
  │   │   ├── ...
  │   │   └── 2_06_17_11_032.jpg
  │   ├── 2_09_09_11/
  │   │   ├── 2_09_09_11_001.jpg
  │   │   ├── ...
  │   │   └── 2_09_09_11_032.jpg
  │   ├── 2_09_09_12/
  │   │   ├── 2_09_09_12_001.jpg
  │   │   ├── ...
  │   │   └── 2_09_09_12_032.jpg
  │   ├── 2_12_03_11/
  │   │   ├── 2_12_03_11_001.jpg
  │   │   ├── ...
  │   │   └── 2_12_03_11_032.jpg
  │   ├── 2_12_03_12/
  │   │   ├── 2_12_03_12_001.jpg
  │   │   ├── ...
  │   │   └── 2_12_03_12_032.jpg
  │   ├── 3_04_15_10/
  │   │   ├── 3_04_15_10_001.jpg
  │   │   ├── ...
  │   │   └── 3_04_15_10_032.jpg
  │   ├── 3_04_15_14/
  │   │   ├── 3_04_15_14_001.jpg
  │   │   ├── ...
  │   │   └── 3_04_15_14_032.jpg
  │   ├── 3_07_08_11/
  │   │   ├── 3_07_08_11_001.jpg
  │   │   ├── ...
  │   │   └── 3_07_08_11_032.jpg
  │   ├── 3_07_08_12/
  │   │   ├── 3_07_08_12_001.jpg
  │   │   ├── ...
  │   │   └── 3_07_08_12_032.jpg
  │   ├── 3_09_30_10/
  │   │   ├── 3_09_30_10_001.jpg
  │   │   ├── ...
  │   │   └── 3_09_30_10_032.jpg
  │   └── 3_09_30_11/
  │   │   ├── 3_09_30_11_001.jpg
  │   │   ├── ...
  │   │   └── 3_09_30_11_032.jpg
  ├── trackingAnalysis.csv
  └── Videos/
      ├── 1_02_04_11_merged.mp4
      ├── 1_02_04_12_merged.mp4
      ├── 2_02_25_14_merged.mp4
      ├── 2_02_25_15_merged.mp4
      ├── 2_06_17_10_merged.mp4
      ├── 2_06_17_11_merged.mp4
      ├── 2_09_09_11_merged.mp4
      ├── 2_09_09_12_merged.mp4
      ├── 2_12_03_11_merged.mp4
      ├── 2_12_03_12_merged.mp4
      ├── 3_04_15_10_merged.mp4
      ├── 3_04_15_14_merged.mp4
      ├── 3_07_08_11_merged.mp4
      ├── 3_07_08_12_merged.mp4
      ├── 3_09_30_10_merged.mp4
      └── 3_09_30_11_merged.mp4
```

### File Details

-   **`trackingAnalysis.csv`**: This CSV file contains the analysis of the tracking results. It includes the video name, image name, the number of animals detected, the number of mistakes made by the tracking algorithm, and the precision of the tracking.
-   **`Frames/`**: This subdirectory contains a set of JPEG image frames extracted from videos. The frames are organized into subdirectories named according to the Pen ID, month, day, and hour (PenID_MM_DD_HH) of the recording. Each frame is named following the same convention, appended with a frame ID.
    -   `1_02_04_11/`: Frames from Pen 1, recorded on February 4th at 11:00.
    -   `1_02_04_12/`: Frames from Pen 1, recorded on February 4th at 12:00.
    -   `2_02_25_14/`: Frames from Pen 2, recorded on February 25th at 14:00.
    -   `2_02_25_15/`: Frames from Pen 2, recorded on February 25th at 15:00.
    -   `2_06_17_10/`: Frames from Pen 2, recorded on June 17th at 10:00.
    -   `2_06_17_11/`: Frames from Pen 2, recorded on June 17th at 11:00.
    -   `2_09_09_11/`: Frames from Pen 2, recorded on September 9th at 11:00.
    -   `2_09_09_12/`: Frames from Pen 2, recorded on September 9th at 12:00.
    -   `2_12_03_11/`: Frames from Pen 2, recorded on December 3rd at 11:00.
    -   `2_12_03_12/`: Frames from Pen 2, recorded on December 3rd at 12:00.
    -   `3_04_15_10/`: Frames from Pen 3, recorded on April 15th at 10:00.
    -   `3_04_15_14/`: Frames from Pen 3, recorded on April 15th at 14:00.
    -   `3_07_08_11/`: Frames from Pen 3, recorded on July 8th at 11:00.
    -   `3_07_08_12/`: Frames from Pen 3, recorded on July 8th at 12:00.
    -   `3_09_30_10/`: Frames from Pen 3, recorded on September 30th at 10:00.
    -   `3_09_30_11/`: Frames from Pen 3, recorded on September 30th at 11:00.
-   **`Videos/`**: This subdirectory contains merged video files (four cams in same video file). The naming convention for the videos is PenID_MM_DD_HH_merged.mp4, providing information about the Pen ID, recording date (month and day), and hour.
    -   `1_02_04_11_merged.mp4`: Merged video from Pen 1, recorded on February 4th at 11:00.
    -   `1_02_04_12_merged.mp4`: Merged video from Pen 1, recorded on February 4th at 12:00.
    -   `2_02_25_14_merged.mp4`: Merged video from Pen 2, recorded on February 25th at 14:00.
    -   `2_02_25_15_merged.mp4`: Merged video from Pen 2, recorded on February 25th at 15:00.
    -   `2_06_17_10_merged.mp4`: Merged video from Pen 2, recorded on June 17th at 10:00.
    -   `2_06_17_11_merged.mp4`: Merged video from Pen 2, recorded on June 17th at 11:00.
    -   `2_09_09_11_merged.mp4`: Merged video from Pen 2, recorded on September 9th at 11:00.
    -   `2_09_09_12_merged.mp4`: Merged video from Pen 2, recorded on September 9th at 12:00.
    -   `2_12_03_11_merged.mp4`: Merged video from Pen 2, recorded on December 3rd at 11:00.
    -   `2_12_03_12_merged.mp4`: Merged video from Pen 2, recorded on December 3rd at 12:00.
    -   `3_04_15_10_merged.mp4`: Merged video from Pen 3, recorded on April 15th at 10:00.
    -   `3_04_15_14_merged.mp4`: Merged video from Pen 3, recorded on April 15th at 14:00.
    -   `3_07_08_11_merged.mp4`: Merged video from Pen 3, recorded on July 8th at 11:00.
    -   `3_07_08_12_merged.mp4`: Merged video from Pen 3, recorded on July 8th at
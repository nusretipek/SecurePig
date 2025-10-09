# Automated video analysis

## Role

This module provides the core video processing and analysis components for the `SecurePig` project. It takes raw video data as input and generates tracking data, segmentation masks, and behavioral classifications used for further analysis in the `Data analysis` directory and behavior samples in the `Behaviour samples` directory, while being evaluated in the `Tracking evaluation` directory.

## Overview

This directory contains the scripts and modules necessary for automated video analysis of pig behavior. It includes functionalities for:

-   **Frame Extraction and Cosine Similarity Analysis:** Identifying key frames based on content changes.
-   **Inference:** Performing segmentation using a U-Net model.
-   **Re-identification:** Assisting in the re-identification of individual animals across frames.
-   **Tracking:** Tracking individual animals within the video.
-   **U-Net Segmentation:** Implementing and training U-Net models for image segmentation.

## Directory and File Details

### DINO/

*   `main_dino.py`: Python script for running DINO. Data loading component is changed for the purpose of our project.
*   `train.sh`: Shell script for training purposes.

### Inference/

*   `calculateInferenceTime.sh`: Shell script to measure the inference time of the segmentation model.
*   `inference.py`: Python script for performing inference using a trained segmentation model.
*   `segmentServer.py`: Python script implementing a server for segmentation tasks.
*   `unet.py`: Python script defining the U-Net model architecture.

### Re-identification/

*   `annotationStreamlitApp.py`: Python script for a Streamlit application, used for manual annotation to aid re-identification.
*   `MaxViT.ipynb`: Jupyter Notebook for training the MaxViT architecture for re-identification.

### Tracking/

*   `securePigTracker.py`: Python script implementing the core tracking logic.
*   `securePigVideo.py`: Python script for generating track-annotated videos.
*   `track.sh`: Shell script to execute the tracking process.
*   `trackGenerator.py`: Python script for generating tracking data or configurations.
*   `trackPiecewise.py`: Python script implementing piecewise tracking algorithms.

### U-Net/

*   `U-Net.ipynb`: Jupyter Notebook for training a standard U-Net model.
*   `U-Net_Distill.ipynb`: Jupyter Notebook for U-Net model self-iteration.

### Files in the current directory:

*   `extractFramesCosineSimilarity.py`: Python script for extracting frames from a video based on cosine similarity of image embeddings.
*   `saveEvaluationSet.sh`: Shell script to save a set of frames for evaluation purposes.
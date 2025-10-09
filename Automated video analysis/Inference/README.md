# Automated video analysis\Inference

## Overview

This directory contains the scripts and modules necessary for performing inference on video data, primarily focusing on segmentation using U-Net and SAM2 models. It includes tools for running inference, serving segmentation models, and calculating inference time.

## Directory and File Details

*   `calculateInferenceTime.sh`: Shell script to measure the inference time of the segmentation model.
*   `inference.py`: Python script for performing inference using a trained segmentation model. It uses both U-Net and SAM2 models for segmentation and includes re-identification functionality.
*   `segmentServer.py`: Python script implementing a server for segmentation tasks. It serves both U-Net and SAM2 models through a Flask API.
*   `unet.py`: Python script defining the U-Net model architecture.
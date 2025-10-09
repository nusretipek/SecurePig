# SecurePig/

## Overview

This project focuses on the automated analysis of sow behavior and welfare under pig appeasing pheromone (PAP) using computer vision techniques. It encompasses various aspects, from tracking and re-identification to behavior recognition and data analysis.

### Role

This directory (`SecurePig/`) serves as the root directory for the SecurePig project. It contains subdirectories organizing the different modules of the project, including automated video analysis, data analysis, tracking evaluation, and zero-shot computer vision methods.

## File Structure

```
SecurePig/
├── Agonistic interaction evaluation/
├── Automated video analysis/
├── Behaviour samples/
├── Data analysis/
├── Tracking evaluation/
├── license.txt
└── requirements.txt
```

## Directory and File Details

- **Agonistic interaction evaluation/:** Contains CSV files relating to evaluation of agonistic interactions of the pigs.
- **Automated video analysis/:** Contains scripts and models for automated video analysis, including object detection, segmentation, tracking, re-identification and tracking.
- **Behaviour samples/:** Contains sample video clips of different pig behaviors.
- **Data analysis/:** Contains scripts for analyzing the skin injuries and data generated from video analysis, including statistical analysis and visualization.
- **Tracking evaluation/:** Contains data and scripts used to evaluate the performance of the tracking algorithms.
- **license.txt:** Contains licence for the code.
- **requirements.txt:** Lists the Python packages required to run the scripts in this directory.

### Requirements

The project relies on the dependencies listed in `requirements.txt`. To install these dependencies, run:

```bash
pip install -r requirements.txt
```

### External tools

The project requires the DINO and SAMv2 models, which must be installed from their official repositories, following their specific installation instructions.
1. DINO: https://github.com/facebookresearch/dino , note that main_dino.py file is updated for task-specific input parsing via new data class (other contents kept same) and given in DINO folder.
2. SAMv2: https://github.com/facebookresearch/sam2 

### Model Weights

The model weights used in this project are too large to upload to the repository. They can be made available upon reasonable request.

## Citation

TO BE FILLED.
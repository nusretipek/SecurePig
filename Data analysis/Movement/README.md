# Data analysis\Movement

## Overview

This directory contains scripts specifically designed for analyzing movement patterns of sows within the experimental setup. It includes tools for processing tracking data to quantify movement and performing statistical analysis to identify significant factors influencing movement.

## File Structure and Details

-   `movement.py`: Main script for movement analysis, calculates individual movements and groups.
-   `movementAnalysis.py`: Script for performing specific analyses on movement data, including summarizing movement metrics across experiments.
-   `movementExperiment[2-9].sh`: Shell scripts to run movement analysis for different experiments. These scripts automate the execution of `movement.py` and `movementAnalysis.py`.
-   `movementLMM.py`: Script for Linear Mixed Model (LMM) analysis of movement data, used for statistical analysis of movement behavior.

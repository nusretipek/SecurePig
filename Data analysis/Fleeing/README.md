# Data analysis\Fleeing

## Overview

This directory contains the scripts for analyzing fleeing behavior among sows. It includes modules for detecting fleeing events, performing statistical analysis, and running linear mixed models to understand factors influencing fleeing.

## Directory and File Details

-   `fleeing.py`: Main script for analyzing fleeing behavior. It loads track data, adjusts time, calculates pairwise distances between animals, and identifies fleeing events.
-   `fleeingAnalysis.py`: Script for performing specific analyses on fleeing data.
-   `fleeingBehavior.sh`: Shell script to automate the execution of the fleeing behavior analysis pipeline.
-   `fleeingLMM.py`: Script for Linear Mixed Model (LMM) analysis of fleeing data. It reads data from CSV files, performs statistical analysis using the `statsmodels` library, and generates plots.

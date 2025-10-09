# Data analysis\Displacement

## Overview

This directory contains the scripts for analyzing displacement behavior among sows. It includes modules for detecting displacement events, performing statistical analysis, and running linear mixed models to understand factors influencing displacement.

## Directory and File Details

-   `displacement.py`: Main script for displacement analysis. It detects displacement events based on track data. The script processes track files, identifies animal IDs.
-   `displacementAnalysis.py`: Script for performing specific analyses on displacement data, including generating plots.
-   `displacementBehavior.sh`: Shell script to run displacement behavior analysis. This script automates the execution of the Python analysis scripts.
-   `displacementLMM.py`: Script for Linear Mixed Model (LMM) analysis of displacement data. It reads data from CSV files, performs statistical analysis using the `statsmodels` library, and generates plots.
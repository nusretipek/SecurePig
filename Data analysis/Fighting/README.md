# Data analysis\Fighting

## Overview

This module focuses on the analysis of sow fighting behavior data. It provides scripts for statistical analysis, data processing, and visualization to derive insights from tracking data. The module uses tracking data to quantify different behaviours of sows, identifies patterns, and evaluates factors influencing these behaviours.

## Directory and File Details

This directory contains scripts for analyzing fighting behavior in sows, including detection, analysis, and statistical modeling.

-   `fighting.py`: Main script for fighting analysis. It includes functionalities for reading track data, adjusting time, and calculating pairwise distances between animals.
-   `fightingAnalysis.py`: Script for performing specific analyses on fighting data, including generating a fighting behavior bar chart and correlation analysis.
-   `fightingBehavior.sh`: Shell script to run fighting behavior analysis.
-   `fightingLMM.py`: Script for Linear Mixed Model (LMM) analysis of fighting data. It reads data from CSV files, performs statistical analysis using the `statsmodels` library, and generates plots.
# Automated video analysis\Tracking

## Overview

This directory contains the scripts and modules necessary for tracking individual animals within video footage. It includes functionalities for initializing tracks, piecewise tracking, and generating video outputs with overlaid tracking information.

## Directory and File Details

*   `securePigTracker.py`: Python script implementing the core tracking logic, including track splitting and identification.
*   `securePigVideo.py`: Python script for generation of videos, including generating videos with overlaid tracking information.
*   `track.sh`: Shell script to execute the tracking process.
*   `trackGenerator.py`: Python script for generating initial tracks from detection data and applying blackout masks.
*   `trackPiecewise.py`: Python script implementing piecewise tracking algorithms for handling track fragmentation.
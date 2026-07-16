#
# Spectrum and Cascade Analysis
July 2026 - Florent Brient (florent.brient@lmd.ipsl.fr)

A Python toolkit for analyzing spectral data.
The model used in Meso-NH, outputs are stored on the Jean-Zay HPC

## Motivation / Overview

This project provides tools and scripts to:

- Perform TKE budget in physical and Fourier spaces
- Compute spectral analyses (e.g. power spectral densities, spectral moments)  
- Plot figures for TKE budget and spectral analysis

The repository is structured into folders:

- `src/`: main source modules and analysis routines  
- `run/`: example run scripts or driver scripts  
- `data/`: sample data or test input data  
- `infos/`: metadata, constants, or supporting files  

## Features [To update]

- Flexible loader for different file formats  
- Preprocessing routines: smoothing, baseline removal, filtering  
- Spectral analysis: FFT, Welch method, Lomb–Scargle, etc.  
- Spectral fitting and model comparison  
- Visualization utilities (spectra, residuals, overlaid models)  
- Batch processing of multiple datasets  

## Installation & Requirements

You can clone this repository, then install required Python packages. For example:

```bash
git clone https://github.com/florentbrient/cascade.git




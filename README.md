# EB Threshold Comparison Analysis

Analysis of thresholding affects Ellerman Bomb properties (lifetime, area, contrast) across multiple solar datasets.

## Features
- Multi-dataset analysis of EB properties
- Comparison of detection result with Ellerman bomb detection code vs. thresholded results
- Area conversion (pixels → arcsec² → Mm²)
- Publication-ready visualizations
- Comprehensive statistical summary

## Quick Start
```bash
git clone https://github.com/AroojFaryad
cd EB-Threshold-Comparison-Analysis
pip install -r requirements.txt
```

## Data Structure
Place your data in the `data/` directory:
```
data/
├── YYYYMMDD/
│   ├── wings.fits
│   ├── qs.txt
│   └── sources_*.npy
```

##  Outputs
- `results/eb_property_comparison.pdf`: Triple-panel comparison plot
- `results/analysis_results.npz`: Numerical results
- Console output: Statistical summary

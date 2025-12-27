# Purkinje Cells Decoding and Characterization

## Abstract

This thesis aims to decode neural processes underlying Purkinje cell population activity and characterize how these processes change across different stimulus conditions using an integrated temporal-spectral approach. Understanding how cerebellar neural networks organize and coordinate their activity to process information remains a central challenge in neuroscience, particularly regarding how Purkinje cells, the primary cortical cerebellar output neurons, encode and integrate complex stimulus information.

The study used computational cerebellar network models that were analyzed through Independent Component Analysis (ICA) for signal decomposition, spectral parameterization using FOOOF, and comprehensive correlation analyses between Purkinje cells and molecular layer interneurons. This model-based approach enabled precise stimulus control and complete validation against known inputs.

ICA successfully decoded cerebellar stimuli solely from Purkinje cell activity, producing components that mapped directly onto network anatomy and revealed neuronal functional organization. The aperiodic exponent emerged as a robust single-parameter "network state decoder," quantifying excitation-inhibition balance across stimulus conditions. This finding received cellular-level validation through Purkinje cell-molecular layer interneuron coordination patterns, demonstrating enhanced inhibitory control in more active network states. Energy distribution and phase analyses revealed distinct coordination strategies: stimulus-driven states exhibited widespread synchrony and focused spectral energy, while baseline states showed dispersed patterns and broadband distributions.

The results demonstrate the power of dimensionality reduction methods for decomposing neural time series and uncovering underlying signals in complex population outputs. The establishment of a multi-scale framework linking temporal decomposition, spectral analysis, and spatial organization advances the understanding of neural computational processes underlying the network states identified by the decoder.

## Repository Structure

### Data Files

**Note:** Due to GitHub's file size limitations, the large data files (CSV and HDF5 files containing simulation outputs) are not included in this repository. These files exceed 100MB and must be generated locally by running the cerebellar network simulations or obtained separately.

### Analysis Notebooks

The repository contains **5 Jupyter notebooks** for comprehensive analysis:

#### Condition-Specific Analyses
1. **`analysisBaseline.ipynb`** - Analysis of baseline condition (spontaneous activity)
2. **`analysisStep.ipynb`** - Analysis of step stimulus condition
3. **`analysisBurst.ipynb`** - Analysis of burst stimulus condition

#### Comprehensive Analyses
4. **`analysisStimulusT.ipynb`** (or `analysisICAStimulus.ipynb`) - Analysis of the entire recording period, integrating all stimulus conditions
5. **`results.ipynb`** - Final comprehensive analysis bringing together results from all conditions

#### Helper Notebooks
- **`panels.ipynb`** - Helper notebook for generating publication-quality figure panels for the thesis
- **`test_structure.ipynb`** - Testing and validation notebook for creating the 3d visualization of the network

### Utility Modules

The repository includes **5 Python utility modules** (`*.py` files) containing all helper functions:

1. **`utils.py`** - Core utility functions for data processing and analysis
2. **`utils_analysis.py`** - Analysis-specific helper functions (ICA, PCA, clustering)
3. **`utils_spectral.py`** - Spectral analysis functions (FOOOF parameterization, power spectral density)
4. **`utils_BK_corr.py`** - Functions for Purkinje cell - molecular layer interneuron correlation analyses
5. **`color_manager.py`** - Color scheme management for consistent visualization across conditions
6. **`dynamics_functions.py`** - Functions for 3D visualization and correlation matrix generation


## Key Methods

### Independent Component Analysis (ICA)
- Signal decomposition of Purkinje cell population activity
- Extraction of independent temporal components
- Mapping of components to network anatomy

### Spectral Parameterization (FOOOF)
- Separation of periodic and aperiodic components
- Extraction of aperiodic exponent as network state decoder
- Quantification of excitation-inhibition balance

### Correlation Analysis
- Purkinje cell - molecular layer interneuron coordination
- Assessment of inhibitory control across network states
- Spatial organization of functional connectivity

### Multi-Scale Framework
- Temporal decomposition (ICA)
- Spectral analysis (FOOOF)
- Spatial organization (3D network visualization)

## Dependencies

```python
numpy
pandas
matplotlib
seaborn
scipy
sklearn
fooof
neo
plotly
```

## Usage

1. Generate or obtain the simulation data files (CSV and HDF5 formats)
2. Place data files in the appropriate directory structure
3. Run condition-specific analysis notebooks (`analysisBaseline.ipynb`, `analysisStep.ipynb`, `analysisBurst.ipynb`)
4. Run comprehensive analysis (`analysisStimulusT.ipynb`)
5. Generate final results and figures (`results.ipynb`)

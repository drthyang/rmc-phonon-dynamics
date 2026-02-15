# Phonon Calculation Code for RMCProfile

## Overview
**rmcph** is a research code repository for conducting phonon calculations in solid materials, designed to integrate with [RMCProfile](https://rmcprofile.ornl.gov) for analyzing total scattering data. The toolkit calculates phonon band structures and irreducible representations using structural ensembles generated via Reverse Monte Carlo (RMC) modeling.

### Key Features
- **Phonon Band Structure:** Calculate and plot standard dispersion relations.
- **Weighted Bands:** Analysis of phonon band weighting.
- **Phonon DOS:** Calculate total Phonon Density of States.
- **Partial DOS:** Decompose DOS into partial contributions by element.
- **Inelastic Neutron Scattering:** Calculate phonon DOS specifically for INS comparison.
- **Irreducible Representations (IRs):** Compute and visualize IRs in real-space.

## System Requirements

This repository contains both CPU (`src/`) and GPU (`src_gpu/`) implementations.

* **Standard Version (`src/`):** Compatible with **macOS**, **Linux**, and **Windows**. Runs on standard CPUs.
* **GPU Version (`src_gpu/`):** Currently optimized for **macOS (Apple Silicon)** using the Metal framework.

## Prerequisites

Ensure you have the following installed:

* [RMCProfile](https://rmcprofile.ornl.gov) (Essential for generating input configurations)
* **Python 3.x**
* **Python Dependencies:**
    * `numpy`
    * `scipy`
    * `matplotlib`
    * `seaborn`
    * `tqdm`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/drthyang/rmcph.git](https://github.com/drthyang/rmcph.git)
    cd rmcph
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install numpy scipy matplotlib seaborn tqdm
    ```

## Code Structure

The codebase is split between CPU and GPU implementations:

- **`src/`**: Main source code (CPU-based, compatible with all systems)
  - `main.py`: Core logic and entry point for standard phonon calculations.
  - `Readers.py`: Handles parsing of RMCProfile output files and structural configuration data.
  - `Calculators.py`: Contains the core physics algorithms for dynamical matrices and DOS computations.
  - `Writers.py`: Manages file output for logs, data tables, and visualization exports.
  - `Visualization.py`: Plotting utilities for band structures and densities of states.

- **`src_gpu/`**: High-performance GPU source code
  - `main.py`: Primary script for CUDA-accelerated phonon calculations.

- **`data/`**: Input data files and configuration templates.
- **`results/`**: Directory where output plots and calculation results are saved.

## Usage

The workflow involves generating ensembles, performing RMC modeling, and then running the phonon analysis.

**1. Generate Initial Files**
Create the initial configuration files required for the ensemble calculations.
```bash
./gen_configs.sh
```
**2. Run RMC modelings**
Create the initial configuration files required for the ensemble calculations.
```bash
./submit_seq.sh
```
**3. Run RMC modelings**
Create the initial configuration files required for the ensemble calculations.
```bash
./gen_coords.sh
```
**4. Run RMC modelings**
Create the initial configuration files required for the ensemble calculations.
```bash
python ./src_gpu/main.py
```
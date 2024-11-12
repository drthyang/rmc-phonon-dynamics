# rmcph - Phonon Calculation Code for RMCProfile
## Overview
This repository contains code for conducting phonon calculations in solid materials, integrating RMCProfile to analyze total scattering data. The code calculates phonon band structure and irreducible representations using structural ensemble calculated by reverse Monte Carlo modelings.

Main features:

- Phonon band structure
- Weighted phonon band structure
- Phonon density of states
- Irreducible representations (IRs)
- Visualize IRs in real-space
- Partial phonon density of states
- Calculate phonon density of states for inelatsic neutron scattering 

## Prerequisites

To use this code, ensure that you have the following dependencies installed:

- RMCProfile (https://rmcprofile.ornl.gov)
- Python 3.x
- NumPy, SciPy
- Matplotlib
- seaborn
- tqdm

## Installation

## Code Structure

## Usage
1. Generating initial files for ensemble calculations.
   ```python
   ./gen_configs.sh
2. Performing RMC modelings.
   ```python
   ./submit_seq.sh
3. Obtaining fractional coordinations for phonon calculations.
   ```python
   ./gen_coords.sh
4. Setting up input files for phonon calculation
   ```python
   python ./rmc_phonon_calc.py
   
## Example


Project Repository - MLDM Project
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Logo_de_l%27Université_Jean_Monnet_Saint-Etienne.png/640px-Logo_de_l%27Université_Jean_Monnet_Saint-Etienne.png" alt="Université Jean Monnet" title="Université Jean Monnet">
============
# Learning PDEs from Data: Application to Surface Engineering

## Overview

This project focuses on discovering Partial Differential Equations (PDEs) that describe the dynamics of nickel surfaces after laser exposure, leading to self-organization of particles. We propose two alignment methods to address the lack of register in experimental images and introduce a sparse regression technique, Iterative Huber Ridge, to estimate the PDEs. Our method effectively handles outliers and misalignment, identifying derivative terms in the PDE and achieving lower Relative Mean Absolute errors compared to existing methods.

## Authors

- Erick Gomez 
- Milan Jankovic
- Mohamed Moudjahed
- Hedi Zeghidi

## Files

- `genetic_algo.py`: Implementation of the Genetic Algorithm for image alignment.
- `genetic_algo_parallel.py`: Parallelized version of the Genetic Algorithm.
- `genetic_algo_synthetic_data.py`: Genetic Algorithm applied to synthetic data.
- `square_align.py`: Implementation of the square alignment method.
- `square_align_parallel.py`: Parallelized version of the square alignment method.
- `square_align_synthetic_data.py`: Square alignment applied to synthetic data.
- `PDE_FIND.py`: Main script for PDE discovery using sparse regression.
- `PDE_MLDM.ipynb`: Jupyter Notebook integrating all Python files.

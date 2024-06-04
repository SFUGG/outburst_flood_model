# Subglacial outburst flood modelling

Tim Hill
tim_hill_2@sfu.ca
December 2021, updated June 2024

## Installation

The code was tested using python 3.11.9 on Ubuntu with packages listed in `requirements.txt`

## Report

A report describing the model and numerical experiments, including Latex source and compiled PDF, is in the `report/` directory.

## Model

The model code is in the `python/` directory. The numerical models (incompressible and compressible) are in `models.py`. The scripts `beta_sensitivity.py`, `compare_hypsometry.py`, `test_imposed_discharge.py` and `test_pressure_coupled.py` carry out various numerical experiments on the core models.

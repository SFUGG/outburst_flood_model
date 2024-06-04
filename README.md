# Subglacial outburst flood modelling

Tim Hill
tim_hill_2@sfu.ca
December 2021, updated June 2024

## Installation

The code was tested using python 3.11.9 on Ubuntu with packages listed in `requirements.txt`.

## Report

A report describing the model and numerical experiments, including Latex source and compiled PDF, is in the `report/` directory.

## Model

The model code is in the `python/` directory. The numerical models (incompressible and compressible) are in `models.py`. The scripts to run experiments are as follows:

 * `test_imposed_discharge.py`: simplest experiment with prescribed lake discharge into the head of the conduit. Compares compressible and incompressible conduit models.
 * `test_pressure_coupled.py` : test compressible and incompressible conduit models with lake discharge computed as a function of lake water level.
 * `sensitivity_beta.py` : sensitivity to the compressibility value beta
 * `sensitivity_hypsometry.py` : sensitivity to lake surface area--height relationship


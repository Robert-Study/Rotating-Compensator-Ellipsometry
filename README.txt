# Ellipsometry Analysis Pipeline

This repository contains the Python analysis code I developed for the ellipsometry section of a university optical thin-film characterisation project.
The workflow was built to take raw rotating-compensator ellipsometry measurements through to fitted harmonic coefficients, extracted \(\Psi\) and \(\Delta\), instrument calibration, thin-film simulation, and final estimation of film thickness and optical constants.

A major part of this work was building code that could handle real experimental data rather than only idealised theory. This included fitting non-ideal intensity traces, calibrating against a silicon reference sample, and comparing simulated and measured behaviour across multiple incidence angles.

## What the code does

The repository is split into separate files covering the main stages of the analysis:

- **data import and preprocessing**
- **harmonic fitting of measured intensity sweeps**
- **PCSA instrument modelling**
- **Fresnel--Airy thin-film simulation**
- **\(\Psi\) and \(\Delta\) extraction**
- **instrument calibration**
- **film property fitting**
- **plotting and output generation**
- **end-to-end workflow execution**

## Project contribution

For the ellipsometry section, my work focused on the modelling and computational analysis used to connect raw detector signals to physically meaningful thin-film parameters.

This included developing the fitting pipeline, implementing the simulation and inversion routines, calibrating the setup using a reference sample, and generating the plots used to interpret the results.

## Skills demonstrated

This project demonstrates:

- scientific programming in Python
- numerical fitting and optimisation
- optical modelling
- thin-film analysis
- data visualisation
- calibration and model validation
- structuring a multi-stage analysis workflow for experimental physics

## Report

Project report: **[add link here when available]**

## Notes

This repository is presented as a portfolio item for the ellipsometry analysis component of the wider project. Some filenames and structure have been cleaned up for presentation, but the code reflects the main analysis logic used in the work.

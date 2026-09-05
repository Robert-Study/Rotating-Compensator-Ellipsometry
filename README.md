# Rotating-Compensator Ellipsometry

Python analysis pipeline developed for a third-year experimental physics project investigating the optical properties of thin films and surface plasmon resonance.

I led the wider eight-person nanophysics project and developed the ellipsometry analysis workflow used to process experimental rotating-compensator measurements.

> **Group project mark:** **71% — First-Class mark**  
> **Individual mark:** **75% — First-Class mark**

📄 **[View the final project report](https://1drv.ms/b/c/4a8cd531de3d2eb8/IQDELQqEGltQTbL7tnMde-ugARVe41tXLblHZuhFeltOyTY?e=LybArG)**

## Overview

Rotating-compensator ellipsometry determines the optical properties of a sample by analysing the change in polarisation of reflected light.

This project developed a complete analysis workflow for experimental measurements from gold and silver thin films, combining signal processing, instrument calibration, optical modelling and numerical optimisation.

The analysis was used to extract the ellipsometric parameters **Ψ** and **Δ**, before fitting thin-film properties using Fresnel-based optical models.

## Analysis Pipeline

The workflow covers:

1. Import and preprocessing of experimental measurements
2. Harmonic fitting of rotating-compensator intensity data
3. Instrument calibration
4. Extraction of ellipsometric parameters Ψ and Δ
5. Fresnel–Airy thin-film modelling
6. Numerical fitting of film thickness and optical constants
7. Uncertainty analysis and visualisation of results

## Technical Methods

- Scientific programming in Python
- Harmonic fitting and signal processing
- Numerical optimisation and curve fitting
- Instrument calibration
- Fresnel–Airy optical modelling
- Thin-film parameter estimation
- Experimental uncertainty analysis
- Data visualisation

## Project Context

The analysis formed one technical component of an eight-person nanophysics group project investigating thin films and surface plasmon resonance.

As **Project Leader**, I coordinated experimental planning, project milestones and technical discussions across the group while developing the ellipsometry analysis pipeline.

The wider project investigated gold and silver thin films produced using magnetron sputtering and compared experimental measurements with optical simulations.

## Repository Structure

```text
Rotating-Compensator-Ellipsometry/
├── calibration_fit.py
├── ellipsometry_common.py
├── ellipsometry_io.py
├── film_property_fit.py
├── fresnel_sim.py
├── harmonics_fit.py
├── pcsa_model.py
├── plotting_and_output.py
├── psi_delta_extraction.py
├── run_ellipsometry_pipeline.py
├── README.md
└── ...

# Rotating-Compensator Ellipsometry

Python analysis pipeline developed as part of an eight-person experimental nanophysics project investigating **thin-film optical properties and surface plasmon resonance**.

I served as **Project Leader** and developed a substantial part of the ellipsometry modelling and analysis workflow, including harmonic fitting, Ψ–Δ extraction, instrument calibration, Fresnel–Airy modelling and thin-film parameter estimation.

> **Group project mark: 71% — First-Class mark**  
> **Individual mark: 75% — First-Class mark**

📄 **[View the final project report](https://1drv.ms/b/c/4a8cd531de3d2eb8/IQDELQqEGltQTbL7tnMde-ugARVe41tXLblHZuhFeltOyTY?e=aEIhof)**

---

## Project Highlights

- Developed an end-to-end analysis pipeline for rotating-compensator ellipsometry
- Extracted ellipsometric parameters **Ψ** and **Δ** from experimental intensity waveforms
- Built and validated a silicon-reference calibration across a measured incidence-angle sweep
- Recovered a silicon-oxide thickness of **55 ± 7 nm**, consistent with the certified **53.30 nm** reference value
- Cross-validated extracted parameters against a **high-resolution industrial ellipsometer**
- Measured gold-film thicknesses that tracked profilometry with **R² = 0.985** across the sample set
- Extracted gold optical constants of **n = 0.195 ± 0.009** and **k = 3.51 ± 0.03** at 632.8 nm, close to published values
- Led project planning, milestones and technical coordination across the wider eight-person group

---

## My Contribution

The wider project combined thin-film fabrication, profilometry, ellipsometry and plasmon imaging. My work focused primarily on the **ellipsometry modelling, calibration and gold-film analysis**, while I also led the overall project.

My technical contributions included:

- Joint development of the Fresnel-coefficient, Jones-matrix and harmonic intensity models
- Development of the **Ψ and Δ extraction** methodology
- Construction of the full analysis and simulation workflow
- Silicon-reference calibration and validation across incidence angle
- Comparison with independent industrial-ellipsometer measurements
- Experimental waveform reconstruction and model comparison
- Determination of the pseudo-Brewster angle for the gold samples
- Analysis of gold-film thicknesses and complex refractive indices

---

## Analysis Pipeline

The repository implements the main stages of the experimental analysis:

1. **Import and preprocessing** of rotating-compensator measurements
2. **Harmonic fitting** of measured intensity waveforms
3. **Instrument calibration** using a silicon-oxide reference sample
4. **Ψ–Δ extraction** from calibrated harmonic coefficients
5. **Fresnel–Airy thin-film modelling**
6. **Numerical fitting** of film thickness and optical constants
7. **Validation and uncertainty analysis** against reference and independent measurements

---

## Validation Results

### Silicon reference

Calibration was tested across the full measured incidence-angle sweep rather than only at the 70° calibration point. The corrected Ψ–Δ values moved consistently towards the simulated reference trajectory, while the extracted film properties remained approximately constant across angle.

A constant fit gave:

- **Thickness:** `55 ± 7 nm` vs certified `53.30 nm`
- **Refractive index:** `1.464 ± 0.004` vs certificate value `1.455`

An independent industrial ellipsometer measured a thickness of **55.48 ± 0.07 nm** at 70°, providing an external cross-check on the calibration.

### Gold thin films

Six gold-coated prisms were analysed at their experimentally determined pseudo-Brewster angle of **71.2 ± 0.2°**.

The most meaningful validation was the direct comparison between ellipsometric and profilometric thickness measurements. The two methods showed a strong correlation:

- **R² = 0.985**
- Best-fit slope through the origin: **1.46 ± 0.08**

This showed that both methods tracked the same relative thickness changes across the sample set, while also revealing a systematic scale difference consistent with known step-height underestimation in profilometry.

---

## Technical Methods

`Python` · `NumPy` · `SciPy` · `Matplotlib`

- Harmonic fitting and signal processing
- Instrument calibration
- Fresnel coefficients and Jones matrices
- Fresnel–Airy thin-film modelling
- Numerical optimisation and curve fitting
- Thin-film thickness and optical-constant estimation
- Uncertainty propagation and model validation
- Experimental data visualisation

---

## Repository Structure

```text
Rotating-Compensator-Ellipsometry/
├── calibration_fit.py            # Instrument calibration
├── ellipsometry_common.py        # Shared utilities and parameters
├── ellipsometry_io.py            # Experimental data import
├── film_property_fit.py          # Thin-film parameter fitting
├── fresnel_sim.py                # Fresnel-based simulation
├── harmonics_fit.py              # Harmonic waveform fitting
├── pcsa_model.py                 # Rotating-compensator intensity model
├── plotting_and_output.py        # Results and visualisation
├── psi_delta_extraction.py       # Ψ and Δ extraction
├── run_ellipsometry_pipeline.py  # End-to-end analysis entry point
└── README.md
```

---

## Wider Project Context

The full nanophysics project investigated sputtered **gold and silver thin films** using magnetron sputtering, profilometry, rotating-compensator ellipsometry and surface-plasmon measurements in the Kretschmann configuration.

The combined project successfully produced and characterised thin films, extracted film thicknesses and optical constants, and observed surface plasmon resonance for both gold and silver samples.

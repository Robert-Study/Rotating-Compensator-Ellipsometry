rm reconstruction and model comparison
- Determination of the pseudo-Brewster angle for the gold samples
- Analysis of gold-film thicknesses
- Extraction of complex refractive indices
- Uncertainty propagation and physical validation of fitted results

As Project Leader, I also coordinated experimental priorities, milestones and technical discussions across the wider fabrication, profilometry, ellipsometry and plasmon-imaging work.

---

## Analysis Pipeline

The repository implements the main stages required to convert raw rotating-compensator measurements into physical film properties:

1. **Import and preprocessing** of experimental intensity sweeps
2. **Harmonic fitting** of the periodic rotating-compensator waveform
3. **Instrument calibration** using a certified silicon-oxide reference
4. **Ψ–Δ extraction** from calibrated harmonic coefficients
5. **Fresnel–Airy thin-film modelling**
6. **Numerical fitting** of film thickness and complex refractive index
7. **Validation and uncertainty analysis** against reference samples and independent measurements

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

`Python` · `NumPy` · `SciPy` · `Pandas` · `Matplotlib`

- Harmonic fitting and signal processing
- Instrument calibration
- Fresnel coefficients and Jones matrices
- Fresnel–Airy thin-film modelling
- Numerical optimisation and curve fitting
- Thin-film thickness and optical-constant estimation
- Uncertainty propagation and model validation
- Experimental data visualisation

### Setup

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The analysis modules are designed around experimental rotating-compensator sweeps and can be combined through `run_ellipsometry_pipeline.py` for an end-to-end workflow.

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
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Wider Project Context

The full nanophysics project investigated sputtered **gold and silver thin films** using magnetron sputtering, profilometry, rotating-compensator ellipsometry and surface-plasmon measurements in the Kretschmann configuration.

The combined project successfully produced and characterised thin films, extracted film thicknesses and optical constants, and observed surface plasmon resonance for both gold and silver samples.

# Rotating-Compensator Ellipsometry

Python analysis pipeline developed as part of an eight-person experimental nanophysics project investigating **thin-film optical properties and surface plasmon resonance**.

I served as **Project Leader** and developed a substantial part of the ellipsometry modelling and analysis workflow, including harmonic fitting, Ψ–Δ extraction, instrument calibration, Fresnel–Airy modelling and thin-film parameter estimation.

> **Group project mark: 71% — First-Class mark**  
> **Individual mark: 75% — First-Class mark**

📄 **[View the final project report](https://1drv.ms/b/c/4a8cd531de3d2eb8/IQDELQqEGltQTbL7tnMde-ugARVe41tXLblHZuhFeltOyTY?e=aEIhof)**

---

## A Glimpse of the Analysis

One of the clearest outcomes of the project is shown below:

<p align="center">
  <img
    width="850"
    alt="Ellipsometry calibration result showing measured and simulated Psi-Delta behaviour"
    src="https://github.com/user-attachments/assets/5d684c04-eeb7-4e07-8c3b-778e508ebeae"
  />
</p>

<p align="center">
  <em>Figure 1. Extracted Ψ and Δ values for the silicon-oxide reference sample before and after calibration, compared with the simulated incidence sweep.</em>
</p>

The aim of the analysis was to turn raw rotating-compensator intensity measurements into physically meaningful thin-film properties.

Compared with previous-year implementations of the experiment, the improved optical setup and analysis tracked the highly sensitive **Ψ and Δ parameters approximately 25× more closely**. I then developed a custom instrument calibration which reduced the remaining Ψ–Δ tracking error by approximately a **further factor of two**.

The result was a sufficiently accurate and stable measurement pipeline to recover **film thickness and complex optical properties**, something previous iterations of the project had not successfully achieved.

Crucially, the calibration was not limited to one measurement or one sample. It was derived using a silicon-oxide reference, validated across a full incidence-angle sweep, and then carried forward to the **gold and silver thin-film measurements**. This made it an instrument-level correction rather than a one-off fit to a particular dataset.

---

## Project Highlights

- Led an **eight-person experimental nanophysics project**
- Developed an end-to-end rotating-compensator ellipsometry analysis pipeline
- Improved Ψ–Δ tracking by approximately **25× compared with previous-year implementations**
- Developed a custom calibration that reduced the remaining tracking error by approximately **2×**
- Built a calibration that generalised across **different incidence angles and thin-film samples**
- Extracted ellipsometric parameters **Ψ** and **Δ** from experimental intensity waveforms
- Recovered a silicon-oxide thickness of **55 ± 7 nm**, consistent with the certified **53.30 nm**
- Cross-validated extracted parameters against a **high-resolution industrial ellipsometer**
- Measured gold-film thicknesses that tracked profilometry with **R² = 0.985**
- Extracted gold optical constants of **n = 0.195 ± 0.009** and **k = 3.51 ± 0.03** at 632.8 nm
- Enabled successful estimation of thin-film **thickness, refractive index and extinction coefficient**

---

## Why the Calibration Mattered

Ellipsometry is extremely sensitive to small experimental imperfections.

Slight errors in:

- polariser alignment
- compensator orientation
- retardance
- incidence angle
- optical extinction
- rotational wobble
- detector response
- normalisation

can produce measurable shifts in Ψ and Δ.

Rather than treating these effects independently, I developed an empirical instrument calibration using a certified silicon-oxide reference sample.

The measured harmonic coefficients were mapped onto the expected optical response of the reference, producing a correction that characterised the **measurement system itself**.

This distinction was important.

The calibration was first determined at the reference measurement, but when applied across the full incidence-angle sweep it continued to move the extracted Ψ–Δ values towards the simulated trajectory rather than only improving the original calibration point.

It could then be applied unchanged to the gold and silver measurements.

This provided strong evidence that the calibration was correcting **systematic instrument behaviour**, rather than simply overfitting one sample.

---

## My Contribution

The wider project combined thin-film fabrication, profilometry, ellipsometry and surface-plasmon imaging.

My work focused primarily on the **ellipsometry modelling, calibration and gold-film analysis**, while I also coordinated the overall eight-person project.

My technical contributions included:

- Joint development of the Fresnel-coefficient, Jones-matrix and harmonic intensity models
- Development of the **Ψ and Δ extraction methodology**
- Construction of the full analysis and simulation workflow
- Design of the silicon-reference calibration
- Validation of calibration across incidence angle
- Comparison with independent industrial-ellipsometer measurements
- Experimental waveform reconstruction and model comparison
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

# Nanophysics Group Project

**Rotating-compensator ellipsometry, thin films and surface plasmon resonance**

I led the experimental group project and worked mainly on the ellipsometry modelling, instrument calibration and gold-film analysis. This repository contains the Python analysis workflow. The wider project also covered thin-film fabrication, profilometry and surface-plasmon measurements.

**Group mark: 71% · Individual mark: 75%** — both First-Class marks.

[Read the Nanophysics Group Project report](https://1drv.ms/b/c/4a8cd531de3d2eb8/IQDELQqEGltQTbL7tnMde-ugARVe41tXLblHZuhFeltOyTY?e=aEIhof) · [Analysis notes](docs/analysis-notes.md) · [Tests](https://github.com/Robert-Study/Rotating-Compensator-Ellipsometry/actions)

## What the calibration changed

Small alignment errors in a rotating-compensator instrument can shift the extracted Ψ and Δ values enough to affect the recovered film properties. I developed a calibration using the silicon-oxide reference and checked whether the correction still helped away from the 70° calibration point.

![Silicon-reference Psi and Delta before and after calibration, compared with the simulated incidence sweep](assets/calibration-reference.png)

*Figure 17 from the report, §4.5.2. The orange points show the calibrated measurements; the red points show the uncalibrated results.*

The correction improved agreement across the measured incidence sweep and was carried forward to the gold and silver measurements. The useful result was a calibration that worked beyond its original reference point. A residual offset in Δ remained, which is discussed in the report.

## Results from the experiment

| Check | Reported result | What it establishes |
| --- | --- | --- |
| Silicon-oxide thickness across incidence angles | **55 ± 7 nm**; certified value **53.30 nm** | Agreement within the reported uncertainty |
| Independent industrial ellipsometer at 70° | **55.48 ± 0.07 nm** | An external thickness comparison |
| Silicon-oxide refractive index | **1.464 ± 0.004**; certificate **1.455** | A small remaining discrepancy |
| Gold thickness against profilometry | **R² = 0.985**, slope **1.46 ± 0.08** through the origin | Strong tracking, with different absolute thickness scales |
| Gold optical constants at 632.8 nm | **n = 0.195 ± 0.009**, **k = 3.51 ± 0.03** | Estimates conditional on the thin-film model |

The high correlation with profilometry does not establish absolute accuracy: the slope differs substantially from one. The measured film profiles provide evidence for step-height underestimation, but roughness, interfaces and model assumptions also need consideration. At 75°, the calibrated silicon thickness was **68 ± 6 nm**, showing that performance was not uniform across all angles.

## My contribution

The report identifies authors by initials. My individual sections cover Ψ–Δ extraction, the analysis workflow, silicon calibration and validation, the industrial-ellipsometer comparison, and the gold-film results (§§4.4.2, 4.4.4, 4.5 and 4.6.1). I also contributed jointly to the Fresnel, Jones-matrix and intensity modelling.

As project leader, I coordinated experimental priorities, milestones and discussion between the fabrication, profilometry, ellipsometry and plasmon groups. The report credits the other contributors for their work.

## Run an example

Use Python 3.12. From a fresh checkout:

```bash
git clone https://github.com/Robert-Study/Rotating-Compensator-Ellipsometry.git
cd Rotating-Compensator-Ellipsometry
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
```

The example writes synthetic input sweeps, waveform plots, calibration tables and `synthetic-recovery.json` to `outputs/synthetic-demo/`. Its known film has **d = 83 nm, n = 1.7, k = 0.1**. The saved [example result](assets/synthetic-recovery.json) shows the recovered parameters for seed 2026 with added noise.

The example checks that the software runs and recovers a known model. It does not reproduce the laboratory measurements: the raw experimental sweeps are not included.

## Using measured data

The input files contain two numeric columns: compensator angle in degrees and intensity. Subtract a measured detector dark signal before normalising. Filenames can use an explicit angle such as `gold_250s_70deg.txt`; sputtering time is not interpreted as incidence angle.

`run_example_pipeline` accepts reference sweeps and measurements of **one unknown film at multiple incidence angles**. The current model fits three film parameters, so a single Ψ–Δ pair is insufficient. See [the analysis notes](docs/analysis-notes.md) for conventions, assumptions and the module map.

The report records the original assessed work. The modular code, synthetic example and tests have been revised since assessment.

# Nanophysics Group Project

**Rotating-compensator ellipsometry, thin films and surface plasmon resonance**

I led an **eight-person experimental nanophysics project**, coordinating the work across fabrication, profilometry, ellipsometry and surface-plasmon measurements. My technical work focused on the ellipsometry modelling, Ψ–Δ extraction and instrument calibration.

**71% group mark · 75% individual mark — First Class**

[Read the project report](https://1drv.ms/b/c/4a8cd531de3d2eb8/IQDELQqEGltQTbL7tnMde-ugARVe41tXLblHZuhFeltOyTY?e=aEIhof)

## Improving the measurement

![Silicon-reference Psi and Delta before and after calibration, compared with the simulated incidence sweep](assets/calibration-reference.png)

*Silicon-oxide reference measurements before and after calibration, compared with the predicted incidence sweep. Figure 17 from the project report.*

Compared with previous-year implementations, the improved optical setup and analysis tracked the highly sensitive **Ψ and Δ parameters approximately 25× more closely**. I then developed a custom instrument calibration that reduced the remaining tracking error by approximately a **further factor of two**.

The calibration was derived from a silicon-oxide reference, checked across the measured incidence-angle sweep and carried forward to the gold and silver measurements. This allowed the analysis to recover film thickness and complex optical properties beyond the original reference measurement.

The silicon-reference analysis gave **55 ± 7 nm** for film thickness, compared with the certified **53.30 nm**. An independent industrial ellipsometer supplied a further comparison. The full report discusses the residual Δ offset and the variation in accuracy with incidence angle.

## My contribution

I developed the Ψ–Δ extraction methodology and the silicon-reference calibration, connected the analysis stages, and checked the recovered parameters against reference measurements. I also contributed to the Fresnel, Jones-matrix and harmonic intensity models and worked on the gold-film analysis.

The report identifies my individual contributions in §§4.4.2, 4.4.4, 4.5 and 4.6.1. The wider project and the thickness comparison with profilometry were shared work.

As project leader, I coordinated experimental priorities, milestones and technical discussions between the different measurement groups.

## From intensity sweeps to film properties

1. Import and normalise the rotating-compensator intensity measurements.
2. Fit the periodic waveform and extract its harmonic content.
3. Calibrate the instrument against the known silicon-oxide reference.
4. Recover the ellipsometric parameters Ψ and Δ.
5. Use Fresnel–Airy thin-film modelling to estimate thickness, refractive index and extinction coefficient.
6. Compare the recovered parameters with reference samples and independent measurements.

The repository contains the analysis modules associated with this work: experimental input handling, harmonic fitting, the optical instrument model, calibration, film fitting and plotting. [Analysis notes](docs/analysis-notes.md) describe the conventions and fitting assumptions.

**Methods:** Python, NumPy, SciPy, pandas, Matplotlib, harmonic analysis, nonlinear fitting, optical modelling and uncertainty propagation.

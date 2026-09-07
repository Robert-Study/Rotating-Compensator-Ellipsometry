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

# Ellipsometry analysis notes

## Experimental evidence and current software

The [Nanophysics Group Project report](https://1drv.ms/b/c/4a8cd531de3d2eb8/IQDELQqEGltQTbL7tnMde-ugARVe41tXLblHZuhFeltOyTY?e=aEIhof) is the source for the experimental values in the README. Useful sections are:

| Report section | Evidence |
| --- | --- |
| §4.4.2 | Ψ–Δ extraction |
| §§4.5.1–4.5.2 | Empirical reference calibration and checks across incidence angle |
| §4.5.3 | Independent industrial-ellipsometer comparison |
| §4.5.4 | Waveform comparison, including the less accurate 75° thickness result |
| §4.6.1 | Gold thickness comparison and optical constants |
| Appendix §9.1.4 | Programs used in the experimental project |

The current modules implement a reusable physical instrument model and joint film fitting. They should not be treated as an exact reconstruction of the report's empirical calibration. The original measured sweeps and calibration inputs belong to the experimental project.

## Optical convention

Refractive indices use **n + i k**, with **k ≥ 0** for passive absorption. The field convention is `exp(i kz z − i ωt)`. The round-trip factor inside the film is therefore `exp(2 i β)`, with `β = 2π kz d / λ` when kz denotes the normal refractive-index component. The forward branch is chosen so the imaginary part of that component is nonnegative.

For a thick absorbing film, propagation to the rear interface vanishes and the result approaches the ambient–film Fresnel interface. The zero-thickness limit, Brewster reflection and total internal reflection provide additional physical checks on this convention.

The convention is consistent with the forward-wave discussion in [Byrnes' transfer-matrix implementation](https://github.com/sbyrnes321/tmm/blob/master/tmm_core.py). The implementation here is limited to a single isotropic film with a nonabsorbing incident medium.

## Data and identifiability

Subtract a measured dark signal before normalising the intensity. A freely fitted detector offset can make Ψ, Δ and gain nonidentifiable from a rotating-compensator waveform. The current extraction fits Ψ, Δ and positive gain with the detector offset fixed at zero after correction.

The film model has three unknowns: thickness, real refractive index and extinction coefficient. A single Ψ–Δ pair gives only two measured quantities. Joint fitting therefore requires at least two distinct incidence angles of the same film; more angles, sensible bounds and checks from different initial guesses are preferable. Multiple angles do not guarantee uniqueness.

Fits use equal weights in Ψ and Δ measured in degrees. Reported covariance estimates are local and conditional on the assumed optical model and calibration. They do not include surface roughness, thickness variation, incidence-angle error, reference uncertainty or propagated calibration uncertainty. Rank-deficient fits do not receive finite covariance errors; film fits at a parameter bound return undefined symmetric errors rather than misleading precision.

The Jacobian condition number is recorded as a diagnostic. It depends on parameter units and scaling, so it is not a universal acceptance threshold.

## Input format

Each text file has two numeric columns, separated by whitespace or commas:

```text
# compensator_angle_deg intensity_after_dark_correction
0.0 0.031
5.0 0.033
10.0 0.039
```

Use at least ten observations spanning the waveform. Numeric metadata embedded in prose is ignored. Zero-valued readings are retained by default; filtering is explicit and the remaining sample count is checked.

Use filenames such as `reference_70deg.txt` or supply the incidence angle directly to `build_sweep`. All sweeps for one film must share the same sample name. Do not combine different sputtered samples into one multi-angle fit.

## Module map

| Module | Purpose |
| --- | --- |
| `ellipsometry_io.py` | Load and normalise measurements |
| `harmonics_fit.py` | Inspect harmonic content |
| `pcsa_model.py` | Polariser–compensator–sample–analyser intensity model |
| `calibration_fit.py` | Fit instrument parameters against known reference stacks |
| `psi_delta_extraction.py` | Recover Ψ and Δ from dark-corrected sweeps |
| `fresnel_sim.py` | Passive single-film Fresnel–Airy model |
| `film_property_fit.py` | Jointly fit thickness, n and k across incidence angles |
| `run_ellipsometry_pipeline.py` | Connect the analysis stages and save outputs |

The defaults leave wobble and retardance fixed. Extra instrument parameters should be introduced only when reference data support their identification.


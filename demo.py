"""Run a small synthetic ellipsometry example without laboratory files.

The generated data are labelled synthetic throughout. Recovery checks numerical
consistency of the implemented model; it does not reproduce the report experiment.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json

import numpy as np

from ellipsometry_common import FilmStack, InstrumentParameters, SweepData
from fresnel_sim import rho_from_stack
from pcsa_model import instrument_intensity_from_rho
from run_ellipsometry_pipeline import run_example_pipeline


def make_sweeps(stack, instrument, angles, sample_name, rng, noise=0.0):
    theta = np.linspace(0, 360, 73, endpoint=False)
    sweeps = []
    for angle in angles:
        intensity = instrument_intensity_from_rho(theta, rho_from_stack(stack, angle), instrument)
        intensity /= intensity.max()
        intensity += rng.normal(0, noise, len(theta))
        sweeps.append(SweepData(sample_name, float(angle), theta, intensity, intensity,
                                f'synthetic:{sample_name}:{angle}', 1.0,
                                {'synthetic': True, 'noise_std': noise, 'dark_offset': 0.0}))
    return sweeps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/synthetic-demo'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--noise', type=float, default=0.0002,
                        help='Gaussian noise SD in units of normalised intensity.')
    args = parser.parse_args()
    if not np.isfinite(args.noise) or args.noise < 0:
        parser.error('--noise must be finite and nonnegative')
    rng = np.random.default_rng(args.seed)
    instrument = InstrumentParameters(polariser_deg=-44, analyser_deg=46, compensator_zero_deg=2)
    reference = FilmStack(1+0j, 1.46+0j, 3.88+0.02j, 53.3, 632.8)
    unknown = FilmStack(1+0j, 1.7+0.1j, 1.52+0j, 83.0, 632.8)
    angles = [45, 50, 55, 60, 65, 70]
    refs = make_sweeps(reference, instrument, angles, 'synthetic_reference', rng, args.noise)
    measurements = make_sweeps(unknown, instrument, angles, 'synthetic_film', rng, args.noise)
    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / 'synthetic-inputs'
    raw_dir.mkdir(exist_ok=True)
    for sweep in refs + measurements:
        np.savetxt(raw_dir / f'{sweep.sample_name}_{sweep.incidence_angle_deg:g}deg.txt',
                   np.column_stack([sweep.compensator_angle_deg, sweep.intensity_norm]),
                   header='SYNTHETIC: compensator_angle_deg normalised_intensity')
    result = run_example_pipeline(
        refs, measurements, lambda angle: reference,
        {'thickness_nm': 70, 'n_real': 1.6, 'k_imag': 0.05,
         'psi_deg': 25, 'delta_deg': -140, 'wavelength_nm': 632.8},
        ambient_n=unknown.ambient, substrate_n=unknown.substrate, output_dir=args.output,
    )
    summary = {'synthetic': True, 'seed': args.seed, 'noise_std': args.noise,
               'known_input': {'thickness_nm': unknown.thickness_nm, 'n_real': unknown.film.real,
                               'k_imag': unknown.film.imag}, 'recovered': asdict(result['film_fit'])}
    (args.output / 'synthetic-recovery.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

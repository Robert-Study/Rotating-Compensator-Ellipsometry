from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from calibration_fit import calibrate_instrument_from_reference
from ellipsometry_common import InstrumentParameters, save_json
from film_property_fit import fit_film_properties_from_psidelta
from plotting_and_output import (
    ensure_output_dir,
    plot_metric_vs_incidence,
    plot_psi_delta_vs_incidence,
    plot_sweep_with_fit,
    psi_delta_table,
    summarise_harmonics,
)
from psi_delta_extraction import fit_psi_delta_for_sweep


def run_example_pipeline(reference_sweeps, unknown_sweeps, reference_stack_builder, unknown_fit_guesses, ambient_n, substrate_n, output_dir='ellipsometry_pipeline_output'):
    out_dir = ensure_output_dir(output_dir)

    instrument, cal_table = calibrate_instrument_from_reference(
        reference_sweeps=reference_sweeps,
        reference_stack_builder=reference_stack_builder,
        initial_instrument=InstrumentParameters(),
        fit_retardance=False,
        fit_wobble=True,
    )
    cal_table.to_csv(out_dir / 'reference_calibration_summary.csv', index=False)
    save_json(asdict(instrument), out_dir / 'instrument_parameters.json')

    fit_list = []
    psi_guess = float(unknown_fit_guesses.get('psi_deg', 45.0))
    delta_guess = float(unknown_fit_guesses.get('delta_deg', 90.0))

    for sweep in unknown_sweeps:
        fit = fit_psi_delta_for_sweep(sweep, instrument, psi_guess_deg=psi_guess, delta_guess_deg=delta_guess, allow_scale_and_offset=True)
        fit_list.append(fit)
        psi_guess, delta_guess = fit.psi_deg, fit.delta_deg  # last fit is usually a decent next guess

        plot_sweep_with_fit(
            sweep,
            instrument,
            fit.psi_deg,
            fit.delta_deg,
            fit.y_scale,
            fit.y_offset,
            output_path=out_dir / f'{sweep.sample_name}_{sweep.incidence_angle_deg:.1f}deg_waveform_fit.png',
        )

    psi_delta_df = psi_delta_table(fit_list)
    psi_delta_df.to_csv(out_dir / 'unknown_sample_psi_delta.csv', index=False)
    plot_psi_delta_vs_incidence(psi_delta_df, output_path_prefix=out_dir / 'psi_delta')

    film_fit = fit_film_properties_from_psidelta(
        sample_name=str(psi_delta_df['sample_name'].iloc[0]) if len(psi_delta_df) else 'unknown',
        incidence_angles_deg=psi_delta_df['incidence_angle_deg'],
        psi_measured_deg=psi_delta_df['psi_deg'],
        delta_measured_deg=psi_delta_df['delta_deg'],
        wavelength_nm=float(unknown_fit_guesses.get('wavelength_nm', 632.8)),
        ambient_n=ambient_n,
        substrate_n=substrate_n,
        thickness_guess_nm=float(unknown_fit_guesses['thickness_nm']),
        n_guess=float(unknown_fit_guesses['n_real']),
        k_guess=float(unknown_fit_guesses['k_imag']),
    )
    pd.DataFrame([asdict(film_fit)]).to_csv(out_dir / 'unknown_sample_film_fit.csv', index=False)

    harmonic_df = summarise_harmonics(unknown_sweeps)
    harmonic_df.to_csv(out_dir / 'unknown_sample_harmonics.csv', index=False)
    if len(harmonic_df) >= 5:
        plot_metric_vs_incidence(
            harmonic_df['incidence_angle_deg'],
            harmonic_df['y0'],
            ylabel='Fitted DC term / a.u.',
            title='Pseudo-Brewster estimate from harmonic minimum',
            output_path=out_dir / 'pseudo_brewster_from_dc_term.png',
        )

    return {
        'instrument': instrument,
        'reference_calibration': cal_table,
        'psi_delta_table': psi_delta_df,
        'film_fit': film_fit,
        'harmonic_table': harmonic_df,
        'output_dir': out_dir,
    }

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from harmonics_fit import fit_fourier_harmonics
from pcsa_model import instrument_intensity_from_psidelta


def ensure_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def psi_delta_table(fits):
    rows = [asdict(fit) for fit in fits]
    return pd.DataFrame(rows).sort_values(['sample_name', 'incidence_angle_deg']).reset_index(drop=True)


def summarise_harmonics(sweeps):
    rows = []
    for sweep in sweeps:
        fit = fit_fourier_harmonics(sweep.compensator_angle_deg, sweep.intensity_norm)
        fit.update({
            'sample_name': sweep.sample_name,
            'incidence_angle_deg': sweep.incidence_angle_deg,
            'source_file': sweep.source_file,
        })
        rows.append(fit)
    return pd.DataFrame(rows).sort_values(['sample_name', 'incidence_angle_deg']).reset_index(drop=True)


def quartic_argmin(x_vals, y_vals):
    x_vals = np.asarray(x_vals, float)
    y_vals = np.asarray(y_vals, float)
    coeffs = np.polyfit(x_vals, y_vals, 4)
    roots = np.roots(np.polyder(coeffs))
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots >= np.min(x_vals)) & (roots <= np.max(x_vals))]

    if len(roots) == 0:
        x_min = float(x_vals[np.argmin(y_vals)])
        return coeffs, x_min, float(np.polyval(coeffs, x_min))

    y_roots = np.polyval(coeffs, roots)
    best = int(np.argmin(y_roots))
    x_min = float(roots[best])
    return coeffs, x_min, float(y_roots[best])


def estimate_pseudo_brewster_angle(incidence_angles_deg, metric_values):
    coeffs, x_min, y_min = quartic_argmin(incidence_angles_deg, metric_values)
    return {
        'quartic_coefficients': coeffs,
        'pseudo_brewster_angle_deg': x_min,
        'metric_at_minimum': y_min,
    }


def plot_sweep_with_fit(sweep, instrument, psi_deg, delta_deg, y_scale, y_offset, output_path=None):
    angle_deg = sweep.compensator_angle_deg
    y_data = sweep.intensity_norm
    y_fit = instrument_intensity_from_psidelta(angle_deg, psi_deg, delta_deg, instrument, y_scale=y_scale, y_offset=y_offset)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(angle_deg, y_data, label='Measured')
    ax.plot(angle_deg, y_fit, label='Best fit')
    ax.set_xlabel('Compensator angle / deg')
    ax.set_ylabel('Normalised intensity / a.u.')
    ax.set_title(f"{sweep.sample_name}, incidence {sweep.incidence_angle_deg:.1f}$^{{\circ}}$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
    return fig, ax


def plot_psi_delta_vs_incidence(psi_delta_df, output_path_prefix=None):
    fig1, ax1 = plt.subplots(figsize=(7.2, 4.5))
    ax1.plot(psi_delta_df['incidence_angle_deg'], psi_delta_df['psi_deg'], 'o-')
    ax1.set_xlabel('Incidence angle / deg')
    ax1.set_ylabel(r'$\Psi$ / deg')
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.5))
    ax2.plot(psi_delta_df['incidence_angle_deg'], psi_delta_df['delta_deg'], 'o-')
    ax2.set_xlabel('Incidence angle / deg')
    ax2.set_ylabel(r'$\Delta$ / deg')
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    if output_path_prefix is not None:
        fig1.savefig(f'{output_path_prefix}_psi.png', dpi=200, bbox_inches='tight')
        fig2.savefig(f'{output_path_prefix}_delta.png', dpi=200, bbox_inches='tight')

    return (fig1, ax1), (fig2, ax2)


def plot_metric_vs_incidence(incidence_angles_deg, metric_values, ylabel, title, output_path=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(incidence_angles_deg, metric_values, 'o-')
    ax.set_xlabel('Incidence angle / deg')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
    return fig, ax

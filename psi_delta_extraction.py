from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ellipsometry_common import InstrumentParameters, PsiDeltaFit, fit_stds, r2_score, rmse, tidy_psi_delta
from pcsa_model import instrument_intensity_from_psidelta


def fit_psi_delta_for_sweep(sweep, instrument: InstrumentParameters, psi_guess_deg=45.0, delta_guess_deg=90.0, allow_scale_and_offset=False):
    """Fit Psi, Delta and positive gain to a dark-corrected intensity sweep.

    A free detector offset can make the inverse problem unidentifiable. Measure
    and subtract the dark signal before normalisation instead of fitting it.
    """
    if allow_scale_and_offset:
        raise ValueError('Subtract a measured dark signal first; a free offset can make Psi and Delta unidentifiable.')
    angles = np.asarray(sweep.compensator_angle_deg, float)
    y_data = np.asarray(sweep.intensity_norm, float)
    if angles.ndim != 1 or y_data.shape != angles.shape or len(angles) < 10:
        raise ValueError('A sweep needs at least ten matching angle/intensity observations.')
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(y_data)) or np.ptp(y_data) == 0:
        raise ValueError('Sweep data must be finite and contain a varying signal.')
    guess = np.array([psi_guess_deg, delta_guess_deg, 1.0], float)
    low = np.array([1e-6, -180.0, 0.0], float)
    high = np.array([89.999999, 180.0, np.inf], float)

    def unpack(vals):
        return float(vals[0]), float(vals[1]), float(vals[2]), 0.0

    def residuals(vals):
        psi_deg, delta_deg, scale, offset = unpack(vals)
        y_fit = instrument_intensity_from_psidelta(angles, psi_deg, delta_deg, instrument, y_scale=scale, y_offset=offset)
        return y_fit - y_data

    result = least_squares(residuals, guess, bounds=(low, high), method='trf')
    if not result.success:
        raise RuntimeError('Psi-Delta fit failed: ' + result.message)
    psi_deg, delta_deg, scale, offset = unpack(result.x)
    psi_deg, delta_deg = tidy_psi_delta(psi_deg, delta_deg)
    y_fit = instrument_intensity_from_psidelta(angles, psi_deg, delta_deg, instrument, y_scale=scale, y_offset=offset)
    errs = fit_stds(result, len(y_data), len(result.x))

    return PsiDeltaFit(
        sample_name=sweep.sample_name,
        incidence_angle_deg=sweep.incidence_angle_deg,
        psi_deg=psi_deg,
        delta_deg=delta_deg,
        y_scale=float(scale),
        y_offset=float(offset),
        rmse=rmse(y_data, y_fit),
        r2=r2_score(y_data, y_fit),
        success=bool(result.success and np.linalg.matrix_rank(result.jac) == len(result.x)),
        n_points=len(y_data),
        psi_std_deg=float(errs[0]) if len(errs) > 0 else float('nan'),
        delta_std_deg=float(errs[1]) if len(errs) > 1 else float('nan'),
    )

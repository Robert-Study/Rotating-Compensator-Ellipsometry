from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ellipsometry_common import InstrumentParameters, PsiDeltaFit, fit_stds, r2_score, rmse, tidy_psi_delta
from pcsa_model import instrument_intensity_from_psidelta


def fit_psi_delta_for_sweep(sweep, instrument: InstrumentParameters, psi_guess_deg=45.0, delta_guess_deg=90.0, allow_scale_and_offset=True):
    angles = sweep.compensator_angle_deg
    y_data = sweep.intensity_norm

    if allow_scale_and_offset:
        guess = np.array([psi_guess_deg, delta_guess_deg, 1.0, 0.0], float)
        low = np.array([0.0, -180.0, 0.0, -2.0], float)
        high = np.array([90.0, 180.0, 10.0, 2.0], float)

        def unpack(vals):
            return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
    else:
        guess = np.array([psi_guess_deg, delta_guess_deg], float)
        low = np.array([0.0, -180.0], float)
        high = np.array([90.0, 180.0], float)

        def unpack(vals):
            return float(vals[0]), float(vals[1]), 1.0, 0.0

    def residuals(vals):
        psi_deg, delta_deg, scale, offset = unpack(vals)
        y_fit = instrument_intensity_from_psidelta(angles, psi_deg, delta_deg, instrument, y_scale=scale, y_offset=offset)
        return y_fit - y_data

    result = least_squares(residuals, guess, bounds=(low, high), method='trf')
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
        success=bool(result.success),
        n_points=len(y_data),
        psi_std_deg=float(errs[0]) if len(errs) > 0 else float('nan'),
        delta_std_deg=float(errs[1]) if len(errs) > 1 else float('nan'),
    )

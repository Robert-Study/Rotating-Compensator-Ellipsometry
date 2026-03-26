from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ellipsometry_common import FilmFitResult, fit_stds, wrap_pm180
from fresnel_sim import FilmStack, psi_delta_from_stack


def fit_film_properties_from_psidelta(sample_name, incidence_angles_deg, psi_measured_deg, delta_measured_deg, wavelength_nm, ambient_n, substrate_n, thickness_guess_nm, n_guess, k_guess, thickness_bounds_nm=(0.1, 1000.0), n_bounds=(0.1, 10.0), k_bounds=(0.0, 10.0)):
    angles = np.asarray(incidence_angles_deg, float)
    psi_data = np.asarray(psi_measured_deg, float)
    delta_data = np.asarray(delta_measured_deg, float)

    guess = np.array([thickness_guess_nm, n_guess, k_guess], float)
    low = np.array([thickness_bounds_nm[0], n_bounds[0], k_bounds[0]], float)
    high = np.array([thickness_bounds_nm[1], n_bounds[1], k_bounds[1]], float)

    def residuals(vals):
        thickness_nm, n_real, k_imag = map(float, vals)
        film_n = complex(n_real, k_imag)
        psi_res = []
        delta_res = []
        for ang, psi_here, delta_here in zip(angles, psi_data, delta_data):
            stack = FilmStack(ambient=ambient_n, film=film_n, substrate=substrate_n, thickness_nm=thickness_nm, wavelength_nm=wavelength_nm)
            psi_fit, delta_fit = psi_delta_from_stack(stack, ang)
            psi_res.append(psi_fit - psi_here)
            delta_res.append(wrap_pm180(delta_fit - delta_here))  # keeps delta on the nearby branch
        return np.concatenate([psi_res, delta_res])

    result = least_squares(residuals, guess, bounds=(low, high), method='trf')
    thickness_nm, n_real, k_imag = map(float, result.x)
    errs = fit_stds(result, 2 * len(angles), 3)
    final = residuals(result.x)
    psi_res = final[:len(angles)]
    delta_res = final[len(angles):]

    return FilmFitResult(
        sample_name=sample_name,
        thickness_nm=thickness_nm,
        n_real=n_real,
        k_imag=k_imag,
        rmse_psi_deg=float(np.sqrt(np.mean(np.asarray(psi_res) ** 2))),
        rmse_delta_deg=float(np.sqrt(np.mean(np.asarray(delta_res) ** 2))),
        success=bool(result.success),
        thickness_std_nm=float(errs[0]) if len(errs) > 0 else float('nan'),
        n_real_std=float(errs[1]) if len(errs) > 1 else float('nan'),
        k_imag_std=float(errs[2]) if len(errs) > 2 else float('nan'),
    )

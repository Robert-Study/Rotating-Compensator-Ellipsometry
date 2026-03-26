from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from ellipsometry_common import bic_from_rss, r2_score, rmse, rss


def fit_fourier_harmonics(theta_deg: np.ndarray, intensity: np.ndarray) -> dict[str, float]:
    theta_deg = np.asarray(theta_deg, float).ravel()
    intensity = np.asarray(intensity, float).ravel()
    keep = np.isfinite(theta_deg) & np.isfinite(intensity)
    theta_deg = theta_deg[keep]
    intensity = intensity[keep]

    th = np.deg2rad(theta_deg)
    design = np.column_stack([
        np.sin(4 * th),
        np.cos(4 * th),
        np.sin(2 * th),
        np.cos(2 * th),
        np.ones_like(th),
    ])
    coeffs, *_ = np.linalg.lstsq(design, intensity, rcond=None)
    a, b, c, d, y0 = map(float, coeffs)
    y_fit = design @ coeffs
    rss_val = rss(intensity, y_fit)

    return {
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'y0': y0,
        'rmse': rmse(intensity, y_fit),
        'r2': r2_score(intensity, y_fit),
        'rss': rss_val,
        'bic': bic_from_rss(rss_val, len(intensity), 5),
    }


def fit_alignment_harmonics(theta_deg: np.ndarray, intensity: np.ndarray) -> dict[str, float]:
    theta_deg = np.asarray(theta_deg, float).ravel()
    intensity = np.asarray(intensity, float).ravel()
    keep = np.isfinite(theta_deg) & np.isfinite(intensity)
    theta_deg = theta_deg[keep]
    intensity = intensity[keep]

    def model(theta_deg, a, b, c, d, e, f, y0, s1, phi1_deg, s2, phi2_deg):
        th = np.deg2rad(theta_deg)
        phi1 = np.deg2rad(phi1_deg)
        phi2 = np.deg2rad(phi2_deg)
        th_w = th + np.deg2rad(s1) * np.cos(th + phi1)  # angular wobble bit
        base = (
            y0
            + a * np.sin(4 * th_w)
            + b * np.cos(4 * th_w)
            + c * np.sin(2 * th_w)
            + d * np.cos(2 * th_w)
            + e * np.sin(th_w)
            + f * np.cos(th_w)
        )
        return (1 + s2 * np.sin(th + phi2)) * base  # slow modulation on top

    y0_guess = float(np.mean(intensity))
    amp_guess = 0.5 * float(np.nanmax(intensity) - np.nanmin(intensity))
    p0 = [0.3 * amp_guess, 0.0, 0.1 * amp_guess, 0.1 * amp_guess, 0.0, 0.0, y0_guess, 0.2, 0.0, 0.02, 0.0]
    low = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.0, -180.0, -0.3, -180.0]
    high = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2.0, 180.0, 0.3, 180.0]

    best, _ = curve_fit(model, theta_deg, intensity, p0=p0, bounds=(low, high), maxfev=50000)
    y_fit = model(theta_deg, *best)
    rss_val = rss(intensity, y_fit)
    a, b, c, d, e, f, y0, s1, phi1_deg, s2, phi2_deg = map(float, best)

    return {
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'e': e,
        'f': f,
        'y0': y0,
        's1': s1,
        'phi1_deg': phi1_deg,
        's2': s2,
        'phi2_deg': phi2_deg,
        'rmse': rmse(intensity, y_fit),
        'r2': r2_score(intensity, y_fit),
        'rss': rss_val,
        'bic': bic_from_rss(rss_val, len(intensity), len(best)),
    }

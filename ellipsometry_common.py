from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re

import numpy as np


def deg2rad(x):
    return np.deg2rad(x)


def rad2deg(x):
    return np.rad2deg(x)


def wrap_pm180(angle_deg):
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def tidy_psi_delta(psi_deg: float, delta_deg: float) -> tuple[float, float]:
    psi = float(psi_deg)
    delta = float(wrap_pm180(delta_deg))
    if psi < 0:
        psi = -psi
        delta += 180.0
    if psi > 90.0:
        psi = 180.0 - psi
        delta += 180.0
    return psi, float(wrap_pm180(delta))


def rmse(y_true, y_fit):
    y_true = np.asarray(y_true, float)
    y_fit = np.asarray(y_fit, float)
    return float(np.sqrt(np.mean((y_true - y_fit) ** 2)))


def rss(y_true, y_fit):
    y_true = np.asarray(y_true, float)
    y_fit = np.asarray(y_fit, float)
    return float(np.sum((y_true - y_fit) ** 2))


def r2_score(y_true, y_fit):
    y_true = np.asarray(y_true, float)
    y_fit = np.asarray(y_fit, float)
    ss_res = np.sum((y_true - y_fit) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def bic_from_rss(rss_val: float, n_pts: int, n_pars: int) -> float:
    if n_pts <= 0:
        return float('nan')
    rss_val = max(float(rss_val), 1e-300)
    return float(n_pts * np.log(rss_val / n_pts) + n_pars * np.log(n_pts))


def fit_stds(result, n_pts: int, n_pars: int) -> np.ndarray:
    jac = getattr(result, 'jac', None)
    if jac is None:
        return np.full(n_pars, np.nan)

    jac = np.asarray(jac, float)
    if jac.ndim != 2 or n_pts <= n_pars:
        return np.full(n_pars, np.nan)

    try:
        _, s, vt = np.linalg.svd(jac, full_matrices=False)
        if len(s) == 0:
            return np.full(n_pars, np.nan)
        tol = np.finfo(float).eps * max(jac.shape) * s[0]
        keep = s > tol
        if not np.any(keep):
            return np.full(n_pars, np.nan)
        jtj_inv = (vt[keep].T / (s[keep] ** 2)) @ vt[keep]
        dof = max(n_pts - n_pars, 1)
        sigma2 = 2.0 * float(result.cost) / dof
        cov = jtj_inv * sigma2
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except Exception:
        return np.full(n_pars, np.nan)


NUM_RE = re.compile(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?')


@dataclass
class SweepData:
    sample_name: str
    incidence_angle_deg: float
    compensator_angle_deg: np.ndarray
    intensity_raw: np.ndarray
    intensity_norm: np.ndarray
    source_file: str
    normalisation_peak: float
    metadata: dict = field(default_factory=dict)


@dataclass
class FilmStack:
    ambient: complex
    film: complex
    substrate: complex
    thickness_nm: float
    wavelength_nm: float


@dataclass
class InstrumentParameters:
    polariser_deg: float = -45.0
    analyser_deg: float = 45.0
    compensator_retardance_deg: float = 90.0
    compensator_zero_deg: float = 0.0
    wobble_amp_deg: float = 0.0
    wobble_phase_deg: float = 0.0


@dataclass
class PsiDeltaFit:
    sample_name: str
    incidence_angle_deg: float
    psi_deg: float
    delta_deg: float
    y_scale: float
    y_offset: float
    rmse: float
    r2: float
    success: bool
    n_points: int
    psi_std_deg: float = float('nan')
    delta_std_deg: float = float('nan')


@dataclass
class FilmFitResult:
    sample_name: str
    thickness_nm: float
    n_real: float
    k_imag: float
    rmse_psi_deg: float
    rmse_delta_deg: float
    success: bool
    thickness_std_nm: float = float('nan')
    n_real_std: float = float('nan')
    k_imag_std: float = float('nan')


@dataclass
class ProjectConfig:
    base_dir: str = '.'
    output_dir: str = 'ellipsometry_pipeline_output'
    file_glob: str = '*.txt'
    theta_min_deg: float | None = None
    theta_max_deg: float | None = None
    drop_zero_voltage: bool = True
    zero_tol: float = 0.0
    peak_window_deg: tuple[float, float] | None = None


def save_json(data: dict, path: str | Path):
    def clean(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        return obj

    with Path(path).open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=clean)

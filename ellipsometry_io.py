from __future__ import annotations

from pathlib import Path

import numpy as np

from ellipsometry_common import NUM_RE, ProjectConfig, SweepData


def load_two_column_sweep(path, theta_min_deg=None, theta_max_deg=None, drop_zero=True, zero_tol=0.0):
    path = Path(path)
    angles, volts = [], []

    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith('#'):
                continue
            nums = NUM_RE.findall(text)
            if len(nums) < 2:
                continue
            ang, sig = map(float, nums[:2])
            if np.isfinite(ang) and np.isfinite(sig):
                angles.append(ang)
                volts.append(sig)

    if len(angles) < 10:
        raise ValueError(f'Too few valid rows found in {path}')

    angles = np.asarray(angles, float)
    volts = np.asarray(volts, float)

    if theta_min_deg is not None:
        keep = angles >= float(theta_min_deg)
        angles, volts = angles[keep], volts[keep]
    if theta_max_deg is not None:
        keep = angles <= float(theta_max_deg)
        angles, volts = angles[keep], volts[keep]
    if drop_zero:
        keep = np.abs(volts) > float(zero_tol)  # handy for dead points at the start/end
        angles, volts = angles[keep], volts[keep]

    order = np.argsort(angles)
    return angles[order], volts[order]


def peak_normalise(angle_deg, signal, lo_deg=None, hi_deg=None):
    angle_deg = np.asarray(angle_deg, float)
    signal = np.asarray(signal, float)

    if lo_deg is None or hi_deg is None:
        peak = float(np.nanmax(signal))
    else:
        keep = (angle_deg >= lo_deg) & (angle_deg <= hi_deg)
        peak = float(np.nanmax(signal[keep])) if np.any(keep) else float(np.nanmax(signal))

    if not np.isfinite(peak) or abs(peak) < 1e-15:
        peak = 1.0
    return signal / peak, peak


def first_number(text: str):
    hit = NUM_RE.search(text)
    return float(hit.group(0)) if hit else None


def parse_incidence_angle_from_filename(path):
    return first_number(Path(path).stem)


def build_sweep(path, sample_name, incidence_angle_deg, config: ProjectConfig):
    angle_deg, raw_signal = load_two_column_sweep(
        path,
        theta_min_deg=config.theta_min_deg,
        theta_max_deg=config.theta_max_deg,
        drop_zero=config.drop_zero_voltage,
        zero_tol=config.zero_tol,
    )

    if config.peak_window_deg is None:
        norm_signal, peak = peak_normalise(angle_deg, raw_signal)
    else:
        lo, hi = config.peak_window_deg
        norm_signal, peak = peak_normalise(angle_deg, raw_signal, lo_deg=lo, hi_deg=hi)

    return SweepData(
        sample_name=str(sample_name),
        incidence_angle_deg=float(incidence_angle_deg),
        compensator_angle_deg=np.asarray(angle_deg, float),
        intensity_raw=np.asarray(raw_signal, float),
        intensity_norm=np.asarray(norm_signal, float),
        source_file=str(Path(path)),
        normalisation_peak=float(peak),
    )


def load_sweeps_from_folder(config: ProjectConfig, sample_name_from_file=None, angle_from_file=True):
    base = Path(config.base_dir)
    sweeps = []

    for path in sorted(base.glob(config.file_glob)):
        sample_name = sample_name_from_file(path) if sample_name_from_file else path.stem
        angle_deg = parse_incidence_angle_from_filename(path) if angle_from_file else None
        if angle_deg is None:
            continue
        sweeps.append(build_sweep(path, sample_name, angle_deg, config))

    return sweeps

from __future__ import annotations

import numpy as np

from ellipsometry_common import InstrumentParameters, deg2rad


def rot(angle_rad: float):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=complex)


def linear_polariser(angle_rad: float):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c * c, c * s], [c * s, s * s]], dtype=complex)


def linear_retarder(angle_rad: float, retardance_rad: float):
    r = rot(angle_rad)
    rinv = rot(-angle_rad)
    phase = np.array([
        [np.exp(-0.5j * retardance_rad), 0.0],
        [0.0, np.exp(0.5j * retardance_rad)],
    ], dtype=complex)
    return r @ phase @ rinv


def sample_matrix_from_rho(rho: complex):
    return np.array([[rho, 0.0], [0.0, 1.0]], dtype=complex)


def instrument_intensity_from_rho(theta_deg, rho, instrument: InstrumentParameters, y_scale=1.0, y_offset=0.0):
    theta_deg = np.asarray(theta_deg, float)
    pol = deg2rad(instrument.polariser_deg)
    ana = deg2rad(instrument.analyser_deg)
    retardance = deg2rad(instrument.compensator_retardance_deg)
    zero = deg2rad(instrument.compensator_zero_deg)
    wobble = deg2rad(instrument.wobble_amp_deg)
    wobble_phase = deg2rad(instrument.wobble_phase_deg)

    p_mat = linear_polariser(pol)
    a_mat = linear_polariser(ana)
    s_mat = sample_matrix_from_rho(rho)
    e_in = np.array([1.0, 0.0], dtype=complex)

    out = np.empty_like(theta_deg, dtype=float)
    for i, ang_deg in enumerate(theta_deg):
        ang = deg2rad(ang_deg) + zero
        ang += wobble * np.cos(ang + wobble_phase)  # keeps a simple wobble correction in one place
        c_mat = linear_retarder(ang, retardance)
        e_out = a_mat @ s_mat @ c_mat @ p_mat @ e_in
        out[i] = np.real(np.vdot(e_out, e_out))

    return y_scale * out + y_offset


def instrument_intensity_from_psidelta(theta_deg, psi_deg, delta_deg, instrument: InstrumentParameters, y_scale=1.0, y_offset=0.0):
    rho = np.tan(deg2rad(psi_deg)) * np.exp(1j * deg2rad(delta_deg))
    return instrument_intensity_from_rho(theta_deg, rho, instrument, y_scale=y_scale, y_offset=y_offset)

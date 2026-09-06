from __future__ import annotations

import numpy as np

from ellipsometry_common import FilmStack, deg2rad, rad2deg, tidy_psi_delta


def snell(n1: complex, n2: complex, theta1_rad: float):
    """Complex Snell angle; retain evanescent solutions above the critical angle."""
    return np.arcsin(complex(n1 / n2) * np.sin(theta1_rad))


def fresnel_rs(n1: complex, n2: complex, theta1_rad: float, theta2_rad: float):
    return (n1 * np.cos(theta1_rad) - n2 * np.cos(theta2_rad)) / (n1 * np.cos(theta1_rad) + n2 * np.cos(theta2_rad))


def fresnel_rp(n1: complex, n2: complex, theta1_rad: float, theta2_rad: float):
    return (n2 * np.cos(theta1_rad) - n1 * np.cos(theta2_rad)) / (n2 * np.cos(theta1_rad) + n1 * np.cos(theta2_rad))


def rho_from_stack(stack: FilmStack, incidence_angle_deg: float):
    """Return rp/rs for a passive, isotropic, single-layer stack.

    Convention: n + i*k (k >= 0), fields exp(i*kz*z - i*omega*t).
    The forward kz branch has nonnegative imaginary part, so absorption
    attenuates the round-trip factor exp(2j*beta).
    """
    if not np.isfinite(stack.thickness_nm) or stack.thickness_nm < 0:
        raise ValueError('Film thickness must be finite and nonnegative.')
    if not np.isfinite(stack.wavelength_nm) or stack.wavelength_nm <= 0:
        raise ValueError('Wavelength must be finite and positive.')
    if not np.isfinite(incidence_angle_deg) or not 0 <= incidence_angle_deg < 90:
        raise ValueError('Incidence angle must lie in [0, 90) degrees.')
    indices = np.asarray([stack.ambient, stack.film, stack.substrate], complex)
    if not np.all(np.isfinite(indices)) or np.any(indices.real <= 0) or np.any(indices.imag < 0):
        raise ValueError('Use passive refractive indices n + i*k, with n > 0 and k >= 0.')
    if indices[0].imag != 0:
        raise ValueError('This implementation requires a nonabsorbing incident medium.')
    theta0 = deg2rad(incidence_angle_deg)
    n0, n1, n2 = indices
    q = np.sqrt(indices**2 - (n0 * np.sin(theta0))**2 + 0j)
    reverse = (q.imag < 0) | ((q.imag == 0) & (q.real < 0))
    q[reverse] *= -1
    q0, q1, q2 = q
    rs01 = (q0 - q1) / (q0 + q1)
    rp01 = (n1**2*q0 - n0**2*q1) / (n1**2*q0 + n0**2*q1)
    rs12 = (q1 - q2) / (q1 + q2)
    rp12 = (n2**2*q1 - n1**2*q2) / (n2**2*q1 + n1**2*q2)
    beta = 2 * np.pi * q1 * stack.thickness_nm / stack.wavelength_nm
    phase = np.exp(2j * beta)

    rs = (rs01 + rs12 * phase) / (1 + rs01 * rs12 * phase)
    rp = (rp01 + rp12 * phase) / (1 + rp01 * rp12 * phase)
    return rp / rs


def psi_delta_from_rho(rho: complex):
    psi_deg = rad2deg(np.arctan(np.abs(rho)))
    delta_deg = rad2deg(np.angle(rho))
    return tidy_psi_delta(psi_deg, delta_deg)


def psi_delta_from_stack(stack: FilmStack, incidence_angle_deg: float):
    return psi_delta_from_rho(rho_from_stack(stack, incidence_angle_deg))

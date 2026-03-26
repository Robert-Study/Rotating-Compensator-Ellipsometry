from __future__ import annotations

import numpy as np

from ellipsometry_common import FilmStack, deg2rad, rad2deg, tidy_psi_delta


def snell(n1: complex, n2: complex, theta1_rad: float):
    return np.arcsin((n1 / n2) * np.sin(theta1_rad))


def fresnel_rs(n1: complex, n2: complex, theta1_rad: float, theta2_rad: float):
    return (n1 * np.cos(theta1_rad) - n2 * np.cos(theta2_rad)) / (n1 * np.cos(theta1_rad) + n2 * np.cos(theta2_rad))


def fresnel_rp(n1: complex, n2: complex, theta1_rad: float, theta2_rad: float):
    return (n2 * np.cos(theta1_rad) - n1 * np.cos(theta2_rad)) / (n2 * np.cos(theta1_rad) + n1 * np.cos(theta2_rad))


def rho_from_stack(stack: FilmStack, incidence_angle_deg: float):
    theta0 = deg2rad(incidence_angle_deg)
    n0, n1, n2 = stack.ambient, stack.film, stack.substrate
    theta1 = snell(n0, n1, theta0)
    theta2 = snell(n1, n2, theta1)

    rs01 = fresnel_rs(n0, n1, theta0, theta1)
    rp01 = fresnel_rp(n0, n1, theta0, theta1)
    rs12 = fresnel_rs(n1, n2, theta1, theta2)
    rp12 = fresnel_rp(n1, n2, theta1, theta2)

    beta = 2 * np.pi * n1 * stack.thickness_nm * np.cos(theta1) / stack.wavelength_nm
    phase = np.exp(-2j * beta)

    rs = (rs01 + rs12 * phase) / (1 + rs01 * rs12 * phase)
    rp = (rp01 + rp12 * phase) / (1 + rp01 * rp12 * phase)
    return rp / rs


def psi_delta_from_rho(rho: complex):
    psi_deg = rad2deg(np.arctan(np.abs(rho)))
    delta_deg = rad2deg(np.angle(rho))
    return tidy_psi_delta(psi_deg, delta_deg)


def psi_delta_from_stack(stack: FilmStack, incidence_angle_deg: float):
    return psi_delta_from_rho(rho_from_stack(stack, incidence_angle_deg))

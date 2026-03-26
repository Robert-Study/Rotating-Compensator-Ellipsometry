from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ellipsometry_common import InstrumentParameters, fit_stds
from fresnel_sim import rho_from_stack
from pcsa_model import instrument_intensity_from_rho


def calibrate_instrument_from_reference(reference_sweeps, reference_stack_builder, initial_instrument=None, fit_retardance=False, fit_wobble=True):
    if initial_instrument is None:
        initial_instrument = InstrumentParameters()

    guess = [
        initial_instrument.polariser_deg,
        initial_instrument.analyser_deg,
        initial_instrument.compensator_zero_deg,
    ]
    low = [-90.0, -90.0, -30.0]
    high = [90.0, 90.0, 30.0]

    if fit_retardance:
        guess.append(initial_instrument.compensator_retardance_deg)
        low.append(60.0)
        high.append(120.0)
    if fit_wobble:
        guess += [initial_instrument.wobble_amp_deg, initial_instrument.wobble_phase_deg]
        low += [0.0, -180.0]
        high += [5.0, 180.0]

    guess = np.asarray(guess, float)
    low = np.asarray(low, float)
    high = np.asarray(high, float)

    def unpack(vals):
        j = 0
        pol = float(vals[j]); j += 1
        ana = float(vals[j]); j += 1
        zero = float(vals[j]); j += 1
        if fit_retardance:
            ret = float(vals[j]); j += 1
        else:
            ret = initial_instrument.compensator_retardance_deg
        if fit_wobble:
            wobble = float(vals[j]); j += 1
            wobble_phase = float(vals[j]); j += 1
        else:
            wobble = initial_instrument.wobble_amp_deg
            wobble_phase = initial_instrument.wobble_phase_deg
        return InstrumentParameters(
            polariser_deg=pol,
            analyser_deg=ana,
            compensator_retardance_deg=ret,
            compensator_zero_deg=zero,
            wobble_amp_deg=wobble,
            wobble_phase_deg=wobble_phase,
        )

    def residuals(vals):
        inst = unpack(vals)
        pieces = []
        for sweep in reference_sweeps:
            stack = reference_stack_builder(float(sweep.incidence_angle_deg))
            rho = rho_from_stack(stack, sweep.incidence_angle_deg)
            y_model = instrument_intensity_from_rho(sweep.compensator_angle_deg, rho, inst, y_scale=1.0, y_offset=0.0)
            design = np.column_stack([y_model, np.ones_like(y_model)])
            scale, offset = np.linalg.lstsq(design, sweep.intensity_norm, rcond=None)[0]  # lets each sweep float a bit
            pieces.append(scale * y_model + offset - sweep.intensity_norm)
        return np.concatenate(pieces)

    result = least_squares(residuals, guess, bounds=(low, high), method='trf')
    best_inst = unpack(result.x)
    errs = fit_stds(result, len(residuals(result.x)), len(result.x))

    rows = []
    for sweep in reference_sweeps:
        stack = reference_stack_builder(float(sweep.incidence_angle_deg))
        rho = rho_from_stack(stack, sweep.incidence_angle_deg)
        y_model = instrument_intensity_from_rho(sweep.compensator_angle_deg, rho, best_inst, y_scale=1.0, y_offset=0.0)
        design = np.column_stack([y_model, np.ones_like(y_model)])
        scale, offset = np.linalg.lstsq(design, sweep.intensity_norm, rcond=None)[0]
        y_fit = scale * y_model + offset
        rows.append({
            'sample_name': sweep.sample_name,
            'incidence_angle_deg': sweep.incidence_angle_deg,
            'scale': float(scale),
            'offset': float(offset),
            'rmse': float(np.sqrt(np.mean((y_fit - sweep.intensity_norm) ** 2))),
        })

    summary = pd.DataFrame(rows).sort_values(['sample_name', 'incidence_angle_deg']).reset_index(drop=True)
    summary.attrs['parameter_stds'] = {
        'polariser_deg': float(errs[0]) if len(errs) > 0 else np.nan,
        'analyser_deg': float(errs[1]) if len(errs) > 1 else np.nan,
        'compensator_zero_deg': float(errs[2]) if len(errs) > 2 else np.nan,
    }
    return best_inst, summary

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from ellipsometry_common import FilmStack, InstrumentParameters, fit_stds
from ellipsometry_io import load_two_column_sweep, parse_incidence_angle_from_filename
from film_property_fit import fit_film_properties_from_psidelta
from fresnel_sim import rho_from_stack, psi_delta_from_stack, snell, fresnel_rp, fresnel_rs
from psi_delta_extraction import fit_psi_delta_for_sweep
from demo import make_sweeps


class OpticalPhysicsTests(unittest.TestCase):
    def test_zero_thickness_reduces_to_bare_interface(self):
        angle = np.deg2rad(60)
        transmitted = np.arcsin(np.sin(angle) / 1.52)
        expected = fresnel_rp(1, 1.52, angle, transmitted) / fresnel_rs(1, 1.52, angle, transmitted)
        for film in [1.46, 0.2+3.5j]:
            self.assertAlmostEqual(rho_from_stack(FilmStack(1, film, 1.52, 0, 632.8), 60), expected)

    def test_thick_absorbing_film_approaches_bulk_surface(self):
        angle = np.deg2rad(60)
        film = 0.2+3.5j
        transmitted = snell(1, film, angle)
        expected = fresnel_rp(1, film, angle, transmitted) / fresnel_rs(1, film, angle, transmitted)
        actual = rho_from_stack(FilmStack(1, film, 1.52, 10000, 632.8), 60)
        self.assertTrue(np.isfinite(actual))
        self.assertAlmostEqual(actual, expected)

    def test_brewster_angle_has_zero_p_reflection(self):
        angle = np.rad2deg(np.arctan(1.5))
        self.assertLess(abs(rho_from_stack(FilmStack(1, 1.3, 1.5, 0, 633), angle)), 1e-12)

    def test_total_internal_reflection_preserves_amplitudes(self):
        angle = np.deg2rad(60)
        transmitted = snell(1.5, 1, angle)
        for function in [fresnel_rp, fresnel_rs]:
            self.assertAlmostEqual(abs(function(1.5, 1, angle, transmitted)), 1)

    def test_single_angle_cannot_fit_three_unknowns(self):
        with self.assertRaisesRegex(ValueError, 'two distinct'):
            fit_film_properties_from_psidelta('film', [60], [30], [-120], 633, 1, 1.5, 80, 1.7, .1)

    def test_multiangle_recovers_known_synthetic_film(self):
        angles = np.linspace(45, 75, 9)
        stack = FilmStack(1, 1.7+.1j, 1.52, 83, 632.8)
        psi, delta = np.array([psi_delta_from_stack(stack, a) for a in angles]).T
        result = fit_film_properties_from_psidelta('synthetic', angles, psi, delta, 632.8, 1, 1.52, 70, 1.6, .05)
        self.assertTrue(result.success)
        np.testing.assert_allclose([result.thickness_nm, result.n_real, result.k_imag], [83, 1.7, .1], rtol=1e-5)

    def test_rank_deficiency_does_not_produce_false_precision(self):
        fit = SimpleNamespace(jac=np.ones((12, 3)), cost=1.0)
        self.assertTrue(np.all(np.isnan(fit_stds(fit, 12, 3))))

    def test_dark_corrected_sweep_recovers_psi_delta(self):
        inst = InstrumentParameters()
        stack = FilmStack(1, 1.7+.1j, 1.52, 83, 632.8)
        sweep = make_sweeps(stack, inst, [60], 'synthetic', np.random.default_rng(2))[0]
        expected = psi_delta_from_stack(stack, 60)
        fit = fit_psi_delta_for_sweep(sweep, inst, psi_guess_deg=25, delta_guess_deg=-140)
        self.assertTrue(fit.success)
        np.testing.assert_allclose([fit.psi_deg, fit.delta_deg], expected, atol=1e-5)
        with self.assertRaisesRegex(ValueError, 'dark signal'):
            fit_psi_delta_for_sweep(sweep, inst, allow_scale_and_offset=True)

    def test_angle_parser_does_not_confuse_sputter_time(self):
        self.assertEqual(parse_incidence_angle_from_filename('gold_250s_70deg.txt'), 70)
        self.assertIsNone(parse_incidence_angle_from_filename('gold_250s.txt'))

    def test_data_reader_ignores_numeric_metadata_and_keeps_zeros(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'sweep.txt'
            p.write_text('Sample 12 at 70 degrees\n' + '\n'.join(f'{i}, {i}' for i in range(12)))
            angles, volts = load_two_column_sweep(p)
            self.assertEqual(len(angles), 12)
            self.assertEqual(volts[0], 0)
            with self.assertRaisesRegex(ValueError, 'filtering'):
                load_two_column_sweep(p, theta_min_deg=10)


if __name__ == '__main__':
    unittest.main()

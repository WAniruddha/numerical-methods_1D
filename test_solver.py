import unittest
import numpy as np
from solver import step, solve, exact_solution, grid_study, sharp_profile, time_step_study


class SolverTests(unittest.TestCase):
    def test_constant_field_stays_constant(self):
        field = np.ones(32) * 2.5
        result = step(field, 1 / 32, 0.0001, 1.0, 0.02)
        np.testing.assert_allclose(result, field, atol=1e-12)

    def test_mass_conserved_periodic(self):
        field = np.arange(32, dtype=float) / 32
        result = step(field, 1 / 32, 0.0001, 1.0, 0.02)
        self.assertAlmostEqual(field.sum(), result.sum(), places=12)

    def test_initial_condition(self):
        x, phi = solve(cells=32, final_time=0)
        np.testing.assert_allclose(phi, np.sin(2 * np.pi * x))

    def test_refinement_reduces_error(self):
        for scheme in ('upwind', 'central'):
            errs = grid_study(scheme=scheme, sizes=(25, 50, 100), time=0.1)
            self.assertLess(errs[1], errs[0])
            self.assertLess(errs[2], errs[1])

    def test_central_has_smaller_error_on_smooth_problem(self):
        x, upwind = solve(cells=100, scheme='upwind')
        _, central = solve(cells=100, scheme='central')
        truth = exact_solution(x, 0.1)
        self.assertLess(np.linalg.norm(central - truth), np.linalg.norm(upwind - truth))

    def test_central_without_diffusion_rejected(self):
        with self.assertRaises(ValueError):
            solve(scheme='central', diffusion=0)

    def test_sharp_profile_mass_and_bounds(self):
        # Pure upwind transport should not create new max/min values.
        x, phi = solve(cells=120, diffusion=0, scheme='upwind',
                       initial=sharp_profile, final_time=0.08)
        self.assertAlmostEqual(phi.mean(), sharp_profile(x).mean(), places=12)
        self.assertGreaterEqual(phi.min(), -1e-12)
        self.assertLessEqual(phi.max(), 1 + 1e-12)

    def test_both_schemes_conserve_mass(self):
        for scheme in ('upwind', 'central'):
            x, phi = solve(cells=75, scheme=scheme, initial=sharp_profile)
            self.assertAlmostEqual(phi.mean(), sharp_profile(x).mean(), places=11)

    def test_timestep_study_is_finite(self):
        for scheme in ('upwind', 'central'):
            vals = time_step_study(cells=80, scheme=scheme)
            self.assertEqual(len(vals), 3)
            self.assertTrue(all(np.isfinite(e) for _, e in vals))

    def test_reject_bad_cfl_and_initial_shape(self):
        with self.assertRaises(ValueError):
            solve(cfl=0)
        with self.assertRaises(ValueError):
            solve(initial=lambda x: x[:-1])

    def test_reject_invalid_input(self):
        with self.assertRaises(ValueError):
            solve(cells=2)
        with self.assertRaises(ValueError):
            step(np.ones(8), 0, 0.01, 1, 0.01)


if __name__ == '__main__':
    unittest.main()

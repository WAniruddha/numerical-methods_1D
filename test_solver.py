import unittest
import numpy as np
from solver import step, solve


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

    def test_reject_invalid_input(self):
        with self.assertRaises(ValueError):
            solve(cells=2)
        with self.assertRaises(ValueError):
            step(np.ones(8), 0, 0.01, 1, 0.01)


if __name__ == '__main__':
    unittest.main()

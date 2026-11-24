"""Незалежні аналітичні перевірки похідних двох тестових функцій."""
import unittest
import numpy as np
from optimizer.problems import problem_suite


class AnalyticReferenceTests(unittest.TestCase):
    def test_extended_rosenbrock_against_explicit_formulas(self):
        for p in problem_suite():
            if p.name not in ('rosenbrock', 'extended_rosenbrock'):
                continue
            x = np.linspace(-1.3, 1.2, p.dimension)
            g, h, f = np.zeros(p.dimension), np.zeros((p.dimension, p.dimension)), 0.
            for i in range(0, p.dimension, 2):
                a, b = x[i], x[i+1]
                f += 100 * (b-a*a)**2 + (1-a)**2
                g[i], g[i+1] = -400*a*(b-a*a) + 2*(a-1), 200*(b-a*a)
                h[i, i] = 1200*a*a - 400*b + 2
                h[i, i+1] = h[i+1, i] = -400*a
                h[i+1, i+1] = 200
            self.assertAlmostEqual(p.value(x), f, places=10)
            np.testing.assert_allclose(p.derivatives(x), g, rtol=1e-13, atol=1e-11)
            np.testing.assert_allclose(p.derivatives(x, 2), h, rtol=1e-13, atol=1e-11)

    def test_himmelblau_against_explicit_formulas(self):
        p = next(p for p in problem_suite() if p.name == 'himmelblau')
        for x, y in ((.7, -1.1), (-3., 3.), (3., 2.)):
            a, b = x*x + y - 11, x + y*y - 7
            expected_g = [4*x*a + 2*b, 2*a + 4*y*b]
            expected_h = [[4*a + 8*x*x + 2, 4*x + 4*y],
                          [4*x + 4*y, 2 + 4*b + 8*y*y]]
            self.assertAlmostEqual(p.value([x, y]), a*a+b*b, places=12)
            np.testing.assert_allclose(p.derivatives([x, y]), expected_g, atol=1e-12)
            np.testing.assert_allclose(p.derivatives([x, y], 2), expected_h, atol=1e-12)


if __name__ == '__main__':
    unittest.main()

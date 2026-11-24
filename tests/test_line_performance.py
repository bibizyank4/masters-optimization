"""Швидка побудова полінома має зберігати результат початкової реалізації."""
import unittest
from unittest.mock import patch
import numpy as np
from numpy.polynomial import Polynomial
from optimizer.methods import Config, minimize
from optimizer.problems import Problem, problem_suite


def original_line(problem, point, direction):
    x, d = problem._check(point), problem._check(direction)
    return problem.expression([Polynomial([a, b]) for a, b in zip(x, d)])


class LinePolynomialRegressionTests(unittest.TestCase):
    def test_coefficients_match_original_bitwise(self):
        rng = np.random.default_rng(42026)
        for p in problem_suite():
            if p.name not in ('rosenbrock', 'extended_rosenbrock', 'course_rosenbrock'):
                continue
            cases = [(rng.normal(size=p.dimension), rng.normal(size=p.dimension)) for _ in range(100)]
            cases += [(np.zeros(p.dimension), np.zeros(p.dimension)),
                      (np.ones(p.dimension), np.zeros(p.dimension)),
                      (np.zeros(p.dimension), np.ones(p.dimension)),
                      (np.full(p.dimension, -0.), np.full(p.dimension, -0.))]
            for x, d in cases:
                expected = original_line(p, x, d).coef
                actual = p.line_polynomial(x, d).coef
                self.assertEqual(actual.tobytes(), expected.tobytes(), (p.name, p.dimension, x, d))

    def test_sensitive_trajectories_match_original(self):
        p = next(x for x in problem_suite() if x.name == 'extended_rosenbrock' and x.dimension == 20)
        starts = (p.starts[0], [-1., 1., 0., 0.] * 5, [0.] * 20)
        for start in starts:
            for method in ('three_step', 'qn_three_step', 'bfgs_three_step'):
                cfg = Config()
                with patch.object(Problem, 'line_polynomial', original_line):
                    original = minimize(p, method, start, cfg)
                improved = minimize(p, method, start, cfg)
                self.assertEqual((original.status, original.iterations, original.counts),
                                 (improved.status, improved.iterations, improved.counts))
                self.assertEqual(len(original.history), len(improved.history))
                for a, b in zip(original.history, improved.history):
                    self.assertEqual(a['x'].tobytes(), b['x'].tobytes())
                    self.assertEqual((a['f'], a['grad_norm'], a['alpha']), (b['f'], b['grad_norm'], b['alpha']))


if __name__ == '__main__':
    unittest.main()

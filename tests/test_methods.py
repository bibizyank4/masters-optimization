import contextlib
import csv
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from optimizer.problems import problem_suite, Problem
from optimizer.methods import (METHODS, Config, Evaluator, NumericalFailure, armijo,
                               broyden_update, bfgs_update, divided_difference, minimize, polynomial_line_minimum,
                               monotone_search)
from optimizer.experiments import run_suite, assess
from optimizer.results import Dataset


class MathematicsTests(unittest.TestCase):
    def test_periodic_broyden_restart_is_optional_and_counted(self):
        p = problem_suite()[4]
        cfg = Config(grad_tol=1e-15, max_iterations=3, b0='hessian', broyden_restart_interval=1)
        r = minimize(p, 'qn_gradient_correction', [-3., 3.], cfg)
        self.assertEqual(r.iterations, 3)
        self.assertEqual(r.counts['broyden_restarts'], 2)
        self.assertEqual(r.counts['hess'], 3)
        self.assertEqual(sum(bool(x.get('broyden_restart_next')) for x in r.history), 2)
        plain = minimize(p, 'qn_gradient_correction', [-3., 3.], Config(grad_tol=1e-15, max_iterations=3))
        self.assertEqual(plain.counts['broyden_restarts'], 0)
        self.assertEqual(plain.counts['hess'], 1)
        unaffected = minimize(p, 'bfgs_gradient_correction', [-3., 3.], cfg)
        self.assertEqual(unaffected.counts['broyden_restarts'], 0)
        with self.assertRaises(ValueError):
            Config(broyden_restart_interval=-1).validate()

    def test_bfgs_secant_symmetry_and_positive_definiteness(self):
        b = np.array([[2., .3], [.3, 1.]])
        s, y = np.array([1., 2.]), np.array([3., 4.])
        updated, action = bfgs_update(b, s, y)
        self.assertEqual(action, 'applied')
        np.testing.assert_allclose(updated @ s, y, atol=1e-12)
        np.testing.assert_allclose(updated, updated.T, atol=1e-12)
        np.linalg.cholesky(updated)

    def test_bfgs_skips_or_damps_negative_curvature(self):
        b, s, y = np.eye(2), np.array([1., 0.]), np.array([-1., 1.])
        skipped, action = bfgs_update(b, s, y, 'skip')
        self.assertEqual(action, 'skipped')
        np.testing.assert_array_equal(skipped, b)
        damped, action = bfgs_update(b, s, y, 'damp')
        self.assertEqual(action, 'damped')
        np.linalg.cholesky(damped)
        self.assertAlmostEqual(float(s @ damped @ s), .2)

    def test_bfgs_initial_matrix_check_and_identity(self):
        p = problem_suite()[2]
        result = minimize(p, 'bfgs_two_step_newton', [1., 1.], Config(b0='hessian'))
        self.assertEqual(result.status, 'bfgs_initial_not_positive_definite')
        result = minimize(problem_suite()[0], 'bfgs_three_step', [2., 2.], Config(b0='identity'))
        self.assertNotEqual(result.status, 'bfgs_initial_not_positive_definite')
        self.assertGreater(sum(result.counts['bfgs_' + key] for key in ('applied', 'skipped', 'damped')), 0)

    def test_known_minima_and_derivatives(self):
        for p in problem_suite():
            for x in p.minima:
                with self.subTest(p=p.name, n=p.dimension, x=x):
                    self.assertLess(abs(p.value(x)), 1e-24)
                    self.assertLess(np.linalg.norm(p.derivatives(x)), 1e-10)

    def test_derivatives_against_independent_central_differences(self):
        for p in problem_suite():
            x = np.asarray(p.starts[0], dtype=float) * .83 + .13
            h = 2e-5
            basis = np.eye(p.dimension)
            numeric_gradient = np.array([(p.value(x+h*d)-p.value(x-h*d))/(2*h) for d in basis])
            numeric_hessian = np.column_stack([(p.derivatives(x+h*d)-p.derivatives(x-h*d))/(2*h) for d in basis])
            with self.subTest(p=p.name, n=p.dimension):
                np.testing.assert_allclose(p.derivatives(x), numeric_gradient, rtol=1e-6, atol=2e-6)
                np.testing.assert_allclose(p.derivatives(x, 2), numeric_hessian, rtol=2e-6, atol=2e-5)
                np.testing.assert_allclose(p.derivatives(x, 2), p.derivatives(x, 2).T, atol=1e-11)

    def test_integrated_hessian_secant_identity(self):
        for p in problem_suite():
            x = np.asarray(p.starts[0], dtype=float)
            u = x + np.linspace(-.2, .3, p.dimension)
            ev = Evaluator(p)
            matrix = divided_difference(ev, x, u)
            with self.subTest(p=p.name, n=p.dimension):
                np.testing.assert_allclose(matrix @ (u-x), p.derivatives(u)-p.derivatives(x), rtol=1e-10, atol=2e-9)
                self.assertEqual(ev.hess_calls, 4)

    def test_line_polynomial_and_global_candidates(self):
        rng = np.random.default_rng(4)
        for p in problem_suite():
            x = np.asarray(p.starts[0], dtype=float)
            direction = rng.normal(size=p.dimension)
            direction /= np.linalg.norm(direction)
            poly = p.line_polynomial(x, direction)
            for t in (-2., -.1, 0., .7, 2.):
                self.assertAlmostEqual(poly(t), p.value(x+t*direction), delta=1e-8 * max(1., abs(poly(t))))
            ev = Evaluator(p)
            point, value, diag = polynomial_line_minimum(ev, x, x+direction)
            self.assertLess(abs(float(p.derivatives(point) @ direction)), 1e-5 * max(1., np.linalg.norm(p.derivatives(point))))
            self.assertLessEqual(value, min(p.value(x+t*direction) for t in np.linspace(-8, 8, 161)) + 1e-7)
            self.assertGreater(diag['line_root_count'], 0)

    def test_broyden_is_given_nonsymmetric_update(self):
        b = np.array([[2., .7], [.7, 3.]])
        s, y = np.array([1., 2.]), np.array([4., -1.])
        updated = broyden_update(b, s, y)
        np.testing.assert_allclose(updated @ s, y)
        self.assertFalse(np.allclose(updated, updated.T))
        with self.assertRaises(NumericalFailure):
            broyden_update(b, np.zeros(2), y)

    def test_each_method_first_step_against_formulas(self):
        p = problem_suite()[0]
        x = np.array([2., 2.])
        g, h = p.derivatives(x), p.derivatives(x, 2)
        for name in METHODS:
            cfg = Config(grad_tol=1e-14, max_iterations=1)
            r = minimize(p, name, x, cfg)
            self.assertEqual(r.iterations, 1, (name, r.message))
            if name.endswith('gradient_correction'):
                u = x-np.linalg.solve(h, g)
                direction = -np.linalg.solve(h, p.derivatives((x+u)/2))
                alpha = r.history[-1]['alpha']
                expected = x + alpha * direction
            elif name.endswith('two_step_newton'):
                u = x-np.linalg.solve(h, g)
                expected = u-np.linalg.solve(h, p.derivatives(u))
            else:
                alpha = r.history[-1]['alpha']
                u = x-alpha*g
                v = x-np.linalg.solve(h, g)  # Hessian квадратичної функції сталий
                d = v-u
                t = -float(p.derivatives(u) @ d) / float(d @ h @ d)
                expected = u+t*d
            np.testing.assert_allclose(r.x, expected, atol=1e-10)

    def test_scaled_corrector_reproduces_coursework_powell(self):
        p = problem_suite()[3]
        r = minimize(p, 'gradient_correction', (3., -1., 0., 1.),
                     Config(grad_tol=1e-5, max_iterations=2000))
        self.assertEqual(r.status, 'stationary')
        self.assertEqual(r.iterations, 28)
        np.testing.assert_allclose(r.x, [0.00589791, -0.00058979, 0.00094366, 0.00094367], atol=1e-7)

    def test_scaled_corrector_reproduces_coursework_himmelblau(self):
        p = problem_suite()[4]
        r = minimize(p, 'gradient_correction', (1., 1.), Config(grad_tol=1e-5, max_iterations=2000))
        self.assertEqual((r.status, r.iterations), ('stationary', 20))
        np.testing.assert_allclose(r.x, [3, 2], atol=1e-5)

    def test_qn_scaled_corrector_uses_updated_matrix(self):
        p = problem_suite()[4]
        cfg = Config(grad_tol=1e-15, max_iterations=2, b0='hessian')
        r = minimize(p, 'qn_gradient_correction', (-3., 3.), cfg)
        self.assertEqual(r.iterations, 2, r.message)
        x0 = np.array([-3., 3.]); x1 = r.history[1]['x']; x2 = r.history[2]['x']
        g0, g1 = p.derivatives(x0), p.derivatives(x1)
        b1 = broyden_update(p.derivatives(x0, 2), x1-x0, g1-g0)
        self.assertGreater(np.linalg.norm(b1 - p.derivatives(x1, 2)), 1e-4)
        u1 = x1 - np.linalg.solve(b1, g1)
        expected = x1 - r.history[2]['alpha'] * np.linalg.solve(b1, p.derivatives((x1+u1)/2))
        np.testing.assert_allclose(x2, expected, atol=1e-12)

    def test_monotone_search_does_not_reject_non_descent_before_search(self):
        class Quartic:
            def f(self, x):
                return float((x[0]**2 - 1)**2)
        x = np.array([.1])
        grad = np.array([4*x[0]*(x[0]**2-1)])
        direction = np.array([-1.])
        self.assertGreater(float(grad @ direction), 0.)
        ev = Quartic()
        alpha, candidate, value = monotone_search(ev, x, ev.f(x), direction, Config())
        self.assertEqual(alpha, 1.)
        self.assertLess(value, ev.f(x))
        with self.assertRaises(NumericalFailure):
            armijo(ev, x, ev.f(x), grad, direction, Config())

    def test_zero_iteration_and_failures_are_explicit(self):
        p = problem_suite()[0]
        for method in METHODS:
            r = minimize(p, method, [0, 0])
            self.assertEqual(r.status, 'stationary')
            self.assertEqual(r.iterations, 0)
        ev = Evaluator(p)
        with self.assertRaises(NumericalFailure) as cm:
            armijo(ev, np.array([1., 1.]), p.value([1, 1]), p.derivatives([1, 1]), p.derivatives([1, 1]), Config())
        self.assertEqual(cm.exception.code, 'not_descent')
        singular = minimize(problem_suite()[3], 'two_step_newton', [1., 0., 0., 1.])
        self.assertEqual(singular.status, 'ill_conditioned_matrix')

    def test_stationary_is_not_automatically_minimum(self):
        p = problem_suite()[4]
        # x=(0,0) нестаціонарна; тут перевіряємо позитивно визначений гессіан у відомому мінімумі.
        result = minimize(p, 'three_step', [3, 2])
        self.assertEqual(assess(p, result)['stationary_type'], 'positive_definite_hessian')
        p2 = problem_suite()[3]
        result2 = minimize(p2, 'three_step', [0, 0, 0, 0])
        self.assertEqual(assess(p2, result2)['stationary_type'], 'inconclusive_hessian')


class IntegrationTests(unittest.TestCase):
    def test_warmups_are_excluded_from_all_measured_results(self):
        observed = []
        def instrumented(problem, method, start, cfg):
            observed.append((method, tuple(start)))
            result = minimize(problem, method, start, cfg)
            result.elapsed = float(len(observed))
            return result
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            with patch('optimizer.experiments.minimize', side_effect=instrumented):
                rows = run_suite(temp, Config(max_iterations=100), 3, 'quadratic', 'three_step', 0,
                                 verbose=False, warmups=2)
            self.assertEqual(len(observed), 5)
            self.assertTrue(all(call == observed[0] for call in observed))
            self.assertEqual(rows[0]['time_mean_seconds'], 4.0)
            self.assertEqual(rows[0]['time_median_seconds'], 4.0)
            self.assertEqual(rows[0]['time_stdev_seconds'], 1.0)
            with (Path(temp) / 'timings.csv').open(encoding='utf-8-sig', newline='') as f:
                timings = list(csv.DictReader(f))
            self.assertEqual([float(row['elapsed_seconds']) for row in timings], [3.0, 4.0, 5.0])
            metadata = json.loads((Path(temp) / 'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['warmups'], 2)
            self.assertTrue(metadata['warmup_timing_excluded'])

    def test_zero_warmups_and_cancellation_during_warmup(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            with patch('optimizer.experiments.minimize', wraps=minimize) as calls:
                run_suite(temp, repeats=1, selection='quadratic', method='three_step', start_index=0,
                          verbose=False, warmups=0)
                self.assertEqual(calls.call_count, 1)
            checks = iter([False, True])
            with patch('optimizer.experiments.minimize', wraps=minimize) as calls:
                rows = run_suite(temp, repeats=1, selection='quadratic', method='three_step', start_index=0,
                                 verbose=False, warmups=2, cancel_check=lambda: next(checks))
                self.assertEqual(calls.call_count, 1)
            self.assertEqual(rows, [])
            metadata = json.loads((Path(temp) / 'metadata.json').read_text(encoding='utf-8'))
            self.assertTrue(metadata['cancelled'])
            self.assertEqual(metadata['completed_cases'], 0)

    def test_invalid_warmups_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            for value in (-1, 1.5, True):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    run_suite(temp, warmups=value)

    def test_csv_outputs_and_repeated_runs(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp, contextlib.redirect_stdout(io.StringIO()):
            rows = run_suite(temp, Config(max_iterations=100), 2, 'quadratic', 'three_step', 0)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]['repeat_consistent'])
            self.assertEqual(rows[0]['status'], 'stationary')
            for filename in ('summary.csv', 'history.csv', 'timings.csv', 'line_diagnostics.csv', 'metadata.json'):
                self.assertTrue((Path(temp)/filename).is_file())

    def test_invalid_selection(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp, self.assertRaises(ValueError):
            run_suite(temp, selection='none')

    def test_cancel_saves_only_completed_cases_and_identifies_version(self):
        cancelled = [False]
        def progress(done, total, row):
            cancelled[0] = True
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp, contextlib.redirect_stdout(io.StringIO()):
            rows = run_suite(temp, Config(max_iterations=100), 1, 'quadratic', progress=progress,
                             cancel_check=lambda: cancelled[0], verbose=False)
            self.assertEqual(len(rows), 1)
            dataset = Dataset(temp)
            self.assertFalse(dataset.legacy)
            self.assertTrue(dataset.metadata['cancelled'])
            self.assertEqual(dataset.metadata['completed_cases'], 1)
            self.assertEqual(dataset.metadata['planned_cases'], 18)
            self.assertEqual(len(dataset.histories[rows[0]['case_id']]), rows[0]['iterations'] + 1)

    def test_old_results_labeled_as_legacy(self):
        import json
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp, contextlib.redirect_stdout(io.StringIO()):
            run_suite(temp, Config(max_iterations=100), 1, 'quadratic', 'gradient_correction', 0, verbose=False)
            path = Path(temp)/'metadata.json'
            metadata = json.loads(path.read_text(encoding='utf-8'))
            metadata.pop('corrector')
            path.write_text(json.dumps(metadata), encoding='utf-8')
            dataset = Dataset(temp)
            self.assertTrue(dataset.legacy)
            self.assertIn('v1', dataset.method_label(dataset.rows[0]))


if __name__ == '__main__':
    unittest.main()

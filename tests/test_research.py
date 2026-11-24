import csv
import json
import tempfile
import unittest
from pathlib import Path

from optimizer.experiments import run_suite
from optimizer.methods import Config
from optimizer.research import sensitivity, matrix_comparison, compare_environments
from optimizer.starts import parse_extra_starts, validate_extra_starts
from optimizer.problems import problem_suite


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class ResearchTests(unittest.TestCase):
    def test_parse_and_validate_extra_starts(self):
        self.assertEqual(parse_extra_starts('(-1,1)*2; (0,0)*2'), [[-1., 1., -1., 1.], [0., 0., 0., 0.]])
        self.assertEqual(parse_extra_starts('[1, 2]'), [[1., 2.]])
        with self.assertRaises(ValueError):
            parse_extra_starts('(1,2)*1001')
        with self.assertRaises(ValueError):
            validate_extra_starts([problem_suite()[0]], [[1.]])
        with self.assertRaises(ValueError):
            validate_extra_starts([problem_suite()[0]], [[2., 2.]])
        with self.assertRaises(ValueError):
            validate_extra_starts(problem_suite(), [[1., 2.]])

    def test_multiple_seeds_and_starts_are_paired_and_reproducible(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            args = dict(config=Config(max_iterations=30), selection='quadratic', method='three_step_family',
                        start_index=0, scales=(1e-8,), samples=2, seeds=[17, 42], extra_starts=[[3., 1.]])
            for name in ('a', 'b'):
                meta = sensitivity(root / name, **args)
                self.assertEqual(meta['completed_cases'], 2 * 3 * (1 + 2 * 2))
                self.assertEqual(meta['seeds'], (17, 42))
            rows = read(root / 'a' / 'trials.csv')
            self.assertEqual(rows, read(root / 'b' / 'trials.csv'))
            starts = {}
            for row in rows:
                starts.setdefault((row['start_index'], row['scale'], row['seed'], row['sample']), set()).add(row['start'])
            self.assertTrue(all(len(values) == 1 for values in starts.values()))
            self.assertEqual({r['start_index'] for r in rows}, {'0', '2'})
            self.assertEqual({r['seed'] for r in rows}, {'', '17', '42'})
            aggregate = read(root / 'a' / 'aggregate.csv')
            self.assertEqual(len(aggregate), 2 * 3 * 3)

    def test_extra_starts_work_in_regular_and_matrix_series(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            rows = run_suite(root / 'regular', Config(), 1, 'quadratic', 'three_step_family', 0,
                             verbose=False, warmups=0, extra_starts=[[3., 1.]])
            self.assertEqual(len(rows), 6)
            self.assertEqual({r['start_index'] for r in rows}, {0, 2})
            self.assertEqual(len({r['case_id'] for r in rows}), 6)
            meta = matrix_comparison(root / 'matrix', Config(), 1, 0, 'quadratic', 'three_step_family', 0,
                                     extra_starts=[[3., 1.]])
            self.assertEqual(meta['completed_cases'], 24)
            for sub in ('identity_skip', 'identity_damp', 'hessian_skip', 'hessian_damp'):
                data = read(root / 'matrix' / sub / 'summary.csv')
                self.assertEqual({r['start_index'] for r in data}, {'0', '2'})
            with self.assertRaises(ValueError):
                matrix_comparison(root / 'bad', Config(), 1, 0, 'all', 'all', extra_starts=[[3., 1.]])

    def test_sensitivity_uses_reproducible_paired_starts(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            cfg = Config(max_iterations=50)
            for name in ('a', 'b'):
                meta = sensitivity(root / name, cfg, 'quadratic', 'all', 0, (1e-12, 1e-8), 2, 17)
                self.assertEqual(meta['completed_cases'], 9 * 5)
            a, b = read(root / 'a' / 'trials.csv'), read(root / 'b' / 'trials.csv')
            self.assertEqual(a, b)
            groups = {}
            for row in a:
                groups.setdefault((row['scale'], row['sample']), set()).add(row['start'])
            self.assertTrue(all(len(starts) == 1 for starts in groups.values()))
            self.assertEqual(len(read(root / 'a' / 'aggregate.csv')), 27)
            self.assertTrue((root / 'a' / 'REPORT.md').is_file())

    def test_sensitivity_cancel_and_full_trajectory_output(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            stop = [False]
            def progress(done, total, row):
                stop[0] = True
            result = sensitivity(Path(tmp) / 'cancel', Config(), 'quadratic', 'three_step', 0,
                                 (1e-12,), 2, 42, progress, lambda: stop[0])
            self.assertTrue(result['cancelled'])
            self.assertEqual(result['completed_cases'], 1)
            self.assertEqual(len(read(Path(tmp) / 'cancel' / 'trials.csv')), 1)
            result = sensitivity(Path(tmp) / 'full', Config(), 'quadratic', 'qn_three_step', 0,
                                 (1e-8,), 2, 42)
            rows = read(Path(tmp) / 'full' / 'trials.csv')
            histories = read(Path(tmp) / 'full' / 'trajectories.csv')
            for row in rows:
                self.assertEqual(sum(x['case_id'] == row['case_id'] for x in histories), int(row['iterations']) + 1)

    def test_matrix_four_series_and_cancellation(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            meta = matrix_comparison(root / 'full', Config(), 1, 0, 'quadratic', 'bfgs_three_step', 0)
            self.assertEqual(meta['completed_cases'], 4)
            for b0 in ('identity', 'hessian'):
                for strategy in ('skip', 'damp'):
                    p = root / 'full' / f'{b0}_{strategy}'
                    sub = json.loads((p / 'metadata.json').read_text(encoding='utf-8'))
                    self.assertEqual((sub['config']['b0'], sub['config']['bfgs_strategy']), (b0, strategy))
                    self.assertEqual(len(read(p / 'summary.csv')), 1)
            stop = [False]
            def progress(done, total, row):
                stop[0] = True
            meta = matrix_comparison(root / 'cancel', Config(), 1, 0, 'quadratic', 'all', 0,
                                     progress, lambda: stop[0])
            self.assertTrue(meta['cancelled'])
            self.assertEqual(meta['completed_cases'], 1)
            self.assertIn('НЕПОВНИЙ', (root / 'cancel' / 'REPORT.md').read_text(encoding='utf-8'))

    def test_environment_comparison_and_config_guard(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            for name in ('a', 'b'):
                run_suite(root / name, Config(), 1, 'quadratic', 'three_step', 0, verbose=False, warmups=0)
            meta = compare_environments(root / 'compare', root / 'a', root / 'b')
            self.assertEqual(meta['matched_cases'], 1)
            row = read(root / 'compare' / 'comparison.csv')[0]
            self.assertEqual(row['status_a'], row['status_b'])
            self.assertEqual(float(row['max_trajectory_distance']), 0.)
            path = root / 'b' / 'metadata.json'
            settings = json.loads(path.read_text(encoding='utf-8'))
            settings['config']['grad_tol'] = 1e-3
            path.write_text(json.dumps(settings), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'grad_tol'):
                compare_environments(root / 'invalid', root / 'a', root / 'b')

    def test_invalid_and_existing_destination(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            (root / 'keep.txt').write_text('keep', encoding='utf-8')
            with self.assertRaises(ValueError):
                sensitivity(root, Config(), 'quadratic', 'three_step', 0)
            self.assertEqual((root / 'keep.txt').read_text(encoding='utf-8'), 'keep')
            with self.assertRaises(ValueError):
                sensitivity(root / 'invalid', Config(), 'quadratic', 'three_step', 0, (1e-8, 1e-8))


if __name__ == '__main__':
    unittest.main()

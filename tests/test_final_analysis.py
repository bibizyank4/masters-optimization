import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from optimizer.final_analysis import _batches, _sensitivity_pairs, final_analysis
from optimizer.methods import Config
from optimizer.research import sensitivity


class FinalAnalysisTests(unittest.TestCase):
    def test_sensitivity_restart_pairs_are_aligned(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            for name, interval in [('sensitivity_n10', 0), ('sensitivity_n10_restart', 1)]:
                sensitivity(root / name, Config(max_iterations=3, broyden_restart_interval=interval),
                            'extended_rosenbrock:10', 'qn_three_step', 0, (1e-8,), 1, 42)
            pairs = _sensitivity_pairs(root, (10,))
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(row['paired'] == 1 for row in pairs))
            self.assertTrue(all(row['stationary_both'] + row['only_original'] + row['only_restart'] + row['neither'] == 1 for row in pairs))

    def test_full_plan_counts_all_sensitivity_jobs(self):
        def fake_run(*args, **kwargs):
            return [{} for _ in range(51)]
        def fake_sensitivity(*args, **kwargs):
            method = args[3]
            return {'completed_cases': 84 if method == 'qn_three_step' else 252, 'cancelled': False}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp, \
                patch('optimizer.final_analysis.run_suite', side_effect=fake_run), \
                patch('optimizer.final_analysis.sensitivity', side_effect=fake_sensitivity), \
                patch('optimizer.final_analysis._build_tables', return_value=([], [])):
            meta = final_analysis(Path(tmp) / 'all', 1, 0, validate=False)
            self.assertEqual((meta['status'], meta['planned_cases'], meta['completed_cases']), ('complete', 2049, 2049))
            self.assertEqual(set(meta['sensitivity_results']), {'10', '10_restart', '20', '20_restart'})

    def test_full_plan_has_separate_original_and_restart(self):
        batches = _batches((1e-6, 1e-8, 1e-10), Config(), 50)
        self.assertEqual(len(batches), 27)
        for tol in (1e-6, 1e-8, 1e-10):
            group = [(name, m, c) for name, m, c in batches if c.grad_tol == tol]
            self.assertEqual(len(group), 9)
            self.assertEqual(sum(m == 'classical_family' for _, m, _ in group), 1)
            self.assertEqual(sum(m == 'broyden_family' and c.broyden_restart_interval == 0 for _, m, c in group), 2)
            self.assertEqual(sum(m == 'broyden_family' and c.broyden_restart_interval == 50 for _, m, c in group), 2)
            self.assertEqual(sum(m == 'bfgs_family' for _, m, _ in group), 4)

    def test_small_plan_saves_complete_paired_results(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp) / 'final'
            meta = final_analysis(root, repeats=1, warmups=0, selection='quadratic', start_index=0,
                                  tolerances=(1e-8,), include_sensitivity=False, validate=False)
            self.assertEqual((meta['status'], meta['completed_cases'], meta['planned_cases']), ('complete', 27, 27))
            self.assertEqual(len(meta['completed_batches']), 9)
            with (root / 'matched_comparison.csv').open(encoding='utf-8-sig', newline='') as f:
                pairs = list(csv.DictReader(f))
            self.assertEqual(len(pairs), 18)
            self.assertTrue(all(row['paired_cases'] == '1' for row in pairs))
            a = json.loads((root / 'series' / 'tol_1e-08_broyden_hessian_restart_0' / 'metadata.json').read_text(encoding='utf-8'))
            b = json.loads((root / 'series' / 'tol_1e-08_broyden_hessian_restart_50' / 'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(a['config']['broyden_restart_interval'], 0)
            self.assertEqual(b['config']['broyden_restart_interval'], 50)
            self.assertTrue((root / 'REPORT.md').is_file())

    def test_cancel_preserves_partial_series(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            stop = [False]
            def progress(done, total, row):
                stop[0] = True
            root = Path(tmp) / 'cancel'
            meta = final_analysis(root, 1, 0, progress, lambda: stop[0], selection='quadratic', start_index=0,
                                  tolerances=(1e-8,), include_sensitivity=False, validate=False)
            self.assertEqual(meta['status'], 'cancelled')
            self.assertEqual(meta['completed_cases'], 1)
            self.assertTrue((root / 'series' / 'tol_1e-08_classical' / 'summary.csv').is_file())

    def test_failed_validation_stops_before_experiments(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp, patch('optimizer.final_analysis._validation', return_value=1):
            root = Path(tmp) / 'bad'
            meta = final_analysis(root, 1, 0, selection='quadratic', start_index=0,
                                  tolerances=(1e-8,), include_sensitivity=False)
            self.assertEqual(meta['status'], 'validation_failed')
            self.assertEqual(meta['completed_cases'], 0)
            self.assertFalse((root / 'series').exists())


if __name__ == '__main__':
    unittest.main()

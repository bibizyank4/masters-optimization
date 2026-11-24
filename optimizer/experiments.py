"""Експериментальні серії, історії ітерацій та показники."""
import csv
import json
import platform
import statistics
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from .methods import Config, expand_methods, minimize
from .problems import problem_suite
from .starts import validate_extra_starts


def assess(problem, result):
    """Позачасова діагностика точки та емпіричного порядку."""
    x = result.x
    nearest = min(problem.minima, key=lambda a: np.linalg.norm(x - np.asarray(a)))
    target = np.asarray(nearest)
    error = float(np.linalg.norm(x - target))
    stationary_type = 'not_stationary'
    if result.status == 'stationary':
        eigen = np.linalg.eigvalsh(problem.derivatives(x, 2))
        threshold = 1e-8 * max(1., float(np.max(np.abs(eigen))))
        if eigen[0] < -threshold:
            stationary_type = 'negative_hessian_eigenvalue'
        elif eigen[0] > threshold:
            stationary_type = 'positive_definite_hessian'
        else:
            stationary_type = 'inconclusive_hessian'
    orders = []
    # Є сенс оцінювати порядок лише для збіжності до відомого мінімуму.
    if result.status == 'stationary' and error < 1e-3:
        errors = [float(np.linalg.norm(item['x'] - target)) for item in result.history]
        for a, b, c in zip(errors, errors[1:], errors[2:]):
            if not (1e-12 < c < b < a < 1e-2):
                continue
            denominator = np.log(b/a)
            if abs(denominator) > 1e-12:
                order = float(np.log(c/b) / denominator)
                if np.isfinite(order) and order > 0:
                    orders.append(order)
    return {'distance_to_nearest_minimum': error, 'stationary_type': stationary_type,
            'observed_order': orders[-1] if orders else None,
            'order_estimates': len(orders)}


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def run_suite(outdir, config=None, repeats=3, selection='all', method='all', start_index=None,
              progress=None, cancel_check=None, verbose=True, warmups=2, extra_starts=None):
    cfg = config or Config()
    cfg.validate()
    if repeats < 1:
        raise ValueError('Кількість повторів має бути додатною')
    if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 0:
        raise ValueError('Кількість прогрівів має бути невід’ємним цілим числом')
    suite = problem_suite()
    if selection != 'all':
        suite = [p for p in suite if f'{p.name}:{p.dimension}' == selection or p.name == selection]
    if not suite:
        raise ValueError(f'Не знайдено задачу {selection!r}')
    methods = expand_methods(method)
    if start_index is not None and any(start_index < 0 or start_index >= len(p.starts) for p in suite):
        raise ValueError('Індекс початкової точки поза межами для однієї із задач')
    extra_starts = validate_extra_starts(suite, extra_starts or [])
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    summary, history, repetitions, diagnostics = [], [], [], []
    jobs = [(p, i, x0, m) for p in suite
            for i, x0 in ((list(enumerate(p.starts)) if start_index is None else [(start_index, p.starts[start_index])])
                          + [(len(p.starts) + i, x) for i, x in enumerate(extra_starts)])
            for m in methods]
    cancelled = False
    for problem, index, x0, method_name in jobs:
        for _ in range(warmups):
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            minimize(problem, method_name, x0, cfg)
        if cancelled:
            break
        runs = []
        for _ in range(repeats):
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            runs.append(minimize(problem, method_name, x0, cfg))
        if cancelled:
            break  # не порівнюємо неповну серію повторень
        result = runs[0]
        assessment = assess(problem, result)
        identifier = f'{problem.name}_n{problem.dimension}_s{index}_{method_name}'
        signature = [(r.status, r.iterations, r.counts, r.x.tolist()) for r in runs]
        repeat_consistent = all(s == signature[0] for s in signature[1:])
        row = {'case_id': identifier, 'problem': problem.name, 'dimension': problem.dimension,
               'start_index': index, 'start': json.dumps(x0), 'method': method_name,
               'status': result.status, 'message': result.message, 'iterations': result.iterations,
               'f': result.f, 'grad_norm': result.grad_norm, 'x': json.dumps(result.x.tolist()),
               **assessment, **{f'calls_{key}': value for key, value in result.counts.items()},
               'time_mean_seconds': statistics.mean(r.elapsed for r in runs),
               'time_median_seconds': statistics.median(r.elapsed for r in runs),
               'time_stdev_seconds': statistics.stdev(r.elapsed for r in runs) if repeats > 1 else 0.,
               'repeats': repeats, 'repeat_consistent': repeat_consistent}
        summary.append(row)
        for i, r in enumerate(runs):
            repetitions.append({'case_id': identifier, 'repeat': i+1, 'elapsed_seconds': r.elapsed,
                                'status': r.status, 'iterations': r.iterations, 'f': r.f,
                                'grad_norm': r.grad_norm})
        for record in result.history:
            history.append({'case_id': identifier, **record, 'x': json.dumps(record['x'].tolist())})
        for i, line in enumerate(result.line_diagnostics):
            diagnostics.append({'case_id': identifier, 'iteration': i+1, **line})
        if verbose:
            print(f'{identifier}: {result.status}, ітерацій={result.iterations}, '
                  f'||g||={result.grad_norm:.3g}, час={row["time_mean_seconds"]:.4g} с', flush=True)
        if progress is not None:
            progress(len(summary), len(jobs), row)
    summary_columns = ['case_id', 'problem', 'dimension', 'start_index', 'start', 'method', 'status', 'message',
                       'iterations', 'f', 'grad_norm', 'x', 'distance_to_nearest_minimum', 'stationary_type',
                       'observed_order', 'order_estimates', 'calls_f', 'calls_grad', 'calls_hess',
                       'calls_line_polynomial', 'calls_linear_solves', 'calls_bfgs_applied',
                       'calls_bfgs_skipped', 'calls_bfgs_damped', 'calls_broyden_restarts', 'time_mean_seconds',
                       'time_median_seconds', 'time_stdev_seconds', 'repeats', 'repeat_consistent']
    write_csv(output / 'summary.csv', summary, summary_columns)
    write_csv(output / 'history.csv', history, ['case_id', 'iteration', 'x', 'f', 'grad_norm', 'alpha', 'broyden_restart_next'])
    write_csv(output / 'timings.csv', repetitions, ['case_id', 'repeat', 'elapsed_seconds', 'status', 'iterations', 'f', 'grad_norm'])
    write_csv(output / 'line_diagnostics.csv', diagnostics,
              ['case_id', 'iteration', 'line_root_count', 'line_derivative', 'line_degenerate'])
    metadata = {'program_version': '3.4', 'corrector': 'inverse_matrix_midpoint_gradient',
                'cancelled': cancelled, 'completed_cases': len(summary), 'planned_cases': len(jobs),
                'created_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
                'numpy': np.__version__, 'platform': platform.platform(), 'processor': platform.processor(),
                'config': vars(cfg), 'repeats': repeats, 'warmups': warmups,
                'warmup_timing_excluded': True, 'selection': selection, 'method': method,
                'start_index': start_index, 'extra_starts': extra_starts,
                'notes': 'Прогріви виконуються для кожного методу і старту та не входять до результатів. Час включає підрахунок і збереження історії в пам’яті, але не запис CSV і діагностику.'}
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary

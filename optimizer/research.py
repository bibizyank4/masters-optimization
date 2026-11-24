"""Додаткові відтворювані експерименти та окремі звіти."""
import hashlib
import csv
import json
import os
import platform
import statistics
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .experiments import assess, run_suite, write_csv
from .methods import Config, expand_methods, minimize
from .problems import problem_suite
from .results import METHOD_NAMES, PROBLEM_NAMES, STATUS_NAMES
from .starts import validate_extra_starts


def _environment():
    return {'created_utc': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'numpy': np.__version__,
            'platform': platform.platform(), 'processor': platform.processor(),
            'thread_env': {k: os.environ.get(k) for k in
                           ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}


def _prepare(path):
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Папка для нового експерименту має бути порожньою')
    output.mkdir(parents=True, exist_ok=True)
    return output


def _select(selection, method, start_index):
    suite = problem_suite()
    if selection != 'all':
        suite = [p for p in suite if selection in (p.name, f'{p.name}:{p.dimension}')]
    if not suite:
        raise ValueError('Задачу не знайдено')
    methods = expand_methods(method)
    if start_index is not None and any(start_index < 0 or start_index >= len(p.starts) for p in suite):
        raise ValueError('Індекс старту поза межами однієї із задач')
    return suite, methods


def _case_key(problem, index):
    return f'{problem.name}:n{problem.dimension}:s{index}'


def _direction(seed, key, sample, n):
    digest = hashlib.sha256(f'{seed}|{key}|{sample}'.encode('utf-8')).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:16], 'big'))
    vector = rng.standard_normal(n)
    return vector / np.linalg.norm(vector)


def sensitivity(outdir, config=None, selection='extended_rosenbrock:10', method='all',
                start_index=0, scales=(1e-15, 1e-12, 1e-8), samples=3, seed=42,
                progress=None, cancel_check=None, seeds=None, extra_starts=None):
    """Стартові збурення з однаковими напрямами для всіх методів та масштабів."""
    cfg = config or Config()
    cfg.validate()
    suite, methods = _select(selection, method, start_index)
    extra_starts = validate_extra_starts(suite, extra_starts or [])
    scales = tuple(float(x) for x in scales)
    if not scales or len(set(scales)) != len(scales) or any(not np.isfinite(x) or x <= 0 for x in scales):
        raise ValueError('Масштаби мають бути різними додатними скінченними числами')
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 100:
        raise ValueError('Кількість збурень для кожного масштабу: від 1 до 100')
    seeds = tuple(seeds if seeds is not None else (seed,))
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds):
        raise ValueError('Seed мають бути різними невід’ємними цілими числами')
    output = _prepare(outdir)
    jobs = []
    for problem in suite:
        indices = range(len(problem.starts)) if start_index is None else (start_index,)
        base_points = [(i, problem.starts[i]) for i in indices]
        base_points += [(len(problem.starts) + i, p) for i, p in enumerate(extra_starts)]
        for index, original in base_points:
            base = np.asarray(original, dtype=float)
            key = _case_key(problem, index)
            starts = [(0., 0, None, base)]
            for selected_seed in seeds:
                for scale in scales:
                    for sample in range(1, samples + 1):
                        direction = _direction(selected_seed, key, sample, problem.dimension)
                        starts.append((scale, sample, selected_seed,
                                       base + scale * max(1., float(np.linalg.norm(base))) * direction))
            # Для порівняння з базовою траєкторією кожен метод виконується спочатку без збурення.
            for name in methods:
                for scale, sample, selected_seed, point in starts:
                    jobs.append((problem, index, key, base, name, scale, sample, selected_seed, point))
    rows, trajectories, baselines = [], [], {}
    cancelled = False
    for problem, index, key, base, name, scale, sample, selected_seed, point in jobs:
        if cancel_check and cancel_check():
            cancelled = True
            break
        result = minimize(problem, name, point, cfg)
        details = assess(problem, result)
        ident = (key, name)
        points = [item['x'] for item in result.history]
        if scale == 0:
            baselines[ident] = points
        ref = baselines[ident]
        distances = [float(np.linalg.norm(a - b)) for a, b in zip(points, ref)]
        case_id = f'{key}:{name}:scale={scale:.17g}:seed={selected_seed}:sample={sample}'
        actual = float(np.linalg.norm(point - base))
        row = {'case_id': case_id, 'problem': problem.name, 'dimension': problem.dimension,
               'start_index': index, 'method': name, 'scale': scale, 'seed': selected_seed, 'sample': sample,
               'original_start': json.dumps(base.tolist()), 'start': json.dumps(point.tolist()),
               'actual_perturbation': actual, 'representable': bool(scale == 0 or actual > 0),
               'status': result.status, 'message': result.message, 'stationary_type': details['stationary_type'],
               'iterations': result.iterations, 'f': result.f, 'grad_norm': result.grad_norm,
               'x': json.dumps(result.x.tolist()), 'distance_to_nearest_minimum': details['distance_to_nearest_minimum'],
               'common_iterations': len(distances) - 1, 'max_trajectory_distance': max(distances, default=0.),
               'calls_bfgs_applied': result.counts['bfgs_applied'],
               'calls_bfgs_skipped': result.counts['bfgs_skipped'], 'calls_bfgs_damped': result.counts['bfgs_damped'],
               'calls_broyden_restarts': result.counts['broyden_restarts']}
        rows.append(row)
        for j, item in enumerate(result.history):
            trajectories.append({'case_id': case_id, 'iteration': item['iteration'],
                                 'x': json.dumps(item['x'].tolist()), 'f': item['f'],
                                 'grad_norm': item['grad_norm'], 'alpha': item['alpha'],
                                 'broyden_restart_next': item.get('broyden_restart_next', False),
                                 'distance_to_baseline': distances[j] if j < len(distances) else ''})
        if progress:
            progress(len(rows), len(jobs), row)
    trial_columns = ['case_id', 'problem', 'dimension', 'start_index', 'method', 'scale', 'seed', 'sample',
                     'original_start', 'start', 'actual_perturbation', 'representable', 'status', 'message',
                     'stationary_type', 'iterations', 'f', 'grad_norm', 'x', 'distance_to_nearest_minimum',
                     'common_iterations', 'max_trajectory_distance', 'calls_bfgs_applied',
                     'calls_bfgs_skipped', 'calls_bfgs_damped', 'calls_broyden_restarts']
    write_csv(output / 'trials.csv', rows, trial_columns)
    write_csv(output / 'trajectories.csv', trajectories,
              ['case_id', 'iteration', 'x', 'f', 'grad_norm', 'alpha', 'broyden_restart_next', 'distance_to_baseline'])
    groups = defaultdict(list)
    for row in rows:
        groups[(row['problem'], row['dimension'], row['start_index'], row['method'], row['scale'], row['seed'])].append(row)
    aggregate = []
    for (problem, n, index, name, scale, selected_seed), group in groups.items():
        success = [r for r in group if r['status'] == 'stationary']
        aggregate.append({'problem': problem, 'dimension': n, 'start_index': index, 'method': name, 'scale': scale, 'seed': selected_seed,
                          'trials': len(group), 'stationary': len(success),
                          'positive_definite': sum(r['stationary_type'] == 'positive_definite_hessian' for r in group),
                          'inconclusive': sum(r['stationary_type'] == 'inconclusive_hessian' for r in group),
                          'negative_eigenvalue': sum(r['stationary_type'] == 'negative_hessian_eigenvalue' for r in group),
                          'unchanged_start': sum(not r['representable'] for r in group),
                          'statuses': json.dumps(dict(Counter(r['status'] for r in group)), ensure_ascii=False),
                          'median_iterations_stationary': statistics.median(r['iterations'] for r in success) if success else '',
                          'median_max_trajectory_distance': statistics.median(r['max_trajectory_distance'] for r in group)})
    write_csv(output / 'aggregate.csv', aggregate, ['problem', 'dimension', 'start_index', 'method', 'scale', 'seed',
              'trials', 'stationary', 'positive_definite', 'inconclusive', 'negative_eigenvalue', 'unchanged_start',
              'statuses', 'median_iterations_stationary', 'median_max_trajectory_distance'])
    metadata = {**_environment(), 'experiment': 'initial_point_sensitivity', 'program_version': '3.4',
                'cancelled': cancelled, 'completed_cases': len(rows), 'planned_cases': len(jobs),
                'config': vars(cfg), 'selection': selection, 'method': method, 'start_index': start_index,
                'scales': scales, 'samples': samples, 'seed': seeds[0] if len(seeds) == 1 else None,
                'seeds': seeds, 'extra_starts': extra_starts,
                'perturbation': 'x0 + scale * max(1, norm(x0)) * direction; same directions across methods and scales',
                'interpretation': 'Start perturbations test sensitivity to initial data, not an emulation or proof of floating-point/library differences.'}
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Чутливість до збурення стартової точки', '',
             f"{'НЕПОВНИЙ' if cancelled else 'Повний'} експеримент: {len(rows)}/{len(jobs)} запусків; seed={', '.join(map(str, seeds))}.",
             'Збурення початкових даних не відтворює і не доводить причину розбіжностей між процесорами або бібліотеками.',
             'Масштаб множиться на max(1, ‖x₀‖); однакові напрями використовуються для всіх методів.',
             'Максимальна відстань траєкторій обчислена лише на спільних номерах ітерацій.', '',
             'Додаткові старти та їхні координати збережено в metadata.json і trials.csv.',
             '| Задача / старт | Метод | Масштаб | Seed | Поріг | Типи стаціонарних: + / ? / − | Медіана ітерацій | Макс. відстань: медіана |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:
        lines.append(f"| {PROBLEM_NAMES.get(r['problem'], r['problem'])}, n={r['dimension']}, {r['start_index']} | "
                     f"{METHOD_NAMES.get(r['method'], r['method'])} | {r['scale']:.1g} | {r['seed'] if r['seed'] is not None else '—'} | {r['stationary']}/{r['trials']} | "
                     f"{r['positive_definite']} / {r['inconclusive']} / {r['negative_eigenvalue']} | "
                     f"{r['median_iterations_stationary'] if r['median_iterations_stationary'] != '' else '—'} | {r['median_max_trajectory_distance']:.3g} |")
    if any(r['unchanged_start'] for r in aggregate):
        lines += ['', 'Увага: деякі збурення округлилися до незміненої початкової точки; див. unchanged_start у aggregate.csv.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return metadata


def matrix_comparison(outdir, config=None, repeats=3, warmups=2, selection='all', method='all',
                      start_index=None, progress=None, cancel_check=None, extra_starts=None):
    """Чотири серії з B₀=I/H і BFGS=skip/damp, зі звітом про попарні результати."""
    cfg = config or Config()
    cfg.validate()
    suite, methods = _select(selection, method, start_index)
    extra_starts = validate_extra_starts(suite, extra_starts or [])
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError('Кількість повторів має бути додатним цілим числом')
    if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 0:
        raise ValueError('Кількість прогрівів має бути невід’ємним цілим числом')
    output = _prepare(outdir)
    per_series = (sum(len(p.starts) if start_index is None else 1 for p in suite) + len(extra_starts)) * len(methods)
    total = 4 * per_series
    combinations = [('identity', 'skip'), ('identity', 'damp'), ('hessian', 'skip'), ('hessian', 'damp')]
    all_rows, completed, cancelled = {}, 0, False
    for b0, strategy in combinations:
        if cancel_check and cancel_check():
            cancelled = True
            break
        name = f'{b0}_{strategy}'
        current_cfg = replace(cfg, b0=b0, bfgs_strategy=strategy)
        def notify(done, total_jobs, row):
            if progress:
                progress(completed + done, total, row)
        rows = run_suite(output / name, current_cfg, repeats, selection, method, start_index,
                         progress=notify, cancel_check=cancel_check, verbose=False, warmups=warmups,
                         extra_starts=extra_starts)
        all_rows[name] = rows
        completed += len(rows)
        if len(rows) != per_series:
            cancelled = True
            break
    lines = ['# Вплив B₀ і стратегії BFGS', '',
             f"{'НЕПОВНИЙ' if cancelled else 'Повний'} експеримент: {completed}/{total} запусків.",
             f'Прогрівів: {warmups}; вимірюваних повторів: {repeats}. Стаціонарність не доводить мінімум.',
             'Час порівнюйте лише на спільних стартах, де обидва методи досягли порога.', '']
    pairs = [('Коректор', 'qn_gradient_correction', 'bfgs_gradient_correction'),
             ('Трикроковий', 'qn_three_step', 'bfgs_three_step'),
             ('Двокроковий', 'qn_two_step_newton', 'bfgs_two_step_newton')]
    for name, rows in all_rows.items():
        lines += [f'## {name}', '', '| Метод | Досягли порога | Усього | Типи стаціонарних: + / ? / − |', '|---|---:|---:|---:|']
        for meth in methods:
            data = [r for r in rows if r['method'] == meth]
            if not data:
                continue
            stat = [r for r in data if r['status'] == 'stationary']
            types = [sum(r['stationary_type'] == t for r in stat) for t in
                     ('positive_definite_hessian', 'inconclusive_hessian', 'negative_hessian_eigenvalue')]
            lines.append(f"| {METHOD_NAMES.get(meth, meth)} | {len(stat)} | {len(data)} | " + ' / '.join(map(str, types)) + ' |')
        lines += ['', '| Пара | Спільних успішних стартів | Медіана ітерацій Бройден / BFGS | Медіана часу, мс Бройден / BFGS |',
                  '|---|---:|---:|---:|']
        for label, broyden, bfgs in pairs:
            lookup = {m: {(r['problem'], r['dimension'], r['start_index']): r for r in rows if r['method'] == m}
                      for m in (broyden, bfgs)}
            a, b = lookup[broyden], lookup[bfgs]
            common = [key for key in a.keys() & b.keys() if a[key]['status'] == b[key]['status'] == 'stationary']
            if not common:
                lines.append(f'| {label} | 0 | — | — |')
                continue
            it = [statistics.median(float(d[k]['iterations']) for k in common) for d in (a, b)]
            tm = [statistics.median(float(d[k]['time_mean_seconds']) * 1000 for k in common) for d in (a, b)]
            lines.append(f'| {label} | {len(common)} | {it[0]:g} / {it[1]:g} | {tm[0]:.3f} / {tm[1]:.3f} |')
        lines.append('')
    metadata = {**_environment(), 'experiment': 'matrix_comparison', 'program_version': '3.4',
                'cancelled': cancelled, 'completed_cases': completed, 'planned_cases': total,
                'config_base': vars(cfg), 'repeats': repeats, 'warmups': warmups, 'selection': selection,
                'method': method, 'start_index': start_index, 'extra_starts': extra_starts,
                'completed_series': [name for name, rows in all_rows.items() if len(rows) == per_series],
                'partial_series': [name for name, rows in all_rows.items() if len(rows) != per_series]}
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return metadata


def compare_environments(outdir, first, second):
    """Порівняти спільні випадки та історії двох уже виконаних серій."""
    def read(folder):
        folder = Path(folder)
        meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8-sig'))
        def table(name):
            with (folder / name).open(encoding='utf-8-sig', newline='') as f:
                return list(csv.DictReader(f))
        return meta, table('summary.csv'), table('history.csv')
    meta_a, rows_a, history_a = read(first)
    meta_b, rows_b, history_b = read(second)
    keys = ('grad_tol', 'max_iterations', 'armijo_c', 'backtrack', 'max_backtracks',
            'condition_limit', 'b0', 'correction_search', 'bfgs_strategy', 'bfgs_min_curvature')
    # Нові параметри BFGS не впливають на старі методи; для BFGS їх перевіряємо окремо.
    base_keys = keys[:8]
    differences = {k: (meta_a.get('config', {}).get(k), meta_b.get('config', {}).get(k))
                   for k in base_keys if meta_a.get('config', {}).get(k) != meta_b.get('config', {}).get(k)}
    restart_a = meta_a.get('config', {}).get('broyden_restart_interval', 0)
    restart_b = meta_b.get('config', {}).get('broyden_restart_interval', 0)
    if meta_a.get('corrector') != meta_b.get('corrector'):
        differences['corrector'] = (meta_a.get('corrector'), meta_b.get('corrector'))
    if differences:
        raise ValueError('Параметри серій відрізняються: ' + json.dumps(differences, ensure_ascii=False))
    def by_case(rows):
        result = {}
        for row in rows:
            key = (row['problem'], int(row['dimension']), int(row['start_index']), row['method'])
            if key in result:
                raise ValueError(f'Повторення випадку: {key}')
            result[key] = row
        return result
    a, b = by_case(rows_a), by_case(rows_b)
    ha, hb = defaultdict(list), defaultdict(list)
    for row in history_a:
        ha[row['case_id']].append(row)
    for row in history_b:
        hb[row['case_id']].append(row)
    comparisons, ignored = [], []
    for key in sorted(a.keys() & b.keys()):
        left, right = a[key], b[key]
        if json.loads(left['start']) != json.loads(right['start']):
            ignored.append({'case': key, 'reason': 'different_start'})
            continue
        if key[3].startswith('qn_') and restart_a != restart_b:
            ignored.append({'case': key, 'reason': 'different_broyden_restart'})
            continue
        if key[3].startswith('bfgs_') and any(meta_a.get('config', {}).get(k) != meta_b.get('config', {}).get(k)
                                             for k in keys[8:]):
            ignored.append({'case': key, 'reason': 'different_bfgs_config'})
            continue
        h1 = {int(r['iteration']): np.asarray(json.loads(r['x']), dtype=float) for r in ha[left['case_id']]}
        h2 = {int(r['iteration']): np.asarray(json.loads(r['x']), dtype=float) for r in hb[right['case_id']]}
        common = sorted(h1.keys() & h2.keys())
        distances = [(i, float(np.linalg.norm(h1[i] - h2[i]))) for i in common]
        first_diff = next((i for i, d in distances if d > 1e-12), '')
        comparisons.append({'problem': key[0], 'dimension': key[1], 'start_index': key[2], 'method': key[3],
                            'status_a': left['status'], 'status_b': right['status'],
                            'iterations_a': left['iterations'], 'iterations_b': right['iterations'],
                            'grad_norm_a': left['grad_norm'], 'grad_norm_b': right['grad_norm'],
                            'common_iterations': len(common), 'first_distance_gt_1e-12': first_diff,
                            'max_trajectory_distance': max((d for _, d in distances), default=None)})
    if not comparisons:
        raise ValueError('У серіях немає зіставних випадків')
    output = _prepare(outdir)
    write_csv(output / 'comparison.csv', comparisons,
              ['problem', 'dimension', 'start_index', 'method', 'status_a', 'status_b', 'iterations_a',
               'iterations_b', 'grad_norm_a', 'grad_norm_b', 'common_iterations',
               'first_distance_gt_1e-12', 'max_trajectory_distance'])
    metadata = {**_environment(), 'experiment': 'saved_environment_comparison', 'program_version': '3.4',
                'first': str(Path(first).resolve()), 'second': str(Path(second).resolve()),
                'source_metadata_a': meta_a, 'source_metadata_b': meta_b, 'matched_cases': len(comparisons),
                'excluded_cases': ignored, 'only_in_a': len(a.keys() - b.keys()), 'only_in_b': len(b.keys() - a.keys())}
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    changed = [r for r in comparisons if r['status_a'] != r['status_b'] or r['iterations_a'] != r['iterations_b']]
    lines = ['# Порівняння збережених серій', '', f'Перша серія: {Path(first).resolve()}',
             f'Друга серія: {Path(second).resolve()}',
             f"Серії неповні: A={'так' if meta_a.get('cancelled') else 'ні/не зазначено'}, "
             f"B={'так' if meta_b.get('cancelled') else 'ні/не зазначено'}.",
             f'Зіставлено {len(comparisons)} випадків; різний статус або число ітерацій у {len(changed)}.',
             f'Виключено через різні старти/налаштування BFGS: {len(ignored)}; '
             f'лише в першій серії: {len(a.keys() - b.keys())}, лише в другій: {len(b.keys() - a.keys())}.',
             'Порівняння не встановлює причину розбіжностей. Максимальна відстань стосується лише спільних ітерацій.', '',
             '| Задача | Метод | Статус A → B | Ітерацій A → B | Перше ‖xᴬ−xᴮ‖ > 1e-12 |',
             '|---|---|---|---:|---:|']
    for r in changed:
        lines.append(f"| {PROBLEM_NAMES.get(r['problem'], r['problem'])}, n={r['dimension']}, старт {r['start_index']} | "
                     f"{METHOD_NAMES.get(r['method'], r['method'])} | {r['status_a']} → {r['status_b']} | "
                     f"{r['iterations_a']} → {r['iterations_b']} | {r['first_distance_gt_1e-12'] or '—'} |")
    if not changed:
        lines.append('| Немає відмінностей статусів або кількості ітерацій | — | — | — | — |')
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return metadata

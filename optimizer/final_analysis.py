"""Відтворюваний комплексний експеримент для вбудованого набору задач."""
import contextlib
import csv
import io
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from .experiments import run_suite, write_csv
from .methods import Config
from .research import _environment, _prepare, _select, sensitivity
from .results import METHOD_NAMES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOLERANCES = (1e-6, 1e-8, 1e-10)
RESTART_INTERVAL = 50
SENSITIVITY_SCALES = (1e-15, 1e-12, 1e-8)
SENSITIVITY_SEEDS = (42, 123, 2026)


def _batches(tolerances, base_config, restart_interval):
    batches = []
    for tol in tolerances:
        base = replace(base_config, grad_tol=tol, broyden_restart_interval=0)
        prefix = f'tol_{tol:.0e}'
        batches.append((prefix + '_classical', 'classical_family', base))
        for b0 in ('hessian', 'identity'):
            for interval in (0, restart_interval):
                batches.append((f'{prefix}_broyden_{b0}_restart_{interval}', 'broyden_family',
                                replace(base, b0=b0, broyden_restart_interval=interval)))
        for b0 in ('hessian', 'identity'):
            for strategy in ('skip', 'damp'):
                batches.append((f'{prefix}_bfgs_{b0}_{strategy}', 'bfgs_family',
                                replace(base, b0=b0, bfgs_strategy=strategy)))
    return batches


def _validation(output):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    cmd = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True, encoding='utf-8',
                                errors='replace', capture_output=True, timeout=300)
        content = result.stdout + result.stderr
        code = result.returncode
        match = re.search(r'Ran\s+(\d+)\s+tests?\s+in\s', content)
        if code == 0 and (not match or int(match.group(1)) == 0):
            code = -2
            content += '\nНе вдалося підтвердити виконання хоча б одного тесту.\n'
    except (OSError, subprocess.TimeoutExpired) as exc:
        content, code = f'Не вдалося завершити автоматичні перевірки: {exc}', -1
    (output / 'validation.txt').write_text(content, encoding='utf-8')
    return code


def _build_tables(batches):
    summaries, comparisons = [], []
    for batch, rows in batches.items():
        for method in dict.fromkeys(row['method'] for row in rows):
            group = [r for r in rows if r['method'] == method]
            stationary = [r for r in group if r['status'] == 'stationary']
            summaries.append({'batch': batch, 'method': method, 'cases': len(group),
                              'stationary': len(stationary),
                              'positive_definite': sum(r['stationary_type'] == 'positive_definite_hessian' for r in stationary),
                              'inconclusive': sum(r['stationary_type'] == 'inconclusive_hessian' for r in stationary),
                              'negative_eigenvalue': sum(r['stationary_type'] == 'negative_hessian_eigenvalue' for r in stationary),
                              'statuses': json.dumps(dict(Counter(r['status'] for r in group)), ensure_ascii=False),
                              'median_iterations_stationary': statistics.median(r['iterations'] for r in stationary) if stationary else '',
                              'repeat_inconsistent': sum(not r['repeat_consistent'] for r in group)})
    def compare(label, name_a, name_b, method_a, method_b):
        if name_a not in batches or name_b not in batches:
            return
        def lookup(rows, method):
            return {(r['problem'], r['dimension'], r['start_index']): r for r in rows if r['method'] == method}
        left, right = lookup(batches[name_a], method_a), lookup(batches[name_b], method_b)
        all_keys = left.keys() & right.keys()
        common = [k for k in all_keys if left[k]['status'] == right[k]['status'] == 'stationary']
        only_a = sum(left[k]['status'] == 'stationary' and right[k]['status'] != 'stationary' for k in all_keys)
        only_b = sum(right[k]['status'] == 'stationary' and left[k]['status'] != 'stationary' for k in all_keys)
        row = {'comparison': label, 'batch_a': name_a, 'batch_b': name_b, 'method_a': method_a, 'method_b': method_b,
               'paired_cases': len(all_keys), 'stationary_both': len(common), 'stationary_only_a': only_a,
               'stationary_only_b': only_b}
        for col, field, factor in [('iterations', 'iterations', 1), ('time_ms', 'time_mean_seconds', 1000),
                                   ('hess_calls', 'calls_hess', 1), ('linear_solves', 'calls_linear_solves', 1)]:
            for suffix, data in (('a', left), ('b', right)):
                row[f'median_{col}_{suffix}'] = statistics.median(float(data[k][field]) * factor for k in common) if common else ''
        comparisons.append(row)
    for name in batches:
        if '_broyden_' not in name or not name.endswith('_restart_0'):
            continue
        prefix, rest = name.split('_broyden_', 1)
        b0 = rest.rsplit('_restart_', 1)[0]
        matching = next((n for n in batches if n.startswith(f'{prefix}_broyden_{b0}_restart_') and n != name), None)
        for family in ('gradient_correction', 'three_step', 'two_step_newton'):
            if matching:
                compare('Бройден: без перезапуску / з перезапуском', name, matching, 'qn_' + family, 'qn_' + family)
            for strategy in ('skip', 'damp'):
                compare('Бройден / BFGS', name, f'{prefix}_bfgs_{b0}_{strategy}', 'qn_' + family, 'bfgs_' + family)
    return summaries, comparisons


def _sensitivity_pairs(output, n_values=(10, 20)):
    result = []
    for n in n_values:
        first = output / f'sensitivity_n{n}' / 'trials.csv'
        second = output / f'sensitivity_n{n}_restart' / 'trials.csv'
        if not first.exists() or not second.exists():
            continue
        def read(path):
            with path.open(encoding='utf-8-sig', newline='') as f:
                return {row['case_id']: row for row in csv.DictReader(f) if row['method'] == 'qn_three_step'}
        a, b = read(first), read(second)
        groups = {}
        for key in a.keys() & b.keys():
            if a[key]['start'] != b[key]['start']:
                raise ValueError('Збурені точки для парного порівняння не збігаються')
            group = (a[key]['start_index'], a[key]['scale'])
            groups.setdefault(group, []).append((a[key], b[key]))
        for (index, scale), pairs in sorted(groups.items(), key=lambda item: (int(item[0][0]), float(item[0][1]))):
            result.append({'dimension': n, 'start_index': index, 'scale': scale, 'paired': len(pairs),
                           'stationary_both': sum(x['status'] == y['status'] == 'stationary' for x, y in pairs),
                           'only_original': sum(x['status'] == 'stationary' and y['status'] != 'stationary' for x, y in pairs),
                           'only_restart': sum(y['status'] == 'stationary' and x['status'] != 'stationary' for x, y in pairs),
                           'neither': sum(x['status'] != 'stationary' and y['status'] != 'stationary' for x, y in pairs)})
    return result


def final_analysis(outdir, repeats=20, warmups=2, progress=None, cancel_check=None,
                   *, selection='all', start_index=None, tolerances=TOLERANCES,
                   restart_interval=RESTART_INTERVAL, include_sensitivity=True, validate=True):
    """Повний план; параметри selection/tolerances призначені також для швидкої перевірки."""
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError('Кількість повторень має бути додатним цілим числом')
    if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 0:
        raise ValueError('Кількість прогрівів має бути невід’ємним цілим числом')
    if not isinstance(restart_interval, int) or isinstance(restart_interval, bool) or restart_interval < 1:
        raise ValueError('Інтервал перезапуску має бути додатним цілим числом')
    if not tolerances or len(set(tolerances)) != len(tolerances) or any(not np.isfinite(x) or x <= 0 for x in tolerances):
        raise ValueError('Пороги градієнта мають бути різними додатними числами')
    suite, _ = _select(selection, 'all', start_index)
    if include_sensitivity and (selection != 'all' or start_index is not None):
        raise ValueError('Повне дослідження чутливості вимагає вибору всіх вбудованих задач')
    base = Config()
    batches = _batches(tolerances, base, restart_interval)
    starts = sum(len(p.starts) if start_index is None else 1 for p in suite)
    per_batch = starts * 3
    per_start = 1 + len(SENSITIVITY_SEEDS) * len(SENSITIVITY_SCALES) * 3
    sensitivity_jobs = 2 * 3 * 4 * per_start if include_sensitivity else 0
    total = len(batches) * per_batch + sensitivity_jobs
    output = _prepare(outdir)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        np.show_config()
    for warning in captured:
        buffer.write(f'\n{warning.message}\n')
    (output / 'environment.txt').write_text(buffer.getvalue(), encoding='utf-8')
    validation_code = _validation(output) if validate else None
    completed, data, finished, sensitivity_results = 0, {}, [], {}
    status = 'running'
    error = ''
    if validate and validation_code != 0:
        status, error = 'validation_failed', 'Автоматичні тести не пройшли; див. validation.txt.'
    else:
        try:
            for name, method, cfg in batches:
                if cancel_check and cancel_check():
                    status = 'cancelled'
                    break
                def notify(done, _total, row):
                    if progress:
                        progress(completed + done, total, row)
                rows = run_suite(output / 'series' / name, cfg, repeats, selection, method, start_index,
                                 progress=notify, cancel_check=cancel_check, verbose=False, warmups=warmups)
                data[name] = rows
                completed += len(rows)
                if len(rows) != per_batch:
                    status = 'cancelled'
                    break
                finished.append(name)
            else:
                if include_sensitivity:
                    tasks = [(n, extra, restart) for n, extra in
                             ((10, [[-1., 1.] * 5, [0., 0.] * 5]),
                              (20, [[-1., 1., 0., 0.] * 5, [0., 0.] * 10]))
                             for restart in (False, True)]
                    for n, extra, restart in tasks:
                        if cancel_check and cancel_check():
                            status = 'cancelled'
                            break
                        def notify_sensitivity(done, _total, row):
                            if progress:
                                progress(completed + done, total, row)
                        suffix = '_restart' if restart else ''
                        method = 'qn_three_step' if restart else 'three_step_family'
                        cfg = replace(base, broyden_restart_interval=restart_interval if restart else 0)
                        meta = sensitivity(output / f'sensitivity_n{n}{suffix}', cfg, f'extended_rosenbrock:{n}',
                                           method, 0, SENSITIVITY_SCALES, 3,
                                           progress=notify_sensitivity, cancel_check=cancel_check,
                                           seeds=SENSITIVITY_SEEDS, extra_starts=extra)
                        sensitivity_results[f'{n}{suffix}'] = meta
                        completed += meta['completed_cases']
                        if meta['cancelled']:
                            status = 'cancelled'
                            break
                    else:
                        status = 'complete'
                else:
                    status = 'complete'
        except Exception as exc:
            status, error = 'failed', f'{type(exc).__name__}: {exc}'
    summaries, comparisons = _build_tables(data)
    summary_cols = ['batch', 'method', 'cases', 'stationary', 'positive_definite', 'inconclusive',
                    'negative_eigenvalue', 'statuses', 'median_iterations_stationary', 'repeat_inconsistent']
    compare_cols = ['comparison', 'batch_a', 'batch_b', 'method_a', 'method_b', 'paired_cases',
                    'stationary_both', 'stationary_only_a', 'stationary_only_b',
                    'median_iterations_a', 'median_iterations_b', 'median_time_ms_a', 'median_time_ms_b',
                    'median_hess_calls_a', 'median_hess_calls_b', 'median_linear_solves_a', 'median_linear_solves_b']
    write_csv(output / 'summary_by_method.csv', summaries, summary_cols)
    write_csv(output / 'matched_comparison.csv', comparisons, compare_cols)
    try:
        sensitivity_pairs = _sensitivity_pairs(output)
    except (ValueError, OSError) as exc:
        sensitivity_pairs = []
        status, error = 'failed', f'Не вдалося порівняти чутливість: {exc}'
    write_csv(output / 'sensitivity_restart_pairs.csv', sensitivity_pairs,
              ['dimension', 'start_index', 'scale', 'paired', 'stationary_both', 'only_original', 'only_restart', 'neither'])
    meta = {**_environment(), 'experiment': 'final_analysis', 'program_version': '3.4', 'status': status,
            'cancelled': status == 'cancelled', 'error': error, 'completed_cases': completed, 'planned_cases': total,
            'validation_returncode': validation_code, 'selection': selection, 'start_index': start_index,
            'tolerances': tolerances, 'repeats': repeats, 'warmups': warmups, 'config_base': vars(base),
            'broyden_restart_interval': restart_interval, 'sensitivity_included': include_sensitivity,
            'planned_batches': [name for name, _, _ in batches], 'completed_batches': finished,
            'sensitivity_results': sensitivity_results, 'python_executable': sys.executable}
    (output / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Комплексний аналіз методів', '', f'Стан: {status}. Завершено випадків: {completed}/{total}.',
             f'Автотести: {"пройдено" if validation_code == 0 else "не виконувалися" if validation_code is None else "не пройдено"}. {error}',
             f'Пороги: {", ".join(str(t) for t in tolerances)}; ліміт ітерацій: {base.max_iterations}; '
             f'прогрівів: {warmups}; вимірюваних повторів: {repeats}.',
             f'Бройден: окремо оригінальна формула і варіант з оновленням гессіана кожні {restart_interval} ітерацій.',
             'BFGS: B₀=H(x₀) / I; пропуск / демпфування. Класичні методи запускаються один раз на поріг.',
             'Стаціонарність не доводить мінімум. Час у попарній таблиці порівнюється тільки на спільних успішних стартах.',
             'Оцінки порядку в окремих summary.csv — евристичні; додатковий розрахунок гессіана входить до витрат.', '',
             '## Підсумки серій', '', '| Серія | Метод | Поріг | + / ? / − | Статуси |', '|---|---|---:|---:|---|']
    for row in summaries:
        lines.append(f"| {row['batch']} | {METHOD_NAMES.get(row['method'], row['method'])} | "
                     f"{row['stationary']}/{row['cases']} | {row['positive_definite']} / {row['inconclusive']} / {row['negative_eigenvalue']} | {row['statuses']} |")
    lines += ['', '## Парне порівняння', '', '| Випадки | Метод A → B | Обидва / лише A / лише B | Медіана ітерацій A / B | Медіана часу, мс A / B |', '|---|---|---:|---:|---:|']
    def fmt(value):
        return '—' if value == '' else f'{value:.3g}'
    for row in comparisons:
        lines.append(f"| {row['batch_a']} → {row['batch_b']} | {row['method_a']} → {row['method_b']} | "
                     f"{row['stationary_both']} / {row['stationary_only_a']} / {row['stationary_only_b']} | "
                     f"{fmt(row['median_iterations_a'])} / {fmt(row['median_iterations_b'])} | "
                     f"{fmt(row['median_time_ms_a'])} / {fmt(row['median_time_ms_b'])} |")
    if sensitivity_pairs:
        lines += ['', '## Чутливість Бройдена: оригінал і перезапуск', '',
                  'Результати наведено за однаковими точками; напрями, використані для різних масштабів, пов’язані між собою.',
                  '| n / старт | Масштаб | Пар | Обидва / лише оригінал / лише перезапуск / жоден |',
                  '|---|---:|---:|---:|']
        for row in sensitivity_pairs:
            lines.append(f"| {row['dimension']} / {row['start_index']} | {row['scale']} | {row['paired']} | "
                         f"{row['stationary_both']} / {row['only_original']} / {row['only_restart']} / {row['neither']} |")
    lines += ['', 'Детальні результати й історії — у series/; попарні обчислення — у matched_comparison.csv.',
              'Чутливість для n=10 і n=20, якщо виконана, зберігається окремо у sensitivity_n10/ та sensitivity_n20/.',
              'Вона використовує 3 стартові точки, 3 seed, 3 масштаби, по 3 напрями та три трикрокові методи без перезапуску.',
              'Додатково трикроковий Бройден із перезапуском досліджується на тих самих точках у папках *_restart.',
              'Збурення початкових даних не моделює і не пояснює відмінності між процесорами чи бібліотеками.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return meta

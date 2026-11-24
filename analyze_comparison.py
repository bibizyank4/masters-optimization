"""Створити стислий звіт за чотирма серіями comparison/ (стандартна бібліотека)."""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'comparison'
PAIRS = [('Коректор', 'qn_gradient_correction', 'bfgs_gradient_correction'),
         ('Трикроковий', 'qn_three_step', 'bfgs_three_step'),
         ('Двокроковий Ньютона', 'qn_two_step_newton', 'bfgs_two_step_newton')]


def case(r):
    return r['problem'], r['dimension'], r['start_index']


def read(path):
    meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    with (path / 'summary.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if meta['cancelled'] or len(rows) != meta['planned_cases']:
        raise ValueError(f'Неповна серія: {path}')
    return meta, rows


def success(r):
    return r['status'] == 'stationary'


def main():
    lines = ['# Попереднє порівняння Бройдена і BFGS', '',
             'Три сімейства, 17 стартів на метод, поріг норми градієнта 1e-8, '
             'ліміт 500 ітерацій, 1 прогрів і 3 вимірювання. Стаціонарність не доводить мінімум.',
             'Використано Python 3.12.14, NumPy 2.3.5, Windows, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1. '
             'Кожна з чотирьох серій виконана окремим процесом; повні умови в metadata.json.',
             'Медіани часу та ітерацій наведено тільки для спільних стартів, де обидва методи досягли порога. '
             'Час є орієнтовним: порівнюються реалізації на одній машині, повторів лише три.', '']
    for name in ('identity_skip', 'identity_damp', 'hessian_skip', 'hessian_damp'):
        meta, rows = read(ROOT / name)
        classic = [(label, sum(success(r) for r in rows if r['method'] == method))
                   for label, method in [('коректор', 'gradient_correction'),
                                         ('трикроковий', 'three_step'),
                                         ('двокроковий Ньютона', 'two_step_newton')]]
        lines += [f'## {name}', '',
                  f"B₀={meta['config']['b0']}; стратегія BFGS={meta['config']['bfgs_strategy']}; "
                  f"повторів={meta['repeats']}, прогрівів={meta['warmups']}; "
                  f"результатів={len(rows)}.",
                  'Класичні схеми, досягли порога: ' + ', '.join(f'{label} {n}/17' for label, n in classic) + '.', '',
                  '| Схема | Поріг Бройден | Поріг BFGS | Спільні | Медіана ітерацій Б / BFGS | Медіана часу, мс Б / BFGS |',
                  '|---|---:|---:|---:|---:|---:|']
        for label, broyden, bfgs in PAIRS:
            br = {case(r): r for r in rows if r['method'] == broyden}
            bf = {case(r): r for r in rows if r['method'] == bfgs}
            common = [k for k in br if k in bf and success(br[k]) and success(bf[k])]
            if common:
                med = lambda d, field, factor: statistics.median(float(d[k][field]) * factor for k in common)
                it = f"{med(br, 'iterations', 1):.1f} / {med(bf, 'iterations', 1):.1f}"
                tm = f"{med(br, 'time_mean_seconds', 1000):.3f} / {med(bf, 'time_mean_seconds', 1000):.3f}"
            else:
                it = tm = '—'
            lines.append(f'| {label} | {sum(map(success, br.values()))}/17 | '
                         f'{sum(map(success, bf.values()))}/17 | {len(common)} | {it} | {tm} |')
        lines += ['', 'Статуси BFGS і кількість оновлень у перших повторах:', '']
        for label, _, method in PAIRS:
            group = [r for r in rows if r['method'] == method]
            statuses = ', '.join(f'{s}: {n}' for s, n in sorted(Counter(r['status'] for r in group).items()))
            classes = ', '.join(f'{s}: {n}' for s, n in sorted(Counter(r['stationary_type'] for r in group if success(r)).items())) or 'немає'
            counts = [sum(int(r['calls_bfgs_' + key]) for r in group) for key in ('applied', 'skipped', 'damped')]
            lines.append(f'- {label}: {statuses}; типи стаціонарних точок: {classes}; '
                         f'оновлення звичайні / пропущені / демпфовані: {counts[0]} / {counts[1]} / {counts[2]}.')
        lines.append('')
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(ROOT / 'REPORT.md')


if __name__ == '__main__':
    main()

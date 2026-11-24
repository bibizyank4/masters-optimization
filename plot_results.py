"""Побудова PNG за CSV: python plot_results.py --input results --output figures."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optimizer.problems import problem_suite


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='Графіки результатів оптимізації')
    parser.add_argument('--input', default='results')
    parser.add_argument('--output', default='figures')
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (source / 'summary.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    with (source / 'history.csv').open(encoding='utf-8-sig', newline='') as f:
        history_rows = list(csv.DictReader(f))
    histories = defaultdict(list)
    for row in history_rows:
        histories[row['case_id']].append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['problem'], int(row['dimension']), int(row['start_index']))].append(row)
    lookup = {(p.name, p.dimension): p for p in problem_suite()}
    # Логарифм нуля не визначений. Для графіка обмежуємо лише відображення;
    # оригінальні значення зберігаються без змін у CSV.
    display_floor = 1e-16
    for (name, n, index), group in groups.items():
        problem = lookup[(name, n)]
        fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
        for row in group:
            hist = histories[row['case_id']]
            if not hist:
                continue
            ax.semilogy([int(h['iteration']) for h in hist],
                        [max(float(h['grad_norm']), display_floor) for h in hist],
                        label=f"{row['method']} ({row['status']})")
        ax.set(xlabel='Ітерація k', ylabel='Норма градієнта (нижня межа графіка 10⁻¹⁶)',
               title=f'{name}, n={n}, початок {index}: збіжність')
        ax.grid(True, alpha=.3)
        ax.legend(fontsize=7)
        fig.savefig(output / f'{name}_n{n}_s{index}_convergence.png', dpi=160)
        plt.close(fig)
        if n != 2:
            continue
        # Для читабельності включаємо лише точки в помірному околі старту й відомих мінімумів.
        initial = np.asarray(json.loads(group[0]['start']), float)
        anchor = np.asarray(min(problem.minima, key=lambda x: np.linalg.norm(initial - x)))
        span = max(2., float(np.max(np.abs(initial-anchor))) + 1.)
        lo, hi = np.minimum(initial, anchor)-span*.35, np.maximum(initial, anchor)+span*.35
        lo = np.minimum(lo, initial-1.)
        hi = np.maximum(hi, initial+1.)
        xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 180), np.linspace(lo[1], hi[1], 180))
        z = problem.expression([xx, yy])
        positive = z[z > 0]
        levels = np.geomspace(max(float(np.min(positive)), 1e-5), max(float(np.max(z)), 1e-4), 25)
        fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
        ax.contour(xx, yy, z, levels=levels, colors='0.75', linewidths=.6)
        for row in group:
            points = np.asarray([json.loads(h['x']) for h in histories[row['case_id']]])
            if not len(points):
                continue
            ax.plot(points[:, 0], points[:, 1], marker='.', markersize=3, linewidth=1,
                    label=f"{row['method']} ({row['status']})")
        ax.scatter(*initial, c='black', marker='x', s=65, zorder=4, label='Початок')
        for point in problem.minima:
            if lo[0] <= point[0] <= hi[0] and lo[1] <= point[1] <= hi[1]:
                ax.scatter(*point, c='red', marker='*', s=75, zorder=4)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel='x₁', ylabel='x₂',
               title=f'{name}, старт {index}: траєкторії (за межами вікна обрізані)')
        ax.legend(fontsize=7)
        fig.savefig(output / f'{name}_n{n}_s{index}_paths.png', dpi=160)
        plt.close(fig)
    print(f'Графіки збережено у {output.resolve()}')


if __name__ == '__main__':
    main()

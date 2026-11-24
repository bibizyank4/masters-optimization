"""Додаткові експерименти: python research.py --help."""
import argparse
import sys

from optimizer.methods import Config, METHODS, METHOD_GROUPS
from optimizer.research import sensitivity, matrix_comparison, compare_environments
from optimizer.starts import parse_extra_starts


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='Додаткові експерименти для багатокрокових методів')
    parser.add_argument('experiment', choices=('sensitivity', 'matrix', 'compare'))
    parser.add_argument('--output', required=True, help='Нова або порожня папка результатів')
    parser.add_argument('--problem', default='all', help='Задача, наприклад extended_rosenbrock:10')
    parser.add_argument('--method', choices=('all', *METHODS, *METHOD_GROUPS), default='all')
    parser.add_argument('--extra-starts', default='', help='Додаткові точки, наприклад "(-1,1)*10;(0,0)*10"')
    parser.add_argument('--start-index', type=int)
    parser.add_argument('--grad-tol', type=float, default=1e-8)
    parser.add_argument('--max-iterations', type=int, default=500)
    parser.add_argument('--condition-limit', type=float, default=1e14)
    parser.add_argument('--correction-search', choices=('monotone', 'armijo'), default='monotone')
    parser.add_argument('--b0', choices=('identity', 'hessian'), default='hessian')
    parser.add_argument('--bfgs-strategy', choices=('skip', 'damp'), default='skip')
    parser.add_argument('--bfgs-min-curvature', type=float, default=1e-10)
    parser.add_argument('--broyden-restart-interval', type=int, default=0)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--scales', type=float, nargs='+', default=[1e-15, 1e-12, 1e-8])
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--seed', type=int, nargs='+', default=[42], help='Одне або кілька значень seed')
    parser.add_argument('--first', help='Перша папка зі збереженою серією для compare')
    parser.add_argument('--second', help='Друга папка зі збереженою серією для compare')
    args = parser.parse_args()
    cfg = Config(grad_tol=args.grad_tol, max_iterations=args.max_iterations,
                 condition_limit=args.condition_limit, correction_search=args.correction_search,
                 b0=args.b0, bfgs_strategy=args.bfgs_strategy, bfgs_min_curvature=args.bfgs_min_curvature,
                 broyden_restart_interval=args.broyden_restart_interval)
    def progress(done, total, row):
        print(f"{done}/{total}: {row['problem']} · {row['method']}", flush=True)
    if args.experiment == 'sensitivity':
        result = sensitivity(args.output, cfg, args.problem, args.method, args.start_index,
                             args.scales, args.samples, progress=progress, seeds=args.seed,
                             extra_starts=parse_extra_starts(args.extra_starts))
    elif args.experiment == 'matrix':
        result = matrix_comparison(args.output, cfg, args.repeats, args.warmups,
                                   args.problem, args.method, args.start_index, progress,
                                   extra_starts=parse_extra_starts(args.extra_starts))
    else:
        if not args.first or not args.second:
            parser.error('Для compare потрібні --first і --second')
        result = compare_environments(args.output, args.first, args.second)
    print('Звіт:', args.output, '/ REPORT.md')
    print('Стан:', 'неповний' if result.get('cancelled') else 'завершено')


if __name__ == '__main__':
    main()

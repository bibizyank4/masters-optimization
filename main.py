"""Запуск: python main.py --help"""
import argparse
import sys
from optimizer.experiments import run_suite
from optimizer.methods import Config, METHODS, METHOD_GROUPS
from optimizer.starts import parse_extra_starts


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='Порівняння багатокрокових методів мінімізації')
    parser.add_argument('--output', default='results', help='Каталог для результатів')
    parser.add_argument('--problem', default='all', help='Назва задачі або назва:розмірність, наприклад extended_rosenbrock:10')
    parser.add_argument('--method', choices=('all', *METHODS, *METHOD_GROUPS), default='all')
    parser.add_argument('--extra-starts', default='', help='Точки через ;, наприклад "(-1,1)*10;(0,0)*10"')
    parser.add_argument('--start-index', type=int, default=None, help='Номер початкової точки, починаючи з 0')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=2, help='Кількість прогрівів перед кожною серією (0 — вимкнути)')
    parser.add_argument('--max-iterations', type=int, default=500)
    parser.add_argument('--grad-tol', type=float, default=1e-8)
    parser.add_argument('--condition-limit', type=float, default=1e14)
    parser.add_argument('--b0', choices=('hessian', 'identity'), default='hessian')
    parser.add_argument('--bfgs-strategy', choices=('skip', 'damp'), default='skip',
                        help='Пропуск або демпфування Пауелла для BFGS')
    parser.add_argument('--bfgs-min-curvature', type=float, default=1e-10)
    parser.add_argument('--broyden-restart-interval', type=int, default=0,
                        help='0 — формула із завдання; >0 — окремий варіант із гессіаном через N ітерацій')
    parser.add_argument('--correction-search', choices=('monotone', 'armijo'), default='monotone',
                        help='Пошук кроку для коректора із середньою точкою')
    args = parser.parse_args()
    cfg = Config(grad_tol=args.grad_tol, max_iterations=args.max_iterations,
                 condition_limit=args.condition_limit, b0=args.b0, correction_search=args.correction_search,
                 bfgs_strategy=args.bfgs_strategy, bfgs_min_curvature=args.bfgs_min_curvature,
                 broyden_restart_interval=args.broyden_restart_interval)
    run_suite(args.output, cfg, args.repeats, args.problem, args.method, args.start_index,
              warmups=args.warmups, extra_starts=parse_extra_starts(args.extra_starts))


if __name__ == '__main__':
    main()

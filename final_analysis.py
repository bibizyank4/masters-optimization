"""Одноразовий комплексний експеримент: python final_analysis.py --help."""
import argparse
import sys

from optimizer.final_analysis import final_analysis


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='Комплексний аналіз усіх вбудованих задач')
    parser.add_argument('--output', required=True, help='Нова або порожня папка')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--preview', action='store_true',
                        help='Швидка перевірка на одній квадратичній задачі; це не повний аналіз')
    args = parser.parse_args()
    def progress(done, total, row):
        print(f"{done}/{total}: {row['problem']} · {row['method']}", flush=True)
    kwargs = {'selection': 'quadratic', 'start_index': 0, 'tolerances': (1e-8,),
              'include_sensitivity': False} if args.preview else {}
    result = final_analysis(args.output, args.repeats, args.warmups, progress, **kwargs)
    print('Стан:', result['status'])
    print('Звіт:', args.output, '/ REPORT.md')
    if result['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

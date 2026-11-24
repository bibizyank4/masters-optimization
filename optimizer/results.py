"""Читання збереженої серії для інтерфейсу і повторного перегляду."""
import csv
import json
from collections import defaultdict
from pathlib import Path

from .problems import problem_suite

PROBLEM_NAMES = {'quadratic': 'Квадратична', 'rosenbrock': 'Розенброка',
                 'beale': 'Біла', 'powell_singular': 'Пауелла (сингулярна)',
                 'himmelblau': 'Гіммельблау', 'extended_rosenbrock': 'Розширена Розенброка',
                 'course_rosenbrock': 'Розенброка з курсової'}
METHOD_NAMES = {'gradient_correction': 'Коректор H⁻¹', 'qn_gradient_correction': 'Коректор · Бройден',
                'bfgs_gradient_correction': 'Коректор · BFGS',
                'three_step': 'Трикроковий', 'qn_three_step': 'Трикроковий · Бройден',
                'bfgs_three_step': 'Трикроковий · BFGS',
                'two_step_newton': 'Двокроковий Ньютона', 'qn_two_step_newton': 'Двокроковий · Бройден',
                'bfgs_two_step_newton': 'Двокроковий · BFGS'}
STATUS_NAMES = {'stationary': 'Стаціонарна точка', 'max_iterations': 'Ліміт ітерацій',
                'ill_conditioned_matrix': 'Погано обумовлена матриця', 'singular_matrix': 'Вироджена матриця',
                'not_descent': 'Напрям не є напрямом спуску', 'line_search_failed': 'Крок не знайдено',
                'stagnation': 'Застій', 'nonfinite_value': 'Нескінченне значення',
                'numerical_error': 'Чисельна помилка', 'line_roots_failed': 'Помилка коренів на прямій',
                'broyden_update_failed': 'Помилка оновлення Бройдена',
                'bfgs_update_failed': 'Помилка оновлення BFGS',
                'bfgs_initial_not_positive_definite': 'B₀ для BFGS не додатно визначена'}
TYPE_NAMES = {'not_stationary': 'критерію не досягнуто',
              'positive_definite_hessian': 'гессіан додатно визначений',
              'negative_hessian_eigenvalue': 'є від’ємне власне значення — не мінімум',
              'inconclusive_hessian': 'гессіан: висновок неоднозначний'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        meta_path = self.path / 'metadata.json'
        if not meta_path.is_file():
            raise ValueError('У папці не знайдено metadata.json')
        self.metadata = json.loads(meta_path.read_text(encoding='utf-8-sig'))
        self.rows = read_csv(self.path / 'summary.csv')
        entries = read_csv(self.path / 'history.csv')
        if not self.rows:
            raise ValueError('Таблиця результатів порожня')
        required = {'case_id', 'problem', 'dimension', 'start_index', 'method', 'status',
                    'iterations', 'f', 'grad_norm', 'x', 'stationary_type'}
        if not required.issubset(self.rows[0]):
            raise ValueError('У summary.csv відсутні потрібні стовпці')
        self.legacy = self.metadata.get('corrector') != 'inverse_matrix_midpoint_gradient'
        self.by_id = {}
        for row in self.rows:
            if row['case_id'] in self.by_id:
                raise ValueError('У summary.csv повторюються ідентифікатори запусків')
            self.by_id[row['case_id']] = row
        self.histories = defaultdict(list)
        for row in entries:
            if row['case_id'] not in self.by_id:
                raise ValueError('Історія посилається на відсутній результат')
            row['point'] = json.loads(row['x'])
            row['k'] = int(row['iteration'])
            row['f_value'] = float(row['f'])
            row['gradient_norm'] = float(row['grad_norm'])
            self.histories[row['case_id']].append(row)
        for rows in self.histories.values():
            rows.sort(key=lambda r: r['k'])
        self.problems = {(p.name, p.dimension): p for p in problem_suite()}

    def problem_label(self, row):
        return f"{PROBLEM_NAMES.get(row['problem'], row['problem'])} · n={row['dimension']}"

    def method_label(self, row):
        method = row['method']
        if self.legacy and method in ('gradient_correction', 'qn_gradient_correction'):
            return 'Коректор ∇f · v1' if method == 'gradient_correction' else 'Коректор ∇f · B · v1'
        return METHOD_NAMES.get(method, method)

    def comparison(self, row):
        return [other for other in self.rows if other['problem'] == row['problem']
                and other['dimension'] == row['dimension'] and other['start_index'] == row['start_index']]

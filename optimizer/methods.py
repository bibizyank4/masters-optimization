"""Багатокрокові методи; у версії 2 коректор масштабується H⁻¹ або B⁻¹."""
from dataclasses import dataclass, field
import time
import numpy as np


METHODS = ('gradient_correction', 'three_step', 'two_step_newton',
           'qn_gradient_correction', 'qn_three_step', 'qn_two_step_newton',
           'bfgs_gradient_correction', 'bfgs_three_step', 'bfgs_two_step_newton')
METHOD_GROUPS = {'three_step_family': ('three_step', 'qn_three_step', 'bfgs_three_step'),
                 'classical_family': ('gradient_correction', 'three_step', 'two_step_newton'),
                 'broyden_family': ('qn_gradient_correction', 'qn_three_step', 'qn_two_step_newton'),
                 'bfgs_family': ('bfgs_gradient_correction', 'bfgs_three_step', 'bfgs_two_step_newton')}


def expand_methods(selection):
    if selection == 'all':
        return METHODS
    if selection in METHOD_GROUPS:
        return METHOD_GROUPS[selection]
    if selection in METHODS:
        return (selection,)
    raise ValueError(f'Невідомий метод: {selection}')


class NumericalFailure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass
class Config:
    grad_tol: float = 1e-8
    max_iterations: int = 500
    armijo_c: float = 1e-4
    backtrack: float = .5
    max_backtracks: int = 60
    condition_limit: float = 1e14
    b0: str = 'hessian'  # або identity; за замовчуванням приклад із завдання
    correction_search: str = 'monotone'
    bfgs_strategy: str = 'skip'
    bfgs_min_curvature: float = 1e-10
    broyden_restart_interval: int = 0

    def validate(self):
        if not (self.grad_tol > 0 and self.max_iterations > 0 and 0 < self.armijo_c < 1
                and 0 < self.backtrack < 1 and self.max_backtracks > 0 and self.condition_limit > 1
                and self.b0 in ('hessian', 'identity')
                and self.correction_search in ('monotone', 'armijo')
                and self.bfgs_strategy in ('skip', 'damp')
                and 0 < self.bfgs_min_curvature < .2
                and isinstance(self.broyden_restart_interval, int)
                and not isinstance(self.broyden_restart_interval, bool)
                and self.broyden_restart_interval >= 0):
            raise ValueError('Некоректні параметри алгоритму')


@dataclass
class Evaluator:
    problem: object
    f_calls: int = 0
    grad_calls: int = 0
    hess_calls: int = 0
    polynomial_calls: int = 0
    linear_solves: int = 0

    def f(self, x):
        self.f_calls += 1
        return self.problem.value(x)

    def grad(self, x):
        self.grad_calls += 1
        return self.problem.derivatives(x, 1)

    def hess(self, x):
        self.hess_calls += 1
        return self.problem.derivatives(x, 2)


@dataclass
class Result:
    method: str
    status: str
    message: str
    x: np.ndarray
    f: float
    grad_norm: float
    iterations: int
    elapsed: float
    counts: dict
    history: list = field(default_factory=list)
    line_diagnostics: list = field(default_factory=list)


def checked_solve(matrix, rhs, ev, config):
    if not np.all(np.isfinite(matrix)):
        raise NumericalFailure('nonfinite_matrix', 'Матриця містить нескінченні значення')
    cond = np.linalg.cond(matrix)
    if not np.isfinite(cond) or cond > config.condition_limit:
        raise NumericalFailure('ill_conditioned_matrix', f'Число обумовленості матриці: {cond:.3g}')
    try:
        ev.linear_solves += 1
        result = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise NumericalFailure('singular_matrix', str(exc)) from exc
    if not np.all(np.isfinite(result)):
        raise NumericalFailure('nonfinite_step', 'Некоректний розв’язок системи')
    return result


def divided_difference(ev, x, u):
    """∫₀¹ H(x+t(u−x))dt: точна для наших поліномів квадратура Ґаусса.

    Максимальний степінь гессіана на прямій дорівнює 6, тому чотири вузли
    інтегрують його точно з точністю арифметики float64.
    """
    nodes, weights = np.polynomial.legendre.leggauss(4)
    out = np.zeros((x.size, x.size))
    for node, weight in zip(nodes, weights):
        out += (weight / 2) * ev.hess(x + ((node + 1) / 2) * (u - x))
    return out


def armijo(ev, x, fx, gx, direction, cfg):
    slope = float(gx @ direction)
    if not np.isfinite(slope) or slope >= 0:
        raise NumericalFailure('not_descent', 'Напрям не є напрямом спуску; крок Арміхо не визначено')
    alpha = 1.0
    for _ in range(cfg.max_backtracks):
        candidate = x + alpha * direction
        if np.array_equal(candidate, x):
            break
        with np.errstate(over='ignore', invalid='ignore'):
            try:
                f_candidate = ev.f(candidate)
            except ValueError:
                f_candidate = np.inf
        if np.isfinite(f_candidate) and f_candidate <= fx + cfg.armijo_c * alpha * slope:
            return alpha, candidate, f_candidate
        alpha *= cfg.backtrack
    raise NumericalFailure('line_search_failed', 'Пошук кроку Арміхо не знайшов допустимого кроку')


def monotone_search(ev, x, fx, direction, cfg):
    """Просте спадання з курсової; не приймає крок, який не зменшує f."""
    alpha = 1.0
    for _ in range(cfg.max_backtracks):
        if alpha < 1e-10:
            break
        candidate = x + alpha * direction
        if np.array_equal(candidate, x):
            break
        with np.errstate(over='ignore', invalid='ignore'):
            try:
                f_candidate = ev.f(candidate)
            except ValueError:
                f_candidate = np.inf
        if np.isfinite(f_candidate) and f_candidate < fx:
            return alpha, candidate, f_candidate
        alpha *= cfg.backtrack
    raise NumericalFailure('line_search_failed', 'Не знайдено кроку строгого спадання')


def polynomial_line_minimum(ev, u, v):
    """Чисельний глобальний пошук на всій прямій для тестових поліномів.

    Досліджує всі чисельно знайдені дійсні корені похідної полінома. Корені
    отримують через власні значення companion-матриці, отже це не сертифікат.
    Масштабуємо напрям до одиничної довжини для кращої обумовленості.
    """
    d = v - u
    length = float(np.linalg.norm(d))
    if not np.isfinite(length):
        raise NumericalFailure('nonfinite_step', 'Нескінченний напрям пошуку на прямій')
    if length == 0:
        return u.copy(), ev.f(u), {'line_root_count': 0, 'line_derivative': 0.0, 'line_degenerate': True}
    direction = d / length
    ev.polynomial_calls += 1
    poly = ev.problem.line_polynomial(u, direction)
    coef = np.asarray(poly.coef, dtype=float)
    if not np.all(np.isfinite(coef)):
        raise NumericalFailure('line_polynomial_failed', 'Некоректні коефіцієнти полінома на прямій')
    # Відкидаємо лише ТОЧНІ нулі; малі коефіцієнти можуть визначати далекі мінімуми.
    coef = np.trim_zeros(coef, 'b')
    if len(coef) <= 1:
        return u.copy(), ev.f(u), {'line_root_count': 0, 'line_derivative': 0.0, 'line_degenerate': True}
    if (len(coef) - 1) % 2 or coef[-1] <= 0:
        raise NumericalFailure('line_polynomial_failed', 'Поліном на прямій не має очікуваного парного степеня')
    deriv = np.polynomial.Polynomial(coef).deriv()
    try:
        roots = deriv.roots()
    except np.linalg.LinAlgError as exc:
        raise NumericalFailure('line_roots_failed', str(exc)) from exc
    real_roots = [float(z.real) for z in roots if abs(z.imag) <= 1e-7 * max(1., abs(z.real))]
    if not real_roots:
        raise NumericalFailure('line_roots_failed', 'Не знайдено жодного дійсного стаціонарного кореня')
    candidates = []
    # Пряме обчислення f не використовує поліном із потенційно нестійким значенням.
    for t in real_roots:
        with np.errstate(over='ignore', invalid='ignore'):
            try:
                pt = u + t * direction
                value = ev.f(pt)
            except ValueError:
                continue
        if np.isfinite(value):
            candidates.append((value, t, pt))
    if not candidates:
        raise NumericalFailure('line_roots_failed', 'Значення функції в коренях не є скінченними')
    value, t, pt = min(candidates, key=lambda item: item[0])
    # Захист від чисельно втраченої найкращої точки: не видаємо її за мінімум.
    fu, fv = ev.f(u), ev.f(v)
    if value > min(fu, fv) + 1e-10 * max(1., abs(fu), abs(fv)):
        raise NumericalFailure('line_roots_failed', 'Корені не забезпечили мінімум відносно u та v')
    return pt, value, {'line_root_count': len(real_roots), 'line_derivative': float(deriv(t)), 'line_degenerate': False}


def broyden_update(b, s, y):
    """Формула (8), без симетризації та без заміни на BFGS."""
    denominator = float(s @ s)
    if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
        raise NumericalFailure('broyden_update_failed', 'Нульовий або надто малий знаменник оновлення Бройдена')
    updated = b + np.outer(y - b @ s, s) / denominator
    if not np.all(np.isfinite(updated)):
        raise NumericalFailure('broyden_update_failed', 'Нескінченні значення в оновленні Бройдена')
    return updated


def bfgs_update(b, s, y, strategy='skip', min_curvature=1e-10):
    """Оновлення прямої матриці B; damp — демпфування Пауелла з порогом 0.2 sᵀBs."""
    bs = b @ s
    sbs = float(s @ bs)
    sy = float(s @ y)
    scale = float(np.linalg.norm(s) * np.linalg.norm(y))
    if not np.isfinite(sbs) or sbs <= 0:
        raise NumericalFailure('bfgs_update_failed', 'Матриця B втратила додатну визначеність')
    if not np.isfinite(sy) or not np.isfinite(scale):
        raise NumericalFailure('bfgs_update_failed', 'Нескінченні дані оновлення BFGS')
    damped = False
    if strategy == 'damp' and sy < .2 * sbs:
        theta = .8 * sbs / (sbs - sy)
        y = theta * y + (1 - theta) * bs
        sy = float(s @ y)
        scale = float(np.linalg.norm(s) * np.linalg.norm(y))
        damped = True
    if sy <= min_curvature * scale or sy <= np.finfo(float).tiny:
        return b, 'skipped'
    updated = b - np.outer(bs, bs) / sbs + np.outer(y, y) / sy
    updated = (updated + updated.T) / 2
    if not np.all(np.isfinite(updated)):
        raise NumericalFailure('bfgs_update_failed', 'Нескінченні значення в оновленні BFGS')
    try:
        np.linalg.cholesky(updated)
    except np.linalg.LinAlgError as exc:
        raise NumericalFailure('bfgs_update_failed', 'BFGS втратив додатну визначеність') from exc
    return updated, 'damped' if damped else 'applied'


def minimize(problem, method, start, config=None):
    if method not in METHODS:
        raise ValueError(f'Невідомий метод: {method}')
    cfg = config or Config()
    cfg.validate()
    x = problem._check(start).copy()
    ev = Evaluator(problem)
    history, line_info = [], []
    status, message = 'max_iterations', 'Досягнуто ліміт ітерацій'
    bfgs_counts = {'bfgs_applied': 0, 'bfgs_skipped': 0, 'bfgs_damped': 0}
    broyden_restarts = 0
    is_bfgs = method.startswith('bfgs_')
    is_quasi = method.startswith('qn_') or is_bfgs
    fx, gnorm, iteration = float('nan'), float('nan'), 0
    start_time = time.perf_counter()
    try:
        fx = ev.f(x)
        g = ev.grad(x)
        gnorm = float(np.linalg.norm(g))
        if not np.isfinite(fx) or not np.isfinite(gnorm):
            raise NumericalFailure('nonfinite_value', 'Нескінченне значення функції або градієнта')
        history.append({'iteration': 0, 'x': x.copy(), 'f': fx, 'grad_norm': gnorm, 'alpha': None})
        # Обчислюємо B0 лише за потреби і після перевірки початкової стаціонарності.
        b = None
        for k in range(cfg.max_iterations):
            if gnorm <= cfg.grad_tol:
                status, message = 'stationary', 'Досягнуто порога норми градієнта; тип точки слід перевірити'
                break
            if is_quasi and b is None:
                b = ev.hess(x) if cfg.b0 == 'hessian' else np.eye(x.size)
                if is_bfgs:
                    try:
                        np.linalg.cholesky(b)
                    except np.linalg.LinAlgError as exc:
                        raise NumericalFailure('bfgs_initial_not_positive_definite',
                                               'B₀ для BFGS не є додатно визначеною; виберіть B₀=I') from exc
            elif method.startswith('qn_') and cfg.broyden_restart_interval and k % cfg.broyden_restart_interval == 0:
                b = ev.hess(x)
                broyden_restarts += 1
                history[-1]['broyden_restart_next'] = True
            alpha = None
            diagnostic = None
            if method.endswith('gradient_correction'):
                matrix = b if is_quasi else ev.hess(x)
                u = x - checked_solve(matrix, g, ev, cfg)
                mid_grad = ev.grad((x + u) / 2)
                direction = -checked_solve(matrix, mid_grad, ev, cfg)
                if cfg.correction_search == 'monotone':
                    alpha, x_new, f_new = monotone_search(ev, x, fx, direction, cfg)
                else:
                    alpha, x_new, f_new = armijo(ev, x, fx, g, direction, cfg)
            elif method.endswith('two_step_newton'):
                matrix = b if is_quasi else ev.hess(x)
                u = x - checked_solve(matrix, g, ev, cfg)
                gu = ev.grad(u)
                x_new = u - checked_solve(matrix, gu, ev, cfg)
                f_new = ev.f(x_new)
            else:
                alpha, u, _ = armijo(ev, x, fx, g, -g, cfg)
                matrix = b if is_quasi else divided_difference(ev, x, u)
                v = x - checked_solve(matrix, g, ev, cfg)
                x_new, f_new, diagnostic = polynomial_line_minimum(ev, u, v)
            if not np.all(np.isfinite(x_new)) or not np.isfinite(f_new):
                raise NumericalFailure('nonfinite_value', 'Отримано нескінченне значення точки або функції')
            g_new = ev.grad(x_new)
            next_norm = float(np.linalg.norm(g_new))
            if not np.isfinite(next_norm):
                raise NumericalFailure('nonfinite_value', 'Отримано нескінченний градієнт')
            if np.array_equal(x_new, x) and next_norm > cfg.grad_tol:
                raise NumericalFailure('stagnation', 'Ітерація не змінила точку')
            s, y = x_new - x, g_new - g
            x, fx, g, gnorm = x_new, f_new, g_new, next_norm
            iteration = k + 1
            history.append({'iteration': iteration, 'x': x.copy(), 'f': fx, 'grad_norm': gnorm, 'alpha': alpha})
            if diagnostic is not None:
                line_info.append(diagnostic)
            restart_next = (method.startswith('qn_') and cfg.broyden_restart_interval
                            and (k + 1) % cfg.broyden_restart_interval == 0)
            if is_quasi and next_norm > cfg.grad_tol and k + 1 < cfg.max_iterations and not restart_next:
                if is_bfgs:
                    b, update_status = bfgs_update(b, s, y, cfg.bfgs_strategy, cfg.bfgs_min_curvature)
                    bfgs_counts['bfgs_' + update_status] += 1
                else:
                    b = broyden_update(b, s, y)
        else:
            if gnorm <= cfg.grad_tol:
                status, message = 'stationary', 'Досягнуто порога норми градієнта; тип точки слід перевірити'
    except NumericalFailure as exc:
        status, message = exc.code, str(exc)
    except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError) as exc:
        status, message = 'numerical_error', str(exc)
    elapsed = time.perf_counter() - start_time
    return Result(method, status, message, x, fx, gnorm, iteration, elapsed,
                  {'f': ev.f_calls, 'grad': ev.grad_calls, 'hess': ev.hess_calls,
                   'line_polynomial': ev.polynomial_calls, 'linear_solves': ev.linear_solves,
                   'broyden_restarts': broyden_restarts, **bfgs_counts}, history, line_info)

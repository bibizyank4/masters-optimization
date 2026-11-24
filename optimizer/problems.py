"""Тестові поліноми та їхні точні похідні через диференціювання виразів.

Усі задачі подані як сума квадратів поліноміальних нев'язок. Клас Jet
автоматично диференціює ці нев'язки до першого або другого порядку.
Це аналітичні, а не скінченнорізницеві похідні.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Polynomial


def _trim_coefficients(coef):
    end = len(coef)
    while end > 1 and coef[end - 1] == 0:
        end -= 1
    return coef[:end]


def _add_coefficients(left, right, subtract=False):
    """Такий самий порядок операцій, як у polynomial.polyadd/polysub."""
    left, right = _trim_coefficients(left), _trim_coefficients(right)
    if len(left) > len(right):
        result = left.copy()
        if subtract:
            result[:len(right)] -= right
        else:
            result[:len(right)] += right
    else:
        result = (-right if subtract else right).copy()
        result[:len(left)] += left
    return _trim_coefficients(result)


def _multiply_coefficients(left, right):
    return _trim_coefficients(np.convolve(_trim_coefficients(left), _trim_coefficients(right)))


def _rosenbrock_line_coefficients(x, direction, scale):
    """Коефіцієнти для суми пар Розенброка без проміжних Polynomial-об'єктів.

    Порядок множень і додавань відповідає виразу в residuals/expression.
    """
    result = np.array([0.])
    one, multiplier = np.array([1.]), np.array([float(scale)])
    for i in range(0, len(x), 2):
        a = np.array([x[i], direction[i]])
        b = np.array([x[i+1], direction[i+1]])
        r1 = _multiply_coefficients(multiplier, _add_coefficients(b, _multiply_coefficients(a, a), subtract=True))
        r2 = _add_coefficients(one, a, subtract=True)
        result = _add_coefficients(result, _multiply_coefficients(r1, r1))
        result = _add_coefficients(result, _multiply_coefficients(r2, r2))
    return result


class Jet:
    def __init__(self, val, grad, hess=None):
        self.val = float(val)
        self.grad = grad
        self.hess = hess

    def __add__(self, other):
        if not isinstance(other, Jet):
            return Jet(self.val + other, self.grad.copy(), None if self.hess is None else self.hess.copy())
        return Jet(self.val + other.val, self.grad + other.grad,
                   None if self.hess is None else self.hess + other.hess)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.val, -self.grad, None if self.hess is None else -self.hess)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if not isinstance(other, Jet):
            return Jet(self.val * other, self.grad * other,
                       None if self.hess is None else self.hess * other)
        h = None
        if self.hess is not None:
            h = (self.hess * other.val + other.hess * self.val
                 + np.outer(self.grad, other.grad) + np.outer(other.grad, self.grad))
        return Jet(self.val * other.val, self.grad * other.val + other.grad * self.val, h)

    __rmul__ = __mul__

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError('Підтримуються лише цілі невід’ємні степені')
        if power == 0:
            return Jet(1, np.zeros_like(self.grad), None if self.hess is None else np.zeros_like(self.hess))
        out = self
        for _ in range(power - 1):
            out = out * self
        return out


@dataclass(frozen=True)
class Problem:
    name: str
    dimension: int
    starts: tuple
    minima: tuple

    def residuals(self, x):
        if self.name == 'quadratic':
            return [x[0], np.sqrt(10) * x[1]]
        if self.name in ('rosenbrock', 'extended_rosenbrock', 'course_rosenbrock'):
            scale = 1 if self.name == 'course_rosenbrock' else 10
            return [r for i in range(0, self.dimension, 2)
                    for r in (scale * (x[i+1] - x[i]**2), 1 - x[i])]
        if self.name == 'beale':
            return [1.5-x[0]+x[0]*x[1], 2.25-x[0]+x[0]*x[1]**2,
                    2.625-x[0]+x[0]*x[1]**3]
        if self.name == 'powell_singular':
            return [x[0]+10*x[1], np.sqrt(5)*(x[2]-x[3]),
                    (x[1]-2*x[2])**2, np.sqrt(10)*(x[0]-x[3])**2]
        if self.name == 'himmelblau':
            return [x[0]**2+x[1]-11, x[0]+x[1]**2-7]
        raise ValueError(f'Невідома задача: {self.name}')

    def expression(self, x):
        return sum(r*r for r in self.residuals(x))

    def _check(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dimension,) or not np.all(np.isfinite(x)):
            raise ValueError(f'Очікується скінченний вектор розмірності {self.dimension}')
        return x

    def value(self, x):
        return float(self.expression(self._check(x)))

    def derivatives(self, x, order=1):
        x = self._check(x)
        if order not in (1, 2):
            raise ValueError('order має дорівнювати 1 або 2')
        basis = np.eye(self.dimension)
        jets = [Jet(x[i], basis[i], np.zeros((self.dimension, self.dimension)) if order == 2 else None)
                for i in range(self.dimension)]
        out = self.expression(jets)
        return out.grad if order == 1 else out.hess

    def line_polynomial(self, point, direction):
        """Коефіцієнти f(point + t*direction), за зростанням степенів."""
        x = self._check(point)
        d = self._check(direction)
        if self.name in ('rosenbrock', 'extended_rosenbrock', 'course_rosenbrock'):
            scale = 1 if self.name == 'course_rosenbrock' else 10
            return Polynomial(_rosenbrock_line_coefficients(x, d, scale))
        return self.expression([Polynomial([a, b]) for a, b in zip(x, d)])


def problem_suite():
    q = Problem('quadratic', 2, ((2., 2.), (1., .1)), ((0., 0.),))
    r = Problem('rosenbrock', 2, ((-1.2, 1.), (.9, .81)), ((1., 1.),))
    b = Problem('beale', 2, ((1., 1.), (2.9, .49)), ((3., .5),))
    p = Problem('powell_singular', 4, ((3., -1., 0., 1.), (.3, -.1, 0., .1), (1., 1., 1., 1.)), ((0., 0., 0., 0.),))
    # Точніші значення коренів поліноміальної системи; використовуються лише для оцінки похибки.
    h = Problem('himmelblau', 2, ((2., 2.), (-3., 3.), (-4., -3.), (4., -2.), (1., 1.)),
                ((3., 2.), (-2.805118086952745, 3.131312518250573),
                 (-3.779310253377747, -3.28318599128617),
                 (3.584428340330492, -1.848126526964404)))
    course_r = Problem('course_rosenbrock', 2, ((-1., 2.),), ((1., 1.),))
    problems = [q, r, b, p, h, course_r]
    for n in (10, 20):
        problems.append(Problem('extended_rosenbrock', n, (tuple([-1.2, 1.] * (n//2)),),
                                (tuple([1.] * n),)))
    return problems

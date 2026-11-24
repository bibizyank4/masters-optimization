"""Читання і перевірка додаткових початкових точок."""
import re
import numpy as np


def parse_extra_starts(text):
    """Вектори через ;, наприклад (-1,1)*10; (0,0)*10 або 1,2,3."""
    if not text or not text.strip():
        return []
    vectors = []
    for part in text.split(';'):
        part = part.strip()
        if not part:
            raise ValueError('Порожня додаткова точка між роздільниками ;')
        match = re.fullmatch(r'\s*\(([^()]*)\)\s*\*\s*(\d+)\s*', part)
        if match:
            content, times = match.group(1), int(match.group(2))
            if not 1 <= times <= 1000:
                raise ValueError('Кількість повторень координат має бути від 1 до 1000')
        else:
            content, times = part.strip('()[] '), 1
        try:
            vector = [float(x.strip()) for x in content.split(',')] * times
        except ValueError as exc:
            raise ValueError('Координати задаються числами через кому; точки — через ;') from exc
        if not vector or not all(np.isfinite(v) for v in vector):
            raise ValueError('Координати точок мають бути скінченними')
        vectors.append(vector)
    return vectors


def validate_extra_starts(suite, vectors):
    if not vectors:
        return []
    if len(suite) != 1:
        raise ValueError('Для додаткових стартів виберіть одну задачу з конкретною розмірністю')
    problem = suite[0]
    result, seen = [], {tuple(float(x) for x in p) for p in problem.starts}
    for vector in vectors:
        point = problem._check(vector).tolist()
        if tuple(point) in seen:
            raise ValueError('Додаткова точка повторює вбудовану або іншу додаткову точку')
        seen.add(tuple(point))
        result.append(point)
    return result

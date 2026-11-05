"""Статистические тесты для сравнения архитектур.

Модуль предоставляет функции для статистического сравнения метрик
continual learning между архитектурами, а также агрегации метрик
по сидам.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Any


def compare_architectures(
    metric_a: list[float],
    metric_b: list[float],
    test: str = "wilcoxon",
) -> tuple[float, float]:
    """Сравнивает две архитектуры по метрике статистическим тестом.

    Выполняет межгрупповое сравнение значений метрики (например,
    forgetting) для двух архитектур. При n=3 (seeds из
    ``config/training.yaml``) тест всё равно вычисляется — малая
    выборка отражается в высоком p-value.

    Для ``wilcoxon`` используется :func:`scipy.stats.ranksums`
    (Wilcoxon rank-sum / Mann-Whitney U для независимых выборок),
    а не парный :func:`scipy.stats.wilcoxon`.

    Args:
        metric_a: Значения метрики для архитектуры A (по сидам).
            Например, ``[1.8, 1.5, 2.0]`` — forgetting для трёх сидов.
        metric_b: Значения метрики для архитектуры B (по сидам).
        test: Название теста. Допустимые значения:
            ``"wilcoxon"`` — rank-sum (по умолчанию),
            ``"ttest"`` — независимый t-тест.

    Returns:
        Кортеж ``(statistic, p_value)`` — статистика теста и
        двустороннее p-value. Оба значения — ``float``.

    Raises:
        ValueError: Если ``test`` не из ``{"wilcoxon", "ttest"}``.
        ValueError: Если хотя бы один из списков пуст.

    Examples:
        >>> compare_architectures([1.8, 1.5, 2.0], [3.1, 2.9, 3.3])
        (..., ...)  # statistic < 0, p_value < 0.05 при большом эффекте
        >>> compare_architectures([1.0, 1.1, 1.0], [1.0, 1.1, 1.0], test="ttest")
        (0.0, 1.0)

    """
    # Проверка входных данных — списки не должны быть пустыми
    if len(metric_a) == 0 or len(metric_b) == 0:
        raise ValueError("metric_a и metric_b не должны быть пустыми")

    # Приведение к numpy-массивам для унификации типов
    # (списки, кортежи, numpy arrays — всё приводится к float)
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)

    # Выбор статистического теста
    if test == "wilcoxon":
        # Wilcoxon rank-sum для независимых выборок
        # (эквивалент Mann-Whitney U, подходит для n=3)
        result = stats.ranksums(a, b)
    elif test == "ttest":
        # Независимый двухвыборочный t-тест (Student's t)
        # equal_var=True по умолчанию — классический вариант
        result = stats.ttest_ind(a, b)
    else:
        raise ValueError(
            f"Неизвестный тест '{test}'. Допустимые значения: 'wilcoxon', 'ttest'."
        )

    # Приведение к float для стабильного типа возвращаемого значения
    # (scipy может вернуть np.float64 или nan)
    statistic = float(result.statistic) if not np.isnan(result.statistic) else float("nan")
    p_value = float(result.pvalue) if not np.isnan(result.pvalue) else float("nan")

    # Для t-теста при нулевой дисперсии scipy может вернуть nan;
    # возвращаем как есть — вызывающий код интерпретирует p=nan как отсутствие данных
    return statistic, p_value


def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Агрегирует метрики по прогонам (сидам).

    Вычисляет среднее, стандартное отклонение и размер выборки для
    каждой числовой метрики. Поддерживает два формата входа:

    * ``list[dict]`` — список словарей, где каждый словарь — метрики
      одного прогона (сида), например
      ``[{"forgetting": 1.8, "reward": -2.3}, {"forgetting": 1.5, ...}]``.
      Для каждого ключа собираются значения по всем словарям и
      считается агрегация.
    * ``list[float]`` — плоский список значений одной метрики
      (для обратной совместимости).

    Args:
        metrics: Список метрик. Элементы — словари с числовыми
            значениями или числа (flat-режим).

    Returns:
        Словарь с агрегатами. В режиме ``list[dict]`` ключи
        соответствуют исходным метрикам, значения — словари
        ``{"mean": float, "std": float, "n": int}``.
        В flat-режиме возвращается один словарь
        ``{"mean": ..., "std": ..., "n": ...}``.
        Для пустого входа возвращается ``{}``.

    Examples:
        >>> aggregate_metrics([{"forgetting": 1.8}, {"forgetting": 2.0}])
        {'forgetting': {'mean': 1.9, 'std': ..., 'n': 2}}
        >>> aggregate_metrics([1.0, 2.0, 3.0])
        {'mean': 2.0, 'std': ..., 'n': 3}

    """
    # Обработка пустого входа — возвращаем пустой словарь
    if not metrics:
        return {}

    # Flat-режим: список чисел (первый элемент — не словарь)
    first = metrics[0]
    if not isinstance(first, dict):
        # Считаем значения числовыми, игнорируем нечисловые
        values = [float(v) for v in metrics if isinstance(v, (int, float, np.floating))]
        # Фильтруем NaN
        values = [v for v in values if not np.isnan(v)]
        n = len(values)
        if n == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        mean = float(np.mean(values))
        # Выборочное std (ddof=1) при n>1, иначе 0.0
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        return {"mean": mean, "std": std, "n": n}

    # Режим списка словарей — собираем все ключи
    all_keys: set[str] = set()
    for entry in metrics:
        if isinstance(entry, dict):
            all_keys.update(entry.keys())

    result: dict[str, Any] = {}
    for key in sorted(all_keys):
        # Собираем числовые значения ключа по всем словарям
        values_for_key: list[float] = []
        for entry in metrics:
            if not isinstance(entry, dict) or key not in entry:
                continue
            val = entry[key]
            # Поддерживаем только скалярные числовые метрики
            if isinstance(val, (int, float, np.floating, np.integer)):
                fval = float(val)
                if not np.isnan(fval):
                    values_for_key.append(fval)
            # Игнорируем нечисловые значения (строки, вложенные словари)

        n = len(values_for_key)
        if n == 0:
            # Нет числовых данных для ключа — пропускаем или ставим nan
            result[key] = {"mean": float("nan"), "std": float("nan"), "n": 0}
            continue

        mean = float(np.mean(values_for_key))
        std = float(np.std(values_for_key, ddof=1)) if n > 1 else 0.0
        result[key] = {"mean": mean, "std": std, "n": n}

    return result

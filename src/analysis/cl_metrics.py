"""Метрики continual learning: forgetting, forward transfer, final reward.

Модуль предоставляет функции для вычисления метрик катастрофического
забывания на основе словарей результатов прогонов. Поддерживает
робастный парсинг разных форматов JSON (см. ``docs/agent_rules.md``).
"""

from __future__ import annotations

from typing import Any, Dict

# Ключи, обозначающие первый и второй таски по умолчанию.
_DEFAULT_TASK1 = "task1_goto"
_DEFAULT_TASK2 = "task2_avoid"


def _extract_results_dict(results: Dict[str, Any]) -> Dict[str, Any]:
    """Извлекает внутренний словарь с метриками по задачам.

    Поддерживает два формата:
    * обёрнутый — ``{"results": {"task1_goto": {...}, ...}, ...}``
      (формат из ``docs/agent_rules.md``);
    * плоский — ``{"task1_goto": {...}, "task2_avoid": {...}}``
      (возврат ``train_continual``).

    Args:
        results: Словарь одного прогона (внешний JSON или уже
            внутренний ``results``).

    Returns:
        Словарь, содержащий блоки задач. Если ``results`` содержит
        ключ ``"results"`` со вложенным словарём задач — возвращается
        он, иначе исходный ``results``.
    """
    # Проверяем тип входа — должен быть словарём.
    if not isinstance(results, dict):
        return {}
    # Если есть вложенный ``results`` и он содержит хотя бы одну задачу —
    # считаем его основным источником.
    nested = results.get("results")
    if isinstance(nested, dict):
        # Эвристика: содержит ли вложенный словарь ключи задач.
        if any(k in nested for k in (_DEFAULT_TASK1, _DEFAULT_TASK2)):
            return nested
        # Если вложенный словарь не пуст и выглядит как метрики —
        # возвращаем его (покрывает случай кастомных имён задач).
        if len(nested) > 0 and any(isinstance(v, dict) for v in nested.values()):
            # Проверяем наличие характерных ключей внутри.
            for v in nested.values():
                if isinstance(v, dict) and any(
                    kk in v for kk in ("reward_after_task1", "reward_after_task2", "forgetting")
                ):
                    return nested
    # Иначе считаем что передан уже плоский формат.
    return results


def _get_task_block(results: Dict[str, Any], task_name: str) -> Dict[str, Any] | None:
    """Находит блок метрик для указанной задачи с робастным парсингом.

    Ищет в ``results["results"][task_name]`` и ``results[task_name]``.

    Args:
        results: Словарь прогона (внешний или плоский).
        task_name: Имя задачи, например ``"task1_goto"``.

    Returns:
        Словарь метрик задачи или ``None`` если не найден.
    """
    # Сначала пробуем извлечь внутренний словарь (если есть обёртка).
    inner = _extract_results_dict(results)
    # Прямой поиск в inner.
    if task_name in inner and isinstance(inner[task_name], dict):
        return inner[task_name]
    # Резерв: прямой поиск в исходном словаре (на случай если _extract
    # вернул плоский, но задача была во внешнем ``results``).
    if task_name in results and isinstance(results[task_name], dict):
        return results[task_name]  # type: ignore[return-value]
    # Дополнительный резерв: вложенный ``results`` без эвристики выше.
    nested = results.get("results")
    if isinstance(nested, dict) and task_name in nested and isinstance(nested[task_name], dict):
        return nested[task_name]  # type: ignore[return-value]
    return None


def _get_first_value(data: Dict[str, Any], keys: list[str]) -> Any | None:
    """Возвращает первое не-None значение по списку ключей-алиасов.

    Args:
        data: Словарь метрик задачи.
        keys: Список возможных имён ключа в порядке приоритета.

    Returns:
        Значение или ``None`` если ни один ключ не найден.
    """
    for k in keys:
        if k in data and data[k] is not None:
            return data[k]
    return None


def _find_any_task_with_keys(
    results: Dict[str, Any], required_keys: list[str]
) -> Dict[str, Any] | None:
    """Ищет любой блок задачи, содержащий все требуемые ключи.

    Используется как фолбэк, если имя задачи неизвестно.

    Args:
        results: Словарь прогона.
        required_keys: Ключи, которые должны присутствовать в блоке.

    Returns:
        Найденный блок или ``None``.
    """
    inner = _extract_results_dict(results)
    # Проверяем все словарные значения inner.
    for v in inner.values():
        if isinstance(v, dict) and all(k in v and v[k] is not None for k in required_keys):
            return v
    # Также проверяем верхний уровень на случай плоского формата без inner.
    for v in results.values():
        if isinstance(v, dict) and all(k in v and v[k] is not None for k in required_keys):
            return v
    return None


def compute_forgetting(results: Dict[str, Any]) -> float:
    """Вычисляет забывание: R(T1 после T1) - R(T1 после T2).

    Поддерживает форматы из ``docs/agent_rules.md`` и плоский формат
    ``train_continual``:

    * ``results["results"]["task1_goto"]["reward_after_task1"]``
    * ``results["task1_goto"]["reward_after_task1"]``

    Робастный парсинг: проверяет обе локации и несколько алиасов ключей.

    Args:
        results: Метрики одного прогона. Ожидаются ключи
            ``reward_after_task1`` и ``reward_after_task2`` внутри блока
            ``task1_goto`` (или любого блока, содержащего оба ключа).

    Returns:
        Forgetting (``float``): положительное значение означает забывание
        (ухудшение на T1 после обучения T2), отрицательное — улучшение.

    Raises:
        KeyError: Если не найден блок задачи или отсутствуют требуемые
            ключи наград.
        TypeError: Если ``results`` не является словарём.
    """
    # Валидируем тип входа.
    if not isinstance(results, dict):
        raise TypeError(f"results должен быть dict, получено {type(results).__name__}")

    # Пытаемся найти блок первой задачи.
    task_data = _get_task_block(results, _DEFAULT_TASK1)

    # Фолбэк: ищем любой блок с обоими ключами наград.
    if task_data is None:
        task_data = _find_any_task_with_keys(
            results, ["reward_after_task1", "reward_after_task2"]
        )

    # Если блок всё ещё не найден — ошибка.
    if task_data is None:
        raise KeyError(
            "Не найден блок task1_goto с ключами reward_after_task1/reward_after_task2. "
            "Проверьте формат: ожидается results['results']['task1_goto'] или results['task1_goto']"
        )

    # Алиасы для наград — покрывают возможные варианты имён.
    r1_keys = [
        "reward_after_task1",
        "reward_after_t1",
        "r_after_task1",
        "reward_t1_after_t1",
        "mean_reward_after_task1",
    ]
    r2_keys = [
        "reward_after_task2",
        "reward_after_t2",
        "r_after_task2",
        "reward_t1_after_t2",
        "mean_reward_after_task2",
        "reward_t1_after_task2",
    ]

    r1 = _get_first_value(task_data, r1_keys)
    r2 = _get_first_value(task_data, r2_keys)

    # Если прямые ключи не найдены, пробуем взять готовое поле forgetting
    # как резерв (но предпочитаем вычисление через разность).
    if r1 is None or r2 is None:
        # Проверяем наличие готового forgetting — возвращаем его если есть.
        fallback = task_data.get("forgetting")
        if fallback is not None:
            try:
                return float(fallback)
            except (TypeError, ValueError):
                pass
        # Формируем понятное сообщение об отсутствующих ключах.
        missing = []
        if r1 is None:
            missing.append("reward_after_task1")
        if r2 is None:
            missing.append("reward_after_task2")
        raise KeyError(f"Отсутствуют ключи для вычисления forgetting: {missing} в {task_data}")

    # Преобразуем к float и вычисляем разность.
    try:
        r1_f = float(r1)  # type: ignore[arg-type]
        r2_f = float(r2)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Значения наград должны быть числовыми: r1={r1!r}, r2={r2!r}") from exc

    # Forgetting = R(T1 после T1) - R(T1 после T2).
    return r1_f - r2_f


def compute_forward_transfer(results: Dict[str, Any]) -> float | None:
    """Вычисляет forward transfer: Steps(T2 после T1) / Steps(T2 с нуля).

    Значение ``< 1`` означает ускорение обучения (положительный перенос),
    ``> 1`` — замедление.

    Поддерживает робастный парсинг: ищет шаги в блоке ``task2_avoid``
    как в ``results["results"][task_name]``, так и в ``results[task_name]``.
    Если данные о шагах отсутствуют — возвращает ``None``.

    Args:
        results: Метрики одного прогона. Ожидаются ключи шагов, например
            ``learning_steps_to_threshold`` (числитель) и
            ``steps_from_scratch`` / ``baseline_steps`` (знаменатель),
            либо уже готовое отношение ``forward_transfer``.

    Returns:
        Отношение шагов как ``float`` или ``None`` если данных недостаточно
        (нет числителя/знаменателя или знаменатель равен нулю).

    Raises:
        TypeError: Если ``results`` не является словарём.
    """
    # Валидируем тип.
    if not isinstance(results, dict):
        raise TypeError(f"results должен быть dict, получено {type(results).__name__}")

    # Находим блок второй задачи (основной носитель шагов).
    task_data = _get_task_block(results, _DEFAULT_TASK2)

    # Фолбэк: если task2_avoid не найден, пробуем найти любой блок с шагами.
    inner = _extract_results_dict(results)
    if task_data is None:
        # Ищем блок, содержащий хотя бы один из ключей шагов.
        step_indicators = [
            "learning_steps_to_threshold",
            "forward_transfer_steps",
            "steps_after",
            "steps_t2_after_t1",
        ]
        for v in inner.values():
            if isinstance(v, dict) and any(k in v for k in step_indicators):
                task_data = v
                break
        # Проверяем верхний уровень.
        if task_data is None:
            for v in results.values():
                if isinstance(v, dict) and any(k in v for k in step_indicators):
                    task_data = v
                    break

    # Если блок задачи не найден — данных нет.
    if task_data is None:
        return None

    # 1. Если уже есть готовое отношение — возвращаем его.
    ratio_keys = ["forward_transfer", "forward_transfer_ratio", "transfer_ratio"]
    ratio_val = _get_first_value(task_data, ratio_keys)
    if ratio_val is not None:
        try:
            return float(ratio_val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # Если не удалось преобразовать — продолжаем поиск шагов.
            pass

    # 2. Ищем числитель и знаменатель по алиасам.
    numerator_keys = [
        "learning_steps_to_threshold",
        "forward_transfer_steps",
        "steps_after_t1",
        "steps_t2_after_t1",
        "steps_after",
        "steps",
        "t2_steps",
        "learning_steps",
    ]
    denominator_keys = [
        "steps_from_scratch",
        "baseline_steps",
        "steps_scratch",
        "steps_baseline",
        "learning_steps_from_scratch",
        "steps_t2_scratch",
        "scratch_steps",
        "baseline_learning_steps",
        "steps_t2_from_scratch",
    ]

    numerator = _get_first_value(task_data, numerator_keys)
    denominator: Any | None = None

    # Знаменатель может быть в том же блоке или во внешних словарях.
    # Проверяем пространства поиска в порядке приоритета.
    search_spaces: list[Dict[str, Any]] = []
    if isinstance(task_data, dict):
        search_spaces.append(task_data)
    if isinstance(inner, dict):
        search_spaces.append(inner)
    if isinstance(results, dict):
        search_spaces.append(results)
    # Также проверяем вложенный baseline-словарь если есть.
    for space in list(search_spaces):
        baseline = space.get("baseline")
        if isinstance(baseline, dict):
            search_spaces.append(baseline)
        # Поддержка отдельного ключа scratch внутри results.
        baseline2 = space.get("task2_avoid_from_scratch")
        if isinstance(baseline2, dict):
            search_spaces.append(baseline2)

    # Ищем знаменатель по всем пространствам.
    for space in search_spaces:
        val = _get_first_value(space, denominator_keys)
        if val is not None:
            denominator = val
            break
        # Дополнительная эвристика: числитель/знаменатель могут быть
        # вложены как ``{"steps": {"after_t1": ..., "from_scratch": ...}}``.
        # Проверяем вложенные словари с ключом steps.
        steps_dict = space.get("steps")
        if isinstance(steps_dict, dict):
            den2 = _get_first_value(steps_dict, denominator_keys + ["from_scratch", "scratch", "baseline"])
            if den2 is not None:
                denominator = den2
                # Если числитель ещё не найден, пробуем взять оттуда же.
                if numerator is None:
                    num2 = _get_first_value(
                        steps_dict, numerator_keys + ["after_t1", "after", "continual"]
                    )
                    if num2 is not None:
                        numerator = num2
                break

    # Если числитель ещё не найден, но знаменатель есть — пробуем
    # альтернативные пространства для числителя (например, outer).
    if numerator is None:
        for space in search_spaces:
            val = _get_first_value(space, numerator_keys)
            if val is not None:
                numerator = val
                break

    # Если хотя бы одно из значений отсутствует — данных недостаточно.
    if numerator is None or denominator is None:
        return None

    # Преобразуем к float и вычисляем отношение.
    try:
        num_f = float(numerator)  # type: ignore[arg-type]
        den_f = float(denominator)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    # Избегаем деления на ноль.
    if den_f == 0:
        return None

    return num_f / den_f


def compute_final_reward(
    results: Dict[str, Any],
    task_name: str = _DEFAULT_TASK1,
) -> float | None:
    """Вычисляет финальную награду для указанной задачи.

    Финальная награда — это ``R(task)`` после завершения всего цикла
    continual learning (после T2). Использует робастный парсинг аналогично
    :func:`compute_forgetting`.

    Args:
        results: Словарь прогона (внешний JSON или плоский).
        task_name: Имя задачи, для которой извлекается награда.
            По умолчанию ``"task1_goto"``. Если ``"task2_avoid"``,
            обычно возвращается ``reward_after_task2``.

    Returns:
        Финальная награда как ``float`` или ``None`` если данные отсутствуют.

    Raises:
        TypeError: Если ``results`` не является словарём.
    """
    # Валидируем тип.
    if not isinstance(results, dict):
        raise TypeError(f"results должен быть dict, получено {type(results).__name__}")

    # Находим блок задачи.
    task_data = _get_task_block(results, task_name)
    if task_data is None:
        return None

    # Алиасы финальной награды в порядке приоритета.
    # ``final_reward`` — основной, ``reward_after_task2`` — фактическая
    # награда после всего цикла для T1, ``mean_reward`` — резерв.
    final_keys = [
        "final_reward",
        "reward_after_task2",
        "reward_after_task1",
        "mean_reward",
        "reward",
        "final",
    ]
    val = _get_first_value(task_data, final_keys)
    if val is None:
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

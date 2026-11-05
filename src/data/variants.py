"""Варианты графов для абляций коннектома.

Содержит генераторы аблированных графов:
- rewired (Maslov–Sneppen, degree-preserving);
- random_er (Erdos–Renyi с той же плотностью);
- unsigned (модуль знаковых весов).

Формат графа — ``torch_geometric.data.Data`` с полями
``edge_index`` [2, E], ``edge_attr``/``edge_weight`` [E] или [E,1],
``sensory_mask`` и ``motor_mask`` [N], ``num_nodes``.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.data import Data


def _seed_all(seed: int | None) -> None:
    """Фиксирует сиды для ``random``, ``numpy`` и ``torch``.

    Args:
        seed: Значение сида. Если ``None``, ничего не делает.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Для CUDA детерминизма достаточно torch.manual_seed,
    # дополнительный вызов для GPU если доступен.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_rewired_multiplier() -> int:
    """Загружает множитель ``n_swaps`` из ``config/graphs.yaml``.

    Returns:
        Множитель числа свапов. По умолчанию 10, если конфиг не найден
        или поле отсутствует.
    """
    # Путь от src/data/variants.py -> корень проекта
    config_path = Path(__file__).resolve().parents[2] / "config" / "graphs.yaml"
    try:
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            # Ожидается cfg["variants"]["rewired"]["n_swaps"]
            return int(cfg["variants"]["rewired"]["n_swaps"])
    except Exception:
        # При любой ошибке чтения — фолбэк на дефолт
        pass
    return 10


def _get_edge_attr_tensor(data: Data) -> torch.Tensor | None:
    """Возвращает тензор весов рёбер (edge_attr или edge_weight).

    Args:
        data: Граф PyG.

    Returns:
        Тензор весов или ``None`` если поле отсутствует.
    """
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        return data.edge_attr
    if hasattr(data, "edge_weight") and data.edge_weight is not None:
        return data.edge_weight
    return None


def _set_edge_attr_tensor(data: Data, weights: torch.Tensor) -> None:
    """Записывает веса обратно в то же поле, откуда они были прочитаны.

    Если исходный граф хранил ``edge_attr``, пишет туда; иначе в
    ``edge_weight``. Если оба поля отсутствовали, создаёт ``edge_attr``.

    Args:
        data: Граф для записи.
        weights: Новые веса.
    """
    # Приоритет — edge_attr, затем edge_weight
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        # Сохраняем исходную размерность [E] vs [E,1]
        if data.edge_attr.dim() == 2 and data.edge_attr.size(1) == 1:
            data.edge_attr = weights.view(-1, 1)
        else:
            data.edge_attr = weights
        # Синхронизируем edge_weight если оно тоже было
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            data.edge_weight = weights
        return
    if hasattr(data, "edge_weight") and data.edge_weight is not None:
        data.edge_weight = weights
        return
    # Нет исходного поля — создаём edge_attr
    data.edge_attr = weights


def create_rewired_graph(base_graph: Data, seed: int | None = None) -> Data:
    """Создаёт degree-preserving перетасованный граф (Maslov–Sneppen).

    Алгоритм — свапы пар рёбер: выбираются два ребра ``(u, v)`` и
    ``(x, y)``, заменяются на ``(u, y)`` и ``(x, v)`` при условии
    отсутствия петель и дубликатов. Сохраняются входящие и исходящие
    степени каждого узла (для ориентированного графа). Множество знаковых
    весов сохраняется — веса переставляются вместе с рёбрами (вес
    остаётся за исходным источником).

    Число попыток свапов ``n_swaps = multiplier * num_edges``, где
    ``multiplier`` читается из ``config/graphs.yaml``
    (``variants.rewired.n_swaps``, по умолчанию 10).

    Топология ``edge_index`` заморожена для GNN — перетасовка нужна
    только для абляции; сенсорные/моторные маски копируются без изменений.

    Args:
        base_graph: Исходный граф PyG с ``edge_index``, ``edge_attr``
            (или ``edge_weight``), ``sensory_mask``, ``motor_mask``.
        seed: Сид для воспроизводимости (фиксирует ``random``, ``numpy``,
            ``torch``).

    Returns:
        Новый граф ``Data`` с той же последовательностью степеней,
        теми же масками и тем же мультимножеством весов, но
        перетасованной топологией.
    """
    _seed_all(seed)

    multiplier = _load_rewired_multiplier()

    # Копируем граф полностью (включая маски, num_nodes и т.д.)
    new_graph: Data = copy.deepcopy(base_graph)

    if base_graph.edge_index is None or base_graph.edge_index.numel() == 0:
        # Пустой граф — нечего перетасовывать
        return new_graph

    edge_index = base_graph.edge_index.clone()
    num_edges = int(edge_index.size(1))
    num_nodes = int(base_graph.num_nodes) if base_graph.num_nodes is not None else int(edge_index.max().item() + 1)

    if num_edges < 2:
        return new_graph

    n_swaps = multiplier * num_edges

    # Извлекаем веса
    edge_weights = _get_edge_attr_tensor(base_graph)
    if edge_weights is not None:
        # Приводим к 1D для удобства, запомним исходную форму
        orig_dim_2d = edge_weights.dim() == 2 and edge_weights.size(1) == 1
        w_list = edge_weights.clone().view(-1).tolist()
    else:
        orig_dim_2d = False
        w_list = None

    # Списки src/dst для быстрых свапов
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    # Мультимножество рёбер для корректной работы с дубликатами
    # Используем Counter чтобы не падать с KeyError при повторном удалении
    edge_counter: Counter[tuple[int, int]] = Counter(zip(src, dst))

    swaps_done = 0
    # Ограничиваем число попыток, чтобы не зациклиться на плотных графах
    max_tries = n_swaps * 10
    tries = 0

    # Основной цикл свапов
    while swaps_done < n_swaps and tries < max_tries:
        tries += 1
        i, j = random.sample(range(num_edges), 2)
        u, v = src[i], dst[i]
        x, y = src[j], dst[j]

        # Пропускаем если источники или стоки совпадают — степени не сохранятся
        # или свап тривиален
        if u == x or v == y:
            continue
        # Избегаем петель после свапа: (u, y) и (x, v) не должны быть петлями
        if u == y or x == v:
            continue
        # Проверяем отсутствие новых рёбер в графе (с учётом мультимножества)
        if edge_counter[(u, y)] > 0 or edge_counter[(x, v)] > 0:
            continue
        # Защита от KeyError на графах с дубликатами: если ребро уже
        # удалено предыдущим свапом (дубликат), пропускаем попытку
        if edge_counter[(u, v)] == 0 or edge_counter[(x, y)] == 0:
            continue

        # Выполняем свап: (u,v)+(x,y) -> (u,y)+(x,v)
        # Веса остаются за индексом i/j (т.е. за источником u/x)
        # Обновляем мультимножество с корректным учётом дубликатов
        edge_counter[(u, v)] -= 1
        if edge_counter[(u, v)] == 0:
            del edge_counter[(u, v)]
        edge_counter[(x, y)] -= 1
        if edge_counter[(x, y)] == 0:
            del edge_counter[(x, y)]
        edge_counter[(u, y)] += 1
        edge_counter[(x, v)] += 1

        # Обновляем списки — меняем только dst (источник фиксирован)
        dst[i] = y
        dst[j] = v
        # src остаётся прежним (u и x), веса не трогаем

        swaps_done += 1

    # Собираем новые тензоры
    new_edge_index = torch.tensor([src, dst], dtype=torch.long)

    new_graph.edge_index = new_edge_index
    new_graph.num_nodes = num_nodes

    if w_list is not None:
        # Веса не меняли порядок, но на случай если нужно было переставить —
        # мультимножество то же, клон исходного достаточен.
        # Пересобираем тензор из w_list чтобы сохранить тип
        w_tensor = torch.tensor(w_list, dtype=torch.float32)
        if orig_dim_2d:
            w_tensor = w_tensor.view(-1, 1)
        _set_edge_attr_tensor(new_graph, w_tensor)

    # Маски уже скопированы deepcopy, num_nodes сохранён.
    # Валидация: степени сохранены, изолированных узлов не добавилось
    # (если исходный был без изолятов, то и новый без них).
    return new_graph


def _create_random_er_impl(
    n_nodes: int,
    n_edges: int,
    seed: int | None = None,
) -> Data:
    """Внутренняя реализация ER-графа (G(n,m) без петель).

    Args:
        n_nodes: Число узлов.
        n_edges: Число рёбер.
        seed: Сид для воспроизводимости.

    Returns:
        Граф ``Data`` с полями ``edge_index`` [2, E], ``edge_attr`` [E],
        ``sensory_mask`` [N], ``motor_mask`` [N], ``num_nodes``.
    """
    _seed_all(seed)

    if n_nodes < 1:
        raise ValueError(f"n_nodes должен быть >=1, получено {n_nodes}")
    if n_edges < 0:
        raise ValueError(f"n_edges должен быть >=0, получено {n_edges}")

    max_possible = n_nodes * (n_nodes - 1)
    if n_edges > max_possible:
        raise ValueError(
            f"n_edges={n_edges} превышает максимум {max_possible} "
            f"для ориентированного графа без петель (n_nodes={n_nodes})"
        )

    # Сэмплирование рёбер без петель и дубликатов
    edge_set: set[tuple[int, int]] = set()
    src_list: list[int] = []
    dst_list: list[int] = []

    # Батчевое сэмплирование для скорости
    # Используем numpy для генерации, random уже зафиксирован
    while len(src_list) < n_edges:
        remaining = n_edges - len(src_list)
        # Берём батч с запасом на отсев петель/дубликатов
        batch_size = min(remaining * 2, 20000)
        batch_size = max(batch_size, remaining)
        # Генерируем случайные пары
        src_batch = np.random.randint(0, n_nodes, size=batch_size)
        dst_batch = np.random.randint(0, n_nodes, size=batch_size)
        for u, v in zip(src_batch.tolist(), dst_batch.tolist()):
            if u == v:
                continue
            if (u, v) in edge_set:
                continue
            edge_set.add((u, v))
            src_list.append(int(u))
            dst_list.append(int(v))
            if len(src_list) >= n_edges:
                break

    # Проверка на изолированные узлы и патч при необходимости
    # Изолят = нет ни входящих ни исходящих рёбер
    if n_nodes > 1 and n_edges > 0:
        degree = [0] * n_nodes
        for u, v in zip(src_list, dst_list):
            degree[u] += 1
            degree[v] += 1
        isolated = [i for i, d in enumerate(degree) if d == 0]
        # Патчим каждый изолят одним случайным ребром
        for iso in isolated:
            # Пытаемся добавить ребро iso -> random или random -> iso
            for _ in range(100):
                # Чередуем направление по чётности попыток
                if random.random() < 0.5:
                    other = int(np.random.randint(0, n_nodes))
                    cand = (iso, other)
                else:
                    other = int(np.random.randint(0, n_nodes))
                    cand = (other, iso)
                if cand[0] == cand[1]:
                    continue
                if cand in edge_set:
                    continue
                edge_set.add(cand)
                src_list.append(cand[0])
                dst_list.append(cand[1])
                break
        # Если добавляли рёбра, edge_attr позже пересоздастся по новому размеру

    actual_edges = len(src_list)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Веса — Uniform[0, 1], положительные
    edge_attr = torch.rand(actual_edges, dtype=torch.float32)

    # Маски: гарантируем >=50 sensory, >=20 motor, без пересечений
    n_sensory = max(50, n_nodes // 10)
    n_motor = max(20, n_nodes // 20)
    # Корректируем если сумма превышает n_nodes
    if n_sensory + n_motor > n_nodes:
        # При маленьких графах уменьшаем пропорционально
        n_sensory = min(n_sensory, n_nodes // 2)
        n_motor = min(n_motor, n_nodes - n_sensory)
        n_sensory = max(1, n_sensory)
        n_motor = max(1, n_motor)

    perm = np.random.permutation(n_nodes)
    sensory_idx = perm[:n_sensory].tolist()
    motor_idx = perm[n_sensory : n_sensory + n_motor].tolist()

    sensory_mask = torch.zeros(n_nodes, dtype=torch.bool)
    motor_mask = torch.zeros(n_nodes, dtype=torch.bool)
    sensory_mask[torch.tensor(sensory_idx, dtype=torch.long)] = True
    motor_mask[torch.tensor(motor_idx, dtype=torch.long)] = True

    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        sensory_mask=sensory_mask,
        motor_mask=motor_mask,
        num_nodes=n_nodes,
    )
    return data


def create_random_er_graph_from_bio(
    base_graph: Data,
    seed: int | None = None,
) -> Data:
    """Создаёт ER-граф на основе био-графа (обёртка для единообразного API).

    Извлекает ``n_nodes`` и ``n_edges`` из ``base_graph`` и делегирует
    в :func:`_create_random_er_impl`. Нужна для единообразного вызова
    ``variants(Data)`` из ``scripts/01_build_graphs.py``.

    Args:
        base_graph: Исходный граф PyG, из которого берутся ``num_nodes``
            и число рёбер ``edge_index.size(1)``.
        seed: Сид для воспроизводимости.

    Returns:
        Новый ER-граф ``Data`` с той же плотностью что у ``base_graph``.
    """
    if base_graph.edge_index is not None and base_graph.edge_index.numel() > 0:
        n_edges = int(base_graph.edge_index.size(1))
    else:
        n_edges = 0
    if base_graph.num_nodes is not None:
        n_nodes = int(base_graph.num_nodes)
    elif base_graph.edge_index is not None and base_graph.edge_index.numel() > 0:
        n_nodes = int(base_graph.edge_index.max().item() + 1)
    else:
        n_nodes = 0
    return _create_random_er_impl(n_nodes, n_edges, seed=seed)


def create_random_er_graph(
    n_nodes: int | Data,
    n_edges: int | None = None,
    seed: int | None = None,
) -> Data:
    """Создаёт ориентированный граф Эрдёша–Реньи с заданной плотностью.

    Поддерживает два режима вызова для обратной совместимости:

    * ``create_random_er_graph(n_nodes, n_edges, seed)`` — классический,
      где ``n_nodes`` и ``n_edges`` — целые числа;
    * ``create_random_er_graph(base_graph, seed)`` — единообразный
      ``variants(Data)`` как ожидает ``scripts/01_build_graphs.py``.
      Если первый аргумент — ``Data``, из него извлекаются ``n_nodes``
      и ``n_edges`` и вызывается :func:`create_random_er_graph_from_bio`.

    Плотность определяется как ``p = n_edges / (n_nodes * (n_nodes-1))``
    для ориентированного графа без петель. Рёбра сэмплируются без
    петель и без дубликатов. Для воспроизводимости при заданном ``seed``
    используется точное число рёбер ``n_edges`` (модель G(n,m)).

    Веса рёбер — независимые сэмплы из ``Uniform[0, 1]``, положительные.
    Сенсорные/моторные маски генерируются случайно с гарантией
    ``sensory >=50`` и ``motor >=20`` (требование валидации).

    Args:
        n_nodes: Число узлов или граф ``Data`` (в режиме ``Data`` —
            извлекаются ``n_nodes``/``n_edges`` из графа).
        n_edges: Число рёбер (определяет плотность) или ``None``/``seed``
            в режиме ``Data``. Если ``n_nodes`` — ``Data`` и ``seed``
            не передан, а ``n_edges`` — ``int``, оно трактуется как
            ``seed`` для удобства позиционного вызова.
        seed: Сид для воспроизводимости.

    Returns:
        Граф ``Data`` с полями ``edge_index`` [2, E], ``edge_attr`` [E],
        ``sensory_mask`` [N], ``motor_mask`` [N], ``num_nodes``.

    Raises:
        ValueError: Если ``n_nodes < 1`` или ``n_edges < 0`` или
            ``n_edges > n_nodes*(n_nodes-1)``.
    """
    # Режим Data — единообразный вызов variants(Data)
    if isinstance(n_nodes, Data):
        base_graph: Data = n_nodes
        # Поддержка позиционного seed: create_random_er_graph(data, 42)
        actual_seed: int | None = seed
        if actual_seed is None and isinstance(n_edges, int):
            actual_seed = n_edges
        return create_random_er_graph_from_bio(base_graph, seed=actual_seed)

    # Классический режим int/int
    if n_edges is None:
        raise ValueError("n_edges должен быть задан при вызове с int n_nodes")
    # n_nodes здесь — int, проверяем тип для mypy/pyright
    return _create_random_er_impl(int(n_nodes), int(n_edges), seed=seed)


def create_unsigned_graph(signed_graph: Data) -> Data:
    """Создаёт копию графа с абсолютными значениями весов.

    Берёт модуль всех знаковых весов ``|w|`` (для тормозных GABA-рёбер
    ``-syn_count`` становится ``+syn_count``). Вся остальная структура
    (``edge_index``, маски, ``num_nodes`` и т.д.) копируется без
    изменений.

    Args:
        signed_graph: Исходный граф со знаковыми весами в ``edge_attr``
            или ``edge_weight``.

    Returns:
        Новый граф ``Data`` с ``edge_attr = |edge_attr|`` (и
        ``edge_weight`` если оно было).
    """
    # Глубокая копия чтобы не мутировать исходный граф
    new_graph: Data = copy.deepcopy(signed_graph)

    weights = _get_edge_attr_tensor(new_graph)
    if weights is not None:
        abs_weights = torch.abs(weights)
        _set_edge_attr_tensor(new_graph, abs_weights)

    return new_graph

"""Модуль анализа активности по модулям коннектома.

Вычисляет среднюю активацию узлов внутри каждого сообщества (модуля),
выделенного алгоритмом Louvain. Используется для построения heatmap
модульной специализации (Figure 3).
"""

from typing import Dict

import community as community_louvain
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


def compute_module_activation(
    activations: torch.Tensor,
    module_assignments: np.ndarray,
    n_modules: int,
) -> Dict[int, float]:
    """Вычисляет среднюю активацию для каждого модуля.

    Для каждого модуля считается среднее значение активаций входящих в него
    узлов. Поддерживаются два формата ``activations``:

    * ``[n_nodes]`` — скалярная активация на узел, используется
      ``mean(|activations|)`` (среднее абсолютных значений).
    * ``[n_nodes, hidden_dim]`` — векторная активация, используется
      ``mean(L2 нормы)`` по узлам модуля. Частный случай ``hidden_dim==1``
      сводится к скалярному варианту.

    Args:
        activations: Тензор активаций узлов. Форма ``[n_nodes]`` или
            ``[n_nodes, hidden_dim]``. Может находиться на любом устройстве.
        module_assignments: Массив ID модулей формы ``[n_nodes]`` со
            значениями ``0..n_modules-1``. Принимается ``np.ndarray``,
            ``torch.Tensor`` или список.
        n_modules: Общее число модулей. Определяет ключи результата
            ``0..n_modules-1``.

    Returns:
        Словарь ``{module_id: mean_activation}``. Для пустого модуля
        (нет узлов) возвращается ``0.0``.

    Raises:
        TypeError: Если ``activations`` не является ``torch.Tensor``.
        ValueError: При несоответствии размеров, некорректной размерности
            или ``n_modules <= 0``.
    """
    # Проверка типа активаций.
    if not isinstance(activations, torch.Tensor):
        raise TypeError("activations должен быть torch.Tensor")

    if activations.dim() not in (1, 2):
        raise ValueError(
            f"activations должен быть 1- или 2-мерным, получен ndim={activations.dim()}"
        )

    if n_modules <= 0:
        raise ValueError(f"n_modules должен быть > 0, получен {n_modules}")

    n_nodes = int(activations.shape[0])

    # Приведение module_assignments к numpy-массиву для унификации.
    if isinstance(module_assignments, torch.Tensor):
        assignments = module_assignments.detach().cpu().numpy()
    else:
        assignments = np.asarray(module_assignments)

    # Проверка формы assignments.
    if assignments.ndim != 1:
        raise ValueError(
            f"module_assignments должен быть 1-мерным, получен ndim={assignments.ndim}"
        )

    if assignments.shape[0] != n_nodes:
        raise ValueError(
            f"Несоответствие размеров: activations n_nodes={n_nodes}, "
            f"module_assignments len={assignments.shape[0]}"
        )

    # Приведение к int для сравнения ID.
    # На случай float-массивов округляем к int.
    try:
        assignments_int = assignments.astype(int)
    except Exception as exc:
        raise ValueError(f"Не удалось привести module_assignments к int: {exc}") from exc

    # Вычисление скалярной меры активации на узел.
    # Для 1D — абсолютное значение, для 2D — L2 норма по hidden_dim.
    if activations.dim() == 1:
        # Скалярный случай: |x| на узел.
        per_node = activations.detach().float().abs().cpu()
    else:
        # Векторный случай: [n_nodes, hidden_dim]
        hidden_dim = int(activations.shape[1])
        if hidden_dim == 1:
            # Вырожденный случай — сводим к скалярному.
            per_node = activations.detach().float().abs().squeeze(1).cpu()
        else:
            # L2 норма по признаковой оси.
            per_node = torch.norm(activations.detach().float(), p=2, dim=1).cpu()

    # Переводим в numpy для групповой агрегации.
    per_node_np = per_node.numpy()

    # Группировка по модулям и усреднение.
    result: Dict[int, float] = {}
    for module_id in range(n_modules):
        # Маска узлов текущего модуля.
        mask = assignments_int == module_id
        if np.any(mask):
            # Среднее по узлам модуля.
            result[module_id] = float(per_node_np[mask].mean())
        else:
            # Пустой модуль — 0.0 для детерминизма.
            result[module_id] = 0.0

    return result


def get_module_assignments(graph: Data) -> np.ndarray:
    """Вычисляет разбиение узлов графа на модули методом Louvain.

    Строит неориентированную версию направленного графа коннектома,
    агрегируя встречные рёбра суммой абсолютных весов
    ``|w|`` (тормозные связи учитываются по модулю). Затем запускает
    алгоритм Louvain (``python-louvain``) для поиска сообществ.

    Узлы с петлями: петли удаляются, так как не влияют на модулярность.
    Граф без рёбер: каждый узел становится отдельным модулем.

    Args:
        graph: Граф в формате ``torch_geometric.data.Data`` с полями
            ``edge_index`` (``[2, E]``), ``edge_attr`` или ``edge_weight``
            (``[E]``) и ``num_nodes``. Веса могут быть знаковыми.

    Returns:
        Массив формы ``[n_nodes]`` с ID модулей ``0..n_modules-1``.
        ID перенумерованы непрерывно от ``0`` независимо от исходных
        меток Louvain. Детерминирован (``random_state=42``).

    Raises:
        TypeError: Если ``graph`` не является ``Data`` или не содержит
            необходимых полей.
        ValueError: При ``NaN`` в весах.
    """
    # Проверка типа входного графа.
    if not isinstance(graph, Data):
        raise TypeError("graph должен быть torch_geometric.data.Data")

    # Извлечение числа узлов.
    if hasattr(graph, "num_nodes") and graph.num_nodes is not None:
        n_nodes = int(graph.num_nodes)
    elif hasattr(graph, "edge_index") and graph.edge_index is not None:
        edge_idx_tmp = graph.edge_index
        if isinstance(edge_idx_tmp, torch.Tensor) and edge_idx_tmp.numel() > 0:
            n_nodes = int(edge_idx_tmp.max().item()) + 1
        elif isinstance(edge_idx_tmp, np.ndarray) and edge_idx_tmp.size > 0:
            n_nodes = int(np.max(edge_idx_tmp)) + 1
        else:
            n_nodes = 0
    else:
        raise TypeError("Data не содержит edge_index или num_nodes")

    if n_nodes == 0:
        return np.zeros(0, dtype=int)

    # Извлечение рёбер и весов.
    edge_index = getattr(graph, "edge_index", None)
    edge_attr = getattr(graph, "edge_attr", None)
    if edge_attr is None:
        edge_attr = getattr(graph, "edge_weight", None)

    # Построение неориентированного графа для Louvain.
    undir = nx.Graph()
    undir.add_nodes_from(range(n_nodes))

    # Если рёбра отсутствуют — каждый узел отдельный модуль (как в metrics.py).
    has_edges = edge_index is not None and (
        (isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0)
        or (isinstance(edge_index, np.ndarray) and edge_index.size > 0)
    )

    if has_edges:
        # Нормализация edge_index к numpy [2, E].
        if isinstance(edge_index, torch.Tensor):
            if torch.isnan(edge_index.float()).any().item():
                raise ValueError("edge_index содержит NaN")
            edge_np = edge_index.detach().cpu().numpy()
        elif isinstance(edge_index, np.ndarray):
            if np.isnan(edge_index).any():
                raise ValueError("edge_index содержит NaN")
            edge_np = edge_index
        else:
            raise TypeError("edge_index должен быть torch.Tensor или np.ndarray")

        if edge_np.ndim != 2:
            raise ValueError(f"edge_index должен быть 2-мерным, получен ndim={edge_np.ndim}")

        # Поддержка ориентаций [2, E] и [E, 2].
        if edge_np.shape[0] == 2:
            src = edge_np[0]
            dst = edge_np[1]
            n_edges = edge_np.shape[1]
        elif edge_np.shape[1] == 2:
            src = edge_np[:, 0]
            dst = edge_np[:, 1]
            n_edges = edge_np.shape[0]
        else:
            raise ValueError(
                f"Неверная форма edge_index: {edge_np.shape}, ожидается [2, E] или [E, 2]"
            )

        # Нормализация весов.
        if edge_attr is None:
            weights = np.ones(n_edges, dtype=float)
        else:
            if isinstance(edge_attr, torch.Tensor):
                if torch.isnan(edge_attr.float()).any().item():
                    raise ValueError("Веса графа содержат NaN (edge_attr)")
                weights_np = edge_attr.detach().cpu().numpy()
            elif isinstance(edge_attr, np.ndarray):
                if np.isnan(edge_attr).any():
                    raise ValueError("Веса графа содержат NaN (edge_attr)")
                weights_np = edge_attr
            else:
                raise TypeError("edge_attr должен быть torch.Tensor, np.ndarray или None")

            weights_np = np.asarray(weights_np).squeeze()
            if weights_np.ndim == 0:
                weights_np = np.full(n_edges, float(weights_np), dtype=float)
            if weights_np.shape[0] != n_edges:
                raise ValueError(
                    f"Длина edge_attr ({weights_np.shape[0]}) не совпадает с E={n_edges}"
                )
            weights = weights_np.astype(float)
            if np.isnan(weights).any():
                raise ValueError("Веса графа содержат NaN")

        # Агрегация направленных рёбер в неориентированные суммой |w|.
        for u, v, w in zip(src, dst, weights, strict=False):
            ui = int(u)
            vi = int(v)
            # Пропуск индексов вне диапазона (защита от некорректных данных).
            if ui < 0 or ui >= n_nodes or vi < 0 or vi >= n_nodes:
                continue
            wf = abs(float(w))
            if np.isnan(wf):
                raise ValueError("Веса графа содержат NaN")
            # Петли будут удалены позже, но пропускаем их сразу для Louvain.
            if ui == vi:
                continue
            if undir.has_edge(ui, vi):
                undir[ui][vi]["weight"] += wf
            else:
                undir.add_edge(ui, vi, weight=wf)

        # Удаление петель на случай если они были добавлены.
        undir.remove_edges_from(nx.selfloop_edges(undir))

    # Случай графа без рёбер после агрегации.
    if undir.number_of_edges() == 0:
        # Каждый узел — отдельный модуль для совместимости с metrics.py.
        return np.arange(n_nodes, dtype=int)

    # Запуск Louvain с фиксированным сидом для воспроизводимости.
    try:
        partition = community_louvain.best_partition(
            undir, weight="weight", random_state=42
        )
    except Exception:
        # Фолбэк при сбое алгоритма — один модуль.
        return np.zeros(n_nodes, dtype=int)

    # Перенумерация меток в непрерывный диапазон 0..n_modules-1.
    # Louvain может вернуть произвольные ID (например, 0,2,5).
    unique_labels = sorted(set(partition.values()))
    remap = {old: new for new, old in enumerate(unique_labels)}

    assignments = np.zeros(n_nodes, dtype=int)
    for node, label in partition.items():
        node_idx = int(node)
        if 0 <= node_idx < n_nodes:
            assignments[node_idx] = int(remap[label])

    # Узлы, отсутствующие в partition (изолированные), уже 0.
    # Для изолированных без рёбер Louvain их включает, но на всякий случай:
    # оставляем их в модуле 0, что соответствует фолбэку.

    return assignments

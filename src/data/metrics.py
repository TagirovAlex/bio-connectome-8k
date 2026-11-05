"""Метрики структуры графа коннектома.

Вычисляет структурные характеристики направленного графа коннектома
для валидации и сравнения вариантов (bio/rewired/random_er/unsigned).

Использует networkx + python-louvain (community) для модулярности.
Поддерживает входные форматы:
- сетьx.DiGraph / Graph — напрямую,
- torch_geometric.data.Data — из пайплайна построения графов,
- тройку тензоров (edge_index, edge_attr, n_nodes) — низкоуровневый вызов.
"""

from typing import Any, Dict, overload

import community as community_louvain
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


@overload
def compute_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:  # type: ignore[overload-overlap]
    ...


@overload
def compute_graph_metrics(graph: Data) -> Dict[str, Any]:  # type: ignore[overload-overlap]
    ...


@overload
def compute_graph_metrics(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    n_nodes: int,
) -> Dict[str, Any]:  # type: ignore[overload-overlap]
    ...


def compute_graph_metrics(
    graph: nx.Graph | Data | torch.Tensor | np.ndarray | None = None,
    edge_index: torch.Tensor | np.ndarray | None = None,
    edge_attr: torch.Tensor | np.ndarray | None = None,
    n_nodes: int | None = None,
) -> Dict[str, Any]:
    """Вычисляет структурные метрики графа.

    Поддерживает перегрузку по типам аргументов для совместимости
    с ``scripts/01_build_graphs.py`` (ожидает ``Data``) и тестами
    на ``networkx``. Также принимает тройку ``(edge_index, edge_attr, n_nodes)``
    позиционно: ``compute_graph_metrics(edge_index, edge_attr, n_nodes)``.

    Args:
        graph: Граф в одном из форматов — ``nx.Graph``/``nx.DiGraph``,
            ``torch_geometric.data.Data`` или тензор ``edge_index`` при
            позиционном вызове тройкой.
        edge_index: Индексы рёбер формы ``[2, E]`` (PyG) либо ``[E, 2]``.
            Используется только в перегрузке с тензорами.
        edge_attr: Веса рёбер формы ``[E]`` или ``[E, 1]``. Если ``None``,
            все веса считаются равными 1.0.
        n_nodes: Число узлов. Обязательно при тензорной перегрузке.

    Returns:
        Словарь с метриками:
        ``modularity_score``, ``mean_degree``, ``recurrence_fraction``,
        ``reciprocity`` (алиас), ``avg_path_length``, ``n_modules``,
        ``density``, ``n_nodes``, ``n_edges``.

    Raises:
        ValueError: Если веса содержат ``NaN``.
        TypeError: Если сигнатура вызова не распознана.
    """
    # Определение фактического формата вызова.
    di_graph: nx.DiGraph | None = None

    # Случай 1: graph — это nx.Graph / DiGraph
    if isinstance(graph, (nx.Graph, nx.DiGraph)):
        di_graph = _nx_to_digraph(graph)

    # Случай 2: graph — это PyG Data
    elif isinstance(graph, Data):
        di_graph = _data_to_digraph(graph)

    # Случай 3: тензорная перегрузка
    # Поддержка позиционного вызова compute_graph_metrics(edge_index, edge_attr, n_nodes)
    # где graph=edge_index, edge_index=edge_attr, edge_attr=n_nodes(int)
    elif isinstance(graph, (torch.Tensor, np.ndarray)):
        # Детекция сдвинутой тройки: graph, edge_index, edge_attr == (E_index, E_attr, N)
        if isinstance(edge_attr, int) and n_nodes is None and edge_index is not None:
            # graph=edge_index, edge_index=edge_attr, edge_attr=n_nodes
            real_edge_index = graph
            real_edge_attr = edge_index
            real_n_nodes: int | None = edge_attr
            di_graph = _tensors_to_digraph(real_edge_index, real_edge_attr, real_n_nodes)
        elif isinstance(n_nodes, int) and graph is not None and edge_index is not None:
            # graph=edge_index, edge_index=edge_attr, n_nodes=N
            real_edge_index = graph
            real_edge_attr = edge_index
            real_n_nodes = n_nodes
            di_graph = _tensors_to_digraph(real_edge_index, real_edge_attr, real_n_nodes)
        elif n_nodes is not None and edge_index is not None:
            # Смешанный случай с ключевыми аргументами
            real_edge_index = graph
            real_edge_attr = edge_index
            real_n_nodes = n_nodes
            di_graph = _tensors_to_digraph(real_edge_index, real_edge_attr, real_n_nodes)
        elif edge_index is None and edge_attr is None and n_nodes is not None:
            # Только graph как edge_index + n_nodes
            di_graph = _tensors_to_digraph(graph, None, n_nodes)
        else:
            # Попытка интерпретировать как (edge_index, edge_attr, n_nodes) с graph=None
            raise TypeError(
                "Неверная сигнатура для тензорного вызова: "
                "ожидается (edge_index, edge_attr, n_nodes)"
            )

    # Случай 4: вызов через ключевые аргументы без graph
    elif graph is None and edge_index is not None:
        # edge_index передан как именованный аргумент, graph отсутствует
        if n_nodes is None:
            raise TypeError("Параметр n_nodes обязателен при тензорном вызове")
        di_graph = _tensors_to_digraph(edge_index, edge_attr, n_nodes)

    else:
        raise TypeError(
            "Не удалось определить тип графа: передайте nx.Graph, "
            "torch_geometric.data.Data или (edge_index, edge_attr, n_nodes)"
        )

    if di_graph is None:
        raise TypeError("Не удалось построить граф из переданных аргументов")

    return _compute_metrics_from_digraph(di_graph)


def _nx_to_digraph(graph: nx.Graph | nx.DiGraph) -> nx.DiGraph:
    """Конвертирует networkx граф в DiGraph с проверкой NaN.

    Args:
        graph: Исходный граф.

    Returns:
        Направленный граф с атрибутом ``weight``.

    Raises:
        ValueError: При наличии ``NaN`` в весах.
    """
    # Проверка NaN в весах рёбер
    for _u, _v, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        # w может быть torch.Tensor скаляром, numpy float или python float
        if isinstance(w, torch.Tensor):
            if torch.isnan(w).any().item():  # type: ignore[union-attr]
                raise ValueError("Веса графа содержат NaN")
        elif isinstance(w, (float, np.floating)):
            if np.isnan(float(w)):
                raise ValueError("Веса графа содержат NaN")
        elif isinstance(w, np.ndarray):
            if np.isnan(w).any():
                raise ValueError("Веса графа содержат NaN")

    if isinstance(graph, nx.DiGraph):
        # Копируем, сохраняя все узлы (включая изолированные)
        di = nx.DiGraph()
        di.add_nodes_from(graph.nodes())
        for u, v, data in graph.edges(data=True):
            w = float(data.get("weight", 1.0)) if not isinstance(
                data.get("weight", 1.0), torch.Tensor
            ) else float(data.get("weight", 1.0).item())  # type: ignore[union-attr]
            if np.isnan(w):
                raise ValueError("Веса графа содержат NaN")
            di.add_edge(u, v, weight=w)
        return di

    # Неориентированный Graph -> преобразуем в DiGraph с двусторонними рёбрами?
    # Для коннектома граф направленный, но для совместимости:
    # сохраняем каждое неориентированное ребро как одно направленное.
    di = nx.DiGraph()
    di.add_nodes_from(graph.nodes())
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 1.0)) if not isinstance(
            data.get("weight", 1.0), torch.Tensor
        ) else float(data.get("weight", 1.0).item())  # type: ignore[union-attr]
        if np.isnan(w):
            raise ValueError("Веса графа содержат NaN")
        di.add_edge(u, v, weight=w)
    return di


def _data_to_digraph(data: Data) -> nx.DiGraph:
    """Конвертирует PyG Data в networkx DiGraph.

    Args:
        data: Объект графа PyG с ``edge_index``, ``edge_attr`` и ``num_nodes``.

    Returns:
        Направленный граф.

    Raises:
        ValueError: При ``NaN`` в весах.
    """
    # Извлечение числа узлов
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        num_nodes = int(data.num_nodes)
    elif hasattr(data, "edge_index") and data.edge_index is not None:
        # Инференс по максимальному индексу
        ei = data.edge_index
        if isinstance(ei, torch.Tensor):
            num_nodes = int(ei.max().item()) + 1 if ei.numel() > 0 else 0
        else:
            num_nodes = int(np.max(ei)) + 1 if np.size(ei) > 0 else 0
    else:
        raise TypeError("Data не содержит edge_index или num_nodes")

    edge_index = data.edge_index
    edge_attr = getattr(data, "edge_attr", None)

    return _tensors_to_digraph(edge_index, edge_attr, num_nodes)


def _tensors_to_digraph(
    edge_index: torch.Tensor | np.ndarray,
    edge_attr: torch.Tensor | np.ndarray | None,
    n_nodes: int | None,
) -> nx.DiGraph:
    """Строит DiGraph из тензоров.

    Args:
        edge_index: Индексы рёбер ``[2, E]`` или ``[E, 2]``.
        edge_attr: Веса ``[E]`` или ``[E, 1]`` или ``None``.
        n_nodes: Число узлов.

    Returns:
        Направленный граф.

    Raises:
        ValueError: При ``NaN`` в весах или некорректной форме.
    """
    if n_nodes is None:
        raise ValueError("n_nodes обязателен для тензорного построения графа")
    if n_nodes < 0:
        raise ValueError("n_nodes должен быть неотрицательным")

    # Нормализация edge_index к numpy массиву формы [2, E]
    if isinstance(edge_index, torch.Tensor):
        if torch.isnan(edge_index.float()).any().item():
            raise ValueError("edge_index содержит NaN")
        ei = edge_index.detach().cpu().numpy()
    elif isinstance(edge_index, np.ndarray):
        if np.isnan(ei := edge_index).any():  # type: ignore[no-redef]
            raise ValueError("edge_index содержит NaN")
        ei = edge_index
    else:
        raise TypeError("edge_index должен быть torch.Tensor или np.ndarray")

    if ei.ndim != 2:
        raise ValueError(f"edge_index должен быть 2-мерным, получен ndim={ei.ndim}")

    # Поддержка обеих ориентаций: [2, E] (PyG) и [E, 2]
    if ei.shape[0] == 2:
        # [2, E]
        src = ei[0]
        dst = ei[1]
        n_edges = ei.shape[1]
    elif ei.shape[1] == 2:
        # [E, 2]
        src = ei[:, 0]
        dst = ei[:, 1]
        n_edges = ei.shape[0]
    else:
        raise ValueError(f"Неверная форма edge_index: {ei.shape}, ожидается [2, E] или [E, 2]")

    # Нормализация edge_attr
    if edge_attr is None:
        weights = np.ones(n_edges, dtype=float)
    else:
        if isinstance(edge_attr, torch.Tensor):
            if torch.isnan(edge_attr.float()).any().item():
                raise ValueError("Веса графа содержат NaN (edge_attr)")
            w = edge_attr.detach().cpu().numpy()
        elif isinstance(edge_attr, np.ndarray):
            if np.isnan(edge_attr).any():
                raise ValueError("Веса графа содержат NaN (edge_attr)")
            w = edge_attr
        else:
            raise TypeError("edge_attr должен быть torch.Tensor, np.ndarray или None")

        # Сжатие размерности [E, 1] -> [E]
        w = np.asarray(w).squeeze()
        if w.ndim == 0:
            # Скаляр — расширяем
            w = np.full(n_edges, float(w), dtype=float)
        if w.shape[0] != n_edges:
            raise ValueError(f"Длина edge_attr ({w.shape[0]}) не совпадает с E={n_edges}")
        weights = w.astype(float)

        # Дополнительная проверка NaN после конверсии
        if np.isnan(weights).any():
            raise ValueError("Веса графа содержат NaN")

    # Построение DiGraph с изолированными узлами
    di = nx.DiGraph()
    di.add_nodes_from(range(n_nodes))
    for u, v, w in zip(src, dst, weights, strict=False):
        ui = int(u)
        vi = int(v)
        wf = float(w)
        if np.isnan(wf):
            raise ValueError("Веса графа содержат NaN")
        di.add_edge(ui, vi, weight=wf)

    return di


def _compute_metrics_from_digraph(di_graph: nx.DiGraph) -> Dict[str, Any]:
    """Вычисляет все метрики из готового DiGraph.

    Args:
        di_graph: Направленный граф.

    Returns:
        Словарь метрик.

    Raises:
        ValueError: При ``NaN`` в весах (дополнительная проверка).
    """
    n_nodes = int(di_graph.number_of_nodes())
    n_edges = int(di_graph.number_of_edges())

    # Проверка NaN (на случай прямого nx графа без предварительной проверки)
    for _u, _v, data in di_graph.edges(data=True):
        w = float(data.get("weight", 1.0))
        if np.isnan(w):
            raise ValueError("Веса графа содержат NaN")

    # Плотность для ориентированного графа: E / (N*(N-1))
    if n_nodes > 1:
        density = float(nx.density(di_graph))
    else:
        density = 0.0

    # Средняя степень: сумма (in+out) / N = 2*E/N для ориентированного без петель
    # networkx degree для DiGraph уже возвращает in+out
    if n_nodes > 0:
        total_degree = sum(dict(di_graph.degree()).values())
        mean_degree = float(total_degree) / float(n_nodes)
    else:
        mean_degree = 0.0

    # Доля взаимных рёбер (reciprocity / recurrence_fraction)
    if n_edges == 0:
        reciprocity = 0.0
    else:
        edge_set = set(di_graph.edges())
        reciprocal_count = sum(1 for u, v in di_graph.edges() if (v, u) in edge_set)
        reciprocity = float(reciprocal_count) / float(n_edges)
    recurrence_fraction = float(reciprocity)

    # Модулярность Louvain на неориентированной версии с абсолютными весами
    # Агрегируем направленные рёбра в неориентированные суммой |w|
    undir = nx.Graph()
    undir.add_nodes_from(di_graph.nodes())
    for u, v, data in di_graph.edges(data=True):
        w = abs(float(data.get("weight", 1.0)))
        if np.isnan(w):
            raise ValueError("Веса графа содержат NaN")
        if undir.has_edge(u, v):
            undir[u][v]["weight"] += w
        else:
            undir.add_edge(u, v, weight=w)
    # Удаляем петли — они не влияют на модулярность
    undir.remove_edges_from(nx.selfloop_edges(undir))

    if undir.number_of_edges() == 0:
        # Нет рёбер — каждый узел отдельный модуль, модулярность 0
        modularity_score = 0.0
        n_modules = int(n_nodes) if n_nodes > 0 else 0
    else:
        # Louvain требует связности не обязательно, но вес должен быть положительным
        # Используем абсолютные веса (уже)
        try:
            partition = community_louvain.best_partition(
                undir, weight="weight", random_state=42
            )
            modularity_score = float(
                community_louvain.modularity(partition, undir, weight="weight")
            )
            n_modules = int(len(set(partition.values())))
        except Exception:
            # Фолбэк при ошибке алгоритма (например, для очень маленьких графов)
            modularity_score = 0.0
            n_modules = 1

    # Средняя длина пути — на крупнейшей слабой компоненте
    if n_nodes <= 1 or n_edges == 0:
        avg_path_length = 0.0
    else:
        # Находим крупнейшую слабую компоненту DiGraph
        if nx.is_weakly_connected(di_graph):
            largest_nodes = set(di_graph.nodes())
            sub_di = di_graph
        else:
            # Ищем largest weakly connected component
            wcc = list(nx.weakly_connected_components(di_graph))
            if not wcc:
                avg_path_length = 0.0
                sub_di = None  # type: ignore[assignment]
                largest_nodes = set()
            else:
                largest_nodes = max(wcc, key=len)
                sub_di = di_graph.subgraph(largest_nodes).copy()  # type: ignore[union-attr]

        if n_nodes > 1 and len(largest_nodes) > 1:  # type: ignore[arg-type]
            # Переводим в неориентированный для пути
            assert sub_di is not None
            undir_sub = sub_di.to_undirected()  # type: ignore[union-attr]
            undir_sub.remove_edges_from(nx.selfloop_edges(undir_sub))
            # Если после удаления петель граф несвязный — берём largest connected
            if not nx.is_connected(undir_sub):
                ccs = list(nx.connected_components(undir_sub))
                if ccs:
                    largest_cc = max(ccs, key=len)
                    undir_sub = undir_sub.subgraph(largest_cc).copy()
            if undir_sub.number_of_nodes() <= 1:
                avg_path_length = 0.0
            else:
                try:
                    avg_path_length = float(nx.average_shortest_path_length(undir_sub))
                except nx.NetworkXError:
                    # Несвязный — fallback к largest CC, иначе 0
                    avg_path_length = 0.0
                except Exception:
                    avg_path_length = float("inf")
        else:
            avg_path_length = 0.0

    return {
        "modularity_score": float(modularity_score),
        "mean_degree": float(mean_degree),
        "recurrence_fraction": float(recurrence_fraction),
        "reciprocity": float(reciprocity),
        "avg_path_length": float(avg_path_length),
        "n_modules": int(n_modules),
        "density": float(density),
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
    }

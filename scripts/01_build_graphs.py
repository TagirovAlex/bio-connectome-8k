"""Скрипт построения подграфов коннектома (шаг 1 пайплайна).

Строит ``bio_2k`` / ``bio_8k`` из сырых CSV FlyWire v783,
генерирует варианты ``rewired`` / ``random_er`` / ``unsigned``,
вычисляет метрики и сохраняет результаты.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch_geometric.data import Data

# Добавляем корень проекта в sys.path для абсолютных импортов src.*
# при запуске как `python scripts/01_build_graphs.py`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.build_graph import build_subgraph, validate_graph  # noqa: E402
from src.data.download import download_flywire_data  # noqa: E402
from src.data.metrics import compute_graph_metrics  # noqa: E402
from src.data.variants import (  # noqa: E402
    create_random_er_graph_from_bio,
    create_rewired_graph,
    create_unsigned_graph,
)


def _parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки.

    Returns:
        Пространство имён с полем ``rebuild``.
    """
    parser = argparse.ArgumentParser(
        description="Построение подграфов коннектома (шаг 1 пайплайна)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Перестроить даже если файлы существуют",
    )
    return parser.parse_args()


def _generate_synthetic_graph(
    n_nodes: int,
    n_edges: int,
    seed: int = 42,
) -> Data:
    """Генерирует синтетический граф для дебага (модульная структура).

    Создаёт SBM-подобный граф с 4 модулями чтобы модулярность ``bio``
    была >0.3. Гарантирует ``sensory >=50``, ``motor >=20`` и отсутствие
    изолированных узлов.

    Args:
        n_nodes: Число узлов.
        n_edges: Целевое число рёбер.
        seed: Сид для воспроизводимости.

    Returns:
        Синтетический граф ``Data``.
    """
    # Фиксируем сиды для воспроизводимости
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Разбиение на 4 модуля (LA/ME/LO/LOP)
    n_modules = 4
    # Назначение модуля каждому узлу (равномерно)
    module_ids = np.array([i % n_modules for i in range(n_nodes)], dtype=int)
    # Перемешиваем чтобы модули не шли подряд (детерминированно)
    perm = np.random.permutation(n_nodes)
    module_ids = module_ids[perm]
    # Обратная перестановка не нужна — просто используем текущее распределение
    # Для генерации рёбер важен только сам модуль узла по индексу

    # Генерация рёбер с модульным смещением: 80% внутри модуля
    edge_set: set[tuple[int, int]] = set()
    src_list: list[int] = []
    dst_list: list[int] = []
    p_in = 0.8

    # Для быстрого выбора узлов одного модуля — индекс по модулям
    nodes_by_module: dict[int, list[int]] = {m: [] for m in range(n_modules)}
    for idx, mod in enumerate(module_ids):
        nodes_by_module[int(mod)].append(idx)

    max_tries = n_edges * 10
    tries = 0
    while len(src_list) < n_edges and tries < max_tries:
        tries += 1
        u = int(np.random.randint(0, n_nodes))
        mod_u = int(module_ids[u])
        # Выбор цели с модульным смещением
        if random.random() < p_in:
            # Внутри модуля
            candidates = nodes_by_module[mod_u]
            v = int(random.choice(candidates))
        else:
            # Межмодульное ребро — случайный узел вне модуля
            v = int(np.random.randint(0, n_nodes))
            # Если случайно попали в тот же модуль — иногда перевыбираем
            if int(module_ids[v]) == mod_u and random.random() < 0.5:
                v = int(np.random.randint(0, n_nodes))
        if u == v:
            continue
        if (u, v) in edge_set:
            continue
        edge_set.add((u, v))
        src_list.append(u)
        dst_list.append(v)

    # Если не добрали рёбра — добиваем случайными
    while len(src_list) < n_edges:
        u = int(np.random.randint(0, n_nodes))
        v = int(np.random.randint(0, n_nodes))
        if u == v or (u, v) in edge_set:
            continue
        edge_set.add((u, v))
        src_list.append(u)
        dst_list.append(v)

    # Патч изолированных узлов (требование валидации)
    if n_nodes > 1 and src_list:
        degree = [0] * n_nodes
        for u, v in zip(src_list, dst_list):
            degree[u] += 1
            degree[v] += 1
        isolated = [i for i, d in enumerate(degree) if d == 0]
        for iso in isolated:
            for _ in range(100):
                other = int(np.random.randint(0, n_nodes))
                if other == iso:
                    continue
                cand = (iso, other) if random.random() < 0.5 else (other, iso)
                if cand in edge_set:
                    continue
                edge_set.add(cand)
                src_list.append(cand[0])
                dst_list.append(cand[1])
                break

    actual_edges = len(src_list)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Знаковые веса: ~75% возбуждающих (+), ~25% тормозных (-)
    # syn_count 1..5
    syn_counts = np.random.randint(1, 6, size=actual_edges)
    is_inhibitory = np.random.rand(actual_edges) < 0.25
    weights = np.where(is_inhibitory, -syn_counts, syn_counts).astype(float)
    edge_attr = torch.tensor(weights, dtype=torch.float)

    # Маски: гарантируем >=50 sensory, >=20 motor без пересечений
    n_sensory = max(50, n_nodes // 10)
    n_motor = max(20, n_nodes // 20)
    if n_sensory + n_motor > n_nodes:
        n_sensory = min(n_sensory, n_nodes // 2)
        n_motor = min(n_motor, n_nodes - n_sensory)
        n_sensory = max(1, n_sensory)
        n_motor = max(1, n_motor)

    perm2 = np.random.permutation(n_nodes)
    sensory_idx = perm2[:n_sensory].tolist()
    motor_idx = perm2[n_sensory : n_sensory + n_motor].tolist()

    sensory_mask = torch.zeros(n_nodes, dtype=torch.bool)
    motor_mask = torch.zeros(n_nodes, dtype=torch.bool)
    sensory_mask[torch.tensor(sensory_idx, dtype=torch.long)] = True
    motor_mask[torch.tensor(motor_idx, dtype=torch.long)] = True

    # Регионы по модулям (LA/ME/LO/LOP)
    region_names = ["LA", "ME", "LO", "LOP"]
    region_list: list[str] = []
    for idx in range(n_nodes):
        # Моторные узлы помечаем как motor
        if bool(motor_mask[idx].item()):
            region_list.append("motor")
        else:
            mod = int(module_ids[idx])
            region_list.append(region_names[mod % len(region_names)])

    root_ids = torch.arange(n_nodes, dtype=torch.long)

    graph = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_attr.clone(),
        num_nodes=n_nodes,
        sensory_mask=sensory_mask,
        motor_mask=motor_mask,
        region=region_list,
        root_ids=root_ids,
    )
    graph.node_ids = root_ids
    return graph


def main() -> None:
    """Точка входа шага 1 — строит все подграфы и метрики."""
    args = _parse_args()

    # Корень проекта — родитель папки scripts
    root = Path(__file__).resolve().parent.parent
    config_path = root / "config" / "graphs.yaml"

    print("[01_build_graphs] Шаг 1: Построение подграфов")
    print(f"[01_build_graphs] Корень проекта: {root}")
    print(f"[01_build_graphs] Конфиг: {config_path}")

    # Загрузка конфига
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    metrics_dir = root / "results" / "metrics"

    # Создаём необходимые директории
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Проверка наличия сырых данных
    conn_path = raw_dir / "connections.csv"
    annot_path = raw_dir / "neuron_annotations.csv"
    raw_available = conn_path.is_file() and annot_path.is_file()

    synthetic_mode = False

    if not raw_available:
        print("[01_build_graphs] Сырые CSV не найдены — пробуем скачать")
        print(f"[01_build_graphs]   connections.csv: {'есть' if conn_path.is_file() else 'нет'}")
        print(f"[01_build_graphs]   neuron_annotations.csv: {'есть' if annot_path.is_file() else 'нет'}")
        try:
            download_flywire_data(raw_dir)
        except Exception as exc:  # noqa: BLE001
            # Сеть недоступна или скачивание не удалось — перейдём в синтетический режим
            print(f"[01_build_graphs] Предупреждение: не удалось скачать данные: {exc}")

        # Повторная проверка после попытки скачивания
        raw_available = conn_path.is_file() and annot_path.is_file()

    if not raw_available:
        synthetic_mode = True
        print(
            "[01_build_graphs] Предупреждение: data/raw пустой — "
            "генерирую синтетические графы для дебага (smoke-тест). "
            "Для реальных экспериментов скачайте FlyWire v783 CSV в data/raw/."
        )

    # Ожидаемые файлы и соответствия ключей метрик
    # Ключи в graph_stats.json — без расширения .pt
    expected_files: dict[str, Path] = {
        "bio_2k": processed_dir / "bio_2k.pt",
        "bio_8k": processed_dir / "bio_8k.pt",
        "rewired_2k": processed_dir / "rewired_2k.pt",
        "rewired_8k": processed_dir / "rewired_8k.pt",
        "random_er_2k": processed_dir / "random_er_2k.pt",
        "unsigned_2k": processed_dir / "unsigned_2k.pt",
    }

    # Хранилище графов в памяти (для вариантов и метрик)
    graphs: dict[str, Data] = {}

    # Шаг 1: bio графы (2k и 8k)
    for scale in ("2k", "8k"):
        key = f"bio_{scale}"
        out_path = expected_files[key]

        # Проверка --rebuild: пропускаем если файл уже есть
        if out_path.is_file() and not args.rebuild:
            print(f"[01_build_graphs] Пропускаю {key}: файл уже существует ({out_path}), используйте --rebuild для пересборки")
            try:
                graphs[key] = torch.load(str(out_path), weights_only=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[01_build_graphs] Не удалось загрузить {out_path}: {exc} — пересобираю")
                # Сбрасываем флаг чтобы пересобрать ниже
                graphs.pop(key, None)
            else:
                continue

        # Построение графа
        if synthetic_mode:
            # Синтетический дебаг-граф
            target_nodes_cfg = config.get("scales", {}).get(scale, {}).get("target_nodes", 2000 if scale == "2k" else 8000)
            n_nodes = int(target_nodes_cfg)
            # Для быстрого smoke-теста используем умеренное число рёбер
            n_edges = n_nodes * 20
            print(f"[01_build_graphs] Генерация синтетического {key}: узлов={n_nodes}, рёбер~{n_edges}")
            graph = _generate_synthetic_graph(n_nodes, n_edges, seed=42 if scale == "2k" else 123)
        else:
            print(f"[01_build_graphs] Строю {key} из сырых CSV (scale={scale})")
            graph = build_subgraph(raw_dir, config, scale)

        # Валидация
        is_valid = validate_graph(graph)
        if not is_valid:
            print(f"[01_build_graphs] Предупреждение: граф {key} не прошёл валидацию (см. лог выше)")

        # Сохранение
        torch.save(graph, str(out_path))
        print(f"[01_build_graphs] Сохранён {key} -> {out_path} (узлов={int(graph.num_nodes)}, рёбер={int(graph.edge_index.size(1))})")
        graphs[key] = graph

    # Гарантируем что bio графы загружены (если были пропущены выше но нужны для вариантов)
    for key in ("bio_2k", "bio_8k"):
        if key not in graphs:
            out_path = expected_files[key]
            if out_path.is_file():
                try:
                    graphs[key] = torch.load(str(out_path), weights_only=False)
                except Exception as exc:  # noqa: BLE001
                    print(f"[01_build_graphs] Ошибка загрузки {key}: {exc}")
                    raise

    # Шаг 2: варианты на основе bio
    # rewired_2k / rewired_8k
    for scale in ("2k", "8k"):
        key = f"rewired_{scale}"
        base_key = f"bio_{scale}"
        out_path = expected_files[key]
        if out_path.is_file() and not args.rebuild:
            print(f"[01_build_graphs] Пропускаю {key}: файл уже существует ({out_path})")
            try:
                graphs[key] = torch.load(str(out_path), weights_only=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[01_build_graphs] Не удалось загрузить {out_path}: {exc} — пересобираю")
                graphs.pop(key, None)
            else:
                continue
        base_graph = graphs.get(base_key)
        if base_graph is None:
            print(f"[01_build_graphs] Пропускаю {key}: базовый граф {base_key} отсутствует")
            continue
        print(f"[01_build_graphs] Строю {key} из {base_key} (Maslov-Sneppen)")
        seed = 42 if scale == "2k" else 123
        rewired = create_rewired_graph(base_graph, seed=seed)
        is_valid = validate_graph(rewired)
        if not is_valid:
            print(f"[01_build_graphs] Предупреждение: граф {key} не прошёл валидацию")
        torch.save(rewired, str(out_path))
        print(f"[01_build_graphs] Сохранён {key} -> {out_path}")
        graphs[key] = rewired

    # random_er_2k
    key = "random_er_2k"
    out_path = expected_files[key]
    if out_path.is_file() and not args.rebuild:
        print(f"[01_build_graphs] Пропускаю {key}: файл уже существует ({out_path})")
        try:
            graphs[key] = torch.load(str(out_path), weights_only=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[01_build_graphs] Не удалось загрузить {out_path}: {exc} — пересобираю")
            graphs.pop(key, None)
        else:
            pass
    if key not in graphs:
        base_graph = graphs.get("bio_2k")
        if base_graph is None:
            print(f"[01_build_graphs] Пропускаю {key}: базовый граф bio_2k отсутствует")
        else:
            print(f"[01_build_graphs] Строю {key} из bio_2k (Erdos-Renyi)")
            # Используем overload create_random_er_graph_from_bio / create_random_er_graph
            try:
                random_er = create_random_er_graph_from_bio(base_graph, seed=42)
            except Exception:  # noqa: BLE001
                from src.data.variants import create_random_er_graph as _crer

                random_er = _crer(base_graph, 42)  # type: ignore[call-arg]
            is_valid = validate_graph(random_er)
            if not is_valid:
                print(f"[01_build_graphs] Предупреждение: граф {key} не прошёл валидацию")
            torch.save(random_er, str(out_path))
            print(f"[01_build_graphs] Сохранён {key} -> {out_path}")
            graphs[key] = random_er

    # unsigned_2k
    key = "unsigned_2k"
    out_path = expected_files[key]
    if out_path.is_file() and not args.rebuild:
        print(f"[01_build_graphs] Пропускаю {key}: файл уже существует ({out_path})")
        try:
            graphs[key] = torch.load(str(out_path), weights_only=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[01_build_graphs] Не удалось загрузить {out_path}: {exc} — пересобираю")
            graphs.pop(key, None)
        else:
            pass
    if key not in graphs:
        base_graph = graphs.get("bio_2k")
        if base_graph is None:
            print(f"[01_build_graphs] Пропускаю {key}: базовый граф bio_2k отсутствует")
        else:
            print(f"[01_build_graphs] Строю {key} из bio_2k (|w|)")
            unsigned = create_unsigned_graph(base_graph)
            is_valid = validate_graph(unsigned)
            if not is_valid:
                print(f"[01_build_graphs] Предупреждение: граф {key} не прошёл валидацию")
            torch.save(unsigned, str(out_path))
            print(f"[01_build_graphs] Сохранён {key} -> {out_path}")
            graphs[key] = unsigned

    # Шаг 3: метрики для каждого графа
    # Дозагружаем пропущенные графы если нужно
    for k, p in expected_files.items():
        if k not in graphs and p.is_file():
            try:
                graphs[k] = torch.load(str(p), weights_only=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[01_build_graphs] Не удалось загрузить {k} для метрик: {exc}")

    metrics: dict[str, dict] = {}
    for k in ("bio_2k", "rewired_2k", "random_er_2k", "unsigned_2k", "bio_8k", "rewired_8k"):
        graph = graphs.get(k)
        if graph is None:
            print(f"[01_build_graphs] Метрики: пропускаю {k} — граф отсутствует")
            continue
        print(f"[01_build_graphs] Вычисляю метрики для {k}")
        try:
            m = compute_graph_metrics(graph)
        except Exception as exc:  # noqa: BLE001
            print(f"[01_build_graphs] Ошибка вычисления метрик для {k}: {exc}")
            continue
        # Сериализуем в JSON-совместимый вид (numpy -> python)
        clean: dict = {}
        for mk, mv in m.items():
            if isinstance(mv, (np.floating, np.integer)):
                mv = mv.item()
            if isinstance(mv, float) and (np.isnan(mv) or np.isinf(mv)):
                mv = 0.0
            clean[mk] = mv
        metrics[k] = clean
        print(
            f"[01_build_graphs] Метрики {k}: "
            f"modularity={clean.get('modularity_score', 0):.3f}, "
            f"mean_degree={clean.get('mean_degree', 0):.2f}, "
            f"reciprocity={clean.get('reciprocity', 0):.3f}"
        )

    # Сохранение метрик
    metrics_path = metrics_dir / "graph_stats.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[01_build_graphs] Метрики сохранены -> {metrics_path}")
    print("[01_build_graphs] Готово")


if __name__ == "__main__":
    main()

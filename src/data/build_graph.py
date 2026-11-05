"""Модуль построения подграфа зрительного пути дрозофилы.

Загружает сырые CSV FlyWire v783, фильтрует по регионам и классам,
агрегирует синапсы и строит знаковый граф PyG.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _normalize_nt(
    nt_type: Any,
    nt_map: Optional[Dict[str, str]] = None,
) -> str:
    """Нормализует nt_type к нижнему регистру полного имени медиатора.

    Сокращения из connections.csv (ACH/GLUT/GABA/OCT/SER/DA) сопоставляются
    с полными именами через ``nt_map`` из config/graphs.yaml. Если ``nt_map``
    не задан — используется встроенный fallback-маппинг сокращений.
    NaN/пустые значения нормализуются к пустой строке.

    Args:
        nt_type: Тип нейромедиатора (сокращение или полное имя).
        nt_map: Словарь сокращение -> полное имя из конфига.

    Returns:
        Полное имя в нижнем регистре (или пустая строка для NaN).
    """
    nt_normalized = str(nt_type).strip().lower() if pd.notna(nt_type) else ""
    if nt_map:
        for code, full in nt_map.items():
            if nt_normalized == str(code).strip().lower():
                return str(full).strip().lower()
        return nt_normalized
    _code_map = {
        "ach": "acetylcholine",
        "glut": "glutamate",
        "gaba": "gaba",
        "oct": "octopamine",
        "ser": "serotonin",
        "da": "dopamine",
    }
    return _code_map.get(nt_normalized, nt_normalized)


def _is_motor_row(
    row: Optional[pd.Series],
    motor_class: str,
    motor_by: str,
    motor_extra_classes: List[str],
    has_class: bool,
    has_super_class: bool,
) -> bool:
    """Проверяет, является ли строка аннотаций моторным нейроном.

    Критерии (из config/graphs.yaml):
    - motor_by == "super_class": super_class == motor_class;
    - motor_by == "class": class == motor_class;
    - motor_by == "any": оба равенства (объединение);
    - class входит в motor_extra_classes (например "brain_motor_neuron").

    Args:
        row: Строка аннотаций или None (узел вне аннотаций).
        motor_class: Значение класса/суперкласса, помечающее мотор.
        motor_by: Колонка для поиска motor_class.
        motor_extra_classes: Дополнительные классы из колонки class.
        has_class: Есть ли колонка class в аннотациях.
        has_super_class: Есть ли колонка super_class в аннотациях.

    Returns:
        True, если узел — моторный нейрон.
    """
    if row is None:
        return False
    if has_super_class and motor_by in ("any", "super_class"):
        if str(row.get("super_class", "")) == motor_class:
            return True
    if has_class and motor_by in ("any", "class"):
        if str(row.get("class", "")) == motor_class:
            return True
    if has_class and motor_extra_classes:
        if str(row.get("class", "")) in motor_extra_classes:
            return True
    return False


def build_subgraph(raw_data_dir: Path, config: Dict[str, Any], scale: str) -> Data:
    """Строит подграф зрительного пути из сырых CSV.

    Фильтрует по регионам (LA, ME, LO, LOP), агрегирует синапсы,
    назначает знаковые веса (+ для возбуждающих, - для тормозных).

    Поддерживает два режима отбора из config:
    - ``largest_weak_component_then_top_degree`` (для 2k): берёт
      наибольшую слабо-связную компоненту, затем топ-N по степени.
    - ``all_visual_plus_motor`` (для 8k): берёт все visual+motor узлы,
      если их больше целевого размера — топ-N по степени.

    Args:
        raw_data_dir: Путь к папке с CSV файлами (data/raw/).
        config: Параметры из config/graphs.yaml.
        scale: Масштаб подграфа ("2k" или "8k").

    Returns:
        Data: Объект графа PyG с полями edge_index, edge_attr,
            sensory_mask, motor_mask, region, root_ids.

    Raises:
        FileNotFoundError: Если отсутствуют connections.csv или
            neuron_annotations.csv.
        ValueError: Если scale отсутствует в config или пулы узлов пусты.
    """
    raw_data_dir = Path(raw_data_dir)

    # Чтение параметров из конфига без хардкода
    visual_regions: List[str] = list(config.get("visual_regions", []))
    motor_class: str = str(config.get("motor_class", "motor"))
    motor_by: str = str(config.get("motor_by", "any"))
    motor_extra_classes: List[str] = [
        str(x) for x in config.get("motor_extra_classes", [])
    ]
    nt_map: Optional[Dict[str, str]] = config.get("nt_map")
    excitatory_nt: List[str] = list(config.get("excitatory_nt", []))
    inhibitory_nt: List[str] = list(config.get("inhibitory_nt", []))
    scales_cfg: Dict[str, Any] = dict(config.get("scales", {}))

    if scale not in scales_cfg:
        raise ValueError(
            f"Масштаб '{scale}' отсутствует в config/scales. "
            f"Доступны: {list(scales_cfg.keys())}"
        )

    target_nodes: int = int(scales_cfg[scale].get("target_nodes", 2000))
    selection: str = str(scales_cfg[scale].get("selection", ""))

    # Загрузка сырых данных
    conn_path = raw_data_dir / "connections.csv"
    annot_path = raw_data_dir / "neuron_annotations.csv"

    if not conn_path.exists():
        raise FileNotFoundError(f"Не найден файл связей: {conn_path}")
    if not annot_path.exists():
        raise FileNotFoundError(f"Не найден файл аннотаций: {annot_path}")

    # Загрузка аннотаций и связей. connections.csv ~260 МБ — читаем только
    # нужные колонки; nt_type как категорию для скорости/памяти.
    annotations = pd.read_csv(annot_path)
    _conn_cols = pd.read_csv(conn_path, nrows=0).columns.tolist()
    _use_cols = [
        c
        for c in ("pre_root_id", "post_root_id", "syn_count", "nt_type")
        if c in _conn_cols
    ]
    connections = pd.read_csv(
        conn_path,
        usecols=_use_cols,
        dtype={
            "pre_root_id": "int64",
            "post_root_id": "int64",
            "nt_type": "category",
        },
    )

    # Проверка ключевых столбцов аннотаций
    if "root_id" not in annotations.columns:
        raise ValueError("В neuron_annotations.csv отсутствует столбец 'root_id'")
    if "region" not in annotations.columns:
        raise ValueError("В neuron_annotations.csv отсутствует столбец 'region'")

    # Проверка ключевых столбцов связей
    for col in ("pre_root_id", "post_root_id", "syn_count"):
        if col not in connections.columns:
            raise ValueError(f"В connections.csv отсутствует столбец '{col}'")
    # nt_type в связях может отсутствовать — тогда считаем все возбуждающими
    if "nt_type" not in connections.columns:
        connections["nt_type"] = ""

    # Фильтрация нейронов по зрительным регионам и моторному классу
    # visual: region в visual_regions
    visual_mask = annotations["region"].isin(visual_regions)

    has_class = "class" in annotations.columns
    has_super_class = "super_class" in annotations.columns

    # motor: super_class == motor_class (motor_by), class == motor_class
    # или class в motor_extra_classes (напр. brain_motor_neuron).
    # Оба критерия дают >0 узлов — берём объединение.
    motor_mask_series = pd.Series(False, index=annotations.index)
    if has_super_class and motor_by in ("any", "super_class"):
        motor_mask_series = motor_mask_series | (
            annotations["super_class"].astype(str) == motor_class
        )
    if has_class and motor_by in ("any", "class"):
        motor_mask_series = motor_mask_series | (
            annotations["class"].astype(str) == motor_class
        )
    if has_class and motor_extra_classes:
        motor_mask_series = motor_mask_series | annotations["class"].astype(
            str
        ).isin(motor_extra_classes)
    # Дополнительная поддержка если класс записан в колонке hemilineage/class иным именем
    # уже покрыто проверкой class/super_class — fallback не нужен

    pool_mask = visual_mask | motor_mask_series
    pool_annotations = annotations.loc[pool_mask].copy()

    if pool_annotations.empty:
        raise ValueError(
            "Пустой пул нейронов после фильтрации по visual_regions и motor_class. "
            f"visual_regions={visual_regions}, motor_class={motor_class}"
        )

    pool_ids = set(pool_annotations["root_id"].astype(int).tolist())

    # Фильтрация связей: оба конца в пуле
    # Приведение типов для корректного isin
    connections["pre_root_id"] = connections["pre_root_id"].astype(int)
    connections["post_root_id"] = connections["post_root_id"].astype(int)
    filtered_conn = connections[
        connections["pre_root_id"].isin(pool_ids)
        & connections["post_root_id"].isin(pool_ids)
    ].copy()

    if filtered_conn.empty:
        raise ValueError("Нет связей внутри пула visual+motor")

    # Агрегация синапсов до уровня пар нейронов (сумма syn_count)
    # nt_type агрегируем первым значением (медиатор пары обычно один)
    # Для детерминизма сортируем
    agg_dict = {"syn_count": "sum", "nt_type": "first"}
    # Если столбец syn_count содержит NaN — заменим на 0 перед суммой
    filtered_conn["syn_count"] = pd.to_numeric(
        filtered_conn["syn_count"], errors="coerce"
    ).fillna(0)
    grouped = (
        filtered_conn.groupby(["pre_root_id", "post_root_id"], as_index=False)
        .agg(agg_dict)
        .sort_values(["pre_root_id", "post_root_id"])
        .reset_index(drop=True)
    )

    # Построение направленного графа для вычисления компонент и степеней
    directed_g = nx.DiGraph()
    directed_g.add_nodes_from(pool_ids)
    # Добавление рёбер из агрегированных пар
    edge_tuples = list(
        zip(
            grouped["pre_root_id"].astype(int).tolist(),
            grouped["post_root_id"].astype(int).tolist(),
        )
    )
    directed_g.add_edges_from(edge_tuples)

    # Выбор узлов по масштабу
    selected_ids: List[int]

    if selection == "largest_weak_component_then_top_degree":
        # Берём наибольшую слабо-связную компоненту
        if len(directed_g) == 0:
            raise ValueError("Граф пуст, нельзя вычислить слабо-связные компоненты")
        weak_components = list(nx.weakly_connected_components(directed_g))
        if not weak_components:
            raise ValueError("Не найдены слабо-связные компоненты")
        largest_cc = max(weak_components, key=len)
        subgraph_lcc = directed_g.subgraph(largest_cc).copy()

        # Степень как сумма входящей и исходящей
        degree_dict = {
            n: subgraph_lcc.in_degree(n) + subgraph_lcc.out_degree(n)
            for n in subgraph_lcc.nodes()
        }
        # Сортировка по степени убывания, при равенстве — по id для детерминизма
        sorted_by_degree = sorted(
            degree_dict.keys(), key=lambda n: (-degree_dict[n], n)
        )
        # Топ-N, но не больше размера компоненты
        take_n = min(target_nodes, len(sorted_by_degree))
        selected_ids = sorted_by_degree[:take_n]

    elif selection == "all_visual_plus_motor":
        # Все visual+motor, если больше целевого — топ по степени
        if len(pool_ids) <= target_nodes:
            selected_ids = list(pool_ids)
        else:
            degree_dict_all = {
                n: directed_g.in_degree(n) + directed_g.out_degree(n)
                for n in directed_g.nodes()
            }
            sorted_pool = sorted(
                pool_ids, key=lambda n: (-degree_dict_all.get(n, 0), n)
            )
            selected_ids = sorted_pool[:target_nodes]
    else:
        raise ValueError(
            f"Неизвестный метод selection '{selection}' для scale '{scale}'"
        )

    if not selected_ids:
        raise ValueError("Отбор узлов дал пустой список")

    # ---- Гарантия sensory >=50 и motor >=20 с фильтрацией по связности ----
    # Кандидаты отбираются ТОЛЬКО среди узлов, имеющих >=1 ребро к текущему
    # selected_set (проверка через directed_g). Сортировка — по степени в LCC
    # убывающей. Если связанных кандидатов не хватает, изолированные узлы
    # устраняются пост-фильтрацией ниже.
    _pool_annot_index = pool_annotations.drop_duplicates(
        subset=["root_id"]
    ).set_index("root_id")
    _sensory_region = "LA"
    if visual_regions and _sensory_region not in visual_regions:
        _sensory_region = visual_regions[0]

    if selection == "largest_weak_component_then_top_degree":
        _deg_pool = degree_dict
    elif "degree_dict_all" in locals():
        _deg_pool = degree_dict_all
    else:
        _deg_pool = {}

    def _annot_row(rid: int) -> Optional[pd.Series]:
        """Возвращает строку аннотаций узла (первую при дубликатах root_id)."""
        if rid not in _pool_annot_index.index:
            return None
        _row = _pool_annot_index.loc[rid]
        if isinstance(_row, pd.DataFrame):
            _row = _row.iloc[0]
        return _row

    def _is_sensory_rid(rid: int) -> bool:
        """True, если узел относится к сенсорному региону (_sensory_region)."""
        _row = _annot_row(rid)
        return _row is not None and str(_row.get("region", "")) == _sensory_region

    def _is_motor_rid(rid: int) -> bool:
        """True, если узел является моторным нейроном (motor_by/extra_classes)."""
        return _is_motor_row(
            _annot_row(rid),
            motor_class,
            motor_by,
            motor_extra_classes,
            has_class,
            has_super_class,
        )

    def _lcc_deg(rid: int) -> int:
        """Степень узла в LCC/полном графе (для сортировки кандидатов)."""
        if rid in _deg_pool:
            return int(_deg_pool[rid])
        if rid in directed_g:
            return int(directed_g.in_degree(rid)) + int(directed_g.out_degree(rid))
        return 0

    def _neighbors_of_selected(sel_set: set) -> set:
        """Все узлы directed_g, смежные хотя бы с одним узлом sel_set."""
        neigh: set = set()
        for s in sel_set:
            if s in directed_g:
                neigh.update(directed_g.predecessors(s))
                neigh.update(directed_g.successors(s))
        return neigh

    def _connected_candidates(
        sel_set: set, pred: Callable[[int], bool]
    ) -> List[int]:
        """Связанные с sel_set кандидаты из пула по убыванию степени в LCC."""
        neigh = _neighbors_of_selected(sel_set)
        cands = [
            rid
            for rid in pool_ids
            if rid not in sel_set and pred(rid) and rid in neigh
        ]
        cands.sort(key=lambda n: (-_lcc_deg(n), n))
        return cands

    selected_set = set(selected_ids)
    # Вставка гарантийных узлов в цикле: заменяем хвост списка по степени,
    # пока не достигнуты цели (sensory>=50, motor>=20) или не исчерпаны
    # связанные кандидаты (тогда добработка — пост-фильтрацией ниже).
    _tail_pos = 0
    for _guar_round in range(8):
        _need_s = max(0, 50 - sum(1 for rid in selected_ids if _is_sensory_rid(rid)))
        _need_m = max(0, 20 - sum(1 for rid in selected_ids if _is_motor_rid(rid)))
        if _need_s == 0 and _need_m == 0:
            break
        _added_any = False
        if _need_s > 0:
            _sens_take = _connected_candidates(selected_set, _is_sensory_rid)[
                :_need_s
            ]
            for _i, _rid in enumerate(_sens_take):
                selected_ids[-(_tail_pos + _i + 1)] = _rid
            _tail_pos += len(_sens_take)
            _added_any = _added_any or bool(_sens_take)
            selected_set = set(selected_ids)
        if _need_m > 0:
            _motor_take = _connected_candidates(selected_set, _is_motor_rid)[
                :_need_m
            ]
            for _i, _rid in enumerate(_motor_take):
                selected_ids[-(_tail_pos + _i + 1)] = _rid
            _tail_pos += len(_motor_take)
            _added_any = _added_any or bool(_motor_take)
            selected_set = set(selected_ids)
        if not _added_any:
            break

    selected_set = set(selected_ids)

    # ---- Пост-фильтрация изолированных узлов ----
    # Степень в индуцированном подграфе считаем через Counter по рёбрам между
    # отобранными узлами. Изолированные узлы (deg==0) НЕ sensory/motor удаляем
    # и добираем следующих по степени связанных кандидатов из пула; sensory/
    # motor с deg==0 заменяем на связанных кандидатов того же типа. Кандидаты
    # берутся ТОЛЬКО связные с ядром (_neigh), fallback на весь пул исключён.
    # _tmp_edges пересчитывается один раз на раунд. Раунд завершается при
    # отсутствии изолятов, при _replaced==0 (нет прогресса) либо после макс.
    # 20 раундов; оставшиеся изолированные узлы ловит финальный RuntimeError.
    _kept_isolated: set = set()
    for _post_round in range(20):
        _tmp_edges = grouped[
            grouped["pre_root_id"].isin(selected_set)
            & grouped["post_root_id"].isin(selected_set)
        ]
        _deg_counter: Counter = Counter()
        if not _tmp_edges.empty:
            _deg_counter.update(_tmp_edges["pre_root_id"].astype(int).tolist())
            _deg_counter.update(_tmp_edges["post_root_id"].astype(int).tolist())
        _isolated_nodes = [n for n in selected_ids if _deg_counter.get(n, 0) == 0]
        if not _isolated_nodes:
            break
        _core_set = selected_set - set(_isolated_nodes)
        if not _core_set:
            break
        # Множество соседей ядра считаем ОДИН раз на раунд (до генератора
        # кандидатов), проверка членства в set внутри генератора — O(1).
        # Изоляты по определению не смежны с ядром и на пул не влияют.
        _neigh = _neighbors_of_selected(_core_set)
        _cand_pool = sorted(
            (n for n in pool_ids if n not in selected_set and n in _neigh),
            key=lambda n: (-_lcc_deg(n), n),
        )
        _replaced = 0
        for _iso in list(_isolated_nodes):
            if _iso in _kept_isolated:
                continue
            _iso_sensory = _is_sensory_rid(_iso)
            _iso_motor = _is_motor_rid(_iso)
            _replacement: Optional[int] = None
            if _iso_sensory or _iso_motor:
                # sensory/motor изолированные — ищем связанного кандидата того же типа
                for _cand in _cand_pool:
                    if _cand in selected_set:
                        continue
                    if _iso_sensory and not _is_sensory_rid(_cand):
                        continue
                    if _iso_motor and not _is_motor_rid(_cand):
                        continue
                    _replacement = _cand
                    break
                if _replacement is None:
                    print(
                        f"Внимание: изолированный sensory/motor узел {_iso} "
                        f"(sensory={_iso_sensory}, motor={_iso_motor}) оставлен — "
                        f"связанных кандидатов того же типа не найдено"
                    )
                    _kept_isolated.add(_iso)
                    continue
            else:
                # дженерик-узел: заменяем на первого свободного связанного кандидата
                for _cand in _cand_pool:
                    if _cand not in selected_set:
                        _replacement = _cand
                        break
                if _replacement is None:
                    break
            _iso_idx = selected_ids.index(_iso)
            selected_ids[_iso_idx] = _replacement
            selected_set.discard(_iso)
            selected_set.add(_replacement)
            _core_set.add(_replacement)
            _replaced += 1
        if _replaced == 0:
            break
        _kept_isolated.clear()

    # Финальная фильтрация рёбер по отобранным узлам
    final_edges = grouped[
        grouped["pre_root_id"].isin(selected_set)
        & grouped["post_root_id"].isin(selected_set)
    ].copy()

    # Назначение знаковых весов. Правила (см. config/graphs.yaml):
    # +syn_count для excitatory_nt, -syn_count для inhibitory_nt (GABA),
    # иначе +syn_count. Векторизованно (сотни тысяч рёбер) через
    # _normalize_nt + numpy; приоритет возбуждающего типа сохранён.
    nt_norm = final_edges["nt_type"].map(lambda v: _normalize_nt(v, nt_map))
    exc_set = {str(x).strip().lower() for x in excitatory_nt}
    inh_set = {str(x).strip().lower() for x in inhibitory_nt}
    _syn_arr = final_edges["syn_count"].astype(float).to_numpy()
    _is_inh = nt_norm.isin(inh_set).to_numpy()
    _is_exc = nt_norm.isin(exc_set).to_numpy()
    final_edges["weight"] = np.where(_is_inh & ~_is_exc, -_syn_arr, _syn_arr)

    # Детерминированное упорядочивание узлов для отображения id -> индекс
    selected_sorted = sorted(selected_ids)
    id_to_idx = {rid: idx for idx, rid in enumerate(selected_sorted)}

    # Сбор регионов и масок
    # Индексируем аннотации по root_id для быстрого поиска
    # Если есть дубликаты root_id — берём первое вхождение
    annot_indexed = pool_annotations.drop_duplicates(subset=["root_id"]).set_index(
        "root_id"
    )
    # fallback на полный annotations если узел был в пуле но не в pool_annotations
    # (не должно случиться, но для надёжности)
    full_annot_indexed = annotations.drop_duplicates(subset=["root_id"]).set_index(
        "root_id"
    )

    region_list: List[str] = []
    sensory_flags: List[bool] = []
    motor_flags: List[bool] = []

    # Сенсорный регион — LA, если есть в visual_regions, иначе первый из списка
    sensory_region = "LA"
    if visual_regions and sensory_region not in visual_regions:
        sensory_region = visual_regions[0]

    for rid in selected_sorted:
        # Регион узла
        if rid in annot_indexed.index:
            row = annot_indexed.loc[rid]
        elif rid in full_annot_indexed.index:
            row = full_annot_indexed.loc[rid]
        else:
            row = None

        if row is not None:
            # row может быть Series или DataFrame при дубликатах
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            region_val = str(row.get("region", "unknown"))
            # Определение моторности по конфигу (motor_by/motor_extra_classes)
            is_motor = _is_motor_row(
                row,
                motor_class,
                motor_by,
                motor_extra_classes,
                has_class,
                has_super_class,
            )
        else:
            region_val = "unknown"
            is_motor = False

        region_list.append(region_val)
        sensory_flags.append(region_val == sensory_region)
        motor_flags.append(is_motor)

    # Построение тензоров PyG
    if final_edges.empty:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float)
    else:
        src_idx = [id_to_idx[int(r)] for r in final_edges["pre_root_id"].tolist()]
        dst_idx = [id_to_idx[int(r)] for r in final_edges["post_root_id"].tolist()]
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        edge_attr = torch.tensor(
            final_edges["weight"].astype(float).tolist(), dtype=torch.float
        )

    num_nodes = len(selected_sorted)

    sensory_mask = torch.tensor(sensory_flags, dtype=torch.bool)
    motor_mask = torch.tensor(motor_flags, dtype=torch.bool)
    root_ids_tensor = torch.tensor(selected_sorted, dtype=torch.long)

    # Создание объекта Data
    graph = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_attr.clone() if edge_attr.numel() > 0 else edge_attr,
        num_nodes=num_nodes,
        sensory_mask=sensory_mask,
        motor_mask=motor_mask,
        region=region_list,
        root_ids=root_ids_tensor,
    )
    # Дополнительные поля для совместимости
    graph.node_ids = root_ids_tensor
    # Храним отображение root_id -> idx как словарь (для анализа)
    graph.root_id_to_idx = id_to_idx  # type: ignore[attr-defined]

    # Повторная валидация перед возвратом. Изолированные узлы — фатальная
    # ошибка: выбрасываем RuntimeError (не проглатываем, см. data_registry.md).
    validate_graph(graph)
    _isolated_final = _count_isolated(graph)
    if _isolated_final > 0:
        raise RuntimeError(
            f"Валидация провалена: найдено {_isolated_final} изолированных узлов "
            f"(узлов={num_nodes}, рёбер={int(edge_index.size(1))}, "
            f"sensory={int(sensory_mask.sum().item())}, "
            f"motor={int(motor_mask.sum().item())})"
        )

    return graph


def _count_isolated(graph: Data) -> int:
    """Считает число изолированных узлов (степень 0) в графе.

    Args:
        graph: Объект графа PyG.

    Returns:
        Число узлов с нулевой степенью (вход+выход).
    """
    num_nodes = int(graph.num_nodes) if hasattr(graph, "num_nodes") else 0
    edge_index = graph.edge_index
    if edge_index is None or edge_index.numel() == 0:
        return num_nodes
    deg = torch.zeros(num_nodes, dtype=torch.long)
    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)
    valid_src = src[(src >= 0) & (src < num_nodes)]
    valid_dst = dst[(dst >= 0) & (dst < num_nodes)]
    if valid_src.numel() > 0:
        deg.index_add_(0, valid_src, torch.ones_like(valid_src))
    if valid_dst.numel() > 0:
        deg.index_add_(0, valid_dst, torch.ones_like(valid_dst))
    return int((deg == 0).sum().item())


def validate_graph(graph: Data) -> bool:
    """Валидация построенного графа.

    Проверяет критерии из data/data_registry.md:
    - нет изолированных узлов
    - нет NaN в весах
    - sensory_mask >= 50
    - motor_mask >= 20

    Args:
        graph: Объект графа PyG.

    Returns:
        True если граф валиден, иначе False. Причины невалидности
        печатаются в stdout (русский лог, как в data_registry).
    """
    is_valid = True

    # Число узлов и рёбер
    num_nodes = int(graph.num_nodes) if hasattr(graph, "num_nodes") else 0
    # Попытка вывести num_nodes из масок если поле отсутствует
    if num_nodes == 0 and hasattr(graph, "sensory_mask"):
        try:
            num_nodes = int(graph.sensory_mask.size(0))
        except Exception:
            num_nodes = 0

    if not hasattr(graph, "edge_index") or graph.edge_index is None:
        print("Валидация провалена: отсутствует edge_index")
        return False

    edge_index = graph.edge_index
    num_edges = int(edge_index.size(1)) if edge_index.dim() == 2 else 0

    if num_edges == 0:
        print("Валидация провалена: граф без рёбер (edge_index пуст)")
        is_valid = False
    else:
        # Проверка изолированных узлов: степень (вход+выход) > 0
        # Используем bincount через index_add_
        try:
            deg = torch.zeros(num_nodes, dtype=torch.long)
            # Учитываем и истоки и стоки
            src = edge_index[0].to(torch.long)
            dst = edge_index[1].to(torch.long)
            # Фильтруем индексы вне диапазона на случай несоответствия
            valid_src = src[(src >= 0) & (src < num_nodes)]
            valid_dst = dst[(dst >= 0) & (dst < num_nodes)]
            if valid_src.numel() > 0:
                deg.index_add_(0, valid_src, torch.ones_like(valid_src))
            if valid_dst.numel() > 0:
                deg.index_add_(0, valid_dst, torch.ones_like(valid_dst))
            isolated = int((deg == 0).sum().item())
            if isolated > 0:
                print(f"Валидация провалена: найдено {isolated} изолированных узлов")
                is_valid = False
        except Exception as exc:
            print(f"Валидация провалена: ошибка при проверке изолированных узлов: {exc}")
            is_valid = False

    # Проверка NaN в весах
    weight_tensor = None
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        weight_tensor = graph.edge_attr
    elif hasattr(graph, "edge_weight") and graph.edge_weight is not None:
        weight_tensor = graph.edge_weight

    if weight_tensor is not None and weight_tensor.numel() > 0:
        try:
            if torch.isnan(weight_tensor).any().item():
                print("Валидация провалена: найдены NaN в весах рёбер (edge_attr)")
                is_valid = False
        except Exception as exc:
            print(f"Валидация провалена: ошибка при проверке NaN весов: {exc}")
            is_valid = False
    else:
        # Если весов нет вообще — считаем это проблемой только если есть рёбра
        if num_edges > 0 and weight_tensor is None:
            print("Валидация провалена: отсутствуют веса рёбер (edge_attr)")
            is_valid = False

    # Проверка sensory_mask
    if not hasattr(graph, "sensory_mask") or graph.sensory_mask is None:
        print("Валидация провалена: отсутствует sensory_mask")
        is_valid = False
    else:
        try:
            sensory_count = int(graph.sensory_mask.sum().item())
            if sensory_count < 50:
                print(
                    f"Валидация провалена: sensory_mask содержит "
                    f"{sensory_count} узлов, требуется >=50"
                )
                is_valid = False
        except Exception as exc:
            print(f"Валидация провалена: ошибка при проверке sensory_mask: {exc}")
            is_valid = False

    # Проверка motor_mask
    if not hasattr(graph, "motor_mask") or graph.motor_mask is None:
        print("Валидация провалена: отсутствует motor_mask")
        is_valid = False
    else:
        try:
            motor_count = int(graph.motor_mask.sum().item())
            if motor_count < 20:
                print(
                    f"Валидация провалена: motor_mask содержит "
                    f"{motor_count} узлов, требуется >=20"
                )
                is_valid = False
        except Exception as exc:
            print(f"Валидация провалена: ошибка при проверке motor_mask: {exc}")
            is_valid = False

    if is_valid:
        # Успешный лог с ключевыми числами
        try:
            s_cnt = int(graph.sensory_mask.sum().item())
            m_cnt = int(graph.motor_mask.sum().item())
        except Exception:
            s_cnt = -1
            m_cnt = -1
        print(
            f"Валидация успешна: узлов={num_nodes}, рёбер={num_edges}, "
            f"sensory={s_cnt}, motor={m_cnt}"
        )

    return is_valid

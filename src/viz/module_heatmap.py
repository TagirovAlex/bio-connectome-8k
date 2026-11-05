"""Модуль визуализации модульной специализации (Figure 3).

Строит heatmap: строки — модули (top-15 по размеру), столбцы — T1/T2.
Значения — средняя активация модуля, нормализованная 0-1, с аннотациями
в ячейках. Данные берутся из поля ``module_activation`` в JSON-метриках;
при их отсутствии генерируется синтетика через
``src.analysis.module_activity.compute_module_activation`` (smoke).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Количество отображаемых модулей (top-15 по размеру).
_TOP_MODULES = 15
# Подписи задач — столбцы heatmap.
_TASK_LABELS = ["T1 (goto)", "T2 (avoid)"]
# Варианты ключей задач в JSON.
_TASK1_KEYS = ("task1", "t1", "goto", "task1_goto")
_TASK2_KEYS = ("task2", "t2", "avoid", "task2_avoid")


def _load_viz_config(viz_config: Dict[str, Any] | None) -> Dict[str, Any]:
    """Загружает конфиг визуализации.

    Если передан непустой словарь — использует его, иначе пытается прочитать
    ``config/viz.yaml`` относительно корня проекта. Фолбэк — встроенные
    дефолты (300 dpi, DejaVu Sans, фиксированные цвета архитектур).

    Args:
        viz_config: Словарь параметров из ``config/viz.yaml`` или ``None``.

    Returns:
        Непустой словарь параметров визуализации.
    """
    if viz_config is not None and len(viz_config) > 0:
        return viz_config
    default_path = Path(__file__).resolve().parents[2] / "config" / "viz.yaml"
    if default_path.exists():
        try:
            with open(default_path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict) and len(loaded) > 0:
                    return loaded
        except Exception:
            pass
    return {
        "dpi": 300,
        "figsize_wide": [10, 4],
        "figsize_square": [6, 6],
        "font_family": "DejaVu Sans",
        "font_size_axis": 12,
        "font_size_title": 14,
        "font_size_legend": 10,
        "colors": {
            "bio_signed": "#1f77b4",
            "rewired": "#ff7f0e",
            "random_er": "#2ca02c",
            "bio_unsigned": "#d62728",
            "mlp": "#9467bd",
        },
        "line_styles": {
            "bio_signed": "-",
            "rewired": "--",
            "random_er": ":",
            "bio_unsigned": "-.",
            "mlp": "-",
        },
        "ci": 0.95,
    }


def _setup_style(viz_config: Dict[str, Any]) -> None:
    """Настраивает стиль matplotlib из конфига визуализации."""
    font_family = viz_config.get("font_family", "DejaVu Sans")
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = viz_config.get("font_size_axis", 12)
    plt.rcParams["axes.titlesize"] = viz_config.get("font_size_title", 14)
    plt.rcParams["axes.labelsize"] = viz_config.get("font_size_axis", 12)
    plt.rcParams["legend.fontsize"] = viz_config.get("font_size_legend", 10)


def _parse_float(value: Any) -> float | None:
    """Безопасно приводит значение к ``float``.

    Args:
        value: Произвольное значение (число, строка, ``None``, bool).

    Returns:
        ``float`` или ``None`` если приведение невозможно.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_module_id(key: Any) -> int | None:
    """Извлекает числовой ID модуля из ключа.

    Поддерживаются ``module_0``, ``"3"``, целые числа и float-целые.

    Args:
        key: Ключ записи модуля.

    Returns:
        Целочисленный ID модуля или ``None`` если извлечь не удалось.
    """
    if isinstance(key, bool):
        return None
    if isinstance(key, int):
        return key
    if isinstance(key, float) and float(key).is_integer():
        return int(key)
    digits = "".join(ch for ch in str(key) if ch.isdigit())
    return int(digits) if digits else None


def _matches_task(key: Any, variants: Tuple[str, ...]) -> bool:
    """Проверяет, относится ли ключ к одной из задач.

    Args:
        key: Ключ из JSON (например, ``"task1_goto"``).
        variants: Кортеж ключевых фрагментов задачи.

    Returns:
        ``True`` если ключ содержит фрагмент из ``variants``.
    """
    low = str(key).lower().strip()
    return any(v in low for v in variants)


def _extract_value(record: Dict[str, Any], variants: Tuple[str, ...]) -> float | None:
    """Извлекает значение активации по одному из ключей задачи.

    Ищет значение на верхнем уровне записи, а также внутри вложенного
    словаря ``activation`` (например, ``{"activation": {"T1": x, "T2": y}}``).

    Args:
        record: Запись модуля из JSON.
        variants: Варианты ключей задачи (T1 или T2).

    Returns:
        Числовое значение или ``None``.
    """
    for key, val in record.items():
        if _matches_task(key, variants):
            parsed = _parse_float(val)
            if parsed is not None:
                return parsed
    act = record.get("activation")
    if isinstance(act, dict):
        for key, val in act.items():
            if _matches_task(key, variants):
                parsed = _parse_float(val)
                if parsed is not None:
                    return parsed
    return None


def _extract_scalar(record: Dict[str, Any]) -> float | None:
    """Извлекает скалярную активацию из записи.

    Используется для формата ``{"module_id": 0, "size": n, "activation": x}``,
    где активация относится к задаче, заданной внешним ключом.

    Args:
        record: Запись модуля из JSON.

    Returns:
        Скалярное значение или ``None``.
    """
    for key in ("activation", "value"):
        parsed = _parse_float(record.get(key))
        if parsed is not None:
            return parsed
    return None


def _find_module_activation(data: Dict[str, Any]) -> Any | None:
    """Ищет поле ``module_activation`` в JSON метрик.

    Проверяются три места: корень файла, блок ``results`` и вложенные
    блоки задач внутри ``results``.

    Args:
        data: Загруженный JSON-словарь метрик прогона.

    Returns:
        Значение поля ``module_activation`` или ``None``.
    """
    candidates: List[Any] = []
    if isinstance(data.get("module_activation"), (dict, list)):
        candidates.append(data["module_activation"])
    results = data.get("results")
    if isinstance(results, dict):
        if isinstance(results.get("module_activation"), (dict, list)):
            candidates.append(results["module_activation"])
        for task_val in results.values():
            if isinstance(task_val, dict) and isinstance(
                task_val.get("module_activation"), (dict, list)
            ):
                candidates.append(task_val["module_activation"])
    for cand in candidates:
        if cand:
            return cand
    return None


def _normalize_module_records(raw: Any) -> Dict[int, Dict[str, Any]]:
    """Приводит сырое ``module_activation`` к единому формату.

    Поддерживаются три формата:
    * словарь модулей ``{module_0: {"size": n, "activation": {"T1": x, "T2": y}}}``;
    * список записей ``[{"module_id": 0, "size": n, "T1": x, "T2": y}]``;
    * словарь задач ``{"task1_goto": [...], "task2_avoid": [...]}``.

    Args:
        raw: Значение поля ``module_activation`` из JSON.

    Returns:
        Словарь ``{module_id: {"size": float, "t1": float|None, "t2": float|None}}``.
    """
    result: Dict[int, Dict[str, Any]] = {}

    def _add(module_id: int, size: float, t1: float | None, t2: float | None) -> None:
        """Аккумулирует запись модуля, сохраняя первое непустое значение."""
        entry = result.setdefault(module_id, {"size": 0.0, "t1": None, "t2": None})
        entry["size"] = max(entry["size"], float(size))
        if t1 is not None and entry["t1"] is None:
            entry["t1"] = float(t1)
        if t2 is not None and entry["t2"] is None:
            entry["t2"] = float(t2)

    def _record_to_entry(rec: Dict[str, Any]) -> None:
        """Обрабатывает одну запись модуля из списка."""
        mid = _parse_module_id(rec.get("module_id", rec.get("id")))
        size = _parse_float(rec.get("size", rec.get("n_nodes", 0)))
        if mid is None:
            return
        _add(mid, size or 0.0, _extract_value(rec, _TASK1_KEYS), _extract_value(rec, _TASK2_KEYS))

    if isinstance(raw, list):
        for rec in raw:
            if isinstance(rec, dict):
                _record_to_entry(rec)
        return result

    if not isinstance(raw, dict):
        return result

    # Формат «словарь задач»: {task1: [...], task2: [...]}.
    if any(_matches_task(k, _TASK1_KEYS + _TASK2_KEYS) for k in raw) and any(
        isinstance(v, list) for v in raw.values()
    ):
        for task_key, mod_list in raw.items():
            is_t1 = _matches_task(task_key, _TASK1_KEYS)
            is_t2 = _matches_task(task_key, _TASK2_KEYS)
            if not (is_t1 or is_t2) or not isinstance(mod_list, list):
                continue
            for rec in mod_list:
                if not isinstance(rec, dict):
                    continue
                mid = _parse_module_id(rec.get("module_id", rec.get("id")))
                size = _parse_float(rec.get("size", rec.get("n_nodes", 0)))
                if mid is None:
                    continue
                # В этом формате активация относится к задаче внешнего ключа.
                t1 = _extract_value(rec, _TASK1_KEYS)
                t2 = _extract_value(rec, _TASK2_KEYS)
                if t1 is None and is_t1:
                    t1 = _extract_scalar(rec)
                if t2 is None and is_t2:
                    t2 = _extract_scalar(rec)
                _add(mid, size or 0.0, t1, t2)
        return result

    # Формат «словарь модулей»: {module_0: {...}}.
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        mid = _parse_module_id(val.get("module_id", val.get("id", key)))
        size = _parse_float(val.get("size", val.get("n_nodes", 0)))
        if mid is None:
            continue
        _add(mid, size or 0.0, _extract_value(val, _TASK1_KEYS), _extract_value(val, _TASK2_KEYS))
    return result


def _normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """Нормализует матрицу активаций в диапазон 0-1 (min-max).

    Args:
        mat: Матрица активаций ``[n, 2]``.

    Returns:
        Матрица той же формы в диапазоне 0-1. Для константной матрицы
        возвращается 0.5 во всех ячейках.
    """
    vmin = float(np.min(mat))
    vmax = float(np.max(mat))
    if vmax > vmin:
        return (mat - vmin) / (vmax - vmin)
    return np.full_like(mat, 0.5)


def _finalize_matrix(
    aggregated: Dict[int, Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    """Выбирает top-15 модулей по размеру и нормализует матрицу.

    Args:
        aggregated: Словарь ``{module_id: {"size": ..., "t1": [...], "t2": [...]}}``.

    Returns:
        ``(matrix, labels)``: матрица ``[top_n, 2]`` (столбцы T1/T2),
        нормализованная 0-1, и подписи модулей ``["M0", ...]``.
    """
    # Сортировка по размеру модуля (убывание) и обрезка до top-15.
    ordered = sorted(aggregated.items(), key=lambda kv: kv[1]["size"], reverse=True)
    ordered = ordered[:_TOP_MODULES]
    n = len(ordered)
    mat = np.zeros((n, 2), dtype=float)
    labels: List[str] = []
    for i, (module_id, rec) in enumerate(ordered):
        t1 = float(np.mean(rec["t1"])) if rec["t1"] else 0.0
        t2 = float(np.mean(rec["t2"])) if rec["t2"] else 0.0
        mat[i, 0] = t1
        mat[i, 1] = t2
        labels.append(f"M{module_id}")
    return _normalize_matrix(mat), labels


def _build_heatmap_data(metrics_dir: Path) -> Tuple[np.ndarray, List[str]]:
    """Собирает матрицу активаций модулей из JSON-метрик.

    Считываются все ``*.json`` из ``metrics_dir`` (кроме ``graph_stats.json``),
    из каждого извлекается поле ``module_activation``. Данные агрегируются
    по модулям: размер берётся максимальный, активации усредняются по прогонам.

    Args:
        metrics_dir: Папка с JSON-метриками.

    Returns:
        ``(matrix, labels)``: матрица ``[top_n, 2]``, нормализованная 0-1,
        и подписи модулей.
    """
    aggregated: Dict[int, Dict[str, Any]] = {}
    if metrics_dir.exists():
        for jf in sorted(metrics_dir.glob("*.json")):
            if jf.name == "graph_stats.json":
                continue
            try:
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            raw = _find_module_activation(data)
            if raw is None:
                continue
            for module_id, rec in _normalize_module_records(raw).items():
                entry = aggregated.setdefault(
                    module_id, {"size": 0.0, "t1": [], "t2": []}
                )
                entry["size"] = max(entry["size"], float(rec["size"]))
                if rec["t1"] is not None:
                    entry["t1"].append(float(rec["t1"]))
                if rec["t2"] is not None:
                    entry["t2"].append(float(rec["t2"]))
    if aggregated:
        return _finalize_matrix(aggregated)
    # Данных нет — синтетика через compute_module_activation (smoke).
    return _synthetic_module_data()


def _synthetic_module_data() -> Tuple[np.ndarray, List[str]]:
    """Генерирует синтетические активации модулей для smoke-прогона.

    Создаёт 300 узлов, разбитых на 15 модулей (по убыванию размера),
    и активации для двух задач с блочной специализацией: модули 0-4
    активны в T1, модули 5-9 — в T2. Активации вычисляются через
    ``compute_module_activation`` из ``src.analysis.module_activity``.

    Returns:
        ``(matrix, labels)``: матрица ``[15, 2]``, нормализованная 0-1,
        и подписи модулей.
    """
    n_modules = _TOP_MODULES
    n_nodes = 300
    rng = np.random.default_rng(42)
    # Распределение узлов по модулям (размеры модулей убывают).
    assignments = np.sort(rng.integers(0, n_modules, size=n_nodes))
    # Множества модулей, специализированных на каждой задаче.
    t1_modules = set(range(0, 5))
    t2_modules = set(range(5, 10))

    try:
        import torch
        from src.analysis.module_activity import compute_module_activation
    except Exception:
        # Тяжёлые зависимости недоступны — равномерный фолбэк.
        mat = rng.uniform(0.2, 0.8, size=(n_modules, 2))
        for i in range(5):
            mat[i, 0] += 0.2
            mat[i, 1] -= 0.15
        for i in range(5, 10):
            mat[i, 0] -= 0.15
            mat[i, 1] += 0.2
        mat = np.clip(mat, 0, 1)
        labels = [f"M{i}" for i in range(n_modules)]
        return mat, labels

    def _gen_activations(hot_modules: set[int]) -> torch.Tensor:
        """Генерирует 1D активации: гауссов шум плюс бонус горячим модулям."""
        acts = rng.normal(0.0, 0.15, size=n_nodes)
        for i, module_id in enumerate(assignments):
            if module_id in hot_modules:
                acts[i] += 1.0
        return torch.from_numpy(acts.astype(np.float32))

    # Средние активации модулей для каждой задачи.
    act_t1 = _gen_activations(t1_modules)
    act_t2 = _gen_activations(t2_modules)
    map_t1 = compute_module_activation(act_t1, assignments, n_modules)
    map_t2 = compute_module_activation(act_t2, assignments, n_modules)
    # Сортировка модулей по размеру (убывание) для соответствия «top-15 по размеру».
    sizes = np.bincount(assignments, minlength=n_modules).astype(float)
    order = np.argsort(-sizes)
    mat = np.zeros((n_modules, 2), dtype=float)
    labels: List[str] = []
    for row, mid in enumerate(order):
        mat[row, 0] = float(map_t1[int(mid)])
        mat[row, 1] = float(map_t2[int(mid)])
        labels.append(f"M{int(mid)}")
    return _normalize_matrix(mat), labels


def plot_module_heatmap(
    metrics_dir: Path, output_path: Path, viz_config: Dict[str, Any]
) -> None:
    """Рисует Figure 3: модульная специализация.

    Heatmap: строки — модули top-15 по размеру, столбцы — T1/T2. Значения —
    средняя активация модуля, нормализованная 0-1, с аннотациями в ячейках
    (формат ``.2f``). Colormap — viridis.

    Данные берутся из поля ``module_activation`` в JSON-метриках
    (``metrics_dir``); при их отсутствии генерируется синтетика через
    ``compute_module_activation`` (smoke-режим).

    Args:
        metrics_dir: Папка с JSON-метриками.
        output_path: Путь для сохранения PNG (300 dpi).
        viz_config: Параметры из ``config/viz.yaml``.
    """
    cfg = _load_viz_config(viz_config)
    _setup_style(cfg)

    dpi = int(cfg.get("dpi", 300))
    figsize_square = cfg.get("figsize_square", [6, 6])
    try:
        figsize = (float(figsize_square[0]), float(figsize_square[1]))
    except (TypeError, ValueError, IndexError):
        figsize = (6, 6)

    mat, labels = _build_heatmap_data(Path(metrics_dir))
    n_mod = int(mat.shape[0])

    fig, ax = plt.subplots(figsize=figsize)

    # Seaborn даёт более аккуратный heatmap с аннотациями.
    try:
        import seaborn as sns  # type: ignore

        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            xticklabels=_TASK_LABELS,
            yticklabels=labels,
            cbar_kws={"label": "Норм. активация"},
            annot_kws={"fontsize": 8},
            ax=ax,
            linewidths=0.5,
        )
    except Exception:
        # Фолбэк на чистый matplotlib без seaborn.
        im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(_TASK_LABELS, fontsize=cfg.get("font_size_axis", 12))
        ax.set_yticks(np.arange(n_mod))
        ax.set_yticklabels(labels, fontsize=cfg.get("font_size_axis", 12) - 1)
        for i in range(n_mod):
            for j in range(2):
                val = mat[i, j]
                # Цвет текста по контрасту с фоном ячейки.
                text_color = "white" if val < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Норм. активация", fontsize=cfg.get("font_size_axis", 12) - 1)

    ax.set_title("Модульная специализация", fontsize=cfg.get("font_size_title", 14))
    ax.grid(False)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
"""Модуль визуализации кривых обучения (Figure 1).

Строит 2x2 сетку: строки — масштабы 2K/8K, столбцы — задачи T1 (goto) и
T2 (avoid). На каждом подграфике — кривые обучения всех архитектур:
средняя по сидам линия и зона 95% CI (fill_between).

Данные читаются из JSON-метрик (``results/metrics/*.json``): для каждой задачи
ожидается ключ ``learning_curve`` (список средних наград по ходу обучения) и
опционально ``timesteps`` (общее число шагов обучения задачи). Если для
масштаба/задачи кривых нет — подграфик помечается как «Нет данных»: рисуется
только доступное.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Порядок архитектур фиксирован: легенда и цвета стабильны между фигурами.
_ARCH_ORDER = ["bio_signed", "rewired", "random_er", "bio_unsigned", "mlp"]
# Человекочитаемые подписи для легенды.
_ARCH_LABELS = {
    "bio_signed": "Bio (signed)",
    "rewired": "Rewired",
    "random_er": "Random ER",
    "bio_unsigned": "Bio (unsigned)",
    "mlp": "MLP",
}
# Масштабы в порядке строк сетки.
_SCALES = ["2k", "8k"]
# Задачи в порядке столбцов сетки.
_TASKS = ["task1_goto", "task2_avoid"]
# Подписи задач для заголовков подграфиков.
_TASK_LABELS = {
    "task1_goto": "T1 (goto)",
    "task2_avoid": "T2 (avoid)",
}
# Файлы-агрегаты и статистика графов не содержат кривых обучения.
_SKIP_STEMS = {"graph_stats", "2k_summary", "8k_summary", "all_summary"}


def _load_viz_config(viz_config: dict[str, Any] | None) -> dict[str, Any]:
    """Загружает конфиг визуализации, подставляя дефолт из yaml если пусто.

    Args:
        viz_config: Словарь из config/viz.yaml или None.

    Returns:
        Конфиг: переданный, либо прочитанный из config/viz.yaml, либо дефолтный.
    """
    # Если передан непустой словарь — используем его.
    if viz_config is not None and len(viz_config) > 0:
        return viz_config
    # Иначе пытаемся прочитать config/viz.yaml относительно корня проекта.
    # Корень = .../src/viz/ -> parents[2].
    default_path = Path(__file__).resolve().parents[2] / "config" / "viz.yaml"
    if default_path.exists():
        try:
            with open(default_path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict) and len(loaded) > 0:
                return loaded
        except Exception:
            pass
    # Фолбэк-дефолты (зеркалят config/viz.yaml) если файл отсутствует.
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


def _setup_style(viz_config: dict[str, Any]) -> None:
    """Настраивает глобальный стиль matplotlib по viz_config."""
    # Шрифт и размеры — из конфига.
    font_family = viz_config.get("font_family", "DejaVu Sans")
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = viz_config.get("font_size_axis", 12)
    plt.rcParams["axes.titlesize"] = viz_config.get("font_size_title", 14)
    plt.rcParams["axes.labelsize"] = viz_config.get("font_size_axis", 12)
    plt.rcParams["legend.fontsize"] = viz_config.get("font_size_legend", 10)
    plt.rcParams["xtick.labelsize"] = viz_config.get("font_size_axis", 12) - 2
    plt.rcParams["ytick.labelsize"] = viz_config.get("font_size_axis", 12) - 2


def _map_arch_key(raw: str) -> str:
    """Мапит сырое имя архитектуры из метрик к ключу цветов в viz.yaml.

    Правила: bio -> bio_signed, unsigned -> bio_unsigned, random -> random_er,
    rewired -> rewired, mlp -> mlp. Неизвестное имя считаем bio_signed.

    Args:
        raw: Имя архитектуры из поля ``architecture`` или имени файла.

    Returns:
        Канонический ключ архитектуры.
    """
    low = raw.lower()
    if "unsigned" in low:
        return "bio_unsigned"
    if "random" in low:
        return "random_er"
    if "rewired" in low:
        return "rewired"
    if "mlp" in low:
        return "mlp"
    if "bio" in low:
        return "bio_signed"
    # Дефолт — считаем bio_signed если не распознано.
    return "bio_signed"


def _infer_scale(
    file_path: Path, data: dict[str, Any], scale: str | None
) -> str | None:
    """Определяет масштаб файла: по имени (2k_*.json / 8k_*.json) или полю scale.

    Args:
        file_path: Путь к файлу метрик.
        data: Загруженный JSON прогона.
        scale: Явный масштаб из параметра. Если задан — фильтруем файлы по нему.

    Returns:
        "2k" или "8k"; None если файл не относится к запрошенному масштабу.
    """
    stem = file_path.stem.lower()
    # Явный фильтр: имя начинается с {scale}_ или поле scale совпадает.
    if scale is not None:
        req = scale.lower()
        if stem.startswith(f"{req}_") or str(data.get("scale", "")).lower() == req:
            return req
        return None
    # Без параметра определяем масштаб по префиксу имени файла.
    if stem.startswith("2k_"):
        return "2k"
    if stem.startswith("8k_"):
        return "8k"
    # Последний шанс — поле scale внутри JSON.
    field = str(data.get("scale", "")).lower()
    return field if field in ("2k", "8k") else None


def _curve_to_array(curve: Any) -> np.ndarray | None:
    """Конвертирует сырую кривую в одномерный массив наград.

    Поддерживает списки наград, списки пар ``[step, reward]`` и вложенные
    структуры. Нечисловые значения (NaN/inf) удаляются.

    Args:
        curve: Сырое значение кривой из JSON.

    Returns:
        Одномерный массив наград или None, если кривую распарсить не удалось.
    """
    try:
        arr = np.asarray(curve)
    except Exception:
        return None
    if arr.size == 0:
        return None
    # Список пар [step, reward] — берём второй столбец (награды).
    if arr.ndim == 2 and arr.shape[1] >= 2:
        arr = arr[:, 1]
    flat = arr.astype(float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    return flat


def _extract_curve_and_steps(
    data: dict[str, Any], task: str
) -> tuple[np.ndarray | None, int | None]:
    """Извлекает кривую обучения задачи и число шагов из JSON метрик.

    Кривая ищется в ``results[task]["learning_curve"]`` (формат run-метрик)
    или в верхнеуровневых ключах ``learning_curve`` / ``learning_curves``
    (формат словаря, возвращаемого ``train_single_task``).

    Args:
        data: Загруженный JSON метрик прогона.
        task: Имя задачи (``task1_goto`` / ``task2_avoid``).

    Returns:
        Пара (массив наград кривой, число шагов обучения). Если кривой нет —
        (None, None).
    """
    curve: Any = None
    steps: Any = None
    # Приоритет: вложенный блок results[task].
    results = data.get("results")
    if isinstance(results, dict):
        task_block = results.get(task)
        if isinstance(task_block, dict):
            curve = task_block.get("learning_curve")
            steps = task_block.get("timesteps", task_block.get("total_timesteps"))
    # Прямые ключи на верхнем уровне (совместимость с train_single).
    if curve is None:
        for key in ("learning_curves", "learning_curve"):
            top = data.get(key)
            if isinstance(top, dict):
                curve = top.get(task)
            elif key == "learning_curve":
                curve = top
            if curve is not None:
                break
    if steps is None:
        steps = data.get("timesteps", data.get("total_timesteps"))
    arr = _curve_to_array(curve)
    if arr is None:
        return None, None
    try:
        steps_int = int(steps) if steps is not None else None
    except (TypeError, ValueError):
        steps_int = None
    return arr, steps_int


def _collect_curves(
    metrics_dir: Path, scale: str | None
) -> dict[str, dict[str, dict[str, list[tuple[np.ndarray, int | None]]]]]:
    """Собирает кривые обучения из JSON-метрик по (scale, задача, архитектура).

    Args:
        metrics_dir: Папка с JSON-метриками.
        scale: Явный масштаб для фильтрации файлов (None — определить по имени).

    Returns:
        Словарь вида {scale: {task: {arch_key: [(кривая, timesteps), ...]}}}.
    """
    grouped: dict[str, dict[str, dict[str, list[tuple[np.ndarray, int | None]]]]] = {
        "2k": {"task1_goto": {}, "task2_avoid": {}},
        "8k": {"task1_goto": {}, "task2_avoid": {}},
    }
    if not metrics_dir.exists():
        return grouped
    for jf in sorted(metrics_dir.glob("*.json")):
        # Агрегатные файлы и статистика графов кривых не содержат.
        if jf.stem in _SKIP_STEMS:
            continue
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        file_scale = _infer_scale(jf, data, scale)
        if file_scale is None:
            continue
        # Архитектура из поля JSON, иначе — из имени файла.
        arch_key = _map_arch_key(str(data.get("architecture", jf.stem)))
        for task in _TASKS:
            curve, steps = _extract_curve_and_steps(data, task)
            if curve is None:
                continue
            grouped[file_scale][task].setdefault(arch_key, []).append((curve, steps))
    return grouped


def _mean_with_ci(
    curves: list[np.ndarray], ci: float
) -> tuple[np.ndarray, np.ndarray]:
    """Средняя кривая по сидам и полуширина доверительного интервала.

    Все кривые обрезаются до общей длины. При одном сиде CI равен нулю.

    Args:
        curves: Кривые разных сидов.
        ci: Уровень доверия (0.95 соответствует z = 1.96).

    Returns:
        Пара (средняя кривая, полуширина CI).
    """
    min_len = min(len(c) for c in curves)
    stacked = np.vstack([c[:min_len] for c in curves])
    mean = np.mean(stacked, axis=0)
    n = stacked.shape[0]
    if n < 2:
        # Один сид — зона CI вырождается в линию.
        return mean, np.zeros_like(mean)
    std = np.std(stacked, axis=0, ddof=1)
    try:
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + float(ci) / 2.0))
    except Exception:
        # Запасной вариант для 95% доверительного интервала.
        z = 1.96
    return mean, z * std / np.sqrt(n)


def plot_learning_curves(
    metrics_dir: Path,
    output_path: Path,
    viz_config: dict[str, Any],
    scale: str | None = None,
) -> None:
    """Рисует Figure 1: кривые обучения для всех архитектур.

    2x2 grid: строки — масштабы 2K/8K, столбцы — задачи T1 (goto) / T2 (avoid).
    На каждом подграфике по архитектуре рисуется средняя по сидам кривая
    (X — timestep, Y — episode reward) с зоной 95% CI (``fill_between``).
    Если для масштаба/задачи кривых нет — подграфик помечается «Нет данных»:
    отрисовывается только доступное.

    Цвета, стили линий, шрифты, dpi и размер фигуры берутся из ``viz_config``.

    Args:
        metrics_dir: Папка с JSON-метриками (results/metrics).
        output_path: Путь для сохранения PNG.
        viz_config: Параметры из config/viz.yaml.
        scale: Фильтр по масштабу файлов (``"2k"``/``"8k"``). Если None —
            масштаб каждого файла определяется по имени (2k_*.json / 8k_*.json)
            или по полю ``scale`` внутри JSON.
    """
    cfg = _load_viz_config(viz_config)
    _setup_style(cfg)

    dpi = int(cfg.get("dpi", 300))
    colors: dict[str, str] = cfg.get("colors", {})
    line_styles: dict[str, str] = cfg.get("line_styles", {})
    ci_level = float(cfg.get("ci", 0.95))
    # Для 2x2 сетки высота удваивается относительно figsize_wide.
    figsize_wide = cfg.get("figsize_wide", [10, 4])
    try:
        figsize = (float(figsize_wide[0]), float(figsize_wide[1]) * 2.0)
    except Exception:
        figsize = (10.0, 8.0)

    grouped = _collect_curves(Path(metrics_dir), scale)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for row, scale_name in enumerate(_SCALES):
        for col, task in enumerate(_TASKS):
            ax = axes[row, col]
            panel = grouped[scale_name][task]
            # X-координаты: timesteps из метрик (первый найденный) или индекс.
            x_scale: int | None = None
            if panel:
                for entries in panel.values():
                    for _, steps in entries:
                        if steps is not None:
                            x_scale = steps
                            break
                    if x_scale is not None:
                        break
            has_any = False
            for arch in _ARCH_ORDER:
                entries = panel.get(arch, [])
                if not entries:
                    continue
                curves = [c for c, _ in entries]
                mean, ci_half = _mean_with_ci(curves, ci_level)
                n = len(mean)
                if x_scale is not None:
                    x = np.linspace(0, float(x_scale), n)
                else:
                    # Нет информации о шагах — используем индекс точки.
                    x = np.arange(n)
                color = colors.get(arch, "#333333")
                ax.plot(
                    x,
                    mean,
                    label=_ARCH_LABELS.get(arch, arch),
                    color=color,
                    linestyle=line_styles.get(arch, "-"),
                    linewidth=1.8,
                )
                # Зона CI по сидам.
                ax.fill_between(
                    x,
                    mean - ci_half,
                    mean + ci_half,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
                has_any = True
            ax.set_xlabel("Timestep", fontsize=cfg.get("font_size_axis", 12))
            ax.set_ylabel("Episode reward", fontsize=cfg.get("font_size_axis", 12))
            ax.set_title(
                f"{scale_name.upper()} — {_TASK_LABELS[task]}",
                fontsize=cfg.get("font_size_title", 14),
            )
            ax.grid(True, alpha=0.3, color="gray")
            ax.tick_params(labelsize=cfg.get("font_size_axis", 12) - 2)
            if not has_any:
                # Мало данных — помечаем подграфик, доступного нет.
                ax.text(
                    0.5,
                    0.5,
                    "Нет данных",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=cfg.get("font_size_axis", 12),
                )
            # Легенда только на верхнем левом подграфике.
            if row == 0 and col == 0 and has_any:
                ax.legend(
                    fontsize=cfg.get("font_size_legend", 10),
                    loc="lower right",
                    framealpha=0.9,
                )

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
"""Модуль визуализации катастрофического забывания (Figure 2).

Строит 1x2 bar chart: слева — forgetting на 2K, справа — на 8K.
X — архитектуры, Y — forgetting = R(T1|after T1) − R(T1|after T2).
Error bars — 95% CI (или std при n=3); над значимыми различиями
относительно Bio (signed) рисуются звёздочки (Wilcoxon rank-sum, p<0.05).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Набор архитектур фиксирован (см. AGENTS.md); порядок и подписи — здесь.
_ARCH_ORDER = ["bio_signed", "rewired", "random_er", "bio_unsigned", "mlp"]
_ARCH_LABELS = {
    "bio_signed": "Bio\n(signed)",
    "rewired": "Rewired",
    "random_er": "Random ER",
    "bio_unsigned": "Bio\n(unsigned)",
    "mlp": "MLP",
}

# Пороговые уровни значимости для звёздочек: p < 0.001 → "***" и т.д.
_SIG_THRESHOLDS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]

# Масштабы, отображаемые панелями Figure 2.
_SCALES = ["2k", "8k"]

# Эталонная архитектура для статистических сравнений (гипотезы H1-H4).
_BASELINE_ARCH = "bio_signed"

# Суффикс агрегированных summary-файлов — они не являются прогонами.
_SUMMARY_SUFFIX = "_summary"


def _load_viz_config(viz_config: Dict[str, Any] | None) -> Dict[str, Any]:
    """Загружает конфиг визуализации, подставляя дефолт из yaml если пусто."""
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
    """Настраивает стиль matplotlib."""
    font_family = viz_config.get("font_family", "DejaVu Sans")
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = viz_config.get("font_size_axis", 12)
    plt.rcParams["axes.titlesize"] = viz_config.get("font_size_title", 14)
    plt.rcParams["axes.labelsize"] = viz_config.get("font_size_axis", 12)
    plt.rcParams["legend.fontsize"] = viz_config.get("font_size_legend", 10)


def _map_arch_key(raw: str) -> str:
    """Мапит сырое имя к ключу цветов."""
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
    return "bio_signed"


def _infer_arch_and_scale(file_path: Path, data: Dict[str, Any]) -> tuple[str, str]:
    """Определяет архитектуру и масштаб."""
    stem = file_path.stem.lower()
    scale = "2k"
    if "8k" in stem or "8000" in stem:
        scale = "8k"
    elif "2k" in stem or "2000" in stem:
        scale = "2k"
    else:
        graph_field = str(data.get("graph", "")).lower()
        if "8k" in graph_field:
            scale = "8k"
    arch_key = _map_arch_key(stem)
    graph_arch = str(data.get("graph", "")).lower()
    if arch_key == "bio_signed" and "rewired" in graph_arch:
        arch_key = "rewired"
    elif arch_key == "bio_signed" and "random" in graph_arch:
        arch_key = "random_er"
    elif arch_key == "bio_signed" and "unsigned" in graph_arch:
        arch_key = "bio_unsigned"
    elif arch_key == "bio_signed" and "mlp" in graph_arch:
        arch_key = "mlp"
    arch_field = str(data.get("architecture", "")).lower()
    if "rewired" in arch_field:
        arch_key = "rewired"
    elif "random" in arch_field:
        arch_key = "random_er"
    elif "unsigned" in arch_field:
        arch_key = "bio_unsigned"
    elif "mlp" in arch_field:
        arch_key = "mlp"
    return arch_key, scale


def _extract_forgetting(data: Dict[str, Any]) -> float | None:
    """Извлекает forgetting из JSON, используя cl_metrics если доступно.

    Порядок приоритета: ``compute_forgetting`` (вычисление из наград),
    затем готовое поле ``forgetting``, затем разность наград как резерв.
    """
    # Пробуем через src.analysis.cl_metrics.
    try:
        from src.analysis.cl_metrics import compute_forgetting

        try:
            val = compute_forgetting(data)
            if val is not None and np.isfinite(float(val)):
                return float(val)
        except Exception:
            pass
    except Exception:
        pass
    # Прямой поиск поля forgetting.
    for space in [data.get("results", {}), data]:
        if isinstance(space, dict):
            for v in space.values():
                if isinstance(v, dict) and "forgetting" in v:
                    try:
                        fv = float(v["forgetting"])
                        if np.isfinite(fv):
                            return fv
                    except Exception:
                        continue
            # Также проверяем верхний уровень.
            if "forgetting" in space:
                try:
                    fv = float(space["forgetting"])
                    if np.isfinite(fv):
                        return fv
                except Exception:
                    pass
    # Пробуем вычислить как разность наград.
    try:
        results_block = data.get("results", data)
        if isinstance(results_block, dict):
            t1 = results_block.get("task1_goto", {})
            if isinstance(t1, dict):
                r1 = t1.get("reward_after_task1")
                r2 = t1.get("reward_after_task2")
                if r1 is not None and r2 is not None:
                    return float(r1) - float(r2)
    except Exception:
        pass
    return None


def _normalize_scale(scale: str) -> str:
    """Нормализует значение scale: '2k', '8k' или 'all'."""
    s = str(scale).strip().lower()
    if s in ("2k", "2000"):
        return "2k"
    if s in ("8k", "8000"):
        return "8k"
    if s == "all":
        return "all"
    raise ValueError(f"Недопустимый scale={scale!r}. Ожидается '2k', '8k' или 'all'")


def _figsize_wide(cfg: Dict[str, Any]) -> tuple[float, float]:
    """Размер фигуры (широкий) из конфига."""
    fs = cfg.get("figsize_wide", [10, 4])
    try:
        return float(fs[0]), float(fs[1])
    except Exception:
        return 10.0, 4.0


def _collect_forgetting(
    metrics_dir: Path, scale: str
) -> Dict[str, Dict[str, List[float]]]:
    """Собирает forgetting по scale -> arch -> list[float].

    Загружает только файлы прогонов, соответствующие ``scale``:
    ``2k_*.json`` для "2k", ``8k_*.json`` для "8k", все ``*.json``
    для "all". Summary-файлы и ``graph_stats.json`` пропускаются.

    Args:
        metrics_dir: Папка с JSON-метриками прогонов.
        scale: Масштаб загрузки: "2k", "8k" или "all".

    Returns:
        Словарь ``{scale: {arch_key: [forgetting_values]}}``.
    """
    grouped: Dict[str, Dict[str, List[float]]] = {s: {} for s in _SCALES}
    if not metrics_dir.exists():
        return grouped
    patterns = ["*.json"] if scale == "all" else [f"{scale}_*.json"]
    for pattern in patterns:
        for jf in sorted(metrics_dir.glob(pattern)):
            # Пропускаем служебные файлы — это не прогоны.
            if jf.name == "graph_stats.json":
                continue
            if jf.stem.endswith(_SUMMARY_SUFFIX):
                continue
            try:
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            arch_key, inferred_scale = _infer_arch_and_scale(jf, data)
            if inferred_scale not in grouped:
                continue
            forgetting = _extract_forgetting(data)
            if forgetting is not None and np.isfinite(forgetting):
                grouped[inferred_scale].setdefault(arch_key, []).append(float(forgetting))
    return grouped


def _synthetic_forgetting() -> Dict[str, Dict[str, List[float]]]:
    """Синтетические заглушки для smoke-режима (когда прогонов нет)."""
    rng = np.random.default_rng(0)
    synthetic_means = {
        "bio_signed": 0.8,
        "rewired": 1.6,
        "random_er": 1.9,
        "bio_unsigned": 1.2,
        "mlp": 2.5,
    }
    grouped: Dict[str, Dict[str, List[float]]] = {s: {} for s in _SCALES}
    for s in _SCALES:
        for arch in _ARCH_ORDER:
            mean = synthetic_means.get(arch, 1.5)
            # На 8K forgetting чуть выше, чем на 2K.
            if s == "8k":
                mean += 0.3
            grouped[s][arch] = (rng.normal(mean, 0.2, size=3)).tolist()
    return grouped


def _ci95_half(std: float, n: int) -> float:
    """Полуширина 95% доверительного интервала среднего (t-критерий)."""
    if n <= 1 or std == 0.0:
        return 0.0
    from scipy.stats import t as t_dist

    t_crit = float(t_dist.ppf(0.975, df=n - 1))
    return t_crit * std / np.sqrt(n)


def _panel_stats(values: List[float]) -> Dict[str, float]:
    """Считает mean/std/полуширину 95% CI для списка значений forgetting."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci": 0.0, "n": 0.0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci = _ci95_half(std, arr.size)
    return {"mean": mean, "std": std, "ci": ci, "n": float(arr.size)}


def _error_for_bar(stat: Dict[str, float]) -> float:
    """Error bar: std при n<=3, иначе 95% CI (спека: CI или std при n=3)."""
    n = int(stat["n"])
    if n <= 1:
        return 0.0
    if n <= 3:
        return float(stat["std"])
    return float(stat["ci"])


def _compute_p_values(grouped: Dict[str, List[float]]) -> Dict[str, float | None]:
    """Wilcoxon rank-sum: bio_signed против каждой остальной архитектуры.

    Возвращает ``{arch: p_value}`` для всех архитектур, кроме базовой.
    p_value равен None, если сравнение невозможно (мало данных/ошибка).
    """
    p_values: Dict[str, float | None] = {}
    baseline = grouped.get(_BASELINE_ARCH, [])
    if len(baseline) < 2:
        return {arch: None for arch in _ARCH_ORDER if arch != _BASELINE_ARCH}
    from src.analysis.stats import compare_architectures

    for arch in _ARCH_ORDER:
        if arch == _BASELINE_ARCH:
            continue
        other = grouped.get(arch, [])
        if len(other) < 2:
            p_values[arch] = None
            continue
        try:
            _, p_value = compare_architectures(baseline, other, test="wilcoxon")
            p_values[arch] = (
                float(p_value) if p_value is not None and np.isfinite(p_value) else None
            )
        except Exception:
            # Не рушим фигуру из-за одной пары сравнений.
            p_values[arch] = None
    return p_values


def _plan_brackets(
    present: List[Dict[str, Any]],
    p_values: Dict[str, float | None],
    bar_width: float,
) -> tuple[List[Dict[str, Any]], float]:
    """Планирует скобки значимости и верхнюю границу панели.

    Скобка рисуется от базовой архитектуры (Bio signed) до архитектуры,
    значимо отличающейся от неё (p<0.05). Возвращает список скобок и
    верхнюю границу y, достаточную для их отрисовки.

    Args:
        present: Прогонявшиеся на масштабе архитектуры с метриками
            (ключи "arch", "j", "mean", "err").
        p_values: Значимости сравнений относительно базовой архитектуры.
        bar_width: Ширина столбца (для отступов скобки).

    Returns:
        Кортеж ``(brackets, top)``: список скобок
        ``{"x1", "x2", "y", "tick", "star"}`` и верхняя граница y.
    """
    top = float(max(r["mean"] + r["err"] for r in present))
    step = 0.15 * top if top > 0 else 0.15
    tick = 0.25 * step
    base_j = _ARCH_ORDER.index(_BASELINE_ARCH)
    brackets: List[Dict[str, Any]] = []
    for arch, p_value in sorted(p_values.items(), key=lambda kv: _ARCH_ORDER.index(kv[0])):
        if p_value is None or p_value >= _SIG_THRESHOLDS[-1][0]:
            continue
        # Скобка рисуется только если архитектура прогонялась на масштабе.
        row = next((r for r in present if r["arch"] == arch), None)
        if row is None:
            continue
        y = top + step * (len(brackets) + 1)
        star = next(s for thr, s in _SIG_THRESHOLDS if p_value < thr)
        brackets.append(
            {
                "x1": float(base_j - bar_width / 2.0),
                "x2": float(row["j"] + bar_width / 2.0),
                "y": y,
                "tick": tick,
                "star": star,
            }
        )
    if brackets:
        top = max(top, brackets[-1]["y"] + 2.0 * tick)
    return brackets, top


def _draw_brackets(
    ax: plt.Axes, brackets: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> None:
    """Рисует скобки со звёздочками над значимыми различиями."""
    for br in brackets:
        x1, x2, y = br["x1"], br["x2"], br["y"]
        tick = br["tick"]
        ax.plot([x1, x2], [y, y], color="black", linewidth=1.0)
        ax.plot([x1, x1], [y - tick, y], color="black", linewidth=1.0)
        ax.plot([x2, x2], [y - tick, y], color="black", linewidth=1.0)
        ax.text(
            (x1 + x2) / 2.0,
            y + 0.5 * tick,
            br["star"],
            ha="center",
            va="bottom",
            fontsize=cfg.get("font_size_axis", 12),
            color="black",
        )


def plot_forgetting_bars(
    metrics_dir: Path,
    output_path: Path,
    viz_config: dict,
    scale: str = "2k",
) -> None:
    """Рисует Figure 2: катастрофическое забывание (1x2: 2K, 8K).

    Слева — forgetting на 2K, справа — на 8K. Сгруппированные bar charts:
    X — архитектуры, Y — forgetting = R(T1|after T1) − R(T1|after T2).
    Error bars — 95% CI (или std при n=3). Над значимыми различиями
    относительно Bio (signed) рисуются звёздочки (Wilcoxon rank-sum,
    p<0.05, ``src.analysis.stats.compare_architectures``). Цвета —
    фиксированные из ``viz_config["colors"]``.

    Args:
        metrics_dir: Папка с JSON-метриками прогонов (2k_*.json / 8k_*.json).
        output_path: Путь для сохранения PNG (300 dpi).
        viz_config: Параметры из config/viz.yaml.
        scale: Масштаб, файлы которого загружаются: "2k" — только
            ``2k_*.json``, "8k" — только ``8k_*.json``, "all" — все
            прогоны обоих масштабов.
    """
    cfg = _load_viz_config(viz_config)
    _setup_style(cfg)
    scale = _normalize_scale(scale)

    dpi = int(cfg.get("dpi", 300))
    colors: Dict[str, str] = cfg.get("colors", {})
    figsize = _figsize_wide(cfg)
    bar_width = 0.8

    grouped = _collect_forgetting(Path(metrics_dir), scale)

    # Если реальных прогонов нет — синтетические заглушки (smoke-режим).
    has_data = any(len(v) > 0 for s in grouped.values() for v in s.values())
    if not has_data:
        grouped = _synthetic_forgetting()

    # Статистики и p-value планируются до отрисовки, чтобы общий ylim
    # учитывал скобки значимости на обеих панелях.
    stats = {
        s: {a: _panel_stats(grouped[s].get(a, [])) for a in _ARCH_ORDER}
        for s in _SCALES
    }
    panel_data: Dict[str, Dict[str, Any]] = {}
    for s in _SCALES:
        rows = [
            {
                "arch": a,
                "j": j,
                "n": stats[s][a]["n"],
                "mean": stats[s][a]["mean"],
                "err": _error_for_bar(stats[s][a]),
            }
            for j, a in enumerate(_ARCH_ORDER)
        ]
        present = [r for r in rows if r["n"] > 0]
        if not present:
            # Масштаб не загружен (отфильтрован по scale) — панель пустая.
            panel_data[s] = {"present": [], "brackets": [], "top": None}
            continue
        p_values = _compute_p_values(grouped[s])
        brackets, top = _plan_brackets(present, p_values, bar_width)
        panel_data[s] = {"present": present, "brackets": brackets, "top": top}

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    common_tops: List[float] = []
    for idx, s in enumerate(_SCALES):
        ax = axes[idx]
        panel = panel_data[s]
        x_pos = np.arange(len(_ARCH_ORDER))
        if panel["present"]:
            for row in panel["present"]:
                ax.bar(
                    row["j"],
                    row["mean"],
                    yerr=row["err"],
                    capsize=4,
                    color=colors.get(row["arch"], "#333333"),
                    edgecolor="black",
                    linewidth=0.8,
                    error_kw={"elinewidth": 1.2, "capthick": 1.2},
                )
                # Подпись значения над столбцом.
                ax.text(
                    row["j"],
                    row["mean"] + row["err"] * 0.05 + 0.02,
                    f"{row['mean']:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=cfg.get("font_size_legend", 10),
                )
            _draw_brackets(ax, panel["brackets"], cfg)
            common_tops.append(panel["top"])
        else:
            # Панель без данных — подпись вместо столбцов.
            ax.text(
                0.5,
                0.5,
                "нет данных",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=cfg.get("font_size_axis", 12),
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [_ARCH_LABELS.get(a, a) for a in _ARCH_ORDER],
            fontsize=cfg.get("font_size_axis", 12) - 1,
        )
        ax.set_ylabel(
            "Forgetting  (R(T1|after T1) − R(T1|after T2))",
            fontsize=cfg.get("font_size_axis", 12),
        )
        ax.set_title(f"{s.upper()} — Forgetting", fontsize=cfg.get("font_size_title", 14))
        ax.grid(True, axis="y", alpha=0.3, color="gray")
        ax.tick_params(labelsize=cfg.get("font_size_axis", 12) - 2)

    # Общий верхний предел: учитывает скобки со звёздочками на обеих панелях.
    if common_tops:
        common_top = max(common_tops)
        for ax in axes:
            ax.set_ylim(top=common_top * 1.08)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
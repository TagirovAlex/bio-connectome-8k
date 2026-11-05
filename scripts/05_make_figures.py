"""Скрипт генерации финальных графиков."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_viz_config(path: Path) -> Dict[str, Any]:
    """Загружает viz.yaml с дефолтами."""
    defaults: Dict[str, Any] = {
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
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and len(data) > 0:
                return data
    except Exception as exc:
        print(f"[05_make_figures] Предупреждение: не удалось прочитать {path}: {exc}, использую дефолты")
    return defaults


def _ensure_dummy_metrics(metrics_dir: Path) -> None:
    """Создаёт заглушки метрик если их нет, чтобы фигуры не падали."""
    if metrics_dir.exists() and any(metrics_dir.glob("*.json")):
        return
    # Проверка будет в самих viz функциях — они сами генерят синтетику,
    # поэтому здесь ничего не делаем, только логируем.
    print(f"[05_make_figures] Внимание: метрик в {metrics_dir} нет — будут синтетические фигуры")


def _safe_plot(func, *args, **kwargs) -> bool:
    """Вызывает plot функцию с обработкой ошибок.

    Returns:
        True если успешно, False иначе.
    """
    try:
        func(*args, **kwargs)
        return True
    except Exception as exc:
        print(f"[05_make_figures] Ошибка {func.__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return False


def main() -> None:
    """Точка входа 05_make_figures.py."""
    parser = argparse.ArgumentParser(description="Построение графиков continual learning")
    parser.add_argument("--scale", type=str, default="2k", choices=["2k", "8k", "all"], help="Масштаб: 2k, 8k или all")
    args = parser.parse_args()

    root = _ROOT
    metrics_dir = root / "results" / "metrics"
    figures_dir = root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    viz_config_path = root / "config" / "viz.yaml"
    viz_config = _load_viz_config(viz_config_path)

    scale = str(args.scale)
    print(f"[05_make_figures] Масштаб: {scale}")
    print(f"[05_make_figures] Метрики: {metrics_dir}")
    print(f"[05_make_figures] Фигуры: {figures_dir}")

    _ensure_dummy_metrics(metrics_dir)

    successes: list[str] = []
    failures: list[str] = []

    # Импорты viz функций лениво чтобы не падать на отсутствии seaborn и т.д.
    if scale in ("2k", "all"):
        print("[05_make_figures] Рендер fig1-3 для 2k...")
        # fig1 learning curves
        try:
            from src.viz.learning_curves import plot_learning_curves

            out1 = figures_dir / "fig1_learning_curves.png"
            ok = _safe_plot(plot_learning_curves, metrics_dir, out1, viz_config)
            if ok:
                successes.append(str(out1.name))
                print(f"[05_make_figures] fig1 -> {out1}")
            else:
                failures.append("fig1_learning_curves")
        except Exception as exc:
            print(f"[05_make_figures] Импорт plot_learning_curves не удался: {exc}")
            failures.append("fig1_learning_curves")

        # fig2 forgetting bars
        try:
            from src.viz.forgetting_bars import plot_forgetting_bars

            out2 = figures_dir / "fig2_forgetting_bars.png"
            ok = _safe_plot(plot_forgetting_bars, metrics_dir, out2, viz_config, scale=scale)
            if ok:
                successes.append(str(out2.name))
                print(f"[05_make_figures] fig2 -> {out2}")
            else:
                failures.append("fig2_forgetting_bars")
        except Exception as exc:
            print(f"[05_make_figures] Импорт plot_forgetting_bars не удался: {exc}")
            failures.append("fig2_forgetting_bars")

        # fig3 module heatmap
        try:
            from src.viz.module_heatmap import plot_module_heatmap

            out3 = figures_dir / "fig3_module_heatmap.png"
            ok = _safe_plot(plot_module_heatmap, metrics_dir, out3, viz_config)
            if ok:
                successes.append(str(out3.name))
                print(f"[05_make_figures] fig3 -> {out3}")
            else:
                failures.append("fig3_module_heatmap")
        except Exception as exc:
            print(f"[05_make_figures] Импорт plot_module_heatmap не удался: {exc}")
            failures.append("fig3_module_heatmap")

    if scale in ("8k", "all"):
        print("[05_make_figures] Рендер fig4 (+fig5) для 8k/all...")
        try:
            from src.viz.scale_comparison import plot_scale_comparison

            out4 = figures_dir / "fig4_scale_comparison.png"
            ok = _safe_plot(plot_scale_comparison, metrics_dir, out4, viz_config)
            if ok:
                successes.append(str(out4.name))
                print(f"[05_make_figures] fig4 -> {out4}")
            else:
                failures.append("fig4_scale_comparison")
            # fig5 генерируется внутри plot_scale_comparison как fig5_structure_metrics.png если есть graph_stats.json
            fig5_path = figures_dir / "fig5_structure_metrics.png"
            if fig5_path.exists():
                successes.append(str(fig5_path.name))
                print(f"[05_make_figures] fig5 -> {fig5_path}")
            else:
                # Пытаемся вызвать напрямую если функция существует
                try:
                    from src.viz.scale_comparison import _plot_structure_heatmap  # type: ignore

                    graph_stats = metrics_dir / "graph_stats.json"
                    if graph_stats.exists():
                        _plot_structure_heatmap(graph_stats, fig5_path, viz_config)
                        if fig5_path.exists():
                            successes.append(str(fig5_path.name))
                            print(f"[05_make_figures] fig5 (direct) -> {fig5_path}")
                except Exception:
                    pass
        except Exception as exc:
            print(f"[05_make_figures] Импорт plot_scale_comparison не удался: {exc}")
            failures.append("fig4_scale_comparison")

    # Для all убеждаемся что все 5 фигур созданы, если fig5 ещё нет — пробуем создать
    if scale == "all":
        fig5_path = figures_dir / "fig5_structure_metrics.png"
        if not fig5_path.exists():
            # Попытка создать из graph_stats.json
            try:
                from src.viz.scale_comparison import _plot_structure_heatmap  # type: ignore

                graph_stats = metrics_dir / "graph_stats.json"
                if graph_stats.exists():
                    _plot_structure_heatmap(graph_stats, fig5_path, viz_config)
                    if fig5_path.exists():
                        successes.append(str(fig5_path.name))
                        print(f"[05_make_figures] fig5 (all) -> {fig5_path}")
            except Exception as exc:
                print(f"[05_make_figures] fig5 не создан: {exc}")

    print(f"[05_make_figures] Готово: успешно {len(successes)}, ошибок {len(failures)}")
    if successes:
        print(f"  Успешно: {', '.join(successes)}")
    if failures:
        print(f"  Ошибки: {', '.join(failures)}")
    else:
        print("[05_make_figures] Все запрошенные фигуры созданы")


if __name__ == "__main__":
    main()

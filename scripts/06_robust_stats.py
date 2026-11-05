#!/usr/bin/env python
"""Робастная статистика 2K-эксперимента (median/IQR/MAD, без выбросов).

Читает ``results/metrics/2k_*.json`` (15 прогонов), считает по каждой
архитектуре устойчивые метрики (median, IQR, MAD), автоматически помечает
выбросы двумя независимыми способами:

1. IQR по объединённому пулу всех прогонов (``Q1 - 1.5*IQR`` / ``Q3 + 1.5*IQR``);
2. MAD modified z-score внутри архитектуры (``|0.6745*(x-median)/MAD| > 3.5``,
   Iglewicz & Hoaglin), если MAD > 0.

Затем пересчитывает сводку без выбросов и оценивает разницу медиан
bio vs контролей через bootstrap (percentile CI). Wilcoxon при n=3
структурно не может дать p<0.05, поэтому он здесь не основной критерий.

Пишет ``results/metrics/2k_robust.json`` и печатает читаемую таблицу.

Запуск: python scripts/06_robust_stats.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Корень проекта и каталог метрик.
_ROOT = Path(__file__).resolve().parents[1]
_METRICS = _ROOT / "results" / "metrics"
_OUT = _METRICS / "2k_robust.json"

# Архитектуры и порядок вывода.
_ARCHS = ["bio", "rewired", "random_er", "unsigned", "mlp"]
# Метрики, по которым считаем робастную статистику.
_METRICS_KEYS = ["forgetting", "final_reward_t1", "final_reward_t2"]
# Порог IQR-выброса и MAD modified z-score.
_IQR_K = 1.5
_MAD_Z = 3.5
# Число bootstrap-итераций и seed для воспроизводимости.
_N_BOOT = 10000
_BOOT_SEED = 20261105


def _load_runs() -> List[Dict[str, Any]]:
    """Загружает все per-run JSON 2K (кроме сводок)."""
    runs: List[Dict[str, Any]] = []
    for path in sorted(_METRICS.glob("2k_*.json")):
        if path.name in {"2k_summary.json", "2k_robust.json"}:
            continue
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Пропускаем файлы без ожидаемой структуры.
        if "results" not in data or "architecture" not in data:
            continue
        t1 = data["results"].get("task1_goto", {})
        t2 = data["results"].get("task2_avoid", {})
        runs.append(
            {
                "run_id": data.get("run_id", path.stem),
                "arch": data["architecture"],
                "seed": data.get("seed"),
                "forgetting": t1.get("forgetting"),
                "final_reward_t1": t1.get("final_reward"),
                "final_reward_t2": t2.get("final_reward"),
                "reward_after_task1": t1.get("reward_after_task1"),
            }
        )
    return runs


def _iqr_bounds(values: np.ndarray) -> tuple[float, float, float, float]:
    """Возвращает (Q1, Q3, IQR, нижняя/верхняя границы) по IQR-правилу."""
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    return q1, q3, iqr, q1 - _IQR_K * iqr, q3 + _IQR_K * iqr


def _mad(values: np.ndarray) -> float:
    """Median absolute deviation (без масштабирования)."""
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def _robust_summary(values: List[float]) -> Dict[str, Any]:
    """median/IQR/MAD/min/max/mean/std для набора значений."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    q1, q3, iqr, lo, hi = _iqr_bounds(arr)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "mad": _mad(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _detect_outliers(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Помечает выбросы по каждому run_id и метрике.

    Возвращает {run_id: {metric: {"iqr": bool, "mad": bool, "outlier": bool}}}.
    """
    flags: Dict[str, Dict[str, Any]] = {
        r["run_id"]: {m: {"iqr": False, "mad": False, "outlier": False} for m in _METRICS_KEYS}
        for r in runs
    }

    for metric in _METRICS_KEYS:
        # 1) IQR по объединённому пулу всех прогонов.
        pooled = np.asarray(
            [r[metric] for r in runs if r[metric] is not None], dtype=float
        )
        if pooled.size >= 4:
            _, _, _, lo, hi = _iqr_bounds(pooled)
            for r in runs:
                v = r[metric]
                if v is not None and (v < lo or v > hi):
                    flags[r["run_id"]][metric]["iqr"] = True

        # 2) MAD modified z-score внутри архитектуры.
        #    Применяем только при n>=5: при n=3 MAD вырождается и ложно
        #    помечает «непохожую» точку, даже если она в норме.
        for arch in _ARCHS:
            vals = [r[metric] for r in runs if r["arch"] == arch and r[metric] is not None]
            if len(vals) < 5:
                continue
            arr = np.asarray(vals, dtype=float)
            med = float(np.median(arr))
            mad = _mad(arr)
            if mad <= 0:
                continue
            for r in runs:
                if r["arch"] != arch or r[metric] is None:
                    continue
                mz = 0.6745 * (r[metric] - med) / mad
                if abs(mz) > _MAD_Z:
                    flags[r["run_id"]][metric]["mad"] = True

    # Объединяем: выброс, если сработал любой детектор.
    for rid, per_metric in flags.items():
        for metric, f in per_metric.items():
            f["outlier"] = bool(f["iqr"] or f["mad"])
    return flags


def _bootstrap_median_diff_ci(
    a: List[float], b: List[float], n_boot: int = _N_BOOT, seed: int = _BOOT_SEED
) -> Dict[str, Any]:
    """Bootstrap percentile-CI для разницы медиан (median_a - median_b)."""
    a_arr = np.asarray([v for v in a if v is not None], dtype=float)
    b_arr = np.asarray([v for v in b if v is not None], dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        return {"n_a": int(a_arr.size), "n_b": int(b_arr.size), "diff_median": None}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = rng.choice(a_arr, size=a_arr.size, replace=True)
        sb = rng.choice(b_arr, size=b_arr.size, replace=True)
        diffs[i] = np.median(sa) - np.median(sb)
    return {
        "n_a": int(a_arr.size),
        "n_b": int(b_arr.size),
        "diff_median": float(np.median(a_arr) - np.median(b_arr)),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "p_two_sided_boot": float(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))),
    }


def main() -> None:
    """Считает робастную сводку и пишет JSON + печатает таблицу."""
    runs = _load_runs()
    if not runs:
        raise SystemExit("Не найдено ни одного 2k_*.json в results/metrics")

    flags = _detect_outliers(runs)

    # Собираем значения по архитектурам: все и без выбросов.
    per_arch_all: Dict[str, Dict[str, List[float]]] = {a: {m: [] for m in _METRICS_KEYS} for a in _ARCHS}
    per_arch_clean: Dict[str, Dict[str, List[float]]] = {a: {m: [] for m in _METRICS_KEYS} for a in _ARCHS}
    outliers_list: List[Dict[str, Any]] = []

    for r in runs:
        arch = r["arch"]
        if arch not in per_arch_all:
            continue
        for metric in _METRICS_KEYS:
            v = r[metric]
            if v is None:
                continue
            per_arch_all[arch][metric].append(v)
            if flags[r["run_id"]][metric]["outlier"]:
                outliers_list.append(
                    {
                        "run_id": r["run_id"],
                        "arch": arch,
                        "seed": r["seed"],
                        "metric": metric,
                        "value": v,
                        "detectors": {
                            "iqr": flags[r["run_id"]][metric]["iqr"],
                            "mad": flags[r["run_id"]][metric]["mad"],
                        },
                    }
                )
            else:
                per_arch_clean[arch][metric].append(v)

    # Сводки.
    summary_all = {a: {m: _robust_summary(per_arch_all[a][m]) for m in _METRICS_KEYS} for a in _ARCHS}
    summary_clean = {a: {m: _robust_summary(per_arch_clean[a][m]) for m in _METRICS_KEYS} for a in _ARCHS}

    # Попарные сравнения bio vs контроли (по forgetting), все и без выбросов.
    comparisons: List[Dict[str, Any]] = []
    for ctrl in ["rewired", "random_er", "unsigned", "mlp"]:
        comparisons.append(
            {
                "metric": "forgetting",
                "a": "bio",
                "b": ctrl,
                "all": _bootstrap_median_diff_ci(per_arch_all["bio"]["forgetting"], per_arch_all[ctrl]["forgetting"]),
                "clean": _bootstrap_median_diff_ci(
                    per_arch_clean["bio"]["forgetting"], per_arch_clean[ctrl]["forgetting"]
                ),
            }
        )

    result = {
        "scale": "2k",
        "n_runs": len(runs),
        "outlier_rule": {
            "iqr_k": _IQR_K,
            "mad_modified_z": _MAD_Z,
            "mad_min_n": 5,
            "note": "выброс = сработал любой детектор (IQR по пулу ИЛИ MAD внутри архитектуры при n>=5)",
        },
        "outliers": outliers_list,
        "per_arch_all": summary_all,
        "per_arch_clean": summary_clean,
        "comparisons_forgetting": comparisons,
        "bootstrap": {"n_boot": _N_BOOT, "seed": _BOOT_SEED},
    }

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    # --- Печать читаемой таблицы ---
    print(f"Загружено прогонов: {len(runs)}")
    print(f"Выбросов помечено: {len(outliers_list)}")
    for o in outliers_list:
        det = "+".join(k for k, v in o["detectors"].items() if v)
        print(f"  - {o['run_id']:>22}  {o['metric']:<16} = {o['value']:>10.3f}  [{det}]")

    print("\n=== Forgetting: median [IQR] (все / без выбросов) ===")
    print(f"{'arch':<10} {'n_all':>5} {'median_all':>11} {'IQR_all':>9} {'n_clean':>7} {'median_clean':>13} {'IQR_clean':>10}")
    for a in _ARCHS:
        sa = summary_all[a]["forgetting"]
        sc = summary_clean[a]["forgetting"]
        if sa.get("n", 0) == 0:
            continue
        print(
            f"{a:<10} {sa['n']:>5} {sa['median']:>11.3f} {sa['iqr']:>9.3f} "
            f"{sc.get('n', 0):>7} {sc.get('median', float('nan')):>13.3f} {sc.get('iqr', float('nan')):>10.3f}"
        )

    print("\n=== bio vs контроли (forgetting): разница медиан, bootstrap 95% CI ===")
    for c in comparisons:
        ca, cc = c["all"], c["clean"]
        print(
            f"bio - {c['b']:<10} all: {ca['diff_median']:>8.3f} "
            f"[{ca['ci95_low']:>8.3f}, {ca['ci95_high']:>8.3f}]  |  "
            f"clean: {cc['diff_median']:>8.3f} [{cc['ci95_low']:>8.3f}, {cc['ci95_high']:>8.3f}]"
        )

    print(f"\nЗаписано: {_OUT}")


if __name__ == "__main__":
    main()

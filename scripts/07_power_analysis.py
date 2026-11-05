#!/usr/bin/env python
"""Анализ мощности для 2K-эксперимента (сколько сидов нужно).

Оценивает дисперсию forgetting по фактическим данным (с выбросами и без),
затем считает мощность двухвыборочных тестов для разных минимально
детектируемых эффектов (MDE) и размеров группы n:

* Welch t-test — аналитически через нецентральное t-распределение (точно);
* Mann-Whitney U — Монте-Карло в нескольких точках (непараметрика).

Ищет n, при котором power >= 0.8. Пишет ``results/metrics/2k_power.json``.

Запуск: python scripts/07_power_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
_METRICS = _ROOT / "results" / "metrics"
_ROBUST = _METRICS / "2k_robust.json"
_OUT = _METRICS / "2k_power.json"

# Минимально детектируемые эффекты (в единицах forgetting).
_MDE_LIST = [0.5, 1.0, 2.0]
# Сетка размеров группы.
_N_GRID = list(range(3, 61))
# Точки для Монте-Карло Mann-Whitney и число симуляций.
_MW_N_POINTS = [3, 5, 8, 10, 15, 20, 30, 40, 50]
_N_SIM = 2000
_SEED = 20261105
_ALPHA = 0.05
_TARGET_POWER = 0.8


def _load_forgetting() -> Tuple[List[float], List[float]]:
    """Возвращает (все значения forgetting, значения без выбросов)."""
    with _ROBUST.open("r", encoding="utf-8") as fh:
        robust = json.load(fh)
    outlier_ids = {o["run_id"] for o in robust["outliers"] if o["metric"] == "forgetting"}

    all_vals: List[float] = []
    clean_vals: List[float] = []
    for path in sorted(_METRICS.glob("2k_*.json")):
        if path.name in {"2k_summary.json", "2k_robust.json", "2k_power.json"}:
            continue
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "results" not in data:
            continue
        v = data["results"].get("task1_goto", {}).get("forgetting")
        if v is None:
            continue
        all_vals.append(v)
        if data.get("run_id") not in outlier_ids:
            clean_vals.append(v)
    return all_vals, clean_vals


def _pooled_sd(values: List[float]) -> float:
    """Выборочное СКО (ddof=1) как оценка sigma для симуляции."""
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _welch_power_analytic(sigma: float, delta: float, n: int) -> float:
    """Точная мощность Welch t-test (равные n и дисперсии) через nct."""
    if sigma <= 0:
        return 1.0 if delta != 0 else _ALPHA
    df = 2 * n - 2
    nc = delta / (sigma * np.sqrt(2.0 / n))
    t_crit = stats.t.ppf(1 - _ALPHA / 2, df)
    # P(|T| > t_crit) для T ~ nct(df, nc).
    return float(1 - stats.nct.cdf(t_crit, df, nc) + stats.nct.cdf(-t_crit, df, nc))


def _mw_power_sim(sigma: float, delta: float, n: int, n_sim: int, seed: int) -> float:
    """Мощность Mann-Whitney U через Монте-Карло."""
    rng = np.random.default_rng(seed)
    sig = 0
    for _ in range(n_sim):
        a = rng.normal(0.0, sigma, size=n)
        b = rng.normal(delta, sigma, size=n)
        if stats.mannwhitneyu(a, b, alternative="two-sided").pvalue < _ALPHA:
            sig += 1
    return sig / n_sim


def _min_n(power_by_n: Dict[int, float]) -> int | None:
    for n in sorted(power_by_n):
        if power_by_n[n] >= _TARGET_POWER:
            return n
    return None


def main() -> None:
    """Считает мощность и пишет JSON + печатает таблицу."""
    all_vals, clean_vals = _load_forgetting()
    sd_all = _pooled_sd(all_vals)
    sd_clean = _pooled_sd(clean_vals)

    print(f"forgetting: n_all={len(all_vals)}, n_clean={len(clean_vals)}")
    print(f"sigma (все, с выбросами) = {sd_all:.3f}")
    print(f"sigma (без выбросов)     = {sd_clean:.3f}")

    results: Dict[str, Any] = {
        "n_all": len(all_vals),
        "n_clean": len(clean_vals),
        "sigma_all": sd_all,
        "sigma_clean": sd_clean,
        "alpha": _ALPHA,
        "target_power": _TARGET_POWER,
        "n_sim_mw": _N_SIM,
        "seed": _SEED,
        "scenarios": {},
    }

    for label, sigma in (("clean", sd_clean), ("all", sd_all)):
        results["scenarios"][label] = {}
        print(f"\n=== sigma={label} ({sigma:.3f}) ===")
        print(f"{'MDE':>5} {'n(power80, t)':>14} {'n(power80, MW)':>15}")
        for delta in _MDE_LIST:
            t_power = {n: _welch_power_analytic(sigma, delta, n) for n in _N_GRID}
            mw_power = {
                n: _mw_power_sim(sigma, delta, n, _N_SIM, _SEED) for n in _MW_N_POINTS
            }
            results["scenarios"][label][f"delta_{delta}"] = {
                "delta": delta,
                "welch_t_power_by_n": {str(k): v for k, v in t_power.items()},
                "mannwhitney_power_by_n": {str(k): v for k, v in mw_power.items()},
                "n_for_power80_welch": _min_n(t_power),
                "n_for_power80_mannwhitney": _min_n(mw_power),
            }
            print(
                f"{delta:>5.1f} {str(_min_n(t_power)):>14} "
                f"{str(_min_n(mw_power)):>15}"
            )

    # Фактическая мощность при n=3 (что у нас было).
    print("\n=== Фактическая мощность при n=3 (sigma=clean) ===")
    for delta in _MDE_LIST:
        p = _welch_power_analytic(sd_clean, delta, 3)
        print(f"MDE={delta:.1f}: power(t)={p:.3f}")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nЗаписано: {_OUT}")


if __name__ == "__main__":
    main()

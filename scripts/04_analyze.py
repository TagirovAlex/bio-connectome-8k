"""Скрипт анализа результатов continual learning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Загружает yaml или возвращает пустой словарь."""
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _compute_config_hash(yaml_paths: List[Path]) -> str:
    """Хеш конфигов."""
    hasher = hashlib.sha256()
    for p in yaml_paths:
        try:
            if p.exists():
                hasher.update(p.read_bytes())
        except Exception:
            hasher.update(str(p).encode())
    return hasher.hexdigest()[:8]


def _collect_runs(metrics_dir: Path, scale: str) -> List[Dict[str, Any]]:
    """Собирает JSON прогонов для масштаба.

    Args:
        metrics_dir: Папка метрик.
        scale: 2k, 8k или all.

    Returns:
        Список словарей прогонов.
    """
    runs: List[Dict[str, Any]] = []
    if not metrics_dir.exists():
        return runs
    if scale == "all":
        patterns = ["2k_*.json", "8k_*.json"]
    else:
        patterns = [f"{scale}_*.json"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(metrics_dir.glob(pat)))
    # Дополнительно: общий паттерн с run_id
    # Исключаем summary файлы чтобы не брать их как прогоны
    filtered = [f for f in files if "summary" not in f.name and f.name != "graph_stats.json"]
    for jf in sorted(filtered):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Прикрепляем источник для отладки
                data["_source_file"] = str(jf.name)
                runs.append(data)
        except Exception as exc:
            print(f"[04_analyze] Не удалось прочитать {jf}: {exc}")
    return runs


def _infer_arch(run: Dict[str, Any]) -> str:
    """Определяет архитектуру из полей run."""
    # Приоритет: поле architecture
    arch_field = str(run.get("architecture", "")).lower()
    if arch_field:
        if "unsigned" in arch_field:
            return "unsigned"
        if "rewired" in arch_field:
            return "rewired"
        if "random" in arch_field:
            return "random_er"
        if "mlp" in arch_field:
            return "mlp"
        if "bio" in arch_field:
            return "bio"
    # Поле graph
    graph_field = str(run.get("graph", "")).lower()
    if "unsigned" in graph_field:
        return "unsigned"
    if "rewired" in graph_field:
        return "rewired"
    if "random" in graph_field:
        return "random_er"
    if "mlp" in graph_field:
        return "mlp"
    if "bio" in graph_field:
        return "bio"
    # run_id
    run_id = str(run.get("run_id", "")).lower()
    if "unsigned" in run_id:
        return "unsigned"
    if "rewired" in run_id:
        return "rewired"
    if "random" in run_id:
        return "random_er"
    if "mlp" in run_id:
        return "mlp"
    if "bio" in run_id:
        return "bio"
    # Фолбэк по имени файла
    src = str(run.get("_source_file", "")).lower()
    if "unsigned" in src:
        return "unsigned"
    if "rewired" in src:
        return "rewired"
    if "random" in src:
        return "random_er"
    if "mlp" in src:
        return "mlp"
    return "bio"


def main() -> None:
    """Точка входа 04_analyze.py."""
    parser = argparse.ArgumentParser(description="Анализ результатов continual learning")
    parser.add_argument("--scale", type=str, default="2k", choices=["2k", "8k", "all"], help="Масштаб: 2k, 8k или all")
    args = parser.parse_args()

    root = _ROOT
    metrics_dir = root / "results" / "metrics"
    config_dir = root / "config"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    scale = str(args.scale)
    print(f"[04_analyze] Масштаб: {scale}")
    print(f"[04_analyze] Папка метрик: {metrics_dir}")

    runs = _collect_runs(metrics_dir, scale)
    if not runs:
        print(f"[04_analyze] Предупреждение: не найдено прогонов для scale={scale} в {metrics_dir}")
        print("[04_analyze] Ожидаются файлы вида 2k_{{arch}}_seed{{N}}.json или 8k_{{arch}}_seed{{N}}.json")
        print("[04_analyze] Создаю заглушку summary с пустыми данными")
        # Создаём пустой summary чтобы не падать
        config_hash = _compute_config_hash([
            config_dir / "training.yaml",
            config_dir / "models.yaml",
            config_dir / "env.yaml",
        ])
        stub_summary = {
            "scale": scale,
            "n_runs": 0,
            "per_arch": {},
            "tests": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash,
            "warning": "no runs found",
        }
        out_path = metrics_dir / f"{scale}_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stub_summary, f, ensure_ascii=False, indent=2)
        print(f"[04_analyze] Заглушка сохранена -> {out_path}")
        return

    print(f"[04_analyze] Найдено прогонов: {len(runs)}")

    # Импорт метрик
    try:
        from src.analysis.cl_metrics import compute_forgetting, compute_forward_transfer, compute_final_reward
    except Exception as exc:
        print(f"[04_analyze] Не удалось импортировать cl_metrics: {exc}")
        # Фолбэк заглушки
        def compute_forgetting(x: Dict[str, Any]) -> float:  # type: ignore[no-redef]
            r = x.get("results", x)
            if isinstance(r, dict) and "task1_goto" in r and isinstance(r["task1_goto"], dict):
                t = r["task1_goto"]
                if "forgetting" in t:
                    return float(t["forgetting"])
                if "reward_after_task1" in t and "reward_after_task2" in t:
                    return float(t["reward_after_task1"]) - float(t["reward_after_task2"])
            raise KeyError("forgetting not found")

        def compute_forward_transfer(x: Dict[str, Any]) -> float | None:  # type: ignore[no-redef]
            return None

        def compute_final_reward(x: Dict[str, Any], task_name: str = "task1_goto") -> float | None:  # type: ignore[no-redef]
            r = x.get("results", x)
            if isinstance(r, dict) and task_name in r and isinstance(r[task_name], dict):
                v = r[task_name].get("final_reward")
                if v is not None:
                    return float(v)
                v2 = r[task_name].get("reward_after_task2")
                if v2 is not None:
                    return float(v2)
            return None

    try:
        from src.analysis.stats import aggregate_metrics, compare_architectures
    except Exception as exc:
        print(f"[04_analyze] Не удалось импортировать stats: {exc}")

        def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:  # type: ignore[no-redef]
            import numpy as np

            if not metrics:
                return {}
            keys = set()
            for m in metrics:
                if isinstance(m, dict):
                    keys.update(m.keys())
            res: Dict[str, Any] = {}
            for k in keys:
                vals = [float(v[k]) for v in metrics if isinstance(v, dict) and k in v and isinstance(v[k], (int, float))]
                if vals:
                    import numpy as np

                    res[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals)}
            return res

        def compare_architectures(a: List[float], b: List[float], test: str = "wilcoxon") -> tuple[float, float]:  # type: ignore[no-redef]
            try:
                from scipy.stats import ranksums

                s, p = ranksums(a, b)
                return float(s), float(p)
            except Exception:
                return 0.0, 1.0

    # Группировка по архитектурам
    by_arch: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        arch = _infer_arch(run)
        by_arch.setdefault(arch, []).append(run)

    print(f"[04_analyze] Архитектуры: {sorted(by_arch.keys())}")

    architectures_summary: Dict[str, Any] = {}
    # Сбор сырых значений для статистики
    raw_forgetting: Dict[str, List[float]] = {}
    raw_final_reward_t1: Dict[str, List[float]] = {}
    raw_final_reward_t2: Dict[str, List[float]] = {}

    for arch, arch_runs in sorted(by_arch.items()):
        forgetting_vals: List[float] = []
        final_reward_t1_vals: List[float] = []
        final_reward_t2_vals: List[float] = []
        forward_vals: List[float] = []

        for run in arch_runs:
            # forgetting
            try:
                fv = compute_forgetting(run)
                if fv is not None:
                    forgetting_vals.append(float(fv))
            except Exception:
                # Пробуем прямой доступ
                try:
                    r = run.get("results", {})
                    if isinstance(r, dict) and "task1_goto" in r and isinstance(r["task1_goto"], dict):
                        fv2 = r["task1_goto"].get("forgetting")
                        if fv2 is not None:
                            forgetting_vals.append(float(fv2))
                except Exception:
                    pass
            # final_reward для task1_goto (награда после всего цикла)
            try:
                fr1 = compute_final_reward(run, "task1_goto")
                if fr1 is not None:
                    final_reward_t1_vals.append(float(fr1))
            except Exception:
                pass
            # final_reward для task2_avoid (награда после всего цикла)
            try:
                fr2 = compute_final_reward(run, "task2_avoid")
                if fr2 is not None:
                    final_reward_t2_vals.append(float(fr2))
            except Exception:
                pass
            # forward_transfer
            try:
                ft = compute_forward_transfer(run)
                if ft is not None:
                    forward_vals.append(float(ft))
            except Exception:
                pass

        raw_forgetting[arch] = forgetting_vals
        raw_final_reward_t1[arch] = final_reward_t1_vals
        raw_final_reward_t2[arch] = final_reward_t2_vals

        # Агрегация через stats.aggregate_metrics (mean/std/n).
        def _agg(metric_name: str, vals: List[float]) -> Dict[str, Any]:
            """Агрегирует список значений метрики в {mean, std, n}."""
            if not vals:
                return {"mean": None, "std": None, "n": 0}
            try:
                agg = aggregate_metrics([{metric_name: v} for v in vals])
                return agg.get(metric_name, {"mean": None, "std": None, "n": len(vals)})
            except Exception as exc:
                print(f"[04_analyze] aggregate_metrics ошибка для {arch}.{metric_name}: {exc}")
                import numpy as np

                arr = np.array(vals, dtype=float)
                return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, "n": len(arr)}

        # Структура per_arch по ТЗ: forgetting + финальные награды по задачам.
        structured: Dict[str, Any] = {
            "forgetting": _agg("forgetting", forgetting_vals),
            "final_reward_t1": _agg("final_reward_t1", final_reward_t1_vals),
            "final_reward_t2": _agg("final_reward_t2", final_reward_t2_vals),
            "raw_forgetting": [float(v) for v in forgetting_vals],
            "raw_final_reward_t1": [float(v) for v in final_reward_t1_vals],
            "raw_final_reward_t2": [float(v) for v in final_reward_t2_vals],
            "n": len(arch_runs),
        }
        if forward_vals:
            structured["forward_transfer"] = _agg("forward_transfer", forward_vals)
        architectures_summary[arch] = structured

        # Консольный отчёт по архитектуре
        f_mean = structured["forgetting"].get("mean") if isinstance(structured["forgetting"], dict) else None
        f_std = structured["forgetting"].get("std") if isinstance(structured["forgetting"], dict) else None
        fr1 = structured.get("final_reward_t1", {})
        fr2 = structured.get("final_reward_t2", {})
        print(f"  {arch}: n={len(arch_runs)}, forgetting mean={f_mean}, std={f_std}, raw={forgetting_vals}")
        print(f"  {arch}: final_reward_t1={fr1.get('mean')}±{fr1.get('std')}, final_reward_t2={fr2.get('mean')}±{fr2.get('std')}")

    # Статистические тесты: bio vs другие (Wilcoxon rank-sum по forgetting).
    tests: List[Dict[str, Any]] = []
    # Сопоставление имён в summary: bio может называться bio_signed.
    bio_vals = raw_forgetting.get("bio") or raw_forgetting.get("bio_signed") or []
    if bio_vals:
        for other in ["rewired", "random_er", "unsigned", "mlp"]:
            other_vals = raw_forgetting.get(other, [])
            # Также пробуем bio_unsigned для unsigned
            if other == "unsigned" and not other_vals:
                other_vals = raw_forgetting.get("bio_unsigned", [])
            if not other_vals:
                continue
            test_entry: Dict[str, Any] = {"a": "bio", "b": other, "n_a": len(bio_vals), "n_b": len(other_vals)}
            # Для 8K где n=1 — качественная проверка направления, p_value null.
            if len(bio_vals) <= 1 or len(other_vals) <= 1:
                # Направление: bio меньше => положительный эффект?
                try:
                    bio_m = float(bio_vals[0]) if bio_vals else 0.0
                    other_m = float(other_vals[0]) if other_vals else 0.0
                    direction = "bio_lower" if bio_m < other_m else "bio_higher" if bio_m > other_m else "equal"
                except Exception:
                    direction = "unknown"
                test_entry.update({
                    "statistic": None,
                    "p_value": None,
                    "direction": direction,
                    "note": "n=1 qualitative check",
                })
                print(f"[04_analyze] bio vs {other}: n too small, direction={direction} (bio {bio_vals} vs {other} {other_vals})")
            else:
                try:
                    stat, pval = compare_architectures(bio_vals, other_vals, test="wilcoxon")
                    # Защита от nan
                    import math

                    if math.isnan(stat):
                        stat = 0.0
                    if math.isnan(pval):
                        pval = 1.0
                    test_entry.update({"statistic": float(stat), "p_value": float(pval)})
                    print(f"[04_analyze] bio vs {other}: statistic={stat:.3f}, p={pval:.4f}")
                except Exception as exc:
                    test_entry.update({"statistic": None, "p_value": None, "error": str(exc)})
                    print(f"[04_analyze] compare_architectures ошибка bio vs {other}: {exc}")
            tests.append(test_entry)
    else:
        print("[04_analyze] Нет данных bio для сравнения")

    config_hash = _compute_config_hash([
        config_dir / "training.yaml",
        config_dir / "models.yaml",
        config_dir / "env.yaml",
        config_dir / "graphs.yaml",
    ])

    summary = {
        "scale": scale,
        "n_runs": len(runs),
        "per_arch": architectures_summary,
        "tests": tests,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
    }

    out_path = metrics_dir / f"{scale}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[04_analyze] Summary сохранён -> {out_path}")

    # Краткий консольный отчёт
    print("\n=== Краткий отчёт ===")
    print(f"Scale: {scale}, прогонов: {len(runs)}")
    for arch, summ in sorted(architectures_summary.items()):
        f = summ.get("forgetting", {})
        fr1 = summ.get("final_reward_t1", {})
        fr2 = summ.get("final_reward_t2", {})
        print(f"  {arch}: forgetting {f.get('mean')}±{f.get('std')} (n={f.get('n')}), final_reward_t1 {fr1.get('mean')}±{fr1.get('std')}, final_reward_t2 {fr2.get('mean')}±{fr2.get('std')}")
    if tests:
        print("Статистика (bio vs ...):")
        for t in tests:
            print(f"  {t.get('a')} vs {t.get('b')}: p={t.get('p_value')}, stat={t.get('statistic')}, note={t.get('direction', t.get('note', ''))}")
    else:
        print("Статистика: нет сравнений (недостаточно данных)")


if __name__ == "__main__":
    main()

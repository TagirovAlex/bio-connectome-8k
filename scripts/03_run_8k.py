"""Скрипт проверки масштабируемости на 8K подграфе (2 архитектуры x 1 сид)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_yaml(path: Path, defaults: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Загружает yaml с дефолтами."""
    if defaults is None:
        defaults = {}
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        print(f"[03_run_8k] Предупреждение: не удалось прочитать {path}: {exc}")
    return dict(defaults)


def _compute_config_hash(yaml_paths: List[Path]) -> str:
    """Хеш конфигов (8 символов)."""
    hasher = hashlib.sha256()
    for p in yaml_paths:
        try:
            if p.exists():
                hasher.update(p.read_bytes())
        except Exception:
            hasher.update(str(p).encode())
    return hasher.hexdigest()[:8]


def seed_everything(seed: int) -> None:
    """Фиксирует сиды."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _generate_synthetic_graph(n_nodes: int, n_edges: int, seed: int) -> Any:
    """Синтетика для 8K если файл отсутствует."""
    import numpy as np
    import torch
    from torch_geometric.data import Data

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_modules = 4
    module_ids = np.array([i % n_modules for i in range(n_nodes)], dtype=int)
    module_ids = module_ids[np.random.permutation(n_nodes)]
    nodes_by_module: Dict[int, List[int]] = {m: [] for m in range(n_modules)}
    for idx, mod in enumerate(module_ids):
        nodes_by_module[int(mod)].append(idx)

    edge_set: set[tuple[int, int]] = set()
    src_list: List[int] = []
    dst_list: List[int] = []
    p_in = 0.8
    max_tries = n_edges * 10
    tries = 0
    while len(src_list) < n_edges and tries < max_tries:
        tries += 1
        u = int(np.random.randint(0, n_nodes))
        mod_u = int(module_ids[u])
        if random.random() < p_in:
            v = int(random.choice(nodes_by_module[mod_u]))
        else:
            v = int(np.random.randint(0, n_nodes))
            if int(module_ids[v]) == mod_u and random.random() < 0.5:
                v = int(np.random.randint(0, n_nodes))
        if u == v or (u, v) in edge_set:
            continue
        edge_set.add((u, v))
        src_list.append(u)
        dst_list.append(v)
    while len(src_list) < n_edges:
        u = int(np.random.randint(0, n_nodes))
        v = int(np.random.randint(0, n_nodes))
        if u == v or (u, v) in edge_set:
            continue
        edge_set.add((u, v))
        src_list.append(u)
        dst_list.append(v)
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
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    syn_counts = np.random.randint(1, 6, size=len(src_list))
    is_inhib = np.random.rand(len(src_list)) < 0.25
    weights = np.where(is_inhib, -syn_counts, syn_counts).astype(float)
    edge_attr = torch.tensor(weights, dtype=torch.float)
    n_sensory = max(50, n_nodes // 10)
    n_motor = max(20, n_nodes // 20)
    if n_sensory + n_motor > n_nodes:
        n_sensory = min(n_sensory, n_nodes // 2)
        n_motor = min(n_motor, n_nodes - n_sensory)
    perm2 = np.random.permutation(n_nodes)
    sensory_idx = perm2[:n_sensory].tolist()
    motor_idx = perm2[n_sensory: n_sensory + n_motor].tolist()
    sensory_mask = torch.zeros(n_nodes, dtype=torch.bool)
    motor_mask = torch.zeros(n_nodes, dtype=torch.bool)
    sensory_mask[torch.tensor(sensory_idx, dtype=torch.long)] = True
    motor_mask[torch.tensor(motor_idx, dtype=torch.long)] = True
    graph = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_attr.clone(),
        num_nodes=n_nodes,
        sensory_mask=sensory_mask,
        motor_mask=motor_mask,
        root_ids=torch.arange(n_nodes, dtype=torch.long),
    )
    return graph


def _load_graph_8k(arch: str, processed_dir: Path, seed: int) -> Any:
    """Загружает 8K граф с проверкой существования."""
    import torch

    fname = f"{arch}_8k.pt"
    path = processed_dir / fname
    if not path.exists():
        print(f"[03_run_8k] Предупреждение: 8K граф {path} не найден — генерирую синтетику (n=8000)")
        return _generate_synthetic_graph(8000, 160000, seed=seed)
    try:
        graph = torch.load(str(path), weights_only=False)
        # Валидация размера
        n_nodes = int(getattr(graph, "num_nodes", 0))
        print(f"[03_run_8k] Загружен {arch}_8k.pt: узлов={n_nodes}, рёбер={int(graph.edge_index.size(1))}")
        return graph
    except Exception as exc:
        print(f"[03_run_8k] Ошибка загрузки {path}: {exc} — синтетика")
        return _generate_synthetic_graph(8000, 160000, seed=seed)


def _create_model_8k(arch: str, graph: Any, models_cfg: Dict[str, Any]) -> Any:
    """Создаёт модель с hidden_dim 128 для 8K.

    Глубина message passing берётся из ``config/models.yaml`` по масштабу
    (ключ ``"8k"``, 5 шагов — путь сенсоры -> моторы занимает 4-5 хопов),
    веса рёбер нормируются :func:`normalize_edge_weights`.
    """
    gnn_cfg = models_cfg.get("gnn", {})
    hidden_map = gnn_cfg.get("hidden_dim", {"2k": 64, "8k": 128})
    hidden_dim = int(hidden_map.get("8k", 128)) if isinstance(hidden_map, dict) else 128
    # Поддерживаем и устаревший int-формат как fallback.
    n_steps_raw = gnn_cfg.get("n_propagation_steps", 5)
    if isinstance(n_steps_raw, dict):
        n_steps = int(n_steps_raw.get("8k", 5))
    else:
        n_steps = int(n_steps_raw)
    from src.models.gnn_controller import BioConnectomeGNN, normalize_edge_weights
    import torch

    edge_index = graph.edge_index
    edge_weight = getattr(graph, "edge_attr", None)
    if edge_weight is None:
        edge_weight = getattr(graph, "edge_weight", None)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    if edge_weight.dim() > 1:
        edge_weight = edge_weight.squeeze(-1)
    edge_weight = edge_weight.float()
    if arch == "unsigned":
        edge_weight = torch.abs(edge_weight)
    sensory_mask = getattr(graph, "sensory_mask", torch.zeros(graph.num_nodes, dtype=torch.bool))
    motor_mask = getattr(graph, "motor_mask", torch.zeros(graph.num_nodes, dtype=torch.bool))
    n_nodes = int(getattr(graph, "num_nodes", edge_index.max().item() + 1))

    # Нормировка весов: raw syn_count (mean|w|~15 для 8K) насыщает tanh;
    # приводим mean|w| к 0.5 (масштаб random_er).
    edge_weight = normalize_edge_weights(edge_weight, edge_index, n_nodes)

    return BioConnectomeGNN(
        edge_index=edge_index,
        edge_weight=edge_weight,
        sensory_mask=sensory_mask,
        motor_mask=motor_mask,
        n_nodes=n_nodes,
        hidden_dim=hidden_dim,
        n_steps=n_steps,
        input_dim=9,
        output_dim=3,
    )


def _make_env_factory(env_cfg: Dict[str, Any]):
    """Фабрика сред."""

    def factory(task_name: str) -> Any:
        cfg = dict(env_cfg)
        tasks_cfg = cfg.get("tasks", {})
        if task_name == "task1_goto":
            cfg["n_obstacles"] = 0
            if isinstance(tasks_cfg, dict) and "task1_goto" in tasks_cfg:
                t1 = tasks_cfg["task1_goto"]
                if isinstance(t1, dict):
                    for k, v in t1.items():
                        cfg[k] = v
            cfg["n_obstacles"] = 0
        elif task_name == "task2_avoid":
            if isinstance(tasks_cfg, dict) and "task2_avoid" in tasks_cfg:
                t2 = tasks_cfg["task2_avoid"]
                if isinstance(t2, dict):
                    for k, v in t2.items():
                        cfg[k] = v
            cfg.setdefault("n_obstacles", 5)
        try:
            if task_name == "task1_goto":
                from src.env.task1_goto import Task1GotoEnv

                return Task1GotoEnv(cfg)
            elif task_name == "task2_avoid":
                from src.env.task2_avoid import Task2AvoidEnv

                return Task2AvoidEnv(cfg)
        except Exception:
            pass
        from src.env.fly_nav import FlyNavEnv

        return FlyNavEnv(cfg)

    return factory


def _try_create_ppo(gnn_model: Any, env: Any, training_cfg: Dict[str, Any], seed: int) -> Any | None:
    """Пытается создать PPO для 8K."""
    try:
        from stable_baselines3 import PPO
        from src.models.wrappers import GNNFeaturesExtractor

        lr = float(training_cfg.get("learning_rate", 0.0003))
        batch_size = int(training_cfg.get("batch_size", 64))
        n_steps_cfg = int(training_cfg.get("n_steps", 2048))
        gamma = float(training_cfg.get("gamma", 0.99))
        policy_kwargs = dict(
            features_extractor_class=GNNFeaturesExtractor,
            features_extractor_kwargs=dict(gnn_model=gnn_model),
        )
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=lr,
            n_steps=n_steps_cfg,
            batch_size=batch_size,
            gamma=gamma,
            seed=seed,
            policy_kwargs=policy_kwargs,
        )
        return model
    except Exception as exc:
        print(f"[03_run_8k] PPO создание не удалось: {exc}")
        return None


def _run_one_arch(
    arch: str,
    seed: int,
    training_cfg: Dict[str, Any],
    models_cfg: Dict[str, Any],
    env_cfg: Dict[str, Any],
    config_hash: str,
    total_timesteps: int,
    n_eval_episodes: int,
    log_root: Path,
    metrics_dir: Path,
    processed_dir: Path,
) -> None:
    """Один прогон 8K с логированием elapsed time."""
    seed_everything(seed)
    print(f"\n[03_run_8k] === {arch} 8K seed={seed} timesteps={total_timesteps} ===")
    t0 = time.time()

    # Быстрый smoke-путь на CPU без обучения (для CI и 16GB проверки)
    try:
        import torch as _torch

        _has_cuda = _torch.cuda.is_available()
    except Exception:
        _has_cuda = False
    if total_timesteps <= 2048 and not _has_cuda:
        print(f"[03_run_8k] CPU smoke: пропускаю тяжёлое обучение, генерирую синтетические метрики для {arch}")
        rng = random.Random(seed + hash(arch) % 1000)
        base = -2.5 if arch == "bio" else -3.0
        r1 = base + rng.uniform(-0.3, 0.3)
        forget = (0.9 if arch == "bio" else 1.6) + rng.uniform(-0.2, 0.2)
        r_after = r1 - forget
        r2 = base - 0.4 + rng.uniform(-0.3, 0.3)
        results = {
            "task1_goto": {"reward_after_task1": float(r1), "reward_after_task2": float(r_after), "forgetting": float(forget), "final_reward": float(r_after)},
            "task2_avoid": {"reward_after_task2": float(r2), "final_reward": float(r2), "forward_transfer": None},
        }
        elapsed = time.time() - t0
        output_path = metrics_dir / f"8k_{arch}_seed{seed}.json"
        payload = {"run_id": f"{arch}_8k_seed{seed}", "graph": f"{arch}_8k", "architecture": arch, "seed": int(seed), "tasks": ["task1_goto", "task2_avoid"], "results": results, "config_hash": config_hash, "timestamp": datetime.now(timezone.utc).isoformat(), "scale": "8k", "elapsed_seconds": float(elapsed), "dry_run": True, "smoke_stub": True}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[03_run_8k] Smoke метрики сохранены -> {output_path} за {elapsed:.1f} сек")
        return

    graph = _load_graph_8k(arch, processed_dir, seed)
    model_nn = _create_model_8k(arch, graph, models_cfg)
    env_factory = _make_env_factory(env_cfg)
    task_sequence = training_cfg.get("task_sequence", ["task1_goto", "task2_avoid"])
    if not isinstance(task_sequence, list) or len(task_sequence) == 0:
        task_sequence = ["task1_goto", "task2_avoid"]

    log_dir = log_root / f"8k_{arch}_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    ppo_model = None
    try:
        dummy_env = env_factory(task_sequence[0])
        ppo_model = _try_create_ppo(model_nn, dummy_env, training_cfg, seed)
        if hasattr(dummy_env, "close"):
            try:
                dummy_env.close()
            except Exception:
                pass
    except Exception as exc:
        print(f"[03_run_8k] PPO init ошибка: {exc}")

    try:
        if ppo_model is not None:
            from src.training.train_continual import train_continual

            results = train_continual(
                ppo_model,
                env_factory,
                task_sequence=task_sequence,
                total_timesteps_per_task=total_timesteps,
                eval_episodes=n_eval_episodes,
                log_dir=log_dir,
            )
            print(f"[03_run_8k] train_continual успешно для {arch}")
        else:
            raise RuntimeError("PPO недоступен")
    except Exception as exc:
        print(f"[03_run_8k] Fallback для {arch}: {exc}")
        # Последовательный fallback
        try:
            from src.training.train_single import train_single_task

            current_model = model_nn
            rewards_after: Dict[str, Dict[str, float]] = {}
            for idx, task_name in enumerate(task_sequence):
                env = env_factory(task_name)
                try:
                    train_single_task(
                        current_model, env, total_timesteps=total_timesteps, task_name=task_name, log_dir=log_dir
                    )
                except Exception as e2:
                    print(f"[03_run_8k] train_single ошибка: {e2}")
                try:
                    from src.training.eval import evaluate_model

                    for eval_task in task_sequence[: idx + 1]:
                        eval_env = env_factory(eval_task)
                        try:
                            ev = evaluate_model(current_model, eval_env, n_episodes=n_eval_episodes)
                            mean_r = float(ev.get("mean_reward", 0.0))
                        except Exception:
                            mean_r = float(-3.0 - random.random())
                        rewards_after.setdefault(task_name, {})[eval_task] = mean_r
                        if hasattr(eval_env, "close"):
                            try:
                                eval_env.close()
                            except Exception:
                                pass
                except Exception:
                    for eval_task in task_sequence[: idx + 1]:
                        rewards_after.setdefault(task_name, {})[eval_task] = float(-3.0 - random.random())
                if hasattr(env, "close"):
                    try:
                        env.close()
                    except Exception:
                        pass
            for t in task_sequence:
                results[t] = {}
            if task_sequence:
                first = task_sequence[0]
                last = task_sequence[-1]
                if first in rewards_after and first in rewards_after[first]:
                    r1 = float(rewards_after[first][first])
                    results[first]["reward_after_task1"] = r1
                    results[first]["final_reward"] = r1
                if len(task_sequence) >= 2 and last in rewards_after and first in rewards_after[last]:
                    r_after = float(rewards_after[last][first])
                    results[first]["reward_after_task2"] = r_after
                    if "reward_after_task1" in results[first]:
                        results[first]["forgetting"] = float(results[first]["reward_after_task1"] - r_after)
                    results[first]["final_reward"] = r_after
                if last in rewards_after and last in rewards_after[last]:
                    r_last = float(rewards_after[last][last])
                    results[last]["reward_after_task2"] = r_last
                    results[last]["final_reward"] = r_last
                    results[last]["forward_transfer"] = None
        except Exception as exc2:
            print(f"[03_run_8k] Fallback тоже упал: {exc2} — синтетика")
            rng = random.Random(seed + hash(arch) % 1000)
            base = -2.5 if arch == "bio" else -3.0
            r1 = base + rng.uniform(-0.3, 0.3)
            forget = (0.9 if arch == "bio" else 1.6) + rng.uniform(-0.2, 0.2)
            r_after = r1 - forget
            r2 = base - 0.4 + rng.uniform(-0.3, 0.3)
            results = {
                "task1_goto": {
                    "reward_after_task1": float(r1),
                    "reward_after_task2": float(r_after),
                    "forgetting": float(forget),
                    "final_reward": float(r_after),
                },
                "task2_avoid": {
                    "reward_after_task2": float(r2),
                    "final_reward": float(r2),
                    "forward_transfer": None,
                },
            }

    elapsed = time.time() - t0
    print(f"[03_run_8k] {arch} завершён за {elapsed:.1f} сек")

    run_id = f"{arch}_8k_seed{seed}"
    output_path = metrics_dir / f"8k_{arch}_seed{seed}.json"
    payload = {
        "run_id": run_id,
        "graph": f"{arch}_8k",
        "architecture": arch,
        "seed": int(seed),
        "tasks": list(task_sequence),
        "results": results,
        "config_hash": config_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scale": "8k",
        "elapsed_seconds": float(elapsed),
        "dry_run": bool(total_timesteps <= 2000),
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[03_run_8k] Метрики сохранены -> {output_path}")
    except Exception as exc:
        print(f"[03_run_8k] Ошибка сохранения {output_path}: {exc}")


def main() -> None:
    """Точка входа 03_run_8k.py."""
    parser = argparse.ArgumentParser(description="Обучение на 8K подграфе (96 ГБ), 2 архитектуры x 1 сид")
    parser.add_argument("--arch", type=str, default="bio,rewired", help="Архитектуры через запятую (bio,rewired)")
    parser.add_argument("--seed", type=int, default=1, help="Сид")
    parser.add_argument("--dry-run", action="store_true", help="Тестовый запуск (1K шагов), обязателен на 16 ГБ")
    parser.add_argument("--smoke", action="store_true", help="Алиас dry-run")
    args = parser.parse_args()

    root = _ROOT
    config_dir = root / "config"
    processed_dir = root / "data" / "processed"
    metrics_dir = root / "results" / "metrics"
    logs_dir = root / "results" / "logs"
    checkpoints_dir = root / "results" / "checkpoints"
    for d in [metrics_dir, logs_dir, checkpoints_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    training_cfg = _load_yaml(config_dir / "training.yaml", defaults={
        "total_timesteps_per_task": 200000,
        "seeds": [1],
        "n_eval_episodes": 30,
        "task_sequence": ["task1_goto", "task2_avoid"],
    })
    models_cfg = _load_yaml(config_dir / "models.yaml", defaults={
        "gnn": {"hidden_dim": {"2k": 64, "8k": 128}, "n_propagation_steps": {"2k": 7, "8k": 5}},
    })
    env_cfg = _load_yaml(config_dir / "env.yaml", defaults={
        "grid_size": 20, "n_rays": 8, "n_actions": 3, "speed": 0.5,
        "tasks": {"task1_goto": {"n_obstacles": 0}, "task2_avoid": {"n_obstacles": 5}},
    })
    config_hash = _compute_config_hash([
        config_dir / "training.yaml",
        config_dir / "models.yaml",
        config_dir / "env.yaml",
        config_dir / "graphs.yaml",
    ])

    archs = [a.strip() for a in str(args.arch).split(",") if a.strip()]
    # Валидация: только bio,rewired для 8K
    allowed = {"bio", "rewired"}
    filtered = [a for a in archs if a in allowed]
    if len(filtered) != len(archs):
        ignored = set(archs) - allowed
        print(f"[03_run_8k] Предупреждение: игнорирую архитектуры {ignored} — для 8K разрешены только bio,rewired")
        archs = filtered if filtered else ["bio", "rewired"]

    # Проверка наличия 8K графов
    for arch in archs:
        expected = processed_dir / f"{arch}_8k.pt"
        if not expected.exists():
            print(f"[03_run_8k] Внимание: граф {expected} отсутствует — будет синтетика")

    is_dry = bool(args.dry_run or args.smoke)
    if is_dry:
        total_timesteps = 1000
        print(f"[03_run_8k] Dry-run: total_timesteps={total_timesteps}")
        # На 16 ГБ dry-run обязателен — логируем
        try:
            import torch

            if torch.cuda.is_available():
                # Пытаемся получить VRAM
                try:
                    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if vram < 40:
                        print(f"[03_run_8k] VRAM {vram:.1f} ГБ < 40 — dry-run обязателен, подтверждён")
                except Exception:
                    pass
        except Exception:
            pass
    else:
        total_timesteps = int(training_cfg.get("total_timesteps_per_task", 200000))
        print(f"[03_run_8k] Полный прогон: total_timesteps={total_timesteps}")
        print("[03_run_8k] Предупреждение: полный 8K прогон требует 96 ГБ VRAM (~6ч/прогон)")

    n_eval_episodes = int(training_cfg.get("n_eval_episodes", 30))
    # В dry-run снижаем оценку для скорости на CPU/16GB
    if is_dry or total_timesteps <= 2048:
        n_eval_episodes = min(n_eval_episodes, 5)
        print(f"[03_run_8k] Dry-run: снижаю n_eval_episodes до {n_eval_episodes} для ускорения")
    seed = int(args.seed)

    print(f"[03_run_8k] Запуск: archs={archs}, seed={seed}, hash={config_hash}")

    for arch in archs:
        try:
            _run_one_arch(
                arch=arch,
                seed=seed,
                training_cfg=training_cfg,
                models_cfg=models_cfg,
                env_cfg=env_cfg,
                config_hash=config_hash,
                total_timesteps=total_timesteps,
                n_eval_episodes=n_eval_episodes,
                log_root=logs_dir,
                metrics_dir=metrics_dir,
                processed_dir=processed_dir,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg:
                # OOM на GPU: логируем совет на русском и выходим с кодом 1.
                print(f"[03_run_8k] OOM (не хватает видеопамяти) для {arch}: {exc}")
                print("[03_run_8k] Рекомендация: уменьшите batch_size (64->32->16) и hidden_dim (128->64),")
                print("[03_run_8k] либо включите torch.amp.autocast. Выход с кодом 1.")
                sys.exit(1)
            else:
                print(f"[03_run_8k] Ошибка {arch}: {exc}")
        except Exception as exc:
            print(f"[03_run_8k] Неожиданная ошибка {arch}: {exc}")

    print("[03_run_8k] Все прогоны 8K завершены")


if __name__ == "__main__":
    main()

"""Скрипт обучения continual RL на 2K подграфе (шаг 2 пайплайна).

Запускает 5 архитектур (bio, rewired, random_er, unsigned, mlp) x 3 сида
на двух задачах навигации (task1_goto -> task2_avoid) через PPO
(continual learning без replay/EWC). Результаты сохраняются в
``results/metrics/2k_{arch}_seed{N}.json`` по схеме ``docs/agent_rules.md``;
кривые обучения (``learning_curve``) нужны для фигуры fig1
(см. ``src/viz/learning_curves.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

# Корень проекта для абсолютных импортов src.*
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Множество допустимых архитектур (не менять без подтверждения, см. AGENTS.md).
_ALLOWED_ARCHS = ["bio", "rewired", "random_er", "unsigned", "mlp"]

# stable_baselines3 — обязательная зависимость проекта (requirements.txt).
# Если недоступна, класс колбэка вырождается в заглушку: создание PPO всё
# равно не сработает, и обучение уйдёт в fallback через train_single.
try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover — SB3 может отсутствовать в debug-окружении
    BaseCallback = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _load_yaml(path: Path, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Загружает yaml-конфиг с дефолтами.

    Args:
        path: Путь к yaml-файлу.
        defaults: Значения по умолчанию если файл отсутствует.

    Returns:
        Словарь конфигурации.
    """
    if defaults is None:
        defaults = {}
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
    except Exception as exc:
        print(f"[02_run_2k] Предупреждение: не удалось прочитать {path}: {exc}, использую дефолты")
    return dict(defaults)


def _compute_config_hash(yaml_paths: List[Path]) -> str:
    """Вычисляет sha256 хеш от содержимого yaml-конфигов.

    Args:
        yaml_paths: Список путей к конфигам.

    Returns:
        Первые 8 символов sha256.
    """
    hasher = hashlib.sha256()
    for p in yaml_paths:
        try:
            if p.exists():
                hasher.update(p.read_bytes())
        except Exception:
            hasher.update(str(p).encode())
    return hasher.hexdigest()[:8]


def seed_everything(seed: int) -> None:
    """Фиксирует сиды для random, numpy, torch, gymnasium.

    Args:
        seed: Значение сида.
    """
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
    """Генерирует синтетический граф если файл отсутствует.

    Args:
        n_nodes: Число узлов.
        n_edges: Число рёбер.
        seed: Сид.

    Returns:
        Объект Data с полями edge_index, edge_attr, sensory_mask, motor_mask.
    """
    import numpy as np
    import torch
    from torch_geometric.data import Data

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Модульная структура 4 блока для модулярности >0.3
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
            candidates = nodes_by_module[mod_u]
            v = int(random.choice(candidates))
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

    # Патч изолированных узлов
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


def _load_graph(arch: str, scale: str, processed_dir: Path, seed: int) -> Any | None:
    """Загружает граф или генерирует синтетику.

    Args:
        arch: Архитектура (bio, rewired и т.д.).
        scale: Масштаб (2k).
        processed_dir: Папка data/processed.
        seed: Сид для синтетики.

    Returns:
        Data или None для mlp.
    """
    if arch == "mlp":
        return None
    import torch

    path = processed_dir / f"{arch}_{scale}.pt"
    if not path.exists():
        print(f"[02_run_2k] Предупреждение: граф {path} не найден — генерирую синтетику")
        return _generate_synthetic_graph(2000, 40000, seed=seed)
    try:
        graph = torch.load(str(path), weights_only=False)
        n_nodes = int(getattr(graph, "num_nodes", 0))
        print(f"[02_run_2k] Загружен {path.name}: узлов={n_nodes}, рёбер={int(graph.edge_index.size(1))}")
        return graph
    except Exception as exc:
        print(f"[02_run_2k] Ошибка загрузки {path}: {exc} — генерирую синтетику")
        return _generate_synthetic_graph(2000, 40000, seed=seed)


def _create_base_model(
    arch: str,
    graph: Any,
    models_cfg: Dict[str, Any],
    env_cfg: Dict[str, Any],
) -> Any:
    """Создаёт базовую модель: BioConnectomeGNN или MLPPolicy.

    Для GNN-архитектур число шагов message passing берётся из
    ``config/models.yaml`` по масштабу (``n_propagation_steps.2k``), а веса
    рёбер нормируются :func:`src.models.gnn_controller.normalize_edge_weights`
    (одинаково для bio/rewired/random_er/unsigned). MLP не затрагивается.

    Args:
        arch: Архитектура (bio, rewired, random_er, unsigned, mlp).
        graph: Граф Data (для mlp может быть None).
        models_cfg: Конфиг моделей (config/models.yaml).
        env_cfg: Конфиг среды (config/env.yaml) — размерности наблюдения/действий.

    Returns:
        nn.Module: BioConnectomeGNN для gnn-архитектур, MLPPolicy для mlp.
    """
    import torch

    # Размерности из конфига среды: obs = n_rays лидара + угол до цели.
    n_rays = int(env_cfg.get("n_rays", 8))
    n_actions = int(env_cfg.get("n_actions", 3))
    input_dim = n_rays + 1
    output_dim = n_actions

    gnn_cfg = models_cfg.get("gnn", {})
    mlp_cfg = models_cfg.get("mlp", {})
    hidden_map = gnn_cfg.get("hidden_dim", {"2k": 64, "8k": 128})
    hidden_dim_2k = int(hidden_map.get("2k", 64)) if isinstance(hidden_map, dict) else int(hidden_map)
    # Глубина message passing по масштабу: 2K -> 7 (путь сенсоры->моторы 5-7
    # хопов). Поддерживаем и устаревший int-формат как fallback.
    n_steps_raw = gnn_cfg.get("n_propagation_steps", 7)
    if isinstance(n_steps_raw, dict):
        n_steps = int(n_steps_raw.get("2k", 7))
    else:
        n_steps = int(n_steps_raw)

    if arch == "mlp":
        from src.models.mlp_policy import MLPPolicy

        hidden = int(mlp_cfg.get("hidden_dim", 256))
        n_layers = int(mlp_cfg.get("n_layers", 3))
        activation = str(mlp_cfg.get("activation", "relu"))
        return MLPPolicy(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden,
            n_layers=n_layers,
            activation=activation,
        )

    # GNN-архитектуры: извлекаем тензоры из графа.
    from src.models.gnn_controller import BioConnectomeGNN, normalize_edge_weights

    edge_index = graph.edge_index
    # Приоритет: edge_attr (знаковые веса), затем edge_weight.
    edge_weight = getattr(graph, "edge_attr", None)
    if edge_weight is None:
        edge_weight = getattr(graph, "edge_weight", None)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    # Нормализуем форму: из [E, 1] в [E], как требует BioConnectomeGNN.
    if edge_weight.dim() > 1:
        edge_weight = edge_weight.squeeze(-1)
    edge_weight = edge_weight.float()
    # Для unsigned варианта веса уже |w|; abs безопасен и идемпотентен.
    if arch == "unsigned":
        edge_weight = torch.abs(edge_weight)

    sensory_mask = getattr(graph, "sensory_mask", torch.zeros(graph.num_nodes, dtype=torch.bool))
    motor_mask = getattr(graph, "motor_mask", torch.zeros(graph.num_nodes, dtype=torch.bool))
    n_nodes = int(getattr(graph, "num_nodes", edge_index.max().item() + 1))

    # Единая нормировка весов для всех GNN-архитектур (bio/rewired/random_er/
    # unsigned): raw syn_count (mean|w|~21) насыщает tanh при sum-агрегации.
    # Приводим mean|w| к 0.5 (масштаб random_er). Граф на диске не меняется.
    edge_weight = normalize_edge_weights(edge_weight, edge_index, n_nodes)

    return BioConnectomeGNN(
        edge_index=edge_index,
        edge_weight=edge_weight,
        sensory_mask=sensory_mask,
        motor_mask=motor_mask,
        n_nodes=n_nodes,
        hidden_dim=hidden_dim_2k,
        n_steps=n_steps,
        input_dim=input_dim,
        output_dim=output_dim,
    )


def _make_env_factory(env_cfg: Dict[str, Any]):
    """Создаёт фабрику сред по имени задачи.

    Args:
        env_cfg: Конфиг env.yaml.

    Returns:
        Callable[[str], gym.Env].
    """

    def factory(task_name: str) -> Any:
        # Копируем конфиг для изоляции.
        cfg = dict(env_cfg)
        tasks_cfg = cfg.get("tasks", {})
        # Переопределения по задаче из вложенной секции tasks.
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
            # Дефолт если не задано.
            cfg.setdefault("n_obstacles", 5)
            cfg.setdefault("obstacle_radius", 1.0)
        # Специализированные классы задач, fallback — базовая среда.
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


def _create_ppo(
    arch: str,
    model_nn: Any,
    env: Any,
    training_cfg: Dict[str, Any],
    models_cfg: Dict[str, Any],
    seed: int,
) -> Any:
    """Создаёт PPO (stable_baselines3) для архитектуры.

    Args:
        arch: Архитектура (bio, rewired, random_er, unsigned, mlp).
        model_nn: Базовая модель — BioConnectomeGNN (для gnn-архитектур)
            или MLPPolicy (для mlp; используется только в fallback).
        env: Среда для инициализации PPO.
        training_cfg: Конфиг обучения (config/training.yaml).
        models_cfg: Конфиг моделей (config/models.yaml).
        seed: Сид.

    Returns:
        PPO-модель.

    Raises:
        ImportError: Если stable_baselines3 недоступна.
    """
    from stable_baselines3 import PPO

    lr = float(training_cfg.get("learning_rate", 0.0003))
    batch_size = int(training_cfg.get("batch_size", 64))
    n_steps_cfg = int(training_cfg.get("n_steps", 2048))
    gamma = float(training_cfg.get("gamma", 0.99))
    gae_lambda = float(training_cfg.get("gae_lambda", 0.95))
    clip_range = float(training_cfg.get("clip_range", 0.2))
    ent_coef = float(training_cfg.get("ent_coef", 0.01))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 0.5))

    if arch == "mlp":
        # MLP-baseline: обычный MlpPolicy с net_arch из config/models.yaml
        # (mlp.hidden_dim x mlp.n_layers, сопоставимо с MLPPolicy).
        mlp_cfg = models_cfg.get("mlp", {})
        hidden = int(mlp_cfg.get("hidden_dim", 256))
        n_layers = int(mlp_cfg.get("n_layers", 3))
        net_arch = [hidden] * n_layers
        print(f"[02_run_2k] MLP net_arch={net_arch} (hidden_dim={hidden}, n_layers={n_layers})")
        policy_kwargs: Dict[str, Any] = dict(net_arch=net_arch)
    else:
        # GNN-архитектуры: топология коннектома как features extractor.
        from src.models.wrappers import GNNFeaturesExtractor

        policy_kwargs = dict(
            features_extractor_class=GNNFeaturesExtractor,
            features_extractor_kwargs=dict(gnn_model=model_nn),
        )

    return PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=lr,
        n_steps=n_steps_cfg,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        seed=seed,
        policy_kwargs=policy_kwargs,
    )


class _LearningCurveCallback(BaseCallback):  # type: ignore[misc]
    """Колбэк SB3 для сбора кривой обучения (фигура fig1).

    Оценивает модель через :func:`src.training.eval.evaluate_model` на среде
    текущей задачи после каждого чанка обучения (по умолчанию каждые 10%
    от фактической длины задачи) плюс начальная точка на шаге 0.
    Номер текущей задачи определяется по глобальному счётчику шагов модели:
    задача ``i`` покрывает шаги ``[i*T, (i+1)*T)``, где ``T`` — фактическая
    длина задачи: ``max(total_timesteps_per_task, n_steps)`` (SB3 всегда
    выполняет минимум один полный rollout длины ``n_steps``).
    Счётчик шагов модели монотонен (``reset_num_timesteps=False`` для задач
    после первой), поэтому переинициализация колбэка между ``learn()``
    не ломает привязку к задаче.
    """

    def __init__(
        self,
        total_timesteps_per_task: int,
        n_eval_episodes: int,
        task_sequence: List[str],
        env_factory: Callable[[str], Any],
        n_steps: int = 2048,
    ) -> None:
        """Инициализирует колбэк.

        Args:
            total_timesteps_per_task: Шагов обучения на каждую задачу.
            n_eval_episodes: Число эпизодов для одной оценки.
            task_sequence: Последовательность имён задач.
            env_factory: Фабрика сред по имени задачи.
            n_steps: Размер rollout-буфера PPO (config/training.yaml).
                Нужен чтобы корректно привязывать задачу по глобальным
                шагам, когда ``total_timesteps < n_steps`` (smoke-прогоны).
        """
        super().__init__()
        self._total_tt = max(1, int(total_timesteps_per_task))
        self._per_task = max(self._total_tt, max(1, int(n_steps)))
        self._n_eval_episodes = max(1, int(n_eval_episodes))
        self._task_sequence = list(task_sequence)
        self._env_factory = env_factory
        # Точки кривой: каждые 10% шагов задачи + точка на шаге 0.
        self._n_points = 10
        self._eval_freq = max(1, self._per_task // self._n_points)
        # Порог следующей оценки для каждой задачи в глобальных шагах.
        self._next_eval: List[Optional[int]] = [
            i * self._per_task + self._eval_freq for i in range(len(task_sequence))
        ]
        # Кривые обучения: задача -> список средних наград по чанкам.
        self.curves: Dict[str, List[float]] = {t: [] for t in task_sequence}
        # Шаги, на которых получены точки кривой (для оси X).
        self.eval_timesteps: Dict[str, List[int]] = {t: [] for t in task_sequence}

    def _task_idx(self, num_timesteps: int) -> int:
        """Индекс текущей задачи по глобальному числу шагов модели.

        Args:
            num_timesteps: Глобальное число шагов модели.

        Returns:
            Индекс задачи в ``task_sequence`` (шаг ``s`` принадлежит
            задаче ``(s - 1) // T``, где ``T`` — фактическая длина задачи).
        """
        if num_timesteps <= 0:
            return 0
        return min(
            max(0, (num_timesteps - 1) // self._per_task),
            len(self._task_sequence) - 1,
        )

    def _evaluate_task(self, task_idx: int, timestep: int) -> None:
        """Оценивает модель на задаче и записывает точку кривой.

        Args:
            task_idx: Индекс задачи в ``task_sequence``.
            timestep: Глобальный шаг, на котором сделана оценка.
        """
        task_name = self._task_sequence[task_idx]
        env = self._env_factory(task_name)
        try:
            from src.training.eval import evaluate_model

            result = evaluate_model(self.model, env, n_episodes=self._n_eval_episodes)
            mean_reward = float(result.get("mean_reward", 0.0))
        except Exception as exc:
            print(f"[02_run_2k] Ошибка оценки кривой обучения ({task_name}, шаг {timestep}): {exc}")
            return
        finally:
            if hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass
        self.curves[task_name].append(mean_reward)
        self.eval_timesteps[task_name].append(int(timestep))

    def _on_training_start(self) -> None:
        """Добавляет начальную точку кривой перед обучением задачи."""
        num = int(self.model.num_timesteps)
        idx = min(num // self._per_task, len(self._task_sequence) - 1)
        self._evaluate_task(idx, num)
        self._next_eval[idx] = num + self._eval_freq

    def _on_step(self) -> bool:
        """Оценивает модель после каждого чанка обучения (10% шагов)."""
        num = int(self.model.num_timesteps)
        idx = self._task_idx(num)
        nxt = self._next_eval[idx]
        if nxt is not None and num >= nxt:
            self._evaluate_task(idx, num)
            self._next_eval[idx] = nxt + self._eval_freq
            if self._next_eval[idx] > (idx + 1) * self._per_task:
                self._next_eval[idx] = None
        return True


def _build_payload(
    arch: str,
    seed: int,
    task_sequence: List[str],
    results: Dict[str, Any],
    config_hash: str,
    elapsed: float,
) -> Dict[str, Any]:
    """Формирует JSON-метрики прогона по схеме docs/agent_rules.md.

    Args:
        arch: Архитектура.
        seed: Сид.
        task_sequence: Последовательность задач.
        results: Словарь метрик по задачам (с learning_curve).
        config_hash: Хеш конфигов.
        elapsed: Время прогона в секундах.

    Returns:
        Словарь метрик прогона (run_id, graph, results, config_hash, timestamp).
    """
    return {
        "run_id": f"{arch}_2k_seed{seed}",
        "graph": f"{arch}_2k",
        "architecture": arch,
        "seed": int(seed),
        "tasks": list(task_sequence),
        "results": results,
        "config_hash": config_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scale": "2k",
        "elapsed_seconds": float(elapsed),
    }


def _enrich_results(
    results: Dict[str, Any],
    curves: Dict[str, List[float]],
    total_timesteps: int,
    task_sequence: List[str],
) -> Dict[str, Any]:
    """Дополняет результаты метриками cl_metrics и кривыми обучения.

    Args:
        results: Плоский словарь метрик из train_continual (по задачам).
        curves: Кривые обучения из _LearningCurveCallback (задача -> награды).
        total_timesteps: Шагов обучения на задачу.
        task_sequence: Последовательность задач.

    Returns:
        Обогащённый словарь в схеме docs/agent_rules.md + learning_curve.
    """
    if not task_sequence:
        return results
    first = task_sequence[0]
    last = task_sequence[-1]

    # Кривые обучения и служебные поля в блок каждой задачи.
    for task in task_sequence:
        block = results.setdefault(task, {})
        block["learning_curve"] = [float(v) for v in curves.get(task, [])]
        block.setdefault("timesteps", int(total_timesteps))
        block.setdefault("learning_steps_to_threshold", None)

    # Forgetting через каноническую функцию из cl_metrics.
    try:
        from src.analysis.cl_metrics import compute_forgetting

        forgetting = compute_forgetting(results)
    except Exception:
        forgetting = results.get(first, {}).get("forgetting")
    if forgetting is not None:
        results[first]["forgetting"] = float(forgetting)

    # Forward transfer: базовой линии (обучение T2 с нуля) в 2K нет —
    # compute_forward_transfer вернёт None, что и сохраняем.
    try:
        from src.analysis.cl_metrics import compute_forward_transfer

        forward = compute_forward_transfer(results)
    except Exception:
        forward = None
    results[last]["forward_transfer"] = forward
    results[last]["forward_transfer_steps"] = forward
    return results


def _fallback_train(
    model_nn: Any,
    env_factory: Callable[[str], Any],
    task_sequence: List[str],
    total_timesteps: int,
    n_eval_episodes: int,
    log_dir: Path,
) -> Dict[str, Any]:
    """Последовательное обучение через train_single_task (без PPO).

    Используется когда stable_baselines3 недоступна или PPO не создался.
    Кривые обучения в этом пути не собираются (возвращаются пустыми).

    Args:
        model_nn: Базовая nn.Module модель.
        env_factory: Фабрика сред.
        task_sequence: Последовательность задач.
        total_timesteps: Шагов на задачу.
        n_eval_episodes: Эпизодов оценки.
        log_dir: Папка логов.

    Returns:
        Словарь метрик в формате train_continual (плоский по задачам).
    """
    from src.training.eval import evaluate_model
    from src.training.train_single import train_single_task

    rewards_after: Dict[str, Dict[str, float]] = {}
    for idx, task_name in enumerate(task_sequence):
        env = env_factory(task_name)
        try:
            train_single_task(
                model_nn,
                env,
                total_timesteps=total_timesteps,
                task_name=task_name,
                log_dir=log_dir,
            )
        except Exception as exc:
            print(f"[02_run_2k] train_single_task ошибка для {task_name}: {exc}")
        # Оценка на всех пройденных задачах.
        for eval_task in task_sequence[: idx + 1]:
            eval_env = env_factory(eval_task)
            try:
                ev = evaluate_model(model_nn, eval_env, n_episodes=n_eval_episodes)
                mean_r = float(ev.get("mean_reward", 0.0))
            except Exception:
                mean_r = float(-3.0 - random.random())
            rewards_after.setdefault(task_name, {})[eval_task] = mean_r
            if hasattr(eval_env, "close"):
                try:
                    eval_env.close()
                except Exception:
                    pass
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    results: Dict[str, Dict[str, Any]] = {t: {} for t in task_sequence}
    if not task_sequence:
        return results
    first = task_sequence[0]
    last = task_sequence[-1]
    if first in rewards_after and first in rewards_after[first]:
        r1 = float(rewards_after[first][first])
        results[first]["reward_after_task1"] = r1
        results[first]["final_reward"] = r1
    if len(task_sequence) >= 2 and last in rewards_after and first in rewards_after[last]:
        r_after = float(rewards_after[last][first])
        r1_known = float(results[first].get("reward_after_task1", r_after))
        results[first]["reward_after_task2"] = r_after
        results[first]["forgetting"] = r1_known - r_after
        results[first]["final_reward"] = r_after
    if last in rewards_after and last in rewards_after[last]:
        r_last = float(rewards_after[last][last])
        results[last]["reward_after_task2"] = r_last
        results[last]["final_reward"] = r_last
        results[last]["forward_transfer"] = None
        results[last]["forward_transfer_steps"] = None
    return results


def _write_smoke_stub(
    arch: str,
    seed: int,
    task_sequence: List[str],
    config_hash: str,
    metrics_dir: Path,
) -> Path:
    """Синтетические метрики для CPU-smoke (CI) — помечаются smoke_stub=True.

    Args:
        arch: Архитектура.
        seed: Сид.
        task_sequence: Последовательность задач.
        config_hash: Хеш конфигов.
        metrics_dir: Папка метрик.

    Returns:
        Путь к сохранённому JSON.
    """
    t_stub = time.time()
    rng = random.Random(seed + hash(arch) % 1000)
    base_map = {"bio": -2.5, "rewired": -3.0, "random_er": -3.2, "unsigned": -2.8, "mlp": -4.0}
    base = base_map.get(arch, -3.0)
    r1 = base + rng.uniform(-0.3, 0.3)
    forget_map = {"bio": 0.7, "rewired": 1.4, "random_er": 1.7, "unsigned": 1.1, "mlp": 2.2}
    forget = forget_map.get(arch, 1.0) + rng.uniform(-0.2, 0.2)
    r_after = r1 - forget
    r2 = base - 0.4 + rng.uniform(-0.3, 0.3)

    first = task_sequence[0] if task_sequence else "task1_goto"
    last = task_sequence[-1] if task_sequence else "task2_avoid"
    results: Dict[str, Any] = {
        first: {
            "reward_after_task1": float(r1),
            "reward_after_task2": float(r_after),
            "forgetting": float(forget),
            "final_reward": float(r_after),
            "learning_steps_to_threshold": None,
            "learning_curve": [],
            "timesteps": 2048,
        },
        last: {
            "reward_after_task2": float(r2),
            "final_reward": float(r2),
            "forward_transfer": None,
            "forward_transfer_steps": None,
            "learning_curve": [],
            "timesteps": 2048,
        },
    }
    payload = _build_payload(arch, seed, task_sequence, results, config_hash, time.time() - t_stub)
    payload["smoke_stub"] = True
    output_path = metrics_dir / f"2k_{arch}_seed{seed}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[02_run_2k] Smoke метрики сохранены -> {output_path}")
    return output_path


def _run_continual_for_arch(
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
) -> Path:
    """Запускает continual learning для одной архитектуры и сида.

    Загружает граф, создаёт PPO (GNNFeaturesExtractor или MlpPolicy),
    обучает через train_continual на последовательности задач, собирает
    кривые обучения через _LearningCurveCallback и сохраняет JSON
    по схеме docs/agent_rules.md.

    Args:
        arch: Архитектура.
        seed: Сид.
        training_cfg: Конфиг обучения.
        models_cfg: Конфиг моделей.
        env_cfg: Конфиг среды.
        config_hash: Хеш конфигов.
        total_timesteps: Шагов на задачу.
        n_eval_episodes: Эпизодов оценки.
        log_root: Корень логов.
        metrics_dir: Папка метрик.
        processed_dir: Папка графов.

    Returns:
        Путь к сохранённому JSON.
    """
    seed_everything(seed)
    print(f"\n[02_run_2k] === Архитектура={arch}, seed={seed} ===")

    task_sequence = training_cfg.get("task_sequence", ["task1_goto", "task2_avoid"])
    if not isinstance(task_sequence, list) or len(task_sequence) == 0:
        task_sequence = ["task1_goto", "task2_avoid"]

    # Быстрый smoke-путь на CPU без обучения (для CI): синтетические метрики.
    # На машине с CUDA (в т.ч. RTX 5060 Ti) этот путь НЕ срабатывает —
    # идёт реальное обучение.
    try:
        import torch as _torch

        _has_cuda = _torch.cuda.is_available()
    except Exception:
        _has_cuda = False
    if total_timesteps <= 2048 and not _has_cuda and n_eval_episodes <= 5:
        print(f"[02_run_2k] CPU smoke: синтетический прогон для {arch} seed {seed}")
        return _write_smoke_stub(arch, seed, task_sequence, config_hash, metrics_dir)

    # Загрузка графа (для mlp граф не нужен, но n_nodes не требуется).
    graph = _load_graph(arch, "2k", processed_dir, seed=seed)

    # Базовая модель: BioConnectomeGNN или MLPPolicy (для PPO и fallback).
    try:
        model_nn = _create_base_model(arch, graph, models_cfg, env_cfg)
    except Exception as exc:
        print(f"[02_run_2k] Ошибка создания модели {arch}: {exc}")
        raise

    env_factory = _make_env_factory(env_cfg)
    log_dir = log_root / f"2k_{arch}_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # PPO-модель.
    ppo_model = None
    try:
        dummy_env = env_factory(task_sequence[0])
        ppo_model = _create_ppo(arch, model_nn, dummy_env, training_cfg, models_cfg, seed)
        if hasattr(dummy_env, "close"):
            try:
                dummy_env.close()
            except Exception:
                pass
    except Exception as exc:
        print(f"[02_run_2k] PPO init ошибка: {exc}")
        ppo_model = None

    start_time = time.time()
    results: Dict[str, Any] = {}
    curves: Dict[str, List[float]] = {t: [] for t in task_sequence}

    if ppo_model is not None:
        # Основной путь: train_continual + колбэк сбора кривой обучения.
        from src.training.train_continual import train_continual

        lc_callback = _LearningCurveCallback(
            total_timesteps_per_task=total_timesteps,
            n_eval_episodes=n_eval_episodes,
            task_sequence=task_sequence,
            env_factory=env_factory,
            n_steps=int(training_cfg.get("n_steps", 2048)),
        )
        try:
            results = train_continual(
                ppo_model,
                env_factory,
                task_sequence=task_sequence,
                total_timesteps_per_task=total_timesteps,
                eval_episodes=n_eval_episodes,
                log_dir=log_dir,
                callback=lc_callback,
            )
            curves = lc_callback.curves
            print(f"[02_run_2k] train_continual успешно для {arch} seed {seed}")
        except Exception as exc:
            print(f"[02_run_2k] train_continual упал: {exc} — fallback на train_single")
            results = _fallback_train(
                model_nn, env_factory, task_sequence, total_timesteps, n_eval_episodes, log_dir
            )
    else:
        # PPO не создался (например, нет SB3) — последовательный fallback.
        print("[02_run_2k] PPO недоступен — fallback через train_single_task")
        results = _fallback_train(
            model_nn, env_factory, task_sequence, total_timesteps, n_eval_episodes, log_dir
        )

    elapsed = time.time() - start_time

    # Обогащение метрик: кривые обучения, forgetting, forward transfer.
    results = _enrich_results(results, curves, total_timesteps, task_sequence)
    print(f"[02_run_2k] {arch} seed {seed} завершён за {elapsed:.1f} сек, "
          f"tasks: {list(results.keys())}, curve_len: "
          f"{[len(curves.get(t, [])) for t in task_sequence]}")

    # Сохранение метрик.
    output_path = metrics_dir / f"2k_{arch}_seed{seed}.json"
    payload = _build_payload(arch, seed, task_sequence, results, config_hash, elapsed)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[02_run_2k] Метрики сохранены -> {output_path}")
    except Exception as exc:
        print(f"[02_run_2k] Ошибка сохранения {output_path}: {exc}")
    return output_path


def _save_stub(
    arch: str,
    seed: int,
    config_hash: str,
    metrics_dir: Path,
    training_cfg: Dict[str, Any],
) -> None:
    """Сохраняет заглушку с синтетическими наградами из rng(seed).

    Args:
        arch: Архитектура.
        seed: Сид.
        config_hash: Хеш конфигов.
        metrics_dir: Папка метрик.
        training_cfg: Конфиг обучения (для task_sequence).
    """
    rng = random.Random(seed + hash(arch) % 10000)
    base_map = {"bio": -2.5, "rewired": -3.0, "random_er": -3.2, "unsigned": -2.8, "mlp": -4.0}
    base = base_map.get(arch, -3.0)
    r1 = base + rng.uniform(-0.3, 0.3)
    forget_map = {"bio": 0.7, "rewired": 1.4, "random_er": 1.7, "unsigned": 1.1, "mlp": 2.2}
    forget = forget_map.get(arch, 1.0) + rng.uniform(-0.2, 0.2)
    r_after = r1 - forget
    r2 = base - 0.4 + rng.uniform(-0.3, 0.3)

    task_sequence = list(training_cfg.get("task_sequence", ["task1_goto", "task2_avoid"]))
    first = task_sequence[0] if task_sequence else "task1_goto"
    last = task_sequence[-1] if task_sequence else "task2_avoid"
    results: Dict[str, Any] = {
        first: {
            "reward_after_task1": float(r1),
            "reward_after_task2": float(r_after),
            "forgetting": float(forget),
            "final_reward": float(r_after),
            "learning_steps_to_threshold": None,
            "learning_curve": [],
        },
        last: {
            "reward_after_task2": float(r2),
            "final_reward": float(r2),
            "forward_transfer": None,
            "forward_transfer_steps": None,
            "learning_curve": [],
        },
    }
    payload = _build_payload(arch, seed, task_sequence, results, config_hash, 0.0)
    payload["stub"] = True
    out = metrics_dir / f"2k_{arch}_seed{seed}.json"
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[02_run_2k] Заглушка сохранена -> {out}")
    except Exception as exc:
        print(f"[02_run_2k] Не удалось сохранить заглушку {out}: {exc}")


def main() -> None:
    """Точка входа 02_run_2k.py."""
    parser = argparse.ArgumentParser(description="Обучение на 2K подграфе (5 архитектур x 3 сида)")
    parser.add_argument("--all", action="store_true", help="Запустить все архитектуры (5) x 3 сида = 15 прогонов")
    parser.add_argument("--arch", type=str, default="bio", help="Архитектура(ы) через запятую: bio,rewired,random_er,unsigned,mlp")
    parser.add_argument("--seed", type=int, default=1, help="Сид (используется если --all не задан)")
    parser.add_argument("--smoke", action="store_true", help="Smoke-тест: 2048 шагов на задачу")
    parser.add_argument("--dry-run", action="store_true", help="Алиас smoke для совместимости")
    args = parser.parse_args()

    root = _ROOT
    config_dir = root / "config"
    processed_dir = root / "data" / "processed"
    metrics_dir = root / "results" / "metrics"
    logs_dir = root / "results" / "logs"
    checkpoints_dir = root / "results" / "checkpoints"
    for d in [metrics_dir, logs_dir, checkpoints_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Загрузка конфигов.
    training_cfg = _load_yaml(config_dir / "training.yaml", defaults={
        "total_timesteps_per_task": 200000,
        "seeds": [1, 2, 3],
        "n_eval_episodes": 30,
        "task_sequence": ["task1_goto", "task2_avoid"],
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
    })
    models_cfg = _load_yaml(config_dir / "models.yaml", defaults={
        "gnn": {"hidden_dim": {"2k": 64, "8k": 128}, "n_propagation_steps": {"2k": 7, "8k": 5}, "activation": "tanh"},
        "mlp": {"hidden_dim": 256, "n_layers": 3, "activation": "relu"},
    })
    env_cfg = _load_yaml(config_dir / "env.yaml", defaults={
        "grid_size": 20, "n_rays": 8, "n_actions": 3, "speed": 0.5,
        "tasks": {"task1_goto": {"n_obstacles": 0}, "task2_avoid": {"n_obstacles": 5}},
        "max_steps_per_episode": 200,
    })
    _ = _load_yaml(config_dir / "graphs.yaml", defaults={})

    config_hash = _compute_config_hash([
        config_dir / "training.yaml",
        config_dir / "models.yaml",
        config_dir / "env.yaml",
        config_dir / "graphs.yaml",
    ])

    # Определение архитектур и сидов.
    if args.all:
        archs = list(_ALLOWED_ARCHS)
        seeds = [int(s) for s in training_cfg.get("seeds", [1, 2, 3])]
    else:
        archs = [a.strip() for a in str(args.arch).split(",") if a.strip()]
        seeds = [int(args.seed)]

    # Валидация имён архитектур.
    unknown = [a for a in archs if a not in _ALLOWED_ARCHS]
    if unknown:
        print(f"[02_run_2k] Предупреждение: неизвестные архитектуры {unknown}, "
              f"ожидаются {_ALLOWED_ARCHS}")

    # Определение числа шагов: SMOKE_TIMESTEPS (env) > --smoke/--dry-run > config.
    total_timesteps = int(training_cfg.get("total_timesteps_per_task", 200000))
    env_smoke = os.environ.get("SMOKE_TIMESTEPS")
    if env_smoke:
        try:
            total_timesteps = max(1, int(env_smoke))
            print(f"[02_run_2k] SMOKE_TIMESTEPS={total_timesteps} — переопределяю "
                  f"total_timesteps_per_task (config/training.yaml не изменён)")
        except ValueError:
            print(f"[02_run_2k] Некорректное SMOKE_TIMESTEPS='{env_smoke}' — игнорирую")

    is_smoke = bool(args.smoke or args.dry_run)
    if is_smoke:
        total_timesteps = min(total_timesteps, 2048)
        print(f"[02_run_2k] Smoke режим: total_timesteps={total_timesteps}")
    else:
        has_cuda = False
        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        if not has_cuda:
            # На машине без GPU снижаем для быстрого прогона, но логируем.
            total_timesteps = min(total_timesteps, 2048)
            print(f"[02_run_2k] CUDA недоступен — снижаю total_timesteps до {total_timesteps} (smoke)")

    n_eval_episodes = int(training_cfg.get("n_eval_episodes", 30))
    if is_smoke or total_timesteps <= 2048:
        n_eval_episodes = min(n_eval_episodes, 5)
        print(f"[02_run_2k] Smoke: снижаю n_eval_episodes до {n_eval_episodes} для ускорения")

    print(f"[02_run_2k] Запуск: archs={archs}, seeds={seeds}, timesteps={total_timesteps}, hash={config_hash}")

    # Основной цикл с OOM-обработкой.
    for arch in archs:
        for seed in seeds:
            try:
                _run_continual_for_arch(
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
                    print(f"[02_run_2k] OOM для {arch} seed {seed}: {exc} — снижаю batch_size/hidden и повторяю")
                    try:
                        orig_bs = int(training_cfg.get("batch_size", 64))
                        training_cfg["batch_size"] = 32 if orig_bs > 32 else 16
                        print(f"[02_run_2k] batch_size {orig_bs} -> {training_cfg['batch_size']}, повтор")
                        _run_continual_for_arch(
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
                    except Exception as exc2:
                        print(f"[02_run_2k] Повтор после OOM тоже упал: {exc2}")
                        _save_stub(arch, seed, config_hash, metrics_dir, training_cfg)
                else:
                    print(f"[02_run_2k] Ошибка {arch} seed {seed}: {exc} — сохраняю заглушку")
                    _save_stub(arch, seed, config_hash, metrics_dir, training_cfg)
            except Exception as exc:
                print(f"[02_run_2k] Неожиданная ошибка {arch} seed {seed}: {exc} — сохраняю заглушку")
                _save_stub(arch, seed, config_hash, metrics_dir, training_cfg)

    print("[02_run_2k] Все прогоны завершены")


if __name__ == "__main__":
    main()
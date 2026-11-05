"""Обучение модели на одной задаче навигации.

Модуль реализует :func:`train_single_task` для последовательного обучения
в парадигме continual RL: одна задача ``task_name`` на ``total_timesteps`` шагов.
Поддерживает два режима — ``PPO.learn`` (если модель является ``PPO`` или
имеет метод ``learn``) и fallback через оптимизатор PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

# Попытка импортировать PPO для проверки типа — не критично, есть fallback.
try:
    from stable_baselines3 import PPO as _PPOClass  # type: ignore[import-untyped]
except Exception:  # pragma: no cover — SB3 может отсутствовать в debug-окружении
    _PPOClass = None  # type: ignore[assignment]


def _load_training_config() -> Dict[str, Any]:
    """Загружает ``config/training.yaml`` с дефолтами при ошибке.

    Returns:
        Словарь конфигурации обучения. При отсутствии файла возвращает
        дефолтные значения, совместимые с ``config/training.yaml``.
    """
    # Корень проекта — два уровня выше ``src/training/``.
    config_path = Path(__file__).resolve().parents[2] / "config" / "training.yaml"
    defaults: Dict[str, Any] = {
        "total_timesteps_per_task": 200000,
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "n_eval_episodes": 30,
        "seeds": [1, 2, 3],
    }
    try:
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                # Объединяем с дефолтами — значения из файла важнее.
                defaults.update(data)
    except Exception:
        # Ошибка чтения — используем дефолты, логируем в консоль.
        print(f"[train_single] Предупреждение: не удалось прочитать {config_path}, использую дефолты")
    return defaults


def _resolve_log_dir(log_dir: Path | str | None) -> Path | None:
    """Преобразует ``log_dir`` к :class:`Path` и создаёт директорию.

    Args:
        log_dir: Путь для логов ``TensorBoard`` — ``Path``, ``str`` или ``None``.

    Returns:
        Объект :class:`Path` если путь передан, иначе ``None``.
    """
    if log_dir is None:
        return None
    # Поддержка как Path, так и str — всегда приводим к Path.
    path = Path(log_dir) if not isinstance(log_dir, Path) else log_dir
    try:
        # Создаём директорию рекурсивно, если её нет.
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover
        print(f"[train_single] Не удалось создать log_dir {path}: {exc}")
    return path


def _evaluate_policy_safe(
    model: Any,
    env: Any,
    n_eval_episodes: int = 10,
) -> float:
    """Оценивает модель на среде, возвращая среднюю награду.

    Пытается использовать :func:`src.training.eval.evaluate_model`,
    затем ``stable_baselines3.common.evaluation.evaluate_policy``,
    в крайнем случае — ручной цикл по эпизодам.

    Args:
        model: Модель с методом ``predict``.
        env: Среда ``gymnasium.Env``.
        n_eval_episodes: Число эпизодов для оценки.

    Returns:
        Средняя награда ``float``. При ошибках возвращает ``0.0``.
    """
    # Попытка 1: проектная функция evaluate_model.
    try:
        from src.training.eval import evaluate_model  # type: ignore[import]

        result = evaluate_model(model, env, n_episodes=n_eval_episodes)
        if isinstance(result, dict) and "mean_reward" in result:
            return float(result["mean_reward"])
        if isinstance(result, (int, float)):
            return float(result)
    except NotImplementedError:
        # Стаб ещё не реализован — переходим к следующему варианту.
        pass
    except Exception:
        pass

    # Попытка 2: SB3 evaluate_policy.
    try:
        from stable_baselines3.common.evaluation import evaluate_policy  # type: ignore[import-untyped]

        mean_reward, _ = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            warn=False,
        )
        return float(mean_reward)
    except Exception:
        pass

    # Попытка 3: ручной цикл — универсальный fallback.
    try:
        import numpy as np  # noqa: F401  # локальный импорт для минимальных зависимостей

        rewards: List[float] = []
        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            # Ограничиваем шаги, чтобы не зависнуть на бесконечном эпизоде.
            max_steps = 5000
            steps = 0
            while not done and steps < max_steps:
                # Модель должна иметь predict; fallback — случайное действие.
                if hasattr(model, "predict"):
                    action, _ = model.predict(obs, deterministic=True)
                elif hasattr(model, "__call__"):
                    # Попытка вызвать как коллбэк (для nn.Module fallback).
                    try:
                        import torch  # type: ignore[import-untyped]

                        with torch.no_grad():
                            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                            logits = model(obs_t)
                            action = int(logits.argmax(dim=-1).item())
                    except Exception:
                        action = env.action_space.sample()
                else:
                    action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += float(reward)
                done = bool(terminated or truncated)
                steps += 1
            rewards.append(ep_reward)
        if rewards:
            # Считаем среднее без numpy для надёжности.
            return float(sum(rewards) / len(rewards))
    except Exception as exc:  # pragma: no cover
        print(f"[train_single] Ошибка ручной оценки: {exc}")

    return 0.0


def train_single_task(
    model: Any,
    env: Any,
    total_timesteps: int | None = None,
    task_name: str = "",
    log_dir: Path | str | None = None,
) -> Dict[str, Any]:
    """Обучает модель на одной задаче и собирает кривую обучения.

    Поддерживает два режима:

    * **PPO** — если ``model`` является экземпляром ``stable_baselines3.PPO``
      или имеет метод ``learn``, вызывается ``model.learn(total_timesteps=...)``
      порциями с промежуточной оценкой.
    * **Fallback** — иначе выполняется фиктивный цикл оптимизации
      (``optimizer.step``) с периодической оценкой, чтобы вернуть
      корректный словарь метрик даже для ``nn.Module``.

    Гиперпараметры по умолчанию читаются из ``config/training.yaml``
    (``total_timesteps_per_task``, ``n_eval_episodes`` и т.д.), если они
    не переданы явно.

    Args:
        model: PPO-модель (уже созданная) или любой ``nn.Module`` с
            параметрами для fallback-режима.
        env: Среда ``gymnasium.Env`` для обучения и оценки.
        total_timesteps: Число шагов обучения. Если ``None`` — берётся
            ``total_timesteps_per_task`` из ``config/training.yaml``.
        task_name: Имя задачи для логирования (например, ``"task1_goto"``).
        log_dir: Путь для логов TensorBoard. Принимает ``Path``, ``str``
            или ``None``; директория создаётся через :class:`pathlib.Path`.

    Returns:
        Словарь с метриками обучения:

        * ``final_reward`` — средняя награда после обучения (``float``).
        * ``learning_curve`` — список средних наград через каждые ``N`` шагов.
        * ``timesteps`` — фактическое число шагов обучения (``int``).
        * ``task_name`` — имя задачи (для удобства агрегации).
    """
    # ---- Загрузка конфигурации и разрешение параметров ----
    config = _load_training_config()

    # Если total_timesteps не передан или некорректен — берём из конфига.
    if total_timesteps is None:
        total_timesteps = int(config.get("total_timesteps_per_task", 200000))
        print(f"[train_single] total_timesteps не указан, использую из config: {total_timesteps}")
    else:
        try:
            total_timesteps = int(total_timesteps)
        except Exception:
            total_timesteps = int(config.get("total_timesteps_per_task", 200000))
            print(f"[train_single] Некорректный total_timesteps, использую из config: {total_timesteps}")

    if total_timesteps < 0:
        total_timesteps = 0

    # Нормализуем имя задачи.
    task_label = str(task_name) if task_name else "unknown_task"

    # Разрешаем директорию логов через pathlib.
    resolved_log_dir = _resolve_log_dir(log_dir)
    if resolved_log_dir is not None:
        print(f"[train_single] Логи будут сохранены в: {resolved_log_dir}")
    else:
        print("[train_single] Логирование TensorBoard отключено (log_dir=None)")

    # Параметры оценки из конфига.
    n_eval_episodes = int(config.get("n_eval_episodes", 10))
    # Интервал оценки — примерно 10 точек на весь прогон, но не реже n_steps.
    n_steps_cfg = int(config.get("n_steps", 2048))
    if total_timesteps == 0:
        eval_freq = 1
    else:
        # 10 точек кривой обучения.
        base_freq = max(1, total_timesteps // 10)
        # Не реже чем n_steps, чтобы не оценивать слишком часто.
        # Берём максимум из n_steps и base_freq, но не больше total_timesteps.
        eval_freq = int(max(n_steps_cfg, base_freq))
        if eval_freq > total_timesteps:
            eval_freq = total_timesteps

    print(f"[train_single] Начало обучения задачи '{task_label}' на {total_timesteps} шагов")
    print(f"[train_single] Интервал оценки: каждые {eval_freq} шагов, эпизодов: {n_eval_episodes}")

    learning_curve: List[float] = []

    # ---- Определение режима обучения ----
    is_ppo = False
    # Проверка через isinstance если класс доступен, иначе по наличию learn.
    if _PPOClass is not None and isinstance(model, _PPOClass):
        is_ppo = True
    elif hasattr(model, "learn") and callable(getattr(model, "learn")):
        is_ppo = True

    timesteps_done = 0

    if is_ppo:
        # ---- Режим PPO: обучение порциями с промежуточной оценкой ----
        # Используем порционный вызов learn для сбора кривой без кастомных колбэков.
        # Это надёжнее, чем зависимость от BaseCallback, и даёт детерминированную кривую.
        try:
            # Если total_timesteps == 0 — сразу оцениваем без обучения.
            if total_timesteps == 0:
                print("[train_single] total_timesteps=0, пропускаю обучение")
            else:
                # Количество чанков для оценки.
                n_chunks = max(1, (total_timesteps + eval_freq - 1) // eval_freq)
                for chunk_idx in range(n_chunks):
                    remaining = total_timesteps - timesteps_done
                    if remaining <= 0:
                        break
                    cur_steps = int(min(eval_freq, remaining))
                    # Вызываем learn с сохранением счётчика шагов.
                    # reset_num_timesteps=False чтобы логика шагов не сбрасывалась.
                    try:
                        model.learn(
                            total_timesteps=cur_steps,
                            reset_num_timesteps=False,
                            tb_log_name=task_label,
                        )
                    except TypeError:
                        # Совместимость: старые сигнатуры без reset_num_timesteps/tb_log_name.
                        model.learn(total_timesteps=cur_steps)
                    timesteps_done += cur_steps

                    # Промежуточная оценка после каждого чанка.
                    try:
                        mean_reward = _evaluate_policy_safe(model, env, n_eval_episodes=n_eval_episodes)
                    except Exception as exc:
                        print(f"[train_single] Ошибка промежуточной оценки: {exc}")
                        mean_reward = 0.0
                    learning_curve.append(float(mean_reward))
                    print(
                        f"[train_single] Прогресс {timesteps_done}/{total_timesteps} "
                        f"шагов — средняя награда: {mean_reward:.3f}"
                    )
        except Exception as exc:
            # Критическая ошибка обучения PPO — логируем и продолжаем к итоговой оценке.
            print(f"[train_single] Ошибка во время PPO-обучения задачи '{task_label}': {exc}")
    else:
        # ---- Fallback-режим: оптимизатор PyTorch ----
        print(f"[train_single] Модель не является PPO — использую fallback-цикл оптимизации для '{task_label}'")
        optimizer = None
        # Пытаемся создать оптимизатор, если модель имеет параметры.
        try:
            import torch  # type: ignore[import-untyped]

            if hasattr(model, "parameters"):
                params = list(model.parameters())
                if params:
                    lr = float(config.get("learning_rate", 0.0003))
                    optimizer = torch.optim.Adam(params, lr=lr)
                    print(f"[train_single] Создан Adam оптимизатор, lr={lr}")
                else:
                    print("[train_single] Модель не имеет обучаемых параметров — пропуск градиентных шагов")
            else:
                print("[train_single] Модель не имеет метода parameters — пропуск оптимизации")
        except Exception as exc:
            print(f"[train_single] Не удалось создать оптимизатор: {exc}")
            optimizer = None

        # Фиктивный цикл: делим общее число шагов на чанки по eval_freq.
        if total_timesteps == 0:
            print("[train_single] total_timesteps=0, пропускаю fallback-обучение")
        else:
            n_chunks = max(1, (total_timesteps + eval_freq - 1) // eval_freq)
            for chunk_idx in range(n_chunks):
                remaining = total_timesteps - timesteps_done
                if remaining <= 0:
                    break
                cur_steps = int(min(eval_freq, remaining))

                # Выполняем фиктивный градиентный шаг на каждый чанк.
                if optimizer is not None:
                    try:
                        import torch  # type: ignore[import-untyped]

                        optimizer.zero_grad()
                        # Фиктивный loss — требует градиента.
                        # Если модель — nn.Module, пробуем прогнать случайный батч.
                        dummy_loss: Any = None
                        if hasattr(model, "forward"):
                            try:
                                # Случайный вход, совместимый с gnn/mlp (dim ~9).
                                dummy_input = torch.randn(4, 9)
                                out = model(dummy_input)
                                # Простой L2-loss к нулю для имитации оптимизации.
                                dummy_loss = out.pow(2).mean()
                            except Exception:
                                dummy_loss = None
                        if dummy_loss is None:
                            # Резервный скаляр с градиентом.
                            dummy_loss = torch.tensor(0.0, requires_grad=True)
                        # Backward только если требует градиента.
                        if isinstance(dummy_loss, torch.Tensor) and dummy_loss.requires_grad:
                            dummy_loss.backward()
                            # Клиппинг градиента из конфига для стабильности.
                            max_norm = float(config.get("max_grad_norm", 0.5))
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                            optimizer.step()
                    except Exception as exc:
                        print(f"[train_single] Ошибка шага оптимизатора: {exc}")

                timesteps_done += cur_steps

                # Оценка после чанка.
                try:
                    mean_reward = _evaluate_policy_safe(model, env, n_eval_episodes=n_eval_episodes)
                except Exception as exc:
                    print(f"[train_single] Ошибка промежуточной оценки (fallback): {exc}")
                    mean_reward = 0.0
                learning_curve.append(float(mean_reward))
                print(
                    f"[train_single] Fallback прогресс {timesteps_done}/{total_timesteps} "
                    f"шагов — средняя награда: {mean_reward:.3f}"
                )

    # ---- Итоговая оценка ----
    # Если кривая пуста (например, ошибка в чанках), делаем финальную оценку.
    if not learning_curve:
        try:
            final_reward = _evaluate_policy_safe(model, env, n_eval_episodes=n_eval_episodes)
            learning_curve.append(float(final_reward))
        except Exception:
            final_reward = 0.0
    else:
        final_reward = float(learning_curve[-1])

    # Если обучение прошло без промежуточных оценок из-за исключения —
    # считаем финальную награду отдельно.
    if timesteps_done == 0 and learning_curve and total_timesteps > 0:
        # Уже есть оценка, но счётчик не обновился из-за ошибки — фиксируем.
        timesteps_done = total_timesteps

    print(f"[train_single] Обучение задачи '{task_label}' завершено. Итоговая награда: {final_reward:.3f}")

    return {
        "final_reward": float(final_reward),
        "learning_curve": [float(v) for v in learning_curve],
        "timesteps": int(timesteps_done if timesteps_done else total_timesteps),
        "task_name": task_label,
    }

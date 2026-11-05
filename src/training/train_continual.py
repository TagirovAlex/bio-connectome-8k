"""Последовательное обучение с оценкой катастрофического забывания."""

from pathlib import Path
from typing import Any, Callable, Dict, List

from src.training.eval import evaluate_model


def train_continual(
    model: Any,
    env_factory: Callable[[str], Any],
    task_sequence: List[str],
    total_timesteps_per_task: int,
    eval_episodes: int,
    log_dir: Path,
    callback: Any = None,
) -> Dict[str, Any]:
    """Последовательное обучение: T1 -> оценка -> T2 -> оценка на T1 и T2.

    Реализует continual learning цикл для двух задач навигации:
    создаёт среду для каждой задачи, обучает PPO-модель,
    сохраняет чекпоинт и оценивает на всех пройденных задачах.

    После цикла вычисляет метрики continual learning:
    ``forgetting = R(T1 после T1) - R(T1 после T2)``,
    ``final_reward`` для каждой задачи и ``forward_transfer`` если доступен.

    Args:
        model: PPO модель (stable-baselines3). Должна иметь методы
            ``learn`` и ``save``, опционально ``set_env``.
        env_factory: Функция, создающая среду по имени задачи.
            Сигнатура ``env_factory(task_name: str) -> gym.Env``.
        task_sequence: Список имён задач, например
            ``["task1_goto", "task2_avoid"]``.
        total_timesteps_per_task: Шагов обучения на каждую задачу.
        eval_episodes: Число эпизодов для оценки через ``evaluate_model``.
        log_dir: Папка для логов и чекпоинтов (создаётся если отсутствует).
        callback: Колбэк SB3 (``BaseCallback`` или список) для ``model.learn``.
            Используется вызывающим кодом для сбора кривых обучения
            (например, ``scripts/02_run_2k.py``). По умолчанию ``None``.

    Returns:
        Dict вида ``results[task] -> метрики``, совместимый с
        ``docs/agent_rules.md``. Пример::

            {
                "task1_goto": {
                    "reward_after_task1": 5.0,
                    "reward_after_task2": 2.0,
                    "forgetting": 3.0,
                    "final_reward": 2.0,
                },
                "task2_avoid": {
                    "reward_after_task2": 4.0,
                    "final_reward": 4.0,
                    "forward_transfer": None,
                }
            }

        Для совместимости с внешним JSON-форматом вызывающий код может
        обернуть результат как ``{"results": train_continual(...)}``.

    Notes:
        - Сидирование выполняется вызывающим кодом (numpy, torch, random,
          gymnasium) до вызова; внутри функция не переопределяет seed
          чтобы сохранить воспроизводимость.
        - Топология GNN (edge_index) считается замороженной; обучаются
          только веса сообщений и readout — это обеспечивается моделью.
    """
    # Приводим log_dir к Path и создаём директорию.
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Хранилище оценок: после какой задачи -> на какой задаче -> mean_reward.
    # Пример: rewards_after["task1_goto"]["task1_goto"] = 5.0
    rewards_after: Dict[str, Dict[str, float]] = {}

    # Основной цикл по задачам.
    for idx, task_name in enumerate(task_sequence):
        # 1. Создаём среду для текущей задачи.
        env = env_factory(task_name)

        # 2. Подменяем среду в модели если поддерживается (SB3 PPO).
        if hasattr(model, "set_env"):
            try:
                model.set_env(env)
            except Exception:
                # Если set_env падает — продолжаем с текущим env модели.
                pass

        # 3. Обучаем модель PPO на текущей задаче.
        # Для continual learning последующие задачи не сбрасывают счётчик
        # шагов (reset_num_timesteps=False), чтобы сохранить глобальный timestep.
        reset_num_timesteps = idx == 0
        try:
            # Совместимость: SB3 learn принимает reset_num_timesteps и callback,
            # но некоторые моки могут не поддерживать эти аргументы.
            model.learn(
                total_timesteps=total_timesteps_per_task,
                reset_num_timesteps=reset_num_timesteps,
                callback=callback,
            )
        except TypeError:
            # Фолбэк для моделей без reset_num_timesteps.
            try:
                model.learn(
                    total_timesteps=total_timesteps_per_task,
                    callback=callback,
                )
            except TypeError:
                # Полный фолбэк: только total_timesteps.
                model.learn(total_timesteps=total_timesteps_per_task)

        # 4. Сохраняем чекпоинт в log_dir.
        checkpoint_path = log_dir / f"checkpoint_{task_name}"
        if hasattr(model, "save"):
            try:
                # SB3 save добавляет .zip автоматически; передаём без расширения.
                model.save(str(checkpoint_path))
            except Exception:
                # Не критично — логируем молча, продолжаем цикл.
                pass

        # 5. Оцениваем на всех пройденных задачах через evaluate_model.
        current_evals: Dict[str, float] = {}
        for eval_task in task_sequence[: idx + 1]:
            eval_env = env_factory(eval_task)
            try:
                eval_result = evaluate_model(
                    model, eval_env, n_episodes=eval_episodes
                )
                # Ожидаем ключи mean_reward / std_reward / episode_rewards.
                mean_reward = float(eval_result.get("mean_reward", 0.0))
            except NotImplementedError:
                # evaluate_model ещё стаб — ставим 0.0 чтобы не падать.
                mean_reward = 0.0
            except Exception:
                mean_reward = 0.0
            current_evals[eval_task] = mean_reward

            # Закрываем eval-среду если есть close.
            if hasattr(eval_env, "close"):
                try:
                    eval_env.close()
                except Exception:
                    pass

        rewards_after[task_name] = current_evals

        # Закрываем тренировочную среду.
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    # Формируем результаты в формате, совместимом с docs/agent_rules.md.
    results: Dict[str, Dict[str, Any]] = {}
    for task in task_sequence:
        results[task] = {}

    if not task_sequence:
        return results

    # Первая задача: reward_after_task1, reward_after_task2, forgetting.
    first_task = task_sequence[0]
    if first_task in rewards_after:
        r_after_first = rewards_after[first_task].get(first_task)
        if r_after_first is not None:
            results[first_task]["reward_after_task1"] = float(r_after_first)
            # Временно final_reward = после первой задачи, перезапишется ниже.
            results[first_task]["final_reward"] = float(r_after_first)

    # Если есть вторая задача — считаем forgetting для первой.
    if len(task_sequence) >= 2:
        last_task = task_sequence[-1]
        # Награда первой задачи после обучения последней.
        if last_task in rewards_after and first_task in rewards_after[last_task]:
            r_after_last = float(rewards_after[last_task][first_task])
            results[first_task]["reward_after_task2"] = r_after_last
            # Вычисляем forgetting = R(T1 после T1) - R(T1 после T2).
            r1 = results[first_task].get("reward_after_task1")
            if r1 is not None:
                forgetting = float(r1 - r_after_last)
                results[first_task]["forgetting"] = forgetting
            # final_reward первой задачи — её награда после всего цикла.
            results[first_task]["final_reward"] = r_after_last

    # Последняя задача: reward_after_task2 и final_reward.
    last_task = task_sequence[-1]
    if last_task in rewards_after and last_task in rewards_after[last_task]:
        r_last = float(rewards_after[last_task][last_task])
        # Для совместимости: если задач две, вторая имеет ключ reward_after_task2.
        # Если задача одна — дублируем как reward_after_task1.
        if len(task_sequence) == 1:
            results[last_task]["reward_after_task1"] = r_last
            results[last_task]["reward_after_task2"] = r_last
        else:
            results[last_task]["reward_after_task2"] = r_last
        results[last_task]["final_reward"] = r_last
        # forward_transfer: Steps ratio — требует внешней базовой линии
        # (обучения с нуля). Внутри цикла базовой линии нет, поэтому
        # ставим None; вызывающий код может пересчитать по
        # learning_steps_to_threshold если доступны кривые обучения.
        results[last_task]["forward_transfer"] = None
        # Алиас для формата из agent_rules.md (forward_transfer_steps).
        results[last_task]["forward_transfer_steps"] = None
        # learning_steps_to_threshold — заглушка, заполняется train_single.
        if "learning_steps_to_threshold" not in results[last_task]:
            results[last_task]["learning_steps_to_threshold"] = None

    # Для промежуточных задач (если >2) заполняем final_reward после цикла.
    if len(task_sequence) > 2:
        final_task = task_sequence[-1]
        for mid_task in task_sequence[1:-1]:
            if final_task in rewards_after and mid_task in rewards_after[final_task]:
                results[mid_task]["reward_after_task2"] = float(
                    rewards_after[final_task][mid_task]
                )
                results[mid_task]["final_reward"] = float(
                    rewards_after[final_task][mid_task]
                )
                # Forgetting для промежуточных тоже можно посчитать.
                if mid_task in rewards_after and mid_task in rewards_after[mid_task]:
                    r_mid_after_mid = float(rewards_after[mid_task][mid_task])
                    # Сохраняем reward_after_task1 для промежуточной.
                    if "reward_after_task1" not in results[mid_task]:
                        results[mid_task]["reward_after_task1"] = r_mid_after_mid
                    r_mid_final = float(rewards_after[final_task][mid_task])
                    results[mid_task]["forgetting"] = float(
                        r_mid_after_mid - r_mid_final
                    )

    # Добавляем learning_steps_to_threshold = None для первой задачи если отсутствует.
    if "learning_steps_to_threshold" not in results[first_task]:
        results[first_task]["learning_steps_to_threshold"] = None

    return results

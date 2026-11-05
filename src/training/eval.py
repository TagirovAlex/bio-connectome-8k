"""Модуль оценки модели в среде навигации.

Предоставляет функцию :func:`evaluate_model` для запуска нескольких
эпизодов с поддержкой как Stable-Baselines3 PPO (через ``predict``),
так и чистых ``torch.nn.Module`` (через ``argmax`` по логитам).
"""

from typing import Any, Dict, List

import numpy as np
import torch


def _get_device_for_model(model: Any) -> torch.device:
    """Определяет устройство модели по её параметрам.

    Args:
        model: Модель ``nn.Module`` или объект с параметрами.

    Returns:
        Устройство ``torch.device`` модели или ``cpu``, если параметров нет.
    """
    # Пытаемся извлечь устройство из параметров модели.
    try:
        # ``parameters()`` есть у nn.Module.
        params = list(model.parameters())  # type: ignore[attr-defined]
        if len(params) > 0:
            return params[0].device
    except Exception:
        pass
    # Резервный путь — CPU.
    return torch.device("cpu")


def _predict_torch_action(model: Any, obs: np.ndarray) -> int:
    """Получает действие из torch-модели через argmax.

    Args:
        model: Модель ``nn.Module``, принимающая тензор ``[B, obs_dim]``.
        obs: Наблюдение из среды, массив формы ``(obs_dim,)``.

    Returns:
        Дискретное действие как ``int``.
    """
    # Переводим модель в режим оценки (отключает dropout/batchnorm).
    try:
        model.eval()  # type: ignore[attr-defined]
    except Exception:
        pass

    # Конвертируем наблюдение в тензор.
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    # Добавляем батч-измерение если вход одномерный.
    if obs_tensor.dim() == 1:
        obs_tensor = obs_tensor.unsqueeze(0)

    # Перемещаем на устройство модели.
    device = _get_device_for_model(model)
    obs_tensor = obs_tensor.to(device)

    with torch.no_grad():
        # Прямой проход.
        output = model(obs_tensor)  # type: ignore[operator]
        # Если модель вернула кортеж/список (например, (logits, value)), берём первый элемент.
        if isinstance(output, (tuple, list)):
            output = output[0]
        # Приводим к тензору если нужно.
        if not isinstance(output, torch.Tensor):
            output = torch.as_tensor(output, dtype=torch.float32, device=device)

        # Обрабатываем разные формы выхода.
        if output.dim() == 2:
            # [B, n_actions] -> argmax по действиям.
            action_tensor = torch.argmax(output, dim=-1)
        elif output.dim() == 1:
            # [n_actions] для батча 1 или одномерный выход.
            # Если батч был добавлен выше, размер 1 — считаем что это действия.
            if output.shape[0] > 1 and obs_tensor.shape[0] == 1:
                action_tensor = torch.argmax(output, dim=-1).unsqueeze(0)
            else:
                action_tensor = torch.argmax(output, dim=-1)
                # Приводим к батч-форме для единообразия.
                if action_tensor.dim() == 0:
                    action_tensor = action_tensor.unsqueeze(0)
        else:
            # Схлопываем лишние измерения: [B, ...] -> [B, -1].
            flat = output.view(output.shape[0], -1)
            action_tensor = torch.argmax(flat, dim=-1)

        # Извлекаем скаляр для первого элемента батча.
        action = int(action_tensor[0].detach().cpu().item())
    return action


def evaluate_model(model: Any, env: Any, n_episodes: int = 30) -> Dict[str, Any]:
    """Оценивает модель на среде по нескольким эпизодам.

    Запускает ``n_episodes`` эпизодов: ``reset`` -> цикл ``step`` до
    ``terminated`` или ``truncated``. На каждом шаге действие получается
    через ``model.predict(obs, deterministic=True)`` если у модели есть
    метод ``predict`` (путь SB3 PPO), иначе через прямой вызов
    ``model(obs)`` с ``argmax`` (путь ``nn.Module``). Собирает суммарные
    награды и длины эпизодов.

    Args:
        model: Обученная модель. Поддерживаются два интерфейса:
            * SB3 PPO — объект с методом ``predict(obs, deterministic=True)``,
              возвращающим ``(action, state)``.
            * ``torch.nn.Module`` — вызывается как ``model(obs_tensor)``
              и возвращает логиты ``[B, n_actions]``.
        env: Среда Gymnasium с интерфейсом ``reset() -> (obs, info)`` и
            ``step(action) -> (obs, reward, terminated, truncated, info)``.
        n_episodes: Число эпизодов для оценки. По умолчанию 30
            (см. ``config/training.yaml: n_eval_episodes``).

    Returns:
        Словарь с ключами:
            * ``mean_reward`` (``float``) — средняя награда по эпизодам.
            * ``std_reward`` (``float``) — стандартное отклонение наград.
            * ``episode_rewards`` (``list[float]``) — список наград каждого эпизода.
            * ``mean_length`` (``float``) — средняя длина эпизода (число шагов).

    Raises:
        ValueError: Если ``n_episodes`` меньше 1.
    """
    # Валидация числа эпизодов.
    if n_episodes < 1:
        raise ValueError(f"n_episodes должен быть >=1, получено {n_episodes}")

    # Определяем, является ли модель SB3 (есть callable predict).
    is_sb3 = hasattr(model, "predict") and callable(getattr(model, "predict", None))

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    # Основной цикл по эпизодам.
    for _ in range(n_episodes):
        # Сбрасываем среду; gymnasium возвращает (obs, info).
        reset_result = env.reset()
        # Поддержка как нового API (obs, info), так и старого (obs).
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _ = reset_result
        else:
            obs = reset_result  # type: ignore[assignment]

        ep_reward: float = 0.0
        ep_length: int = 0
        done: bool = False

        # Шаги внутри эпизода до терминации.
        while not done:
            # Получаем действие в зависимости от типа модели.
            if is_sb3:
                # Путь SB3 PPO: детерминированный прогноз.
                try:
                    # Основной вызов с deterministic=True.
                    action, _states = model.predict(obs, deterministic=True)  # type: ignore[attr-defined]
                except TypeError:
                    # Резерв: некоторые обёртки не принимают deterministic.
                    action, _states = model.predict(obs)  # type: ignore[attr-defined]
                # SB3 может вернуть массив формы (n_envs,) или скаляр.
                # Для Discrete среды с одним окружением приводим к скаляру.
                if isinstance(action, np.ndarray):
                    # Если батч из одного элемента — извлекаем.
                    if action.size == 1:
                        action = int(action.squeeze())
                    elif action.ndim > 1 and action.shape[0] == 1:
                        # Форма [1, ...] -> берём первый.
                        action = action[0]
                        if isinstance(action, np.ndarray) and action.size == 1:
                            action = int(action.squeeze())
            else:
                # Путь nn.Module: argmax по логитам.
                action = _predict_torch_action(model, np.asarray(obs))

            # Выполняем шаг среды.
            step_result = env.step(action)
            # Ожидаем gymnasium API: (obs, reward, terminated, truncated, info).
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            elif len(step_result) == 4:
                # Резерв для старого API: (obs, reward, done, info).
                obs, reward, done_flag, _ = step_result
                done = bool(done_flag)
            else:
                raise RuntimeError(f"Неожиданный формат возврата env.step: {step_result}")

            ep_reward += float(reward)
            ep_length += 1

        episode_rewards.append(float(ep_reward))
        episode_lengths.append(int(ep_length))

    # Агрегируем статистики.
    rewards_arr = np.array(episode_rewards, dtype=np.float64)
    mean_reward = float(np.mean(rewards_arr)) if len(rewards_arr) > 0 else 0.0
    std_reward = float(np.std(rewards_arr)) if len(rewards_arr) > 0 else 0.0
    mean_length = float(np.mean(episode_lengths)) if len(episode_lengths) > 0 else 0.0

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "episode_rewards": episode_rewards,
        "mean_length": mean_length,
    }

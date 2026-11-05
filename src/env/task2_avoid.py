"""Среда задачи T2: навигация к цели с препятствиями."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.env.fly_nav import FlyNavEnv


class Task2AvoidEnv(FlyNavEnv):
    """Среда задачи T2: навигация с избеганием препятствий.

    Наследует базовую логику :class:`FlyNavEnv` и фиксирует наличие
    препятствий (окружностей). Награда за приближение к цели,
    штраф ``-10`` за столкновение уже реализован в базовом классе.

    Attributes:
        n_obstacles: Число препятствий (кругов) на арене.
        obstacle_radius: Радиус каждого препятствия.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Инициализирует среду T2 с препятствиями.

        Читает ``tasks.task2_avoid.n_obstacles`` (дефолт ``5``) и
        ``tasks.task2_avoid.obstacle_radius`` (дефолт ``1.0``) из
        ``config``. Поддерживает как вложенный формат ``tasks``, так и
        плоский ``n_obstacles`` / ``obstacle_radius`` на верхнем уровне.

        Args:
            config: Словарь конфигурации, как в ``config/env.yaml``.
                Ожидаются ключи ``grid_size``, ``n_rays`` и др., плюс
                ``tasks.task2_avoid.n_obstacles`` и
                ``tasks.task2_avoid.obstacle_radius``. При их отсутствии
                используются дефолты ``5`` и ``1.0``.
        """
        # Копируем конфиг, чтобы не мутировать исходный словарь.
        task_config: Dict[str, Any] = dict(config)

        # Извлекаем вложенную конфигурацию задачи T2, если она есть.
        tasks = config.get("tasks", {})
        task2_cfg: Dict[str, Any] = {}
        if isinstance(tasks, dict):
            raw = tasks.get("task2_avoid", {})
            if isinstance(raw, dict):
                task2_cfg = raw

        # Определяем число препятствий: приоритет у вложенного ключа,
        # затем плоский ключ, затем дефолт 5.
        if "n_obstacles" in task2_cfg:
            n_obstacles = task2_cfg["n_obstacles"]
        elif "n_obstacles" in config:
            n_obstacles = config["n_obstacles"]
        else:
            n_obstacles = 5

        # Определяем радиус препятствия аналогично, дефолт 1.0.
        if "obstacle_radius" in task2_cfg:
            obstacle_radius = task2_cfg["obstacle_radius"]
        elif "obstacle_radius" in config:
            obstacle_radius = config["obstacle_radius"]
        else:
            obstacle_radius = 1.0

        # Кладём разрешённые значения в копию конфига для базы.
        task_config["n_obstacles"] = int(n_obstacles)
        task_config["obstacle_radius"] = float(obstacle_radius)

        # Делегируем инициализацию базовому классу.
        super().__init__(task_config)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Сбрасывает среду T2 в начальное состояние с препятствиями.

        Препятствия (``n_obstacles`` кругов радиуса ``obstacle_radius``)
        сэмплируются равномерно по арене в базовом классе.

        Args:
            seed: Сид для детерминизма.
            options: Дополнительные опции (не используются).

        Returns:
            Кортеж ``(observation, info)``, где ``observation`` имеет
            форму ``(n_rays + 1,)``.
        """
        # Делегируем логику сброса базе — она создаёт препятствия.
        return super().reset(seed=seed, options=options)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Выполняет шаг среды с учётом столкновений.

        Использует базовую логику перемещения, лидара и награды:
        ``-dist_to_goal`` + ``10`` за достижение цели и ``-10`` за
        столкновение с препятствием.

        Args:
            action: Дискретное действие (0=влево, 1=прямо, 2=вправо).

        Returns:
            Кортеж ``(observation, reward, terminated, truncated, info)``,
            где ``info`` содержит ``distance_to_goal``, ``is_success``,
            ``is_collision``.
        """
        # Делегируем шаг базовому классу — штраф за столкновение уже там.
        return super().step(action)

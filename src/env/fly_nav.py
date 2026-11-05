"""Базовая среда навигации для continual RL."""

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from typing import Any, Dict, Optional, Tuple


class FlyNavEnv(gym.Env):
    """Базовая среда навигации агента в 2D пространстве.

    Наблюдение: вектор лучей (n_rays) + угол к цели.
    Действие: дискретное, 3 действия (влево, прямо, вправо).

    Attributes:
        grid_size: Размер квадратной арены.
        n_rays: Число лучей лидара.
        n_actions: Число дискретных действий.
        speed: Скорость перемещения за шаг.
        max_steps: Лимит шагов в эпизоде.
        n_obstacles: Число препятствий.
        obstacle_radius: Радиус препятствия.
        target_distance: Желаемая дистанция до цели при сэмплировании.
        position: Текущая позиция агента (2,).
        goal: Позиция цели (2,).
        obstacles: Массив позиций препятствий (n_obstacles, 2).
        heading: Угол ориентации агента в радианах.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Dict[str, Any]) -> None:
        """Инициализирует среду.

        Args:
            config: Словарь конфигурации с ключами grid_size, n_rays,
                n_actions, speed, max_steps_per_episode, n_obstacles,
                obstacle_radius, target_distance (или вложенный tasks).
        """
        super().__init__()
        # Сохраняем исходный конфиг для воспроизводимости.
        self.config = config
        # Читаем основные параметры через .get с дефолтами.
        self.grid_size: float = float(config.get("grid_size", 20))
        self.n_rays: int = int(config.get("n_rays", 8))
        self.n_actions: int = int(config.get("n_actions", 3))
        self.speed: float = float(config.get("speed", 0.5))
        self.max_steps: int = int(config.get("max_steps_per_episode", 200))

        # Параметры препятствий и цели — поддержка плоских и вложенных ключей.
        self.n_obstacles: int = int(config.get("n_obstacles", 0))
        self.obstacle_radius: float = float(config.get("obstacle_radius", 1.0))
        self.target_distance: Optional[float] = config.get("target_distance", None)

        # Если target_distance вложен в tasks — извлекаем оттуда.
        tasks = config.get("tasks", {})
        if isinstance(tasks, dict) and tasks:
            if self.target_distance is None:
                for task_cfg in tasks.values():
                    if isinstance(task_cfg, dict) and "target_distance" in task_cfg:
                        self.target_distance = float(task_cfg["target_distance"])
                        break

        # Пространства действий и наблюдений.
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_rays + 1,), dtype=np.float32
        )

        # Состояние — инициализируется в reset().
        self.np_random: np.random.Generator
        # gymnasium seeding: super().reset создаст self.np_random, но для
        # детерминизма до первого reset инициализируем через seeding.
        # Используем gymnasium.utils.seeding для соответствия требованию.
        try:
            # Пытаемся создать генератор через gymnasium seeding (совместимость).
            self.np_random, _ = seeding.np_random(None)
        except Exception:
            self.np_random = np.random.default_rng()

        self.position: np.ndarray = np.zeros(2, dtype=np.float64)
        self.goal: np.ndarray = np.zeros(2, dtype=np.float64)
        self.obstacles: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self.heading: float = 0.0
        self._step_count: int = 0

        # Пустые до reset — по требованию спецификации помечаем отсутствие.
        # Переопределяем как None-подобные пустые массивы, но для безопасности
        # оставляем нулевые массивы и флаг _initialized.
        self._initialized: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Сбрасывает среду в начальное состояние.

        Args:
            seed: Сид для детерминизма (используется self.np_random).
            options: Дополнительные опции (не используются).

        Returns:
            Кортеж (observation, info).
        """
        # Делегируем сидирование gymnasium — создаёт self.np_random.
        super().reset(seed=seed)

        # super().reset уже создал self.np_random; fallback через seeding если нет.
        if not hasattr(self, "np_random") or self.np_random is None:
            self.np_random, _ = seeding.np_random(seed)

        # Сэмплируем позицию агента равномерно по арене.
        self.position = self.np_random.uniform(
            0, self.grid_size, size=(2,)
        ).astype(np.float64)

        # Сэмплируем ориентацию агента.
        self.heading = float(self.np_random.uniform(0, 2 * np.pi))

        # Сэмплируем цель.
        if self.target_distance is not None:
            # Цель на фиксированном расстоянии target_distance от агента.
            angle = float(self.np_random.uniform(0, 2 * np.pi))
            offset = np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float64
            ) * float(self.target_distance)
            self.goal = self.position + offset
            # Клипаем в границы арены.
            self.goal = np.clip(self.goal, 0, self.grid_size).astype(np.float64)
        else:
            self.goal = self.np_random.uniform(
                0, self.grid_size, size=(2,)
            ).astype(np.float64)

        # Сэмплируем препятствия.
        if self.n_obstacles > 0:
            self.obstacles = self.np_random.uniform(
                0, self.grid_size, size=(self.n_obstacles, 2)
            ).astype(np.float64)
        else:
            self.obstacles = np.zeros((0, 2), dtype=np.float64)

        self._step_count = 0
        self._initialized = True

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Выполняет шаг среды.

        Args:
            action: Дискретное действие (0=влево -30°, 1=прямо, 2=вправо +30°).

        Returns:
            Кортеж (observation, reward, terminated, truncated, info).
        """
        # Проверка инициализации.
        if not self._initialized:
            raise RuntimeError("Среда не инициализирована — вызовите reset() перед step().")

        # Обновляем ориентацию по действию.
        # 0 -> -30°, 1 -> 0°, 2 -> +30°.
        delta_deg = (int(action) - 1) * 30.0
        delta_rad = np.deg2rad(delta_deg)
        self.heading = float(self.heading + delta_rad)
        # Нормализуем угол к диапазону [-pi, pi] для стабильности.
        self.heading = float((self.heading + np.pi) % (2 * np.pi) - np.pi)

        # Обновляем позицию.
        dx = np.cos(self.heading) * self.speed
        dy = np.sin(self.heading) * self.speed
        self.position = self.position + np.array([dx, dy], dtype=np.float64)
        # Клипаем в границы арены.
        self.position = np.clip(self.position, 0, self.grid_size).astype(np.float64)

        self._step_count += 1

        # Проверяем достижение цели.
        dist_to_goal = float(np.linalg.norm(self.position - self.goal))
        goal_reached = dist_to_goal < 1.0

        # Проверяем столкновение с препятствием.
        is_collision = False
        if self.obstacles.shape[0] > 0:
            dists = np.linalg.norm(self.obstacles - self.position, axis=1)
            is_collision = bool(np.any(dists < self.obstacle_radius))

        # Лимит шагов.
        truncated = self._step_count >= self.max_steps
        terminated = bool(goal_reached or is_collision)

        # Вычисляем награду.
        reward = self._compute_reward()

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "distance_to_goal": dist_to_goal,
            "is_success": bool(goal_reached),
            "is_collision": bool(is_collision),
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Вычисляет наблюдение: лучи лидара + угол к цели.

        Returns:
            Массив формы (n_rays+1,) dtype float32, где первые n_rays —
            инвертированные нормализованные дистанции лидара (1 - dist),
            последний элемент — угол к цели, нормализованный в [-1, 1].
        """
        # Если среда ещё не инициализирована — возвращаем нули.
        if not self._initialized:
            return np.zeros(self.n_rays + 1, dtype=np.float32)

        # Параметры лидара.
        max_range = float(self.grid_size)
        # Для избежания деления на ноль.
        eps = 1e-8
        rays = []

        for i in range(self.n_rays):
            # Угол луча относительно текущего heading.
            ray_angle = self.heading + (2 * np.pi * i / self.n_rays)
            dx = float(np.cos(ray_angle))
            dy = float(np.sin(ray_angle))

            # Дистанция до стены.
            t_wall = float("inf")

            # Пересечение с вертикальными стенами x=0 и x=grid_size.
            if abs(dx) > eps:
                if dx > 0:
                    t_x = (self.grid_size - self.position[0]) / dx
                else:
                    t_x = (0.0 - self.position[0]) / dx
                if t_x > 0:
                    t_wall = min(t_wall, t_x)

            # Пересечение с горизонтальными стенами y=0 и y=grid_size.
            if abs(dy) > eps:
                if dy > 0:
                    t_y = (self.grid_size - self.position[1]) / dy
                else:
                    t_y = (0.0 - self.position[1]) / dy
                if t_y > 0:
                    t_wall = min(t_wall, t_y)

            # Дистанция до ближайшего препятствия (окружности).
            t_min = t_wall
            if self.obstacles.shape[0] > 0:
                for obs_pos in self.obstacles:
                    # Вектор от центра препятствия к агенту.
                    oc_x = self.position[0] - obs_pos[0]
                    oc_y = self.position[1] - obs_pos[1]
                    # Квадратное уравнение: |pos + t*dir - obs|^2 = r^2
                    # a = 1 (dir нормирован), b = 2*dot(dir, oc), c = |oc|^2 - r^2
                    b = 2.0 * (dx * oc_x + dy * oc_y)
                    c = oc_x * oc_x + oc_y * oc_y - self.obstacle_radius**2
                    disc = b * b - 4 * c
                    if disc >= 0:
                        sqrt_disc = np.sqrt(disc)
                        t1 = (-b - sqrt_disc) / 2.0
                        t2 = (-b + sqrt_disc) / 2.0
                        # Берём ближайшее положительное пересечение.
                        if t1 > eps:
                            t_min = min(t_min, t1)
                        elif t2 > eps:
                            t_min = min(t_min, t2)

            # Если не найдено пересечение (не должно случиться), используем max_range.
            if not np.isfinite(t_min):
                t_min = max_range

            # Нормализуем и инвертируем: 1 - dist/max_range, клип [0,1].
            normalized = t_min / max_range if max_range > eps else 0.0
            inv = 1.0 - float(np.clip(normalized, 0.0, 1.0))
            rays.append(inv)

        # Угол к цели, нормализованный в [-1, 1].
        vec_x = float(self.goal[0] - self.position[0])
        vec_y = float(self.goal[1] - self.position[1])
        angle_to_goal = float(np.arctan2(vec_y, vec_x))
        relative_angle = angle_to_goal - self.heading
        # Обёртка в [-pi, pi].
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
        angle_norm = float(relative_angle / np.pi)
        angle_norm = float(np.clip(angle_norm, -1.0, 1.0))

        obs = np.array(rays + [angle_norm], dtype=np.float32)
        # Клипаем всё в [-1, 1] для соответствия Box.
        obs = np.clip(obs, -1.0, 1.0).astype(np.float32)
        return obs

    def _compute_reward(self) -> float:
        """Вычисляет награду за текущее состояние.

        Returns:
            Скалярная награда: -дистанция до цели, +10 за достижение цели,
            -10 за столкновение.
        """
        if not self._initialized:
            return 0.0

        dist = float(np.linalg.norm(self.position - self.goal))
        reward = -dist

        # Бонус за достижение цели.
        if dist < 1.0:
            reward += 10.0

        # Штраф за столкновение.
        if self.obstacles.shape[0] > 0:
            dists = np.linalg.norm(self.obstacles - self.position, axis=1)
            if np.any(dists < self.obstacle_radius):
                reward -= 10.0

        return float(reward)

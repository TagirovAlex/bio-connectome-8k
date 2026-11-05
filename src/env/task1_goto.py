"""Среда задачи T1: навигация к цели без препятствий."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.env.fly_nav import FlyNavEnv


class Task1GotoEnv(FlyNavEnv):
    """Среда задачи T1 — навигация к видимой цели без препятствий.

    Наследует :class:`src.env.fly_nav.FlyNavEnv` и фиксирует отсутствие
    препятствий (``n_obstacles=0``). Наблюдение и динамика идентичны
    базовой среде, но гарантировано ``obstacles`` пустой, а награда
    формируется только за приближение к цели (штраф за столкновения
    отсутствует, так как сталкиваться не с чем).

    Attributes:
        config: Исходный словарь конфигурации (копия).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Инициализирует среду T1.

        Копирует переданный ``config``, принудительно устанавливает
        ``n_obstacles=0`` (задача без препятствий) и прокидывает
        ``target_distance`` из секции ``tasks.task1_goto`` если он не
        задан на верхнем уровне.

        Args:
            config: Словарь конфигурации, совместимый с ``FlyNavEnv``.
                Ожидаются ключи ``grid_size``, ``n_rays``, ``n_actions``,
                ``speed``, ``max_steps_per_episode`` и вложенный
                ``tasks.task1_goto.target_distance``.
        """
        # Копируем конфиг, чтобы не мутировать оригинал, переданный вызывающим кодом.
        task_config: Dict[str, Any] = config.copy()
        # Задача T1 — без препятствий.
        task_config["n_obstacles"] = 0

        # Явно прокидываем target_distance из секции task1_goto, если нужно.
        # Базовый класс уже умеет извлекать его из tasks, но для
        # прозрачности дублируем на верхний уровень.
        tasks = config.get("tasks", {})
        if isinstance(tasks, dict):
            t1_cfg = tasks.get("task1_goto", {})
            if isinstance(t1_cfg, dict) and "target_distance" in t1_cfg:
                # Не переопределяем если уже задано явно в task_config.
                task_config.setdefault("target_distance", float(t1_cfg["target_distance"]))

        # Инициализируем базовую среду с подготовленным конфигом.
        super().__init__(task_config)

        # Дублируем гарантию отсутствия препятствий на уровне атрибута
        # (база уже выставила 0, но фиксируем инвариант задачи T1).
        self.n_obstacles = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Сбрасывает среду T1 в начальное состояние без препятствий.

        Генерирует случайную позицию агента и цели; список препятствий
        всегда остаётся пустым (``obstacles.shape == (0, 2)``).

        Args:
            seed: Сид для детерминизма.
            options: Дополнительные опции (не используются).

        Returns:
            Кортеж ``(observation, info)``, где ``observation`` — массив
            формы ``(n_rays + 1,)`` с лучами лидара и углом к цели,
            ``info`` — пустой словарь.
        """
        # Делегируем базовой реализации — при n_obstacles=0 она уже
        # создаёт только цель и пустой массив препятствий.
        obs, info = super().reset(seed=seed, options=options)

        # Инвариант T1: препятствий нет.
        # Явно гарантируем пустой массив на случай внешних модификаций.
        if self.obstacles.shape[0] != 0:
            # Логируем нарушение инварианта и исправляем.
            # Используем нулевой массив формы (0, 2).
            self.obstacles = np.zeros((0, 2), dtype=np.float64)

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Выполняет шаг среды T1.

        Динамика перемещения и награда наследуются из ``FlyNavEnv``.
        При ``n_obstacles=0`` награда формируется только за приближение
        к цели (``-dist`` + ``+10`` за достижение), без штрафа за
        препятствия.

        Args:
            action: Дискретное действие (0 — влево −30°, 1 — прямо,
                2 — вправо +30°).

        Returns:
            Кортеж ``(observation, reward, terminated, truncated, info)``,
            где ``terminated`` срабатывает только при достижении цели,
            ``truncated`` — по лимиту шагов, ``info`` содержит
            ``distance_to_goal``, ``is_success``, ``is_collision``
            (всегда ``False`` в T1).
        """
        # Делегируем шаг базовой среде — она уже учитывает n_obstacles=0
        # (нет проверки коллизий и штрафа за них).
        obs, reward, terminated, truncated, info = super().step(action)

        # В T1 столкновений быть не может — фиксируем в info.
        # Оставляем флаг из базы, но гарантируем False при непустых препятствиях.
        if self.n_obstacles == 0:
            info["is_collision"] = False

        return obs, reward, terminated, truncated, info

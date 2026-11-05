"""GNN-контроллер на основе топологии коннектома дрозофилы.

Модуль реализует класс :class:`BioConnectomeGNN`, где топология
``edge_index`` фиксирована (зарегистрирована как buffer и не обучается),
а синаптические веса представлены обучаемым параметром ``edge_weight``.
Архитектура следует спецификации из PROJECT_SPEC §4.3 и config/models.yaml:
наблюдение -> сенсорные узлы -> n_steps message passing (tanh) ->
моторные узлы -> readout.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter


def normalize_edge_weights(
    edge_weight: Tensor,
    edge_index: Tensor,
    n_nodes: int,
) -> Tensor:
    """Нормирует веса рёбер так, чтобы средний модуль веса стал ~0.5.

    Сырые ``syn_count`` коннектома имеют большой масштаб (mean ``|w|`` ~ 21
    для 2K, ~15 для 8K, max ~ 2633). При sum-агрегации это насыщает ``tanh``
    почти на всех узлах (до 92% юнитов). Функция делит все веса на

    ``scale = 2 * mean_{e in E} |w_e|``,

    то есть приводит средний модуль веса к 0.5. Обоснование:

    (а) масштаб совпадает с ``random_er`` (веса ``Uniform[0, 1]``,
    mean ``|w|`` ~ 0.5), поэтому bio/rewired/unsigned/random_er находятся
    в одном диапазоне активаций;
    (б) при таком масштабе усиление за шаг message passing
    (``active_fan_in * mean|w| / scale``) близко к 1, поэтому ``tanh`` не
    насыщается, а сигнал не затухает на пути в 5-7 хопов от сенсоров
    к моторам;
    (в) деление на общий положительный скаляр сохраняет знаки
    (возбуждающие/тормозные синапсы) и относительные величины весов.

    Нормировка на среднюю сумму входящих весов (``mean_nodes(sum_in |w|)``
    ~ 445) для этих графов слишком сильна: сигнал падает на порядок за хоп
    и к моторным узлам приходит ~1e-7.

    Знаки, dtype и device сохраняются. Исходный граф на диске не изменяется —
    нормируется только тензор, передаваемый в модель.

    Args:
        edge_weight: Веса рёбер формы ``[E]``, dtype ``float``.
        edge_index: Индексы рёбер формы ``[2, E]``; используется только для
            проверки согласованности числа рёбер (топология не меняется).
        n_nodes: Число узлов графа ``N``; сохранено в сигнатуре для
            совместимости, на масштаб не влияет.

    Returns:
        Нормированные веса формы ``[E]`` с теми же знаками, dtype и device.
        Если весов нет или ``mean|w| == 0``, возвращается исходный тензор
        без изменений (копия не создаётся).

    Raises:
        ValueError: Если число рёбер в ``edge_index`` и ``edge_weight``
            не совпадает.
    """
    if edge_weight.numel() == 0:
        return edge_weight

    if edge_index.size(1) != edge_weight.size(0):
        raise ValueError(
            f"Несовместимые размеры: edge_index E={edge_index.size(1)}, "
            f"edge_weight E={edge_weight.size(0)} (n_nodes={n_nodes})"
        )

    # Средний модуль веса по всем рёбрам (0-dim тензор).
    mean_abs_w = edge_weight.abs().mean().detach()
    if float(mean_abs_w) <= 0.0:
        # Все веса нулевые — нормировать нечего, избегаем деления на 0.
        return edge_weight

    # scale = 2 * mean|w|  =>  mean|w| / scale = 0.5.
    return edge_weight / (2.0 * mean_abs_w)


class BioConnectomeGNN(nn.Module):
    """GNN-контроллер, использующий структуру коннектома как граф вычислений.

    Топология (``edge_index``) зафиксирована и не обучается, а веса
    сообщений (``edge_weight``) являются обучаемыми параметрами,
    инициализированными знаковыми синаптическими весами коннектома.

    Поток данных::

        obs [B, input_dim] -> input_proj -> states[:, sensory_mask] ->
        n_steps x (линейное преобразование + взвешенная агрегация по рёбрам + tanh) ->
        mean по motor_mask -> readout -> logits [B, output_dim]

    Attributes:
        edge_index: Буфер формы ``[2, E]`` с индексами рёбер (заморожен).
        edge_weight: Обучаемый параметр формы ``[E]`` с весами сообщений.
        sensory_mask: Буфер формы ``[N]`` — маска сенсорных узлов.
        motor_mask: Буфер формы ``[N]`` — маска моторных узлов.
        n_nodes: Число узлов графа ``N``.
        hidden_dim: Размерность скрытого состояния узла.
        n_steps: Число шагов message passing.
        input_proj: Линейная проекция наблюдения в скрытое пространство.
        mp_layers: Список линейных слоёв для каждого шага распространения.
        readout: Линейная проекция моторных состояний в логиты действий.
    """

    def __init__(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        sensory_mask: Tensor,
        motor_mask: Tensor,
        n_nodes: int,
        input_dim: int = 9,
        output_dim: int = 3,
        hidden_dim: int = 64,
        n_steps: int = 3,
    ) -> None:
        """Инициализирует GNN-контроллер коннектома.

        Args:
            edge_index: Тензор рёбер формы ``[2, E]``, dtype ``long``.
                Первый ряд — индексы источников (src), второй — приёмников (dst).
            edge_weight: Тензор весов рёбер формы ``[E]``, dtype ``float``.
                Знаковые веса: ``+syn_count`` для возбуждающих медиаторов,
                ``-syn_count`` для тормозных. Используется для инициализации
                обучаемого параметра сообщений.
            sensory_mask: Булева маска формы ``[N]`` — ``True`` для сенсорных
                узлов (регион LA, ожидается >=50 узлов).
            motor_mask: Булева маска формы ``[N]`` — ``True`` для моторных
                узлов (class == "motor", ожидается >=20 узлов).
            n_nodes: Общее число узлов ``N``.
            input_dim: Размерность наблюдения (по умолчанию 9: 8 лучей лидара
                + угол до цели, см. ``config/env.yaml``).
            output_dim: Размерность выхода — число дискретных действий
                (по умолчанию 3).
            hidden_dim: Размерность скрытого состояния узла. По умолчанию 64
                для масштаба 2K (``config/models.yaml: gnn.hidden_dim.2k``),
                128 для 8K.
            n_steps: Число шагов message passing (по умолчанию 3 для совместимости,
                фактическое значение задаётся по масштабу в
                ``config/models.yaml: gnn.n_propagation_steps.{2k,8k}``:
                7 для 2K и 5 для 8K, чтобы покрыть путь сенсоры -> моторы,
                активация ``tanh``).

        Raises:
            ValueError: Если размерности масок или рёбер не согласованы
                с ``n_nodes``.
        """
        super().__init__()

        # ---- Валидация входных размерностей ----
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index должен иметь форму [2, E], получено {tuple(edge_index.shape)}"
            )
        if edge_weight.dim() != 1:
            raise ValueError(
                f"edge_weight должен иметь форму [E], получено {tuple(edge_weight.shape)}"
            )
        if edge_index.size(1) != edge_weight.size(0):
            raise ValueError(
                f"Несовместимые размеры: edge_index E={edge_index.size(1)}, "
                f"edge_weight E={edge_weight.size(0)}"
            )
        if sensory_mask.size(0) != n_nodes:
            raise ValueError(
                f"sensory_mask размер {sensory_mask.size(0)} != n_nodes {n_nodes}"
            )
        if motor_mask.size(0) != n_nodes:
            raise ValueError(
                f"motor_mask размер {motor_mask.size(0)} != n_nodes {n_nodes}"
            )

        # ---- Регистрация непереобучаемой топологии как buffer ----
        # edge_index — целочисленный буфер, не участвует в градиентном спуске.
        self.register_buffer(
            "edge_index",
            edge_index.detach().clone().long(),
            persistent=True,
        )
        # Маски также регистрируем как buffer для корректного перемещения
        # между устройствами (cpu/cuda) вместе с моделью.
        self.register_buffer(
            "sensory_mask",
            sensory_mask.detach().clone().bool(),
            persistent=True,
        )
        self.register_buffer(
            "motor_mask",
            motor_mask.detach().clone().bool(),
            persistent=True,
        )

        # ---- Обучаемые веса сообщений ----
        # Инициализируем из знаковых весов коннектома; градиенты течёт
        # через этот параметр. Копия делается detach+clone для изоляции.
        self.edge_weight = nn.Parameter(
            edge_weight.detach().clone().float(), requires_grad=True
        )

        self.n_nodes = int(n_nodes)
        self.hidden_dim = int(hidden_dim)
        self.n_steps = int(n_steps)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        # ---- Слои сети ----
        # Проекция наблюдения в скрытое пространство сенсорных узлов.
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Отдельный линейный слой без bias для каждого шага message passing.
        # Вес каждого слоя применяется к скрытым состояниям перед агрегацией.
        self.mp_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(n_steps)]
        )
        # Считывание: усреднённые моторные состояния -> логиты действий.
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs: Tensor) -> Tensor:
        """Прямое распространение через коннектом.

        Args:
            obs: Наблюдение среды формы ``[B, input_dim]`` или ``[input_dim]``
                для одиночного примера. Тип ``float``, устройство определяет
                устройство вычислений (все буферы/параметры приводятся к нему).

        Returns:
            Логиты действий формы ``[B, output_dim]`` (или ``[1, output_dim]``
            если вход был одномерным).
        """
        # ---- Нормализация размерности входа ----
        # Поддержка как батча [B, D], так и одиночного вектора [D].
        squeeze_single = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_single = True
        elif obs.dim() != 2:
            raise ValueError(
                f"obs должен иметь форму [B, input_dim] или [input_dim], "
                f"получено {tuple(obs.shape)}"
            )

        batch_size = obs.size(0)
        device = obs.device
        dtype = obs.dtype

        # ---- Приведение буферов/параметров к устройству входа ----
        # edge_index — буфер (long), переносим без нарушения графа.
        edge_index = self.edge_index
        if edge_index.device != device:
            edge_index = edge_index.to(device)

        # edge_weight — обучаемый параметр; .to() сохраняет связь с графом
        # через ToCopy-узел, поэтому градиенты корректно вернутся к параметру.
        edge_weight = self.edge_weight
        if edge_weight.device != device:
            edge_weight = edge_weight.to(device)

        sensory_mask = self.sensory_mask
        if sensory_mask.device != device:
            sensory_mask = sensory_mask.to(device)

        motor_mask = self.motor_mask
        if motor_mask.device != device:
            motor_mask = motor_mask.to(device)

        # ---- Инициализация скрытых состояний узлов нулями ----
        # states: [B, N, H] — состояние каждого узла для каждого примера батча.
        states = torch.zeros(
            batch_size, self.n_nodes, self.hidden_dim, device=device, dtype=dtype
        )

        # ---- Инъекция наблюдения в сенсорные узлы ----
        # input_proj: [B, input_dim] -> [B, H]
        proj = self.input_proj(obs)  # [B, H]
        # Добавляем спроецированное наблюдение ко всем сенсорным узлам.
        # Broadcasting: [B, n_sensory, H] += [B, 1, H]
        n_sensory = int(sensory_mask.sum().item())
        if n_sensory > 0:
            # proj.unsqueeze(1) имеет форму [B, 1, H] и широковещается
            # по измерению сенсорных узлов при сложении.
            states[:, sensory_mask, :] += proj.unsqueeze(1)

        # ---- Message passing ----
        # Каждый шаг: линейное преобразование -> взвешенная агрегация -> tanh.
        if self.n_steps > 0 and edge_index.size(1) > 0:
            src_idx = edge_index[0]  # [E]
            dst_idx = edge_index[1]  # [E]

            for mp_layer in self.mp_layers:
                # Линейное преобразование скрытых состояний:
                # nn.Linear действует по последней размерности [B, N, H] -> [B, N, H].
                h_trans = mp_layer(states)  # [B, N, H]

                # Сбор сообщений с источников рёбер и взвешивание по edge_weight:
                # h_trans[:, src_idx, :] -> [B, E, H]
                # edge_weight [E] -> [1, E, 1] для широковещания по батчу и H.
                msg = h_trans[:, src_idx, :] * edge_weight.view(1, -1, 1)  # [B, E, H]

                # Векторизованная sum-агрегация сообщений по приёмникам:
                # torch_geometric.utils.scatter — оптимизированный CUDA-оператор,
                # заменяет развёрнутый индекс [B, E, H] (огромная память и
                # медленный backward) на индекс только по рёбрам [E].
                states_next = scatter(
                    msg, dst_idx, dim=1, dim_size=self.n_nodes
                )  # [B, N, H]

                # Нелинейность tanh (config/models.yaml: gnn.activation = "tanh").
                states = torch.tanh(states_next)

        # ---- Считывание с моторных узлов ----
        n_motor = int(motor_mask.sum().item())
        if n_motor > 0:
            motor_states = states[:, motor_mask, :]  # [B, n_motor, H]
            # Усреднение по моторным узлам (альтернатива sum; mean стабильнее
            # при разном числе моторных нейронов между графами).
            pooled = motor_states.mean(dim=1)  # [B, H]
        else:
            # Резервный путь если маска пуста (не должно случиться при
            # валидном графе, где motor >=20, но сохраняет работоспособность).
            pooled = states.mean(dim=1)  # [B, H]

        # Линейный readout в пространство действий.
        logits = self.readout(pooled)  # [B, output_dim]

        # Если вход был одномерным, сохраняем батч-размерность 1
        # (вызывающий код ожидает [B, output_dim]; squeeze не делаем
        # для консистентности, но фиксируем флаг на будущее).
        _ = squeeze_single  # флаг зарезервирован для возможного squeeze

        return logits

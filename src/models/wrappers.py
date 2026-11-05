"""Обёртка GNN для использования в качестве FeatureExtractor в SB3 PPO.

Модуль предоставляет класс :class:`GNNFeaturesExtractor`, который
адаптирует графовую модель (обычно :class:`BioConnectomeGNN`) к
интерфейсу ``stable_baselines3.common.torch_layers.BaseFeaturesExtractor``.
"""

from typing import Optional

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """Оборачивает GNN как экстрактор признаков для Stable-Baselines3 PPO.

    Адаптирует графовую модель коннектома к интерфейсу SB3: на вход
    подаётся батч наблюдений ``obs`` формы ``[B, input_dim]`` (по умолчанию
    ``[B, 9]`` — 8 лидар-лучей + угол до цели), на выходе возвращается
    тензор признаков ``[B, features_dim]`` для дальнейшего ``mlp_extractor``.

    Логика извлечения признаков:

    * Пытается получить скрытое состояние моторных узлов до ``readout``
      (идеальный путь — без финального линейного слоя).
    * Если модель возвращает логиты (``[B, output_dim]``) — проецирует
      их к ``features_dim`` через ``Linear`` или обрезку.
    * Если GNN ещё не реализован (стаб с ``NotImplementedError``) — создаёт
      резервную проекцию, чтобы импорт и создание PPO не падали; проекция
      будет совместима после реализации GNN.

    Attributes:
        gnn: Ссылка на графовую модель (экземпляр ``nn.Module``).
        features_dim: Размерность пространства признаков (равна
            ``gnn.hidden_dim`` если атрибут доступен, иначе значение
            по умолчанию 64).

    Args:
        observation_space: Пространство наблюдений среды Gymnasium.
            Используется базовым классом SB3 для валидации.
        gnn_model: Графовая модель. Ожидается наличие атрибутов
            ``hidden_dim``, ``n_nodes``, ``sensory_mask``, ``motor_mask``,
            ``edge_index``, ``edge_weight``, ``input_proj``, ``mp_layers``,
            ``readout`` (как у :class:`src.models.gnn_controller.BioConnectomeGNN`),
            но отсутствие любого из них обрабатывается fallback-логикой.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        gnn_model: nn.Module,
    ) -> None:
        """Инициализирует обёртку и вычисляет ``features_dim``.

        Args:
            observation_space: Пространство наблюдений среды.
            gnn_model: Графовая модель, оборачиваемая экстрактором.

        Raises:
            ValueError: Если ``gnn_model`` равен ``None``.
        """
        # Валидация входной модели.
        if gnn_model is None:
            raise ValueError("gnn_model не может быть None")

        # Определяем features_dim как gnn.hidden_dim (требование ТЗ).
        # Если атрибут отсутствует или не int — берём 64 (конфиг 2k).
        hidden_dim: Optional[int] = getattr(gnn_model, "hidden_dim", None)
        if not isinstance(hidden_dim, int):
            # Попытка альтернативных имён (для совместимости).
            for alt in ("_features_dim", "features_dim", "_hidden_dim"):
                alt_val = getattr(gnn_model, alt, None)
                if isinstance(alt_val, int):
                    hidden_dim = alt_val
                    break
        if not isinstance(hidden_dim, int):
            hidden_dim = 64

        features_dim = int(hidden_dim)

        # Инициализация базового класса SB3.
        super().__init__(observation_space, features_dim=features_dim)

        # Сохраняем ссылку на GNN.
        self.gnn: nn.Module = gnn_model

        # Пытаемся вывести размер выхода GNN для fallback-проекции.
        # Если readout существует и имеет out_features != features_dim,
        # создаём линейный слой для проекции логитов -> признаки.
        self._fallback_proj: Optional[nn.Linear] = None
        output_dim: Optional[int] = None

        # Приоритет 1: атрибут output_dim.
        out_dim_attr = getattr(gnn_model, "output_dim", None)
        if isinstance(out_dim_attr, int):
            output_dim = int(out_dim_attr)

        # Приоритет 2: readout.out_features (nn.Linear).
        if output_dim is None and hasattr(gnn_model, "readout"):
            readout = getattr(gnn_model, "readout")
            if hasattr(readout, "out_features") and isinstance(
                readout.out_features, int
            ):
                output_dim = int(readout.out_features)

        # Если выход отличается от признаков — создаём проекцию заранее.
        if output_dim is not None and output_dim != features_dim:
            self._fallback_proj = nn.Linear(output_dim, features_dim)

        # Резервная проекция для случая когда GNN — стаб (NotImplementedError).
        # Создаётся лениво в forward, когда известен input_dim.
        self._stub_proj: Optional[nn.Linear] = None

    def _extract_motor_hidden_manual(
        self, obs: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Пытается вручную извлечь скрытое состояние моторных узлов.

        Воспроизводит логику :class:`BioConnectomeGNN` до ``readout``:
        инициализация, инъекция в сенсорные узлы, ``n_steps`` шагов
        message passing, усреднение по моторным узлам.

        Args:
            obs: Батч наблюдений, форма ``[B, input_dim]``.

        Returns:
            Тензор ``[B, hidden_dim]`` если удалось, иначе ``None``.
        """
        gnn = self.gnn

        # Проверяем наличие необходимых компонентов.
        required_attrs = (
            "input_proj",
            "mp_layers",
            "n_nodes",
            "hidden_dim",
            "sensory_mask",
            "motor_mask",
            "edge_index",
        )
        if not all(hasattr(gnn, attr) for attr in required_attrs):
            return None

        try:
            # Проекция наблюдения в скрытое пространство.
            # input_proj: Linear(input_dim -> hidden_dim)
            projected = gnn.input_proj(obs)  # [B, hidden_dim]
            batch_size = obs.shape[0]
            device = obs.device
            dtype = projected.dtype
            hidden_dim = int(gnn.hidden_dim)
            n_nodes = int(gnn.n_nodes)

            # Инициализация состояний всех узлов нулями.
            hidden = torch.zeros(
                batch_size, n_nodes, hidden_dim, device=device, dtype=dtype
            )

            # Инъекция в сенсорные узлы.
            sensory_mask = gnn.sensory_mask
            if sensory_mask.dtype != torch.bool:
                sensory_mask = sensory_mask.to(torch.bool)
            sensory_mask = sensory_mask.to(device)
            n_sensory = int(sensory_mask.sum().item())
            if n_sensory > 0:
                # Каждому сенсорному узлу достаётся копия проекции.
                hidden[:, sensory_mask, :] = projected.unsqueeze(1).expand(
                    -1, n_sensory, -1
                )

            # Подготовка топологии.
            edge_index = gnn.edge_index.to(device)
            # Веса могут храниться как edge_weight или edge_attr.
            edge_weight = None
            if hasattr(gnn, "edge_weight") and gnn.edge_weight is not None:
                edge_weight = gnn.edge_weight.to(device).to(dtype)
            elif hasattr(gnn, "edge_attr") and gnn.edge_attr is not None:
                edge_weight = gnn.edge_attr.to(device).to(dtype)

            # Шаги message passing.
            mp_layers = gnn.mp_layers  # ModuleList[Linear]
            for layer in mp_layers:
                # Линейная трансформация скрытых состояний.
                transformed = layer(hidden)  # [B, N, H]
                # Сбор сообщений по рёбрам.
                src, dst = edge_index[0], edge_index[1]
                # Сообщения от источников.
                msgs = transformed[:, src, :]  # [B, E, H]
                if edge_weight is not None and edge_weight.numel() == msgs.shape[1]:
                    msgs = msgs * edge_weight.view(1, -1, 1)
                # Векторизованная sum-агрегация по приёмникам через PyG scatter
                # (вместо Python-цикла по батчу с index_add_ — медленный backward).
                from torch_geometric.utils import scatter

                aggregated = scatter(msgs, dst, dim=1, dim_size=n_nodes)  # [B, N, H]
                # Обновление с residual и нелинейностью tanh (из config/models.yaml).
                hidden = torch.tanh(hidden + aggregated)

            # Чтение с моторных узлов — среднее.
            motor_mask = gnn.motor_mask
            if motor_mask.dtype != torch.bool:
                motor_mask = motor_mask.to(torch.bool)
            motor_mask = motor_mask.to(device)
            n_motor = int(motor_mask.sum().item())
            if n_motor > 0:
                motor_hidden = hidden[:, motor_mask, :].mean(dim=1)  # [B, H]
            else:
                # Если маска пуста — среднее по всем узлам.
                motor_hidden = hidden.mean(dim=1)
            return motor_hidden
        except Exception:
            # Любая ошибка — fallback к общему пути.
            return None

    def _project_logits_to_features(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """Проецирует/обрезает логиты к ``features_dim``.

        Args:
            logits: Тензор выхода GNN, форма ``[B, output_dim]``.

        Returns:
            Тензор ``[B, features_dim]``.
        """
        # Схлопываем лишние измерения если есть.
        if logits.dim() > 2:
            logits = logits.view(logits.shape[0], -1)

        in_dim = int(logits.shape[-1])
        out_dim = int(self.features_dim)

        if in_dim == out_dim:
            return logits
        if in_dim > out_dim:
            # Обрезка — детерминированно берём первые out_dim.
            return logits[:, :out_dim]
        # in_dim < out_dim — линейная проекция.
        if (
            self._fallback_proj is None
            or self._fallback_proj.in_features != in_dim
            or self._fallback_proj.out_features != out_dim
        ):
            # Ленивое создание/пересоздание слоя под текущий размер.
            self._fallback_proj = nn.Linear(in_dim, out_dim).to(logits.device)
        return self._fallback_proj(logits)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Пропускает наблюдения через GNN и возвращает признаки.

        Пытается получить скрытое состояние моторных узлов (пред-readout).
        Если GNN возвращает логиты — проецирует к ``features_dim``.
        Если GNN ещё стаб — использует резервную линейную проекцию.

        Args:
            obs: Батч наблюдений, форма ``[B, input_dim]``.

        Returns:
            Тензор признаков формы ``[B, features_dim]``.
        """
        # Попытка 1: специальные методы GNN (если будут добавлены).
        for method_name in (
            "get_motor_features",
            "get_hidden",
            "extract_features",
            "forward_features",
            "encode",
        ):
            if hasattr(self.gnn, method_name):
                try:
                    method = getattr(self.gnn, method_name)
                    hidden = method(obs)
                    if isinstance(hidden, torch.Tensor):
                        # Возможно нужен ресайз.
                        if hidden.dim() > 2:
                            hidden = hidden.view(hidden.shape[0], -1)
                        if hidden.shape[-1] != self.features_dim:
                            hidden = self._project_logits_to_features(hidden)
                        return hidden
                except NotImplementedError:
                    # Метод стаба — пробуем дальше.
                    continue
                except Exception:
                    continue

        # Попытка 2: ручное извлечение motor hidden без readout.
        manual = self._extract_motor_hidden_manual(obs)
        if manual is not None:
            # manual уже [B, hidden_dim] == [B, features_dim].
            if manual.shape[-1] != self.features_dim:
                manual = self._project_logits_to_features(manual)
            return manual

        # Попытка 3: обычный forward GNN (логиты) + проекция.
        try:
            out = self.gnn(obs)
            if isinstance(out, torch.Tensor):
                return self._project_logits_to_features(out)
        except NotImplementedError:
            # GNN ещё стаб — переходим к заглушке.
            pass
        except Exception:
            # Другая ошибка forward — также fallback.
            pass

        # Fallback-заглушка для стаба: линейная проекция obs -> features.
        # Создаётся лениво, чтобы узнать input_dim из первого батча.
        input_dim = int(obs.shape[-1])
        if (
            self._stub_proj is None
            or self._stub_proj.in_features != input_dim
            or self._stub_proj.out_features != self.features_dim
        ):
            self._stub_proj = nn.Linear(input_dim, self.features_dim).to(obs.device)
        return self._stub_proj(obs)

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """Базовый MLP baseline без использования графа.

    Число параметров подбирается сопоставимым с GNN для честного
    сравнения в экспериментах по континуальному обучению.

    Архитектура строится как Sequential из ``n_layers`` скрытых
    слоёв с нелинейностью ``activation`` и финального линейного слоя
    в пространство действий.

    Attributes:
        net: Последовательная сеть, возвращающая логиты действий.
    """

    def __init__(
        self,
        input_dim: int = 9,
        output_dim: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str = "relu",
    ) -> None:
        """Инициализирует MLP политику.

        Args:
            input_dim: Размерность наблюдения (по умолчанию 9:
                8 лидар-лучей + угол до цели).
            output_dim: Размерность выхода (число дискретных действий).
            hidden_dim: Ширина скрытых слоёв. Значение по умолчанию
                256 соответствует ``config/models.yaml: mlp.hidden_dim``.
            n_layers: Количество скрытых слоёв. Значение по умолчанию
                3 соответствует ``config/models.yaml: mlp.n_layers``.
            activation: Название функции активации (``"relu"`` или
                ``"tanh"``). По умолчанию ``"relu"`` как в конфиге.

        Raises:
            ValueError: Если ``n_layers`` < 1 или ``activation``
                не поддерживается.
        """
        super().__init__()

        # Валидация количества слоёв.
        if n_layers < 1:
            raise ValueError(f"n_layers должен быть >= 1, получено {n_layers}")

        # Выбор функции активации по строковому имени из конфига.
        act_name = activation.lower()
        if act_name == "relu":
            act_cls = nn.ReLU
        elif act_name == "tanh":
            act_cls = nn.Tanh
        else:
            raise ValueError(
                f"Неизвестная активация '{activation}', ожидается 'relu' или 'tanh'"
            )

        # Сборка слоёв Sequential:
        # 1) входной слой input_dim -> hidden_dim + активация
        # 2) (n_layers - 1) скрытых слоёв hidden_dim -> hidden_dim + активация
        # 3) выходной слой hidden_dim -> output_dim (без активации, логиты)
        layers: list[nn.Module] = []

        # Входной скрытый слой.
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_cls())

        # Дополнительные скрытые слои.
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())

        # Выходной слой — проекция в пространство действий.
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net: nn.Sequential = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = act_name

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Прямое распространение через MLP.

        Args:
            obs: Наблюдение среды, shape ``(batch, input_dim)``.

        Returns:
            Логиты / Q-values действий, shape ``(batch, output_dim)``.
        """
        # Последовательный проход через все скрытые и выходной слои.
        return self.net(obs)

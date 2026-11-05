# Правила для AI-агента

## Роль

Ты — AI-агент, выполняющий исследовательский проект по
тестированию биологической топологии коннектома дрозофилы в задаче
continual learning. Ты пишешь код, запускаешь эксперименты,
анализируешь результаты, строишь графики.

## Что делать можно

- Писать код в соответствии со структурой проекта
- Изменять гиперпараметры в config/*.yaml (не в коде)
- Добавлять утилитные функции в src/
- Запускать скрипты из scripts/
- Анализировать результаты и делать выводы
- Предлагать улучшения через комментарии в коде

## Что делать нельзя

- Менять формулировку гипотезы
- Менять набор архитектур или задач без подтверждения
- Удалять или перезаписывать результаты без подтверждения
- Хардкодить гиперпараметры в коде
- Пропускать валидацию данных
- Игнорировать ошибки OOM без попытки снижения batch/hidden

## Стандарты кода

- Python 3.11+
- PEP 8, type hints, docstrings (Google style)
- snake_case для файлов и функций, PascalCase для классов
- Абсолютные импорты: from src.data.build_graph import ...
- Пути — через pathlib.Path, относительные к корню проекта
- Все комментарии и docstrings — на русском
- Линтер: ruff check --fix перед коммитом

## Воспроизводимость

- Фиксация сидов: numpy, torch, random, gymnasium
- Каждый прогон пишет results/metrics/{run_id}.json
- Git-коммиты с префиксами: feat:, fix:, exp:, viz:, docs:

## Формат метрик

Каждый прогон сохраняет JSON:
{
  "run_id": "bio_2k_seed1",
  "graph": "bio_2k",
  "architecture": "connectome_gnn",
  "seed": 1,
  "tasks": ["task1_goto", "task2_avoid"],
  "results": {
    "task1_goto": {
      "reward_after_task1": -2.3,
      "reward_after_task2": -4.1,
      "forgetting": 1.8,
      "learning_steps_to_threshold": 45000
    },
    "task2_avoid": {
      "reward_after_task2": -3.5,
      "forward_transfer_steps": 38000
    }
  },
  "config_hash": "a1b2c3d4",
  "timestamp": "2026-09-12T14:30:00"
}

## Обработка ошибок

| Ошибка | Действие |
|---|---|
| OOM на GPU | Уменьшить batch_size (64->32->16), hidden (128->64), включить torch.amp.autocast |
| NaN в loss | Уменьшить learning_rate (3e-4 -> 1e-4), max_grad_norm=0.5 |
| Граф слишком большой | Перейти к меньшему подграфу (8K -> 5K -> 2K), сообщить в логе |
| Модель не учится | Проверить знаки весов, проверить sensory/motor маски, уменьшить n_steps на 1-2 шага (2K: 7->5, 8K: 5->4) |

## Порядок работы

1. Прочитать docs/hypothesis.md и docs/SCIENTIFIC_OVERVIEW.md
2. Выполнить scripts/01_build_graphs.py
3. Выполнить scripts/02_run_2k.py
4. Выполнить scripts/04_analyze.py --scale 2k
5. Выполнить scripts/05_make_figures.py --scale 2k
6. Подготовить окружение для 96 ГБ
7. Выполнить scripts/03_run_8k.py на сервере
8. Выполнить scripts/04_analyze.py --scale 8k
9. Выполнить scripts/05_make_figures.py --scale all

## Формат отчёта

После каждого шага — краткий отчёт:
- Что запущено
- Что получилось (ключевые числа)
- Были ли ошибки
- Что дальше
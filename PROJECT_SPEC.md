# Bio-Connectome CL: Единая спецификация проекта

## 1. Вводные данные

### 1.1. Источник: FlyWire коннектом

| Параметр | Значение |
|---|---|
| Проект | FlyWire (flywire.ai) |
| Версия данных | v783 |
| Организм | Drosophila melanogaster, взрослая самка |
| Объём | ~139 000 нейронов, ~54,5 млн синапсов |
| Формат выгрузки | CSV (связи), CSV (аннотации нейронов) |
| Лицензия | Открытая, для научных целей |

### 1.2. Файлы для скачивания

| Файл | Содержание | Откуда |
|---|---|---|
| connections.csv | pre_root_id, post_root_id, syn_count, nt_type | flywire.ai -> Downloads, или через caveclient |
| neuron_annotations.csv | root_id, region, class, super_class, hemilineage, nt_type | flywire.ai -> Annotations |

### 1.3. Целевая выборка: зрительный путь

| Регион | Сокращение | Роль в графе |
|---|---|---|
| Lamina | LA | Сенсорный вход |
| Medulla | ME | Промежуточная обработка |
| Lobula | LO | Высшая зрительная интеграция |
| Lobula Plate | LOP | Детекция движения |
| Моторные нейроны | — | Выходной слой (class == "motor") |

### 1.4. Два масштаба подграфа

| Масштаб | Узлов | Рёбер (ожидаемо) | Где используется |
|---|---|---|---|
| 2K | ~2 000 | ~150 000-300 000 | 16 ГБ VRAM, основная статистика |
| 8K | ~8 000 | ~800 000-1 500 000 | 96 ГБ VRAM, проверка масштабируемости |

### 1.5. Признаки рёбер

| Поле | Тип | Значение |
|---|---|---|
| pre_root_id | int | ID пресинаптического нейрона |
| post_root_id | int | ID постсинаптического нейрона |
| weight | float | Знаковый: +syn_count для возбуждающих (acetylcholine, glutamate), -syn_count для тормозных (GABA) |
| nt_type | str | Нейромедиатор (для абляции unsigned) |

### 1.6. Признаки узлов

| Поле | Тип | Значение |
|---|---|---|
| node_id | int | Индекс 0..N-1 |
| region | str | LA / ME / LO / LOP / motor |
| role | str | sensory (LA), intrinsic (ME/LO/LOP), motor |
| module | int | ID модуля по Louvain (вычисляется на этапе анализа) |

---

## 2. Вычислительная инфраструктура

### 2.1. Три платформы

| Платформа | VRAM | Роль | ОС |
|---|---|---|---|
| ПК-1 | 8 ГБ | Разработка, отладка, анализ, визуализация | Windows |
| ПК-2 | 16 ГБ | Основные эксперименты (2K), MLP baseline | Windows |
| Сервер | 96 ГБ | Проверка масштабируемости (8K), одна сессия до 12 ч | любая |

### 2.2. Окружение на Windows

- Python 3.11+ (winget install Python.Python.3.11 или с python.org)
- Виртуальное окружение: python -m venv .venv
- Активация: .venv\Scripts\activate
- CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121
- Пути в коде — через pathlib.Path, не сырые строки со слешами

### 2.3. Запуск на 96 ГБ сервере

1. За день до сессии: на 16 ГБ ПК проверить, что 03_run_8k.py --dry-run не падает
2. На сервере: создать venv, установить зависимости из requirements.txt
3. Запустить python scripts/03_run_8k.py --arch bio,rewired
4. После завершения: скопировать results/metrics/8k_*.json и чекпоинты на локальный ПК

---

## 3. Структура проекта

    bio-connectome-cl/
    ├── README.md
    ├── PROJECT_SPEC.md
    ├── requirements.txt
    ├── .gitignore
    ├── config/
    │   ├── graphs.yaml
    │   ├── env.yaml
    │   ├── models.yaml
    │   ├── training.yaml
    │   └── viz.yaml
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── data_registry.md
    ├── src/
    │   ├── __init__.py
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── download.py
    │   │   ├── build_graph.py
    │   │   ├── variants.py
    │   │   └── metrics.py
    │   ├── env/
    │   │   ├── __init__.py
    │   │   ├── fly_nav.py
    │   │   ├── task1_goto.py
    │   │   └── task2_avoid.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── gnn_controller.py
    │   │   ├── mlp_policy.py
    │   │   └── wrappers.py
    │   ├── training/
    │   │   ├── __init__.py
    │   │   ├── train_single.py
    │   │   ├── train_continual.py
    │   │   └── eval.py
    │   ├── analysis/
    │   │   ├── __init__.py
    │   │   ├── cl_metrics.py
    │   │   ├── module_activity.py
    │   │   └── stats.py
    │   └── viz/
    │       ├── __init__.py
    │       ├── learning_curves.py
    │       ├── forgetting_bars.py
    │       ├── module_heatmap.py
    │       └── scale_comparison.py
    ├── scripts/
    │   ├── 01_build_graphs.py
    │   ├── 02_run_2k.py
    │   ├── 03_run_8k.py
    │   ├── 04_analyze.py
    │   └── 05_make_figures.py
    ├── results/
    │   ├── logs/
    │   ├── checkpoints/
    │   ├── metrics/
    │   └── figures/
    └── docs/
        ├── hypothesis.md
        ├── experiment_plan.md
        ├── agent_rules.md
        └── SCIENTIFIC_OVERVIEW.md

---

## 4. Конфигурационные файлы

Все гиперпараметры вынесены в config/*.yaml. Хардкод в коде запрещён.

---

## 5. Требования к визуализации

### 5.1. Общий формат

| Параметр | Значение |
|---|---|
| Формат | PNG, 300 dpi |
| Размер | 10x4 дюйма (широкие), 6x6 (квадратные) |
| Шрифт | DejaVu Sans, 12 (оси), 14 (заголовок), 10 (легенда) |
| Палитра | tab10 (matplotlib), цвет-кодирование по архитектуре |
| Error bars | 95% CI (или std при n=3) |
| Сетка | alpha=0.3, серая |

### 5.2. Цветовое кодирование

| Архитектура | Цвет (hex) | Метка |
|---|---|---|
| Bio Connectome (signed) | #1f77b4 | Bio (signed) |
| Rewired (degree-preserving) | #ff7f0e | Rewired |
| Random ER | #2ca02c | Random ER |
| Bio (unsigned) | #d62728 | Bio (unsigned) |
| MLP | #9467bd | MLP |

### 5.3. Конкретные графики

Figure 1: fig1_learning_curves — Кривые обучения (2x2 grid, T1/T2 на 2K/8K)
Figure 2: fig2_forgetting_bars — Катастрофическое забывание (bar chart, 2K слева, 8K справа)
Figure 3: fig3_module_heatmap — Модульная специализация (heatmap, модули x задачи)
Figure 4: fig4_scale_comparison — Сравнение масштабов (2K vs 8K)
Figure 5: fig5_structure_metrics — Структурные метрики графов (heatmap-таблица)

---

## 6. Гипотеза

Основная: Модульная топология зрительного пути дрозофилы (2K-8K нейронов) при последовательном обучении двум задачам навигации уменьшает катастрофическое забывание по сравнению с графами той же плотности, но без биологической модульности.

H1: Bio < Rewired (важна конкретная схема связей)
H2: Bio < Random ER (важна структура вообще)
H3: Signed > Unsigned (роль тормозных связей критична)
H4: Эффект масштабируется с 2K до 8K узлов

---

## 7. Правила для AI-агента

Роль: AI-агент, выполняющий исследовательский проект по тестированию биологической топологии коннектома дрозофилы в задаче continual learning.

Можно: писать код, менять config/*.yaml, добавлять функции в src/, запускать scripts/, анализировать результаты.
Нельзя: менять гипотезу, менять набор архитектур без подтверждения, удалять результаты, хардкодить параметры, пропускать валидацию данных.
Стандарты: PEP 8, type hints, docstrings (Google style), комментарии на русском, pathlib.Path, абсолютные импорты.
Воспроизводимость: фиксация сидов (numpy, torch, random, gym), JSON-метрики после каждого прогона.
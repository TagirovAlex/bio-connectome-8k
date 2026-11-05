# Bio-Connectome CL: топология коннектома дрозофилы и катастрофическое забывание

Исследование: снижает ли модульная топология зрительного пути дрозофилы
(FlyWire v783) катастрофическое забывание в последовательном обучении с
подкреплением (continual RL) по сравнению с контрольными графами той же
плотности и размера.

**Статус:** 2K-эксперимент завершён (15/15 прогонов). 8K-эксперимент не
выполнен. Итоговая статья — [`docs/ARTICLE_2K.md`](docs/ARTICLE_2K.md).

---

## Главный результат (2K)

На масштабе 2K **эффект не обнаружен**: медианы forgetting у биологического
графа и контролей статистически неразличимы, все 95 % bootstrap-интервалы
разницы медиан пересекают ноль, Wilcoxon даёт `p = 0.827` во всех сравнениях.

| Архитектура | Forgetting, медиана [IQR] |
|---|---:|
| `bio` (signed) | −1.748 [2.170] |
| `rewired` | +0.101 [1.116] |
| `random_er` | −1.849 [1.375] |
| `unsigned` | +0.386 [0.238] |
| `mlp` | −1.310 [0.850] |

Важные оговорки:

- **Забывание ≈ 0 у всех архитектур** — пара задач `goto → avoid` почти не
  провоцирует забывание (потолок измерения).
- **2 из 15 прогонов (13 %)** не сошлись и дают значения порядка −22 и −146 —
  это срыв обучения, а не забывание.
- **Мощность при n = 3 низкая**: 0.103 для эффекта 1.0 единицы; для
  `power = 0.8` нужно ~34–40 сидов.
- Корректная формулировка — **«эффект не обнаружен»**, а не «доказано
  отсутствие эффекта».
- Результат относится к классу моделей «замороженная топология + обучение
  только весов», а не к биологическому коннектому (см.
  [`docs/LIMITATIONS_AND_POWER.md`](docs/LIMITATIONS_AND_POWER.md), раздел 10).

Подробности: [`docs/ARTICLE_2K.md`](docs/ARTICLE_2K.md),
[`docs/LIMITATIONS_AND_POWER.md`](docs/LIMITATIONS_AND_POWER.md).

---

## Структура репозитория

```
.
├── README.md                  # этот файл
├── LICENSE                    # MIT
├── CITATION.cff               # как цитировать
├── requirements.txt           # полный набор (разработка)
├── requirements-run.txt       # лёгкий набор для прогонов/Docker
├── Dockerfile                 # образ для 8K-прогона (torch cu128)
│
├── config/                    # все гиперпараметры (YAML)
│   ├── graphs.yaml
│   ├── env.yaml
│   ├── models.yaml
│   ├── training.yaml
│   └── viz.yaml
│
├── data/
│   ├── README.md              # происхождение данных и как восстановить
│   ├── data_registry.md       # реестр CSV FlyWire
│   ├── raw/                   # сырые CSV (не в git, ~290 МБ)
│   └── processed/             # готовые графы *.pt (в git, ~14 МБ)
│
├── docs/
│   ├── ARTICLE_2K.md          # итоговая статья
│   ├── LIMITATIONS_AND_POWER.md
│   ├── SCIENTIFIC_OVERVIEW.md
│   ├── hypothesis.md
│   ├── experiment_plan.md
│   ├── RUNBOOK.md
│   ├── RUNBOOK_8K_SERVER.md
│   ├── agent_rules.md
│   └── archive/               # ранние черновики
│
├── results/
│   ├── README.md              # карта результатов и их статус
│   ├── metrics/               # JSON-метрики прогонов
│   └── figures/               # PNG-графики
│
├── scripts/                   # пайплайн 01–07 + серверные скрипты
│   ├── 01_build_graphs.py
│   ├── 02_run_2k.py
│   ├── 03_run_8k.py
│   ├── 04_analyze.py
│   ├── 05_make_figures.py
│   ├── 06_robust_stats.py
│   ├── 07_power_analysis.py
│   ├── bootstrap.sh
│   ├── RUN_8K_SERVER.sh
│   ├── RUN_8K_SERVER_DOCKER.sh
│   ├── run_8k_inside.sh
│   ├── cleanup_server.sh
│   └── 2k_runs/               # PowerShell-обёртки прогонов
│
└── src/                       # библиотека: данные, модели, среда, анализ, viz
```

---

## Воспроизводимость

Окружение, на котором получены результаты (Windows, глобальный Python):

- Python 3.11, PyTorch 2.11.0+cu128 (Blackwell sm_120), torch-geometric 2.8.0
- stable-baselines3 2.3.2, gymnasium 0.29.1, numpy 2.4.6, scipy, matplotlib, seaborn

Установка:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Пайплайн (порядок важен):

```bash
python scripts/01_build_graphs.py          # графы -> data/processed/*.pt
python scripts/02_run_2k.py --all          # 5 арх × 3 сида -> results/metrics/2k_*.json
python scripts/04_analyze.py --scale 2k    # -> 2k_summary.json
python scripts/05_make_figures.py --scale 2k
python scripts/06_robust_stats.py          # -> 2k_robust.json
python scripts/07_power_analysis.py        # -> 2k_power.json
```

Все гиперпараметры — в `config/*.yaml`; сиды `[1, 2, 3]`;
`config_hash = 7d6f4e48` соответствует коду, на котором получены результаты.

---

## Данные

- **Сырые CSV FlyWire v783** (`data/raw/`, ~290 МБ) в репозиторий не входят —
  см. [`data/README.md`](data/README.md) и [`data/data_registry.md`](data/data_registry.md).
- **Готовые графы** (`data/processed/*.pt`, ~14 МБ) включены, чтобы
  эксперименты 2K/8K воспроизводились без повторной сборки из CSV.

---

## Цитирование

См. [`CITATION.cff`](CITATION.cff).

## Лицензия

MIT — см. [`LICENSE`](LICENSE).

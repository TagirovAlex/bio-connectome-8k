# RUNBOOK — продолжение работы без передачи контекста

Цель: новая сессия opencode может продолжить проект по этому файлу. Окружение и данные проверены 2026-09-16; состояние экспериментов обновлено 2026-11-05 (2K перезапущен на фикшенном коде и завершён 15/15, следующий этап — 8K).

## 1. Окружение

- GPU: **NVIDIA RTX 5060 Ti 16GB** (Blackwell sm_120), драйвер 595.79, CUDA 13.2.
- Python: **глобальный** `c:\Python311\python.exe` (3.11.9) — `.venv` не используется.
- PyTorch: `torch 2.11.0+cu128` — cu121-колеса на Blackwell sm_120 **не работают**, не переустанавливать.

Проверка CUDA:

```powershell
c:\Python311\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Ожидаемо: `2.11.0+cu128 True NVIDIA GeForce RTX 5060 Ti`.

## 2. Данные (на месте, не перекачивать)

- `data/raw/` (~290 МБ): connections.csv (261 МБ, **Filtered**, 5 342 446 строк), classification.csv, nt.csv, consolidated_cell_types.csv, neuron_annotations.csv (мердж classification+nt). `check_raw_files(data/raw)` → True.
- `data/processed/`: bio_2k.pt, bio_8k.pt, rewired_2k.pt, rewired_8k.pt, random_er_2k.pt, unsigned_2k.pt — все прошли валидацию (isolated=0, sens 50, motor 20, NaN нет).
- Состав и метрики: `data/data_registry.md`.

## 3. ВАЖНЫЕ предупреждения

- `results/metrics/2k_*.json` — **реальные прогоны (15/15)**, поле `smoke_stub` отсутствует, `timesteps = 200000` у обеих задач. Перезапускать их без причины не нужно; повторный запуск перезапишет файлы.
- `results/metrics/8k_*.json` — **СТАБЫ** (старые smoke-заглушки, `smoke_stub: True`). Реальный прогон 8K их перезапишет. Не путать с готовыми графами `bio_8k.pt` / `rewired_8k.pt`.
- `scripts/02_run_2k.py`: на CUDA идёт реальное обучение (PPO 200K шагов/задача, ~2-3 ч на прогон). Stub-ветка срабатывает ТОЛЬКО без CUDA (`not _has_cuda`) или при `total_timesteps<=2048`.
- `scripts/01_build_graphs.py`: без `--rebuild` при наличии `.pt` ПРОПУСКАЕТ пересборку; при пустом `data/raw/` висит на download. Пересобирать только с `--rebuild` и при наличии данных.
- `results/` не удалять и не перезаписывать без подтверждения (правило AGENTS.md).
- Окружение для CUDA: torch ставится через `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.

## 4. Прогон 2K — завершён

> Обновление 2026-09-17 09:00: message passing оптимизирован (vectorized PyG
> scatter вместо развёрнутого [B,E,H] индекса) — скорость выросла ~15x.
> Фактический замер на RTX 5060 Ti: полный прогон (2 задачи x 200K шагов +
> 22 eval по 30 эпизодов) ~2.3-2.5 ч на GNN-архитектуру.

> Обновление 2026-11-05 (P0-фикс GNN): первый 2K-прогон давал отрицательные
> награды у GNN-архитектур. Причина — неверные гиперпараметры message passing.
> Фикс в `src/models/gnn_controller.py`: `n_propagation_steps` {2k: 7, 8k: 5}
> вместо 3 и нормировка весов рёбер (`normalize_edge_weights`, mean|w| → 0.5),
> применено в `scripts/02_run_2k.py` и `scripts/03_run_8k.py`. После фикса
> **все 15 прогонов 2K перезапущены и завершены** на одной машине без сбоев.
> Числа первого (до-фиксного) прогона устарели — ниже только актуальные.

### Статус: ЗАВЕРШЁН, актуальный прогон (обновлено 2026-11-05)

Все 15 прогонов (5 архитектур × 3 сида) — реальные, на фикшенном коде: у каждого `2k_*.json` поле `smoke_stub` отсутствует, `timesteps = 200000` на обеих задачах, `config_hash = 7d6f4e48`. Прогоны завершены 05.11 04:20:43, машина работала без сбоев.

| Архитектура | Forgetting seed1 | seed2 | seed3 | mean ± std |
|---|---|---|---|---|
| bio | −1.75 | −2.45 | +1.89 | −0.77 ± 2.33 |
| mlp | −0.29 | −1.31 | −1.99 | −1.20 ± 0.86 |
| random_er | −1.85 | +0.48 | −2.27 | −1.21 ± 1.48 |
| rewired | +1.22 | −22.14 | −1.02 | −7.31 ± 12.9 |
| unsigned | +0.62 | +0.15 | −146.08 | −48.4 ± 84.6 |

Forgetting = R(T1|после T1) − R(T1|после T2). Wilcoxon bio vs остальных: p = 0.827 во всех сравнениях — значимых различий нет.

Ключевой результат: после фикса GNN все GNN-архитектуры (bio, unsigned, частично rewired) учатся — награды положительные, forgetting вблизи 0. Исключение — два seed-выброса, где модель не выучила T1: `rewired s2` (R(T1) = −142.96) и `unsigned s3` (forgetting −146.08). Именно они раздувают std у rewired (−7.31±12.9) и unsigned (−48.4±84.6). **Обрабатывать честно**: учитывать при интерпретации, но не как эффект топологии; дополнительно сообщить медианы/результат без выбросов (все ≈0).

### Команды повторного запуска (только при необходимости)

Перезапуск перезаписывает существующие `2k_*.json`. Вариант A — полный, 15 прогонов (5 арх × 3 сида), ~30-45 ч:

```powershell
cd C:\Github\bio-connectome-cl
c:\Python311\python.exe scripts/02_run_2k.py --all
```

Вариант B — сокращённый, 5 прогонов, ~12-15 ч:

```powershell
c:\Python311\python.exe scripts/02_run_2k.py --arch bio,rewired,random_er,unsigned,mlp --seed 1
```

Результат: `results/metrics/2k_*.json` (схема `{arch}_2k_seed{N}.json`).

## 5. Анализ и графики (2K — выполнено 2026-11-05)

Анализ 2K прогнан на всех 15 файлах (фикшенный прогон), результаты на месте (пересозданы 05.11 07:18):

- `results/metrics/2k_summary.json` — mean±std по архитектурам + Wilcoxon-тесты vs bio.
- `results/figures/fig1_learning_curves.png` — кривые обучения по задачам.
- `results/figures/fig2_forgetting_bars.png` — забывание по архитектурам.
- `results/figures/fig3_module_heatmap.png` — модульная структура/heatmap.

Перегенерация (перезапишет summary и графики):

```powershell
c:\Python311\python.exe scripts/04_analyze.py --scale 2k
c:\Python311\python.exe scripts/05_make_figures.py --scale 2k
```

## 6. 8K — следующий этап (16GB dry-run → 96GB full)

Графы готовы: `data/processed/bio_8k.pt`, `data/processed/rewired_8k.pt` (валидацию прошли). Полного прогона 8K ещё не было: текущие `8k_*.json` — dry-run-заглушки от 03.11 (проверка пайплайна, `smoke_stub` не реальный прогон), их перезапишет полный прогон. 8K идёт на 96 ГБ (bio + rewired, `n_steps=5`) и использует тот же фикс GNN, что и 2K.

Порядок:

1. **Dry-run на 16 ГБ** — проверка пайплайна без длительного обучения (за день до сессии на 96 ГБ):

```powershell
c:\Python311\python.exe scripts/03_run_8k.py --arch bio,rewired --dry-run
```

2. **Полный прогон на 96 ГБ** — 2 прогона (bio, rewired), 1 сид, ~6 ч на прогон:

```powershell
c:\Python311\python.exe -u scripts/03_run_8k.py --arch bio,rewired
```

3. Анализ и графики:

```powershell
c:\Python311\python.exe scripts/04_analyze.py --scale 8k
c:\Python311\python.exe scripts/05_make_figures.py --scale all
```

Результат: `results/metrics/8k_bio_seed1.json`, `8k_rewired_seed1.json`, `8k_summary.json`, `fig4-5.png`.

> Примечание 2026-11-05: проблема «GNN не выучивают задачу» (отрицательные награды) была устранена P0-фиксом (увеличение `n_propagation_steps` до {2k:7,8k:5} + нормировка весов). На 2K после фикса GNN-архитектуры учатся. Блокировки интерпретации 8K по этой причине больше нет.

## 7. Полезные файлы

- `AGENTS.md` — окружение, пайплайн, конвенции (раздел «Актуальное окружение (2026-09)»).
- `data/data_registry.md` — состав и валидация данных.
- `docs/experiment_plan.md` — план, статус, таймлайн.
- `docs/agent_rules.md` — правила и схема `results/metrics/*.json`.
- `temp/real_smoke_test.py` — проверенный тест реального continual-обучения на CUDA.
- `config/*.yaml` — все гиперпараметры (не хардкодить).
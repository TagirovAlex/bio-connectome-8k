# Результаты

Карта всех результатов и их статус. Основные числа статьи — в
[`docs/ARTICLE_2K.md`](../docs/ARTICLE_2K.md).

## `metrics/` — JSON-метрики

### Итоговые (2K, используются в статье)

| Файл | Содержимое | Статус |
|---|---|---|
| `2k_{arch}_seed{1..3}.json` | 15 сырых прогонов (5 архитектур × 3 сида): награды, forgetting, learning curves, `elapsed_seconds`, `config_hash` | **финальные** |
| `2k_summary.json` | mean/std по архитектурам, Wilcoxon, `config_hash 7d6f4e48` | **финальные** |
| `2k_robust.json` | медианы/IQR/MAD, список выбросов, bootstrap CI | **финальные** |
| `2k_power.json` | анализ мощности: σ, мощность по n и δ, n для power = 0.8 | **финальные** |
| `graph_stats.json` | структурные метрики графов 2K и 8K (модульность, степень и т.д.) | **финальные** |

Архитектуры: `bio`, `rewired`, `random_er`, `unsigned`, `mlp`.

### Заглушки 8K (НЕ реальные результаты)

| Файл | Содержимое | Статус |
|---|---|---|
| `8k_bio_seed1.json`, `8k_rewired_seed1.json`, `8k_summary.json` | dry-run заглушки (`config_hash b097c956`, `n = 1`, `p = null`) | **не результаты**, 8K не выполнен |

### Сводка

| Файл | Содержимое |
|---|---|
| `all_summary.json` | агрегация 2K + 8K (8K — заглушки) |

### Архив

| Путь | Содержимое |
|---|---|
| `archive_pre_fix_2k/` | старые 2K-результаты **до** P0-фикса (не используются в анализе, сохранены для истории) |

## `figures/` — PNG (300 dpi)

| Файл | Что показывает |
|---|---|
| `fig1_learning_curves.png` | кривые обучения T1/T2 |
| `fig2_forgetting_bars.png` | забывание по архитектурам (с CI) |
| `fig3_module_heatmap.png` | модульная специализация T1/T2 |
| `fig4_scale_comparison.png` | сравнение 2K/8K |
| `fig5_structure_metrics.png` | структурные метрики графов |

## Как пересчитать

```bash
python scripts/04_analyze.py --scale 2k        # 2k_summary.json
python scripts/05_make_figures.py --scale 2k   # fig1–3
python scripts/06_robust_stats.py              # 2k_robust.json
python scripts/07_power_analysis.py            # 2k_power.json
```

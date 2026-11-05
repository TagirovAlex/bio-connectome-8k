# Реестр данных

## FlyWire v783 — фактические файлы (проверено 2026-09-16)

Все файлы скачаны **вручную (Codex)**, автоматический download в `01_build_graphs.py` не использовался.

| Файл | Источник | Размер | Строк | Дата | Версия |
|------|----------|--------|-------|------|--------|
| connections.csv | FlyWire v783, **Filtered connections** (не 2.3 ГБ full — достаточно) | 261 МБ | 5 342 446 | 2026-09-16 | v783 |
| classification.csv | FlyWire v783 | 10 МБ | 139 255 | 2026-09-16 | v783 |
| nt.csv | FlyWire v783 | 8.2 МБ | 139 255 | 2026-09-16 | v783 |
| consolidated_cell_types.csv | FlyWire v783 (используется редко) | 3.6 МБ | — | 2026-09-16 | v783 |
| neuron_annotations.csv | **МЕРДЖ** classification.csv + nt.csv (см. примечание) | 7.8 МБ | 139 255 | 2026-09-16 | — |

`check_raw_files(data/raw)` → True.

### Примечания

- **neuron_annotations.csv создан мерджем** (classification + nt): колонки `root_id, region, class, super_class, hemilineage, nt_type`. `region` выводится из `nt.group` (первый токен до `.`/`_`), `nt_type` переведён в полные имена (ACH→acetylcholine, GLUT→glutamate, GABA→GABA, SER→serotonin, DA→dopamine, OCT→octopamine).
- Распределение region: ME 66.7k, LO 14.5k, LA 7.4k, LOP 3.3k, GNG 5.4k, MB 5k, unknown 5k.
- connections.csv — **Filtered connections (261 МБ)**, НЕ полный датасет 2.3 ГБ; объёма достаточно для проекта.

### Колонки исходных файлов

- `connections.csv`: `pre_root_id, post_root_id, neuropil, syn_count, nt_type`. nt_type: ACH 3.2M / GABA 1.17M / GLUT 0.82M / OCT / SER.
- `classification.csv`: `root_id, flow, super_class, class, sub_class, hemilineage, side, nerve`. super_class: optic 77k, central 32k, sensory 17k, motor 110.
- `nt.csv`: `root_id, group, nt_type, nt_type_score, ...`. group — нейропил (ME, LO, LA, LOP и т.д.), nt_type: ACH/GLUT/GABA/SER/DA/OCT.

## Производные (data/processed/, пересобраны из реальных CSV)

| Файл | Узлов | Рёбер | Назначение |
|------|-------|-------|------------|
| bio_2k.pt | 2000 | 42 023 | Основная статистика |
| bio_8k.pt | 8000 | 224 567 | Проверка масштаба |
| rewired_2k.pt | 2000 | — | H1 baseline |
| rewired_8k.pt | 8000 | — | H1, масштаб |
| random_er_2k.pt | 2000 | — | H2 baseline |
| unsigned_2k.pt | 2000 | — | H3 абляция |

`unsigned_8k.pt` отсутствует — в плане 8K только bio+rewired (см. `docs/experiment_plan.md`).

## Валидация (все 6 графов прошли, 2026-09-16)

- [x] Нет изолированных узлов (isolated = 0)
- [x] Нет NaN в весах
- [x] sensory_mask непустой (>= 50 узлов) — sens 50
- [x] motor_mask непустой (>= 20 узлов) — motor 20
- [x] modularity score > 0.3 (для bio) — см. метрики ниже

### Фактические метрики (`results/metrics/graph_stats.json`)

| Граф | Modularity | mean_degree | density | reciprocity | avg_path |
|------|-----------|-------------|---------|-------------|----------|
| bio_2k | 0.684 | 42 | записано | записано | записано |
| rewired_2k | 0.28 | — | записано | записано | записано |
| random_er_2k | 0.154 | — | записано | записано | записано |
| unsigned_2k | 0.684 | — | записано | записано | записано |
| bio_8k | 0.641 | 56 | записано | записано | записано |
| rewired_8k | 0.19 | — | записано | записано | записано |

Полные значения density/reciprocity/avg_path — в самом `results/metrics/graph_stats.json`.
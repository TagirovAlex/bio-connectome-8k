# 8K continual-RL прогон (коннектом дрозофилы) — Docker-батч для 96 ГБ сервера

Автономный прогон 8K-эксперимента (bio, rewired) на GPU-сервере через Docker:
окружение в образе, обучение → анализ → фигуры → результаты уходят обратно в
этот git-репозиторий, машина выключается.

Подробности и запуск: `docs/RUNBOOK_8K_SERVER.md`.

## Содержимое
- `Dockerfile` — образ: Python 3.11 + torch(cu128) + код + 8K-графы.
- `scripts/run_8k_inside.sh` — точка входа контейнера (обучение+анализ+фигуры).
- `scripts/RUN_8K_SERVER_DOCKER.sh` — host-скрипт: docker+nvidia → build → run → push → poweroff.
- `data/processed/bio_8k.pt`, `rewired_8k.pt` — графы 8K (~10 МБ).
- `results/metrics/2k_*.json`, `2k_summary.json` — метрики 2K (для итоговых фигур).
- `config/`, `src/`, `scripts/03-05*.py` — код эксперимента.

## Быстрый старт (сервер Debian 13, 96 ГБ)
```bash
REPO_URL="https://github.com/TagirovAlex/bio-connectome-8k.git" \
GIT_TOKEN="ghp_..." \
bash scripts/RUN_8K_SERVER_DOCKER.sh 2>&1 | tee /opt/run8k/console.log
```
Собирает образ, обучает 8K, пушит `results/` обратно в репозиторий и выключает машину.
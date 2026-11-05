# RUNBOOK — автономный 8K-прогон на 96 ГБ сервере (Debian 13) через Docker

Цель: одна команда запускает всё на удалённом сервере, результаты возвращаются
в git, машина выключается — без ручного мониторинга (оплата по часам за включённую
машину). Окружение и данные упакованы в Docker-образ; на сервере ничего не нужно,
кроме docker + nvidia драйвера.

Репозиторий: `https://github.com/TagirovAlex/bio-connectome-8k` (публичный).

## Состав репозитория

- `Dockerfile` — образ: Python 3.11 + torch(cu128) + `requirements.txt` + код + 8K-графы.
- `scripts/run_8k_inside.sh` — точка входа контейнера.
- `scripts/RUN_8K_SERVER_DOCKER.sh` — host-скрипт (bootstrap сервера).
- `scripts/03_run_8k.py`, `04_analyze.py`, `05_make_figures.py` — прогон/анализ/фигуры.
- `data/processed/bio_8k.pt`, `rewired_8k.pt` — графы 8K (~10 МБ; CSV не нужны).
- `results/metrics/2k_*.json`, `2k_summary.json` — метрики 2K (для итоговых фигур).
- `config/`, `src/` — конфиги и код эксперимента.

## Поток (без мониторинга)

```
host RUN_8K_SERVER_DOCKER.sh
  1. apt: git/curl/ca-certificates/docker.io
  2. nvidia-container-toolkit (для --gpus)
  3. git clone https://github.com/TagirovAlex/bio-connectome-8k
  4. docker build -t run8k:latest .
  5. docker run --rm --gpus all -v $REPO/results:/opt/run8k/results run8k
       -> внутри: 03_run_8k (bio,rewired; 200K шагов×2) -> 04_analyze 8k
                  -> 05_make_figures all -> маркер results/_done
  6. git add/commit/push results/ -> PUSH_TO (GitHub)
  7. poweroff
```

`collect` даёт лучший-effort push: при любом сбое результаты и лог публикуются
(или грузятся tarball в `OUT_DIR`), и машина выключается.

## Запуск

На сервере (96 ГБ, Debian 13), от root/sudo:
```bash
GIT_TOKEN="ghp_..." bash scripts/RUN_8K_SERVER_DOCKER.sh 2>&1 | tee /opt/run8k/console.log
```
- `REPO_URL` по умолчанию = `https://github.com/TagirovAlex/bio-connectome-8k.git`.
- `GIT_TOKEN` — PAT со scope `repo` для записи результатов обратно в репозиторий.
  Альтернатива: задай `PUSH_TO` как ssh-remote с deploy-ключом (write).
- После того как в GitHub-репо появились `results/8k_*`, `8k_summary.json`,
  `fig4/5` и машина выключена — забрать на контрольной машине: `git pull`.

### Автозапуск по загрузке (опция)
Чтобы сервер сам стартовал работу при каждом включении (например, после перезапуска):
подними systemd-oneshot/cloud-init, который при boot запускает
`bash scripts/RUN_8K_SERVER_DOCKER.sh`. Скрипт идемпотентен: пересоздаёт клон,
пересобирает образ (из кэша — быстро), прогоняет заново и публикует новую порцию.

## Переменные host-скрипта

| Переменная | Назначение | Default |
|---|---|---|
| `REPO_URL` | git-адрес кода+данных | `https://github.com/TagirovAlex/bio-connectome-8k.git` |
| `PUSH_TO` | remote для записи результатов | `=${REPO_URL}` |
| `GIT_TOKEN` | PAT для push | — |
| `BRANCH` | ветка | `main` |
| `IMAGE` | имя Docker-образа | `run8k:latest` |
| `WORK` | рабочая директория | `/opt/run8k` |
| `OUT_DIR` | доп. каталог для tarball результатов | — |
| `NO_SHUTDOWN` | `1` → не выключать машину | `0` |

## Форс-мажоры и сбои

- **OOM во время обучения**: контейнер сам снижает `batch_size` 64→32 (→16 при
  повторном OOM) в `scripts/run_8k_inside.sh` и повторяет.
- **GPU не виден в контейнере**: убедись, что установлен `nvidia-container-toolkit`
  и `nvidia-smi` на хосте работает; без него `--gpus` не отдаст GPU.
- **Python-версия**: в образе фиксирована 3.11 (torch cu128-колеса для неё есть).
- **Сетевые ограничения сервера**: если сервер не тянет к GitHub — используй вариант
  с автономным бандлом (`git bundle`) и/или `OUT_DIR` (tarball), см. старый скрипт
  `scripts/RUN_8K_SERVER.sh` (без Docker, клонирует из `SRC_BUNDLE`).

## Контрольная машина — забрать результаты
```bash
git clone https://github.com/TagirovAlex/bio-connectome-8k.git   # или git pull
# опционально: загрузить таблицы/фигуры, интерпретировать
```
Рекомендуется затем локально собрать сводный отчёт (как для 2K) и зафиксировать
в `docs/experiment_plan.md` с датой.
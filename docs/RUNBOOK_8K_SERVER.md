# Запуск 8K-эксперимента на 96 ГБ сервере (Debian 13)

Автономный прогон: окружение → обучение 8K (bio, rewired) → анализ → графики →
пуш результатов в git → выключение машины (чтобы не платить за простой).

## Состав

- `scripts/RUN_8K_SERVER.sh` — главный bootstrap-скрипт (bash, Debian 13).
- `server-8k.bundle` — git-бандл всего репозитория с данными для прогона
  (код, конфиги, `data/processed/bio_8k.pt` + `rewired_8k.pt`, метрики/фигуры 2K
  для построения итоговых fig4-5). Графы 8K — всего ~10 МБ, тяжёлые CSV НЕ нужны.
- `docs/RUNBOOK_8K_SERVER.md` — этот файл.

## Как это работает (поток, без мониторинга)

```
ENV vars -> RUN_8K_SERVER.sh
    1. apt: git/curl/python3-venv/python3-pip/build-essential
    2. cloning код+данные: REPO_SOURCE_URI ИЛИ SRC_BUNDLE
    3. venv + torch(cu128) + requirements.txt
    4. полный прогон: python scripts/03_run_8k.py --arch bio,rewired  (200K шагов×2 задачи)
    5. python scripts/04_analyze.py --scale 8k
    6. python scripts/05_make_figures.py --scale all
    7. git add/commit результатов -> PUSH_TO (или tarball в OUT_DIR)
    8. poweroff
```

`trap EXIT` гарантирует, что даже при сбое обученные результаты и лог будут
сохранены (push/tarball) и машина выключится.

## Два способа — выбери один

### A. Через удалённый git (рекомендуется, чтобы пушить результаты обратно)

1. Создай приватный репозиторий на GitHub/GitLab, например `bio-8k-runs`.
2. Залей в него содержимое бандла:
   ```
   git clone temp/server_bundle_repo .   # локальная копия уже подготовлена
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
   (или `git pull` бандла: `git clone server-8k.bundle repo && git push origin main`)
3. На сервере (96 ГБ, Debian 13):
   ```
   sudo REPO_SOURCE_URI="https://github.com/<you>/bio-8k-runs.git" \
        PUSH_TO="https://github.com/<you>/bio-8k-runs.git" \
        GIT_TOKEN="ghp_..." bash scripts/RUN_8K_SERVER.sh 2>&1 | tee /opt/run8k/console.log
   ```
   ⚠️ Для push в ту же ветку `main`, с которой клонировали, понадобится токен с
   правом записи (`GIT_TOKEN`) — иначе push сделает сервер в *новую* ветку результата.
   Рекомендуется: клонировать с `PUSH_TO=<ваш-writable-remote>`, а read-источник
   держать отдельным токеном, либо сделать `PUSH_TO` ssh-remote с deploy-ключом.

4. Результаты и лог появятся в удалённом репозитории, машина выключится.

### B. Через бандл + смонтированный OUT_DIR (без интернета у сервера)

1. Передай бандл на сервер: `scp temp/server-8k.bundle root@SERVER:/opt/8k/`
2. Смонтируй каталог с результатами (e.g. в /out), запусти:
   ```
   sudo SRC_BUNDLE=/opt/8k/server-8k.bundle OUT_DIR=/out bash scripts/RUN_8K_SERVER.sh
   ```
3. Результаты (`results_8k_*.tar.gz`) и лог появятся в `/out`, машина выключится.

## Переменные (defaults в скрипте)

| Переменная      | Назначение                                            | Default       |
|-----------------|-------------------------------------------------------|---------------|
| `REPO_SOURCE_URI` | git-адрес чтения (код+данные)                     | — (или bundle)|
| `SRC_BUNDLE`    | путь к бандлу, если нет REPO_SOURCE_URI             | —             |
| `PUSH_TO`       | git-remote для записи результатов                   | — (тогда tarball) |
| `GIT_TOKEN`     | токен для HTTPS push                                | —             |
| `BRANCH`        | ветка                                                | `main`        |
| `WORK`          | рабочая директория                                   | `/opt/run8k`  |
| `OUT_DIR`       | каталог для tarball результатов                     | —             |
| `NO_SHUTDOWN`   | `1` → НЕ выключать машину (отладка)                 | `0`           |
| `PYTHON_BIN`    | интерпретатор для venv                               | автопоиск     |
| `TORCH_INDEX_URL`| индекс torch (cu128 — для Blackwell/Ada и новее)    | cu128         |

## Проверки/важно

- **CUDA**: сервер обязан иметь драйвер NVIDIA (проверяется в скрипте печатью
  `torch.cuda.is_available()` и объёма VRAM). Torch ставится с cu128 индекса.
- **Python**: предпочтительно 3.11/3.12. Если система 3.13 и torch-колёса cu128
  для неё недоступны — укажи `PYTHON_BIN=/usr/bin/python3.11` (или 3.12).
- **VRAM**: default `batch_size=64` (~60 ГБ). При OOM скрипт сам снижает
  `batch_size` 64→32 (→16 при повторном OOM) и повторяет.
- **Графы**: если `data/processed/*_8k.pt` случайно не присутствуют — прогон
  сгенерирует синтетику (логируется как WARN). Убедись, что графы в репозитории.
- **Повторный запуск**: скрипт идемпотентен — пересоздаёт `WORK/repo`, каждый раз
  обучает заново и коммитит новую порцию результатов.
- **Опция запуска по загрузке**: можно обернуть в systemd-oneshot или cloud-init
  `user-data`, чтобы скрипт стартовал автоматически при каждом включении машины.

## Забрать результаты (контрольная машина)

После того как лог «RUN_8K_SERVER finished» ушёл в репозиторий и машина выключилась:
```
git pull origin main        # получишь results/metrics/8k_*.json, 8k_summary.json, figures
```
Либо, при способе B, — из `OUT_DIR` (tarball + лог).
#!/usr/bin/env bash
# =============================================================================
# cleanup_server.sh — удаляет ТОЛЬКО артефакты проекта bio-connectome-8k.
#
# Что удаляется:
#   - контейнер(ы) и образ run8k:latest;
#   - build-кэш Docker (слои torch и пр.);
#   - рабочая папка /opt/run8k (репо, логи, результаты);
#   - nvidia-container-toolkit + его apt-репозиторий и ключ
#     (если не указан --keep-toolkit), с очисткой nvidia-runtime из daemon.json.
#
# Что НЕ трогается:
#   - Docker-движок (docker.io) — остаётся;
#   - ComfyUI, драйвер NVIDIA, git/curl/ca-certificates;
#   - предсуществующий битый nvidia-cuda.sources.disabled (оставляем отключённым).
#
# Запуск (лучше из /tmp, чтобы не удалять скрипт вместе с /opt/run8k):
#   curl -fsSL <raw-url>/scripts/cleanup_server.sh -o /tmp/cleanup_server.sh
#   sudo bash /tmp/cleanup_server.sh
#
# Флаги:
#   -y, --yes        не спрашивать подтверждение;
#   --keep-toolkit   оставить nvidia-container-toolkit (GPU в Docker);
#   -h, --help       справка.
# =============================================================================
set -uo pipefail

WORK="${WORK:-/opt/run8k}"
IMAGE="${IMAGE:-run8k:latest}"
ASSUME_YES=0
KEEP_TOOLKIT=0

for arg in "$@"; do
    case "$arg" in
        -y|--yes) ASSUME_YES=1 ;;
        --keep-toolkit) KEEP_TOOLKIT=1 ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "[cleanup] неизвестный аргумент: $arg" >&2; exit 2 ;;
    esac
done

# Работаем из корня, чтобы удаление /opt/run8k не ломало cwd.
cd / || true

SUDO=""
if [ "${EUID:-$(id -u)}" -ne 0 ]; then SUDO="sudo"; fi

log() { echo "[cleanup] $*"; }

if [ "${ASSUME_YES}" -ne 1 ]; then
    echo "Будет удалено:"
    echo "  - контейнеры/образ ${IMAGE}"
    echo "  - build-кэш Docker"
    echo "  - ${WORK}"
    if [ "${KEEP_TOOLKIT}" -eq 0 ]; then
        echo "  - nvidia-container-toolkit + его apt-репозиторий/ключ"
    else
        echo "  (nvidia-container-toolkit оставляем: --keep-toolkit)"
    fi
    echo "Docker-движок, ComfyUI, драйвер NVIDIA — НЕ трогаются."
    read -r -p "Продолжить? [y/N] " ans
    case "${ans}" in
        y|Y|yes|YES) ;;
        *) echo "Отменено."; exit 0 ;;
    esac
fi

# 1) Контейнер + образ + build-кэш (Docker-движок остаётся).
if command -v docker >/dev/null 2>&1; then
    ids="$(${SUDO} docker ps -aq --filter ancestor="${IMAGE}" 2>/dev/null || true)"
    if [ -n "${ids}" ]; then
        # shellcheck disable=SC2086
        ${SUDO} docker rm -f ${ids} 2>/dev/null || true
        log "удалены контейнеры образа ${IMAGE}"
    fi
    ${SUDO} docker rmi -f "${IMAGE}" 2>/dev/null || true
    ${SUDO} docker builder prune -af 2>/dev/null || true
    log "образ ${IMAGE} и build-кэш удалены"
else
    log "docker не найден — пропускаю образ/кэш"
fi

# 2) Рабочая папка проекта (репо, логи, результаты).
${SUDO} rm -rf "${WORK}"
log "удалено: ${WORK}"

# 3) nvidia-container-toolkit + apt-репозиторий (если не --keep-toolkit).
if [ "${KEEP_TOOLKIT}" -eq 0 ]; then
    ${SUDO} apt-get remove -y nvidia-container-toolkit 2>/dev/null || true
    ${SUDO} rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ${SUDO} rm -f /usr/share/keyrings/libnvidia-container.gpg

    # Убираем nvidia-runtime из daemon.json, чтобы Docker не ссылался на
    # удалённый nvidia-container-runtime.
    if [ -f /etc/docker/daemon.json ]; then
        ${SUDO} python3 - <<'PY' 2>/dev/null || true
import json

path = "/etc/docker/daemon.json"
try:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
except Exception:
    raise SystemExit(0)

runtimes = data.get("runtimes")
if isinstance(runtimes, dict) and "nvidia" in runtimes:
    runtimes.pop("nvidia", None)
    if not runtimes:
        data.pop("runtimes", None)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print("[cleanup] nvidia-runtime убран из /etc/docker/daemon.json")
PY
    fi

    ${SUDO} apt-get autoremove -y 2>/dev/null || true
    log "nvidia-container-toolkit и его apt-репозиторий удалены"
else
    log "nvidia-container-toolkit оставлен (--keep-toolkit)"
fi

# 4) Кэш скачанных apt-пакетов.
${SUDO} apt-get clean 2>/dev/null || true

# 5) Отчёт.
log "готово"
if command -v df >/dev/null 2>&1; then
    df -h / 2>/dev/null | tail -1
fi
if command -v docker >/dev/null 2>&1; then
    ${SUDO} docker system df 2>/dev/null || true
fi

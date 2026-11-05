#!/usr/bin/env bash
# =============================================================================
# RUN_8K_SERVER_DOCKER.sh — автономный 8K-прогон на 96GB-сервере (Debian 13)
# через Docker: docker+nvidia -> build -> train 8K -> analyze/figures -> push -> poweroff.
#
# Запуск (root или через sudo):
#   sudo GIT_TOKEN="ghp_..." bash scripts/RUN_8K_SERVER_DOCKER.sh 2>&1 | tee /opt/run8k/console.log
#
# Переменные:
#   REPO_URL    - git-адрес кода+данных (default: публичный репозиторий проекта).
#   BRANCH      - ветка (default: main).
#   PUSH_TO     - remote для записи результатов (default: REPO_URL).
#   GIT_TOKEN   - PAT для push.
#   WORK        - рабочая директория (default: /opt/run8k).
#   IMAGE       - имя образа (default: run8k:latest).
#   OUT_DIR     - доп. каталог для tarball результатов.
#   NO_SHUTDOWN - "1" -> не выключать машину по завершении (отладка).
#   SHUTDOWN_ON_ERROR - "1" -> гасить машину и при сбое настройки (default: 0).
# =============================================================================
set -uo pipefail

WORK="${WORK:-/opt/run8k}"
LOG="${WORK}/console.log"
REPO="${WORK}/repo"
BRANCH="${BRANCH:-main}"
IMAGE="${IMAGE:-run8k:latest}"
NO_SHUTDOWN="${NO_SHUTDOWN:-0}"
SHUTDOWN_ON_ERROR="${SHUTDOWN_ON_ERROR:-0}"
REPO_URL="${REPO_URL:-https://github.com/TagirovAlex/bio-connectome-8k.git}"
PUSH_TO="${PUSH_TO:-${REPO_URL}}"
OUT_DIR="${OUT_DIR:-}"

mkdir -p "${WORK}"
log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }
warn() { echo "[$(date -Is)] WARN: $*" | tee -a "${LOG}"; }

# Публикуем результаты (лучший-effort). Вызывается только после успешного прогона.
collect() {
    cd "${REPO}"
    git add -A results/ 2>/dev/null || true
    git -c user.name="server-runner" -c user.email="runner@localhost" \
        commit -m "8K results $(date -Is)" 2>/dev/null || warn "ничего нового для коммита"
    local url="${PUSH_TO}"
    if [ -n "${GIT_TOKEN:-}" ]; then
        url="${PUSH_TO/https:\/\//https://x-access-token:${GIT_TOKEN}@}"
    fi
    git remote remove origin 2>/dev/null || true
    git remote add origin "${url}"
    git push -u origin "${BRANCH}" 2>&1 | tee -a "${LOG}" || warn "git push failed"
    if [ -n "${OUT_DIR:-}" ]; then
        mkdir -p "${OUT_DIR}"
        local tb="${OUT_DIR}/results_8k_$(date +%Y%m%d_%H%M%S).tar.gz"
        log "export tarball -> ${tb}"
        (cd "${REPO}" && tar --exclude='results/logs' --exclude='results/checkpoints' --exclude='.git' -czf "${tb}" results/) || true
        cp "${LOG}" "${OUT_DIR}/console_$(date +%Y%m%d_%H%M%S).log" 2>/dev/null || true
    fi
}

shutdown_machine() {
    if [ "${NO_SHUTDOWN}" = "1" ]; then log "NO_SHUTDOWN=1 — не выключаю"; return 0; fi
    log "shutdown in 5s..."
    sleep 5
    (systemctl poweroff --no-block 2>/dev/null \
        || poweroff 2>/dev/null \
        || shutdown -h now 2>/dev/null) || warn "не выключилось"
}

log "=== RUN_8K_SERVER_DOCKER start (work=${WORK}, image=${IMAGE}) ==="
log "host: $(hostname); nproc=$(nproc)"

# 0) права root
if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    warn "запущен не от root — apt/docker могут не работать. Лучше: sudo bash ...sh"
fi

# 1) apt + docker
export DEBIAN_FRONTEND=noninteractive
if ! command -v docker >/dev/null 2>&1; then
    log "apt install docker.io (git/curl/ca-certificates/docker.io)"
    apt-get update -y >>"${LOG}" 2>&1 || { warn "apt-get update не удался (см. лог)"; tail -15 "${LOG}"; }
    apt-get install -y --no-install-recommends git curl ca-certificates docker.io >>"${LOG}" 2>&1 \
        || { warn "apt-get install docker.io не удался (см. лог)"; tail -20 "${LOG}"; }
fi

if ! command -v docker >/dev/null 2>&1; then
    warn "docker НЕ установлен. Проверь сеть/репозитории apt:"
    warn "  sudo apt-get update && sudo apt-get install -y docker.io"
    warn "Либо вручную: см. https://docs.docker.com/engine/install/debian/"
    if [ "${SHUTDOWN_ON_ERROR}" = "1" ]; then shutdown_machine; fi
    exit 2
fi

systemctl enable --now docker >>"${LOG}" 2>&1 || service docker start >>"${LOG}" 2>&1 || true

# nvidia-container-toolkit (GPU в docker). Лучший-effort.
if ! command -v nvidia-ctk >/dev/null 2>&1; then
    log "install nvidia-container-toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --batch --yes --dearmor -o /usr/share/keyrings/libnvidia-container.gpg 2>/dev/null || true
    cat > /etc/apt/sources.list.d/nvidia-container-toolkit.list <<'EOF'
deb [signed-by=/usr/share/keyrings/libnvidia-container.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH) /
EOF
    apt-get update -y >>"${LOG}" 2>&1 || true
    apt-get install -y --no-install-recommends nvidia-container-toolkit >>"${LOG}" 2>&1 \
        || warn "nvidia-container-toolkit install failed (нужна ручная установка)"
    nvidia-ctk runtime configure --runtime=docker >>"${LOG}" 2>&1 || true
    systemctl restart docker >>"${LOG}" 2>&1 || true
fi

command -v nvidia-smi >/dev/null 2>&1 \
    && log "nvidia-smi: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)" \
    || warn "nvidia-smi не найден на хосте — docker --gpus может не сработать (нужны драйверы NVIDIA)"

# 2) клонируем код+данные
rm -rf "${REPO}"
log "clone ${REPO_URL} (branch ${BRANCH})"
GIT_TERMINAL_PROMPT=0 git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO}" || {
    warn "clone не удался (проверь REPO_URL); повтор без --branch"
    GIT_TERMINAL_PROMPT=0 git clone "${REPO_URL}" "${REPO}"
}

# 3) собираем образ
log "docker build -t ${IMAGE}"
if ! docker build -t "${IMAGE}" "${REPO}" >>"${LOG}" 2>&1; then
    warn "docker build не удался (см. лог, хвост ниже)"
    tail -30 "${LOG}"
    if [ "${SHUTDOWN_ON_ERROR}" = "1" ]; then shutdown_machine; fi
    exit 3
fi

# 4) запускаем контейнер с GPU; каталог results монтируем на хост
log "docker run (--gpus all) — обучение 8K в контейнере"
ret=2
if docker run --rm --gpus all \
       -v "${REPO}/results":/opt/run8k/results \
       "${IMAGE}" >>"${LOG}" 2>&1; then
    ret=0
else
    ret=$?
    warn "docker run вернул ${ret} (см. лог, хвост ниже)"
    tail -30 "${LOG}"
fi
run_done=""
[ -f "${REPO}/results/_done" ] && run_done="$(cat "${REPO}/results/_done")"
log "_done marker=${run_done:-не найден} (docker ret=${ret})"

# 5) публикуем результаты и гасим машину (только после попытки прогона)
collect || warn "collect не удался"
log "=== RUN_8K_SERVER_DOCKER finished (ret=${ret}, marker=${run_done:-}) ==="
shutdown_machine
exit 0
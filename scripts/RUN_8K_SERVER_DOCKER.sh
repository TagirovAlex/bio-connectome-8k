#!/usr/bin/env bash
# =============================================================================
# RUN_8K_SERVER_DOCKER.sh — реактивация 96GB-сервера (Debian 13) через Docker.
#
# Один запуск: ставит docker + nvidia-container-toolkit, клонирует код/данные
# из GitHub, собирает образ, запускает обучение 8K в контейнере с GPU, забирает
# результаты, пушит их в GitHub и выключает машину.
#
# Запуск (root/sudo):
#   REPO_URL="https://github.com/TagirovAlex/<repo>.git" GIT_TOKEN="ghp_..." \
#     bash RUN_8K_SERVER_DOCKER.sh 2>&1 | tee /opt/run8k/console.log
#
# Переменные:
#   REPO_URL    - git-адрес GitHub-репозитория (код+Dockerfile+данные), обязателен.
#   BRANCH      - ветка (default: main).
#   PUSH_TO     - remote для записи результатов (default: REPO_URL).
#   GIT_TOKEN   - PAT для push (если PUSH_TO на https).
#   WORK        - рабочая директория (default: /opt/run8k).
#   IMAGE       - имя образа (default: run8k:latest).
#   OUT_DIR     - доп. каталог для tarball результатов (optional).
#   NO_SHUTDOWN - "1" -> не выключать машину по завершении (отладка).
# =============================================================================
set -euo pipefail

WORK="${WORK:-/opt/run8k}"
LOG="${WORK}/console.log"
REPO="${WORK}/repo"
BRANCH="${BRANCH:-main}"
IMAGE="${IMAGE:-run8k:latest}"
NO_SHUTDOWN="${NO_SHUTDOWN:-0}"
: "${REPO_URL:? задай REPO_URL (GitHub-адрес кода+данных)}"
PUSH_TO="${PUSH_TO:-${REPO_URL}}"
OUT_DIR="${OUT_DIR:-}"

mkdir -p "${WORK}"
log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }
warn() { echo "[$(date -Is)] WARN: $*" | tee -a "${LOG}"; }

log "=== RUN_8K_SERVER_DOCKER start (work=${WORK}, image=${IMAGE}) ==="
log "host: $(hostname); nproc=$(nproc)"

# 1) системные зависимости + docker + nvidia-container-runtime
export DEBIAN_FRONTEND=noninteractive
apt-get update -y >>"${LOG}" 2>&1 || warn "apt update failed"
apt-get install -y --no-install-recommends \
    git curl ca-certificates docker.io >>"${LOG}" 2>&1 \
    || warn "apt install частично не удался"

systemctl enable --now docker >>"${LOG}" 2>&1 || service docker start >>"${LOG}" 2>&1 || true

# nvidia-container-toolkit (GPU в docker). Лучший-effort: без него --gpus не сработает.
if ! command -v nvidia-ctk >/dev/null 2>&1; then
    log "install nvidia-container-toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/libnvidia-container.gpg 2>/dev/null || true
    cat > /etc/apt/sources.list.d/nvidia-container-toolkit.list <<'EOF'
deb [signed-by=/usr/share/keyrings/libnvidia-container.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH) /
EOF
    apt-get update -y >>"${LOG}" 2>&1 || true
    apt-get install -y --no-install-recommends nvidia-container-toolkit >>"${LOG}" 2>&1 \
        || warn "nvidia-container-toolkit install failed (нужна ручная установка)"
    nvidia-ctk runtime configure --runtime=docker >>"${LOG}" 2>&1 || true
    systemctl restart docker >>"${LOG}" 2>&1 || true
fi

# 2) клонируем код+данные
rm -rf "${REPO}"
log "clone ${REPO_URL} (branch ${BRANCH})"
GIT_TERMINAL_PROMPT=0 git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO}" || {
    warn "clone не удался (проверь REPO_URL); повтор без --branch"
    GIT_TERMINAL_PROMPT=0 git clone "${REPO_URL}" "${REPO}"
}

# 3) собираем образ (torch уже внутри — большая часть уже прозрачна для сети сервера)
log "docker build -t ${IMAGE}"
docker build -t "${IMAGE}" "${REPO}" || {
    warn "build failed"; code=3; collect 2>/dev/null || true; shutdown_machine; exit 3
}

# 4) запускаем контейнер с GPU, каталог results монтируем на хост (быстрый возврат данных)
log "docker run (--gpus all) — обучение 8K в контейнере"
run_done=""
if docker run --rm --gpus all \
       -v "${REPO}/results":/opt/run8k/results \
       "${IMAGE}"; then
    ret=0
else
    ret=$?
    warn "docker run вернул ${ret}; результаты/маркер смотрим на хосте"
fi
[ -f "${REPO}/results/_done" ] && run_done="$(cat "${REPO}/results/_done")"
log "_done marker=${run_done:-не найден} (docker ret=${ret})"

# 5) пушим результаты обратно в GitHub
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

# 6) сохраняем результат и гасим машину
log "push results"
collect || warn "collect не удался"
log "=== RUN_8K_SERVER_DOCKER finished (ret=${ret:-0}, marker=${run_done:-}) ==="
shutdown_machine
exit 0
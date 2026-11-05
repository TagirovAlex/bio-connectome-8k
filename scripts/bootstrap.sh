#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — единая команда запуска на 96GB-сервере (Debian 13).
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/TagirovAlex/bio-connectome-cl/main/bootstrap.sh) <GIT_TOKEN>
#
# Делает: (1) ставит git+curl, если их нет; (2) клонирует репозиторий;
# (3) запускает RUN_8K_SERVER_DOCKER.sh (docker+nvidia -> build -> train 8K
#     -> analyze -> figures -> push -> poweroff) с переданным токеном.
#
# Аргументы: $1 = GitHub PAT (scope 'repo') для записи результатов обратно.
# Переменные (опц.): REPO_URL, BRANCH, WORK, NO_SHUTDOWN, OUT_DIR.
# =============================================================================
set -euo pipefail

TOKEN="${1:-}"
if [ -z "${TOKEN}" ]; then
    echo "[bootstrap] WARN: не передан GIT_TOKEN аргументом — push результатов не сработает." >&2
    echo "[bootstrap] Использование: <curl ...bootstrap.sh> <GIT_TOKEN>" >&2
fi

REPO_URL="${REPO_URL:-https://github.com/TagirovAlex/bio-connectome-cl.git}"
WORK="${WORK:-/opt/run8k}"
BRANCH="${BRANCH:-main}"

# sudo если не root
SUDO=""
if [ "${EUID:-$(id -u)}" -ne 0 ]; then SUDO="sudo"; fi

export DEBIAN_FRONTEND=noninteractive
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
    echo "[bootstrap] устанавливаю git+curl"
    ${SUDO} apt-get update -y >/dev/null 2>&1
    ${SUDO} apt-get install -y --no-install-recommends git curl >/dev/null 2>&1
fi

mkdir -p "${WORK}"
rm -rf "${WORK}/repo"
echo "[bootstrap] clone ${REPO_URL} (${BRANCH})"
GIT_TERMINAL_PROMPT=0 git clone --branch "${BRANCH}" "${REPO_URL}" "${WORK}/repo"

echo "[bootstrap] запуск RUN_8K_SERVER_DOCKER.sh (GIT_TOKEN=${TOKEN:+...передан})"
exec env GIT_TOKEN="${TOKEN}" bash "${WORK}/repo/scripts/RUN_8K_SERVER_DOCKER.sh" \
    2>&1 | tee /opt/run8k/console.log
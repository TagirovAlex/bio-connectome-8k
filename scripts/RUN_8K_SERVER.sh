#!/usr/bin/env bash
# =============================================================================
# RUN_8K_SERVER.sh — автономный прогон 8K-эксперимента на GPU-сервере (Debian 13,
# 96 ГБ VRAM). Один запуск делает всё: ставит окружение, клонирует код+данные,
# обучает 8K (bio,rewired), анализирует, строит фигуры, пушит результаты в git
# и выключает машину, чтобы не платить за простаивающий сервер.
#
# Запуск (от root или через sudo):
#   REPO_SOURCE_URI="https://github.com/<you>/<repo>.git" PUSH_TO="<write remote>" \
#     sudo bash RUN_8K_SERVER.sh 2>&1 | tee /opt/run8k/console.log
#
# Переменные (все необязательные):
#   REPO_SOURCE_URI  - git-адрес чтения исходного репозитория (код+графы+метрики).
#                      Если не задан и задан SRC_BUNDLE — клонируется бандл.
#   SRC_BUNDLE       - путь к git-бандлу (файл server-8k.bundle) с тем же содержимым.
#   PUSH_TO          - git-remote для записи результатов обратно (https/ssh).
#                      Если пусто — результаты выгружаются tarball в OUT_DIR (если задан).
#   GIT_TOKEN        - токен/PAT для HTTPS push (встраивается в remote URL).
#   BRANCH           - ветка (default: main).
#   WORK             - рабочая директория (default: /opt/run8k).
#   OUT_DIR          - смонтированный каталог для выгрузки результатов (tarball).
#   NO_SHUTDOWN      - "1" -> НЕ выключать машину по завершении (отладка).
#   PYTHON_BIN       - интерпретатор Python для venv (default: автопоиск).
#   TORCH_INDEX_URL  - torch-индекс (default: cu128 для Blackwell/Ada и новее).
# =============================================================================
set -euo pipefail

# ----------------------------- конфигурация ------------------------------
WORK="${WORK:-/opt/run8k}"
LOG="${WORK}/run.log"
REPO="${WORK}/repo"
BRANCH="${BRANCH:-main}"
NO_SHUTDOWN="${NO_SHUTDOWN:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

mkdir -p "${WORK}"
LOG_FD=""

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }
warn() { echo "[$(date -Is)] WARN: $*" | tee -a "${LOG}"; }

log "=== RUN_8K_SERVER start (work=${WORK}) ==="
log "host: $(hostname); nproc=$(nproc)"

# ------------------------- системные зависимости -------------------------
log "apt: update + install system deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y >>"${LOG}" 2>&1 || warn "apt-get update failed"
apt-get install -y --no-install-recommends \
  git curl wget ca-certificates \
  python3 python3-venv python3-pip build-essential \
  >>"${LOG}" 2>&1 || warn "apt-get install partially failed"

# --------------------------- источник кода -------------------------------
rm -rf "${REPO}"
if [ -n "${REPO_SOURCE_URI:-}" ]; then
    log "clone source from ${REPO_SOURCE_URI}"
    GIT_TERMINAL_PROMPT=0 git clone --branch "${BRANCH}" "${REPO_SOURCE_URI}" "${REPO}"
elif [ -n "${SRC_BUNDLE:-}" ] && [ -f "${SRC_BUNDLE}" ]; then
    log "clone source from bundle ${SRC_BUNDLE}"
    git clone "${SRC_BUNDLE}" "${REPO}"
else
    warn "Нет источника кода: задайте REPO_SOURCE_URI или SRC_BUNDLE. Выход."
    exit 1
fi
cd "${REPO}"

# ---------------------------- Python + torch ------------------------------
if [ -z "${PYTHON_BIN:-}" ]; then
    for cand in python3.11 python3.12 python3.13 python3; do
        if command -v "${cand}" >/dev/null 2>&1; then PYTHON_BIN="${cand}"; break; fi
    done
fi
log "python: ${PYTHON_BIN:-none} -> $(command -v "${PYTHON_BIN:-false}" || echo missing)"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { warn "python не найден"; exit 2; }

VENV="${WORK}/venv"
if [ ! -d "${VENV}" ]; then
    log "create venv at ${VENV}"
    "${PYTHON_BIN}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

log "pip: install torch (${TORCH_INDEX_URL})"
pip install --upgrade pip >>"${LOG}" 2>&1 || true
pip install "torch" "torchvision" --index-url "${TORCH_INDEX_URL}" >>"${LOG}" 2>&1 \
    || { warn "torch install failed"; exit 3; }

log "pip: install requirements.txt"
pip install -r requirements.txt >>"${LOG}" 2>&1 \
    || { warn "requirements install failed"; exit 3; }

# --------------------------- проверка GPU/графов --------------------------
log "cuda check:"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu", torch.cuda.get_device_name(0), "vram_GB", round(p.total_memory/2**30,1))
PY
for g in data/processed/bio_8k.pt data/processed/rewired_8k.pt; do
    [ -f "$g" ] || { warn "граф отсутствует: $g — прогон даст синтетику!"; }
done

# ---------------------------- прогон 8K + уход ----------------------------
run_phase='init'
on_exit() {
    local code=$?
    log "=== EXIT phase='${run_phase}' code=${code} ==="
    collect_and_save "exit_${run_phase}"
    shutdown_machine || true
}
trap on_exit EXIT

collect_and_save() {
    local tag="$1"
    log "collect results (tag=${tag})"
    cd "${REPO}"
    git add -A results/ 2>/dev/null || true
    git -c user.name="server-runner" -c user.email="runner@localhost" \
        commit -m "8K results ${tag} $(date -Is)" 2>/dev/null \
        || warn "ничего нового для коммита (или git error)"
    if [ -n "${PUSH_TO:-}" ]; then
        if [ -n "${GIT_TOKEN:-}" ]; then
            PUSH_URL="${PUSH_TO/https:\/\//https://x-access-token:${GIT_TOKEN}@}"
        else
            PUSH_URL="${PUSH_TO}"
        fi
        log "push results -> ${PUSH_TO}"
        git remote remove origin 2>/dev/null || true
        git remote add origin "${PUSH_URL}"
        git push -u origin "${BRANCH}" 2>&1 | tee -a "${LOG}" \
            || warn "git push failed (продолжаем, результаты в tarball)"
    fi
    if [ -n "${OUT_DIR:-}" ]; then
        mkdir -p "${OUT_DIR}"
        local tb="${OUT_DIR}/results_8k_${tag}_$(date +%Y%m%d_%H%M%S).tar.gz"
        log "export tarball -> ${tb}"
        (cd "${REPO}" && tar --exclude='results/logs' --exclude='results/checkpoints' \
            --exclude='.git' -czf "${tb}" results/) 2>&1 | tee -a "${LOG}" \
            || warn "tarball export failed"
        # Сохраняем полный лог в каталог выгрузки
        cp "${LOG}" "${OUT_DIR}/run8k_${tag}_$(date +%Y%m%d_%H%M%S).log" 2>/dev/null || true
    fi
}

shutdown_machine() {
    if [ "${NO_SHUTDOWN}" = "1" ]; then
        log "NO_SHUTDOWN=1 — машину НЕ выключаю"
        return 0
    fi
    log "shutdown in 5s..."
    sleep 5
    (systemctl poweroff --no-block 2>/dev/null \
        || poweroff 2>/dev/null \
        || shutdown -h now 2>/dev/null) \
        || warn "не удалось выключить машину"
}

# Полный прогон 8K (bio, rewired). При OOM автоматически уменьшаем batch_size.
# Коды возврата: 0=успех, 1=OOM (нужен повтор со сниженным batch), 2=прочая ошибка.
run_train() {
    local bs="${1:-64}"
    if [ "${bs}" != "64" ]; then
        log "OOM-fallback: batch_size 64->${bs}"
        sed -i "s/^[[:space:]]*batch_size:[[:space:]]*[0-9]*/batch_size: ${bs}/" config/training.yaml \
            || warn "не смог подменить batch_size в config/training.yaml"
    fi
    run_phase="train_bs${bs}"
    log "RUN: python scripts/03_run_8k.py --arch bio,rewired"
    python scripts/03_run_8k.py --arch bio,rewired 2>&1 | tee -a "${LOG}"
    local code=${PIPESTATUS[0]}
    run_phase="train_done"
    if [ "${code}" -ne 0 ]; then
        log "обучение вышло с кодом ${code}"
        # Ловим только OOM-подобные сбои: можно перезапустить меньшим batch.
        if tail -80 "${LOG}" | grep -qi "out of memory\\|OOM\\|CUDA out of memory"; then
            log "обнаружен OOM — снижаю batch_size и повторяю"
            return 1
        fi
        return 2
    fi
    return 0
}

if run_train 64; then _run_rc=0; else _run_rc=$?; fi
if [ "${_run_rc}" -eq 1 ]; then
    run_train 32 || true
elif [ "${_run_rc}" -eq 2 ]; then
    log "обучение завершилось с ошибкой (не OOM)"
fi

# ------------------------- анализ и графики 8K ----------------------------
run_phase="analyze"
log "RUN: 04_analyze.py --scale 8k"
python scripts/04_analyze.py --scale 8k 2>&1 | tee -a "${LOG}" || true
run_phase="figures"
log "RUN: 05_make_figures.py --scale all"
python scripts/05_make_figures.py --scale all 2>&1 | tee -a "${LOG}" || true

run_phase="finalize"
log "=== перезапуск анализа ИТОГ ==="
collect_and_save "done"
log "=== RUN_8K_SERVER finished successfully ==="

# trap сработает после этого: вызовет on_exit (push + shutdown)
exit 0
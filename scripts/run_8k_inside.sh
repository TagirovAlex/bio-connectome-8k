#!/usr/bin/env bash
# =============================================================================
# run_8k_inside.sh — точка входа контейнера. Обучает 8K (bio,rewired),
# анализирует, строит фигуры и отмечает готовность маркером.
#
# Ожидает, что каталог /opt/run8k/results смонтирован как volume с хоста (rw),
# чтобы результаты вернулись на хост. Маркер готовности: /opt/run8k/results/_done
# (содержит 0 при успехе, код ошибки при сбое).
# =============================================================================
set -uo pipefail

RESULTS="/opt/run8k/results"
LOG="/opt/run8k/results/run8k_container.log"
mkdir -p "${RESULTS}"
: > "${LOG}"

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

mark_done() { echo "$1" > "${RESULTS}/_done"; }
cleanup() { :; }
trap cleanup EXIT

log "=== container start ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu", torch.cuda.get_device_name(0), "vram_GB", round(p.total_memory/2**30,1))
PY

for g in data/processed/bio_8k.pt data/processed/rewired_8k.pt; do
    [ -f "$g" ] || log "WARN: граф отсутствует ${g} — будет синтетика"
done

# Цикл с автоснижением batch_size при OOM.
run_train() {
    local bs="${1:-64}"
    if [ "${bs}" != "64" ]; then
        log "OOM-fallback: batch_size 64->${bs}"
        sed -i "s/^[[:space:]]*batch_size:[[:space:]]*[0-9]*/batch_size: ${bs}/" config/training.yaml
    fi
    log "RUN: python scripts/03_run_8k.py --arch bio,rewired"
    python scripts/03_run_8k.py --arch bio,rewired 2>&1 | tee -a "${LOG}"
    return ${PIPESTATUS[0]}
}

rc=0
if run_train 64; then
    rc=0
else
    rc=$?
    if tail -60 "${LOG}" | grep -qi "out of memory\\|OOM\\|CUDA out of memory"; then
        log "OOM -> retry batch_size 32"
        run_train 32 || true
    fi
fi

log "RUN: analyze 8k"
python scripts/04_analyze.py --scale 8k 2>&1 | tee -a "${LOG}" || true
log "RUN: figures all"
python scripts/05_make_figures.py --scale all 2>&1 | tee -a "${LOG}" || true

log "=== container done (rc=${rc}) ==="
mark_done "${rc}"
exit 0
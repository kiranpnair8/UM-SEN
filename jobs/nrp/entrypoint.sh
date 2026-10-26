#!/bin/bash
set -euo pipefail
MODE="${1:?}"
RUN_ID="${NRP_RUN_ID:?}"
[[ "${RUN_ID}" =~ ^[a-z0-9_]+$ ]] || exit 2
ROOT="${NRP_ROOT:?}"
mkdir -p "${ROOT}/logs/${RUN_ID}"
LOG="${ROOT}/logs/${RUN_ID}/${POD_UID:-local}.log"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED="${NRP_SEED:-42}"
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${ROOT}/runs/${RUN_ID}/matplotlib"
export CUPY_CACHE_DIR="${ROOT}/runs/${RUN_ID}/cupy-cache"
export XDG_CACHE_HOME="${ROOT}/runs/${RUN_ID}/cache"
mkdir -p "${MPLCONFIGDIR}" "${CUPY_CACHE_DIR}" "${XDG_CACHE_HOME}"
python3.10 -u /opt/nrp/nrp_runner.py "$@" > >(tee -a "${LOG}") 2>&1 &
PID=$!
trap 'if [[ "${MODE}" == train ]]; then kill -USR1 "${PID}" 2>/dev/null || true; else kill -TERM "${PID}" 2>/dev/null || true; fi' TERM INT
RC=0
wait "${PID}" || RC=$?
while kill -0 "${PID}" 2>/dev/null; do
  RC=0
  wait "${PID}" || RC=$?
done
exit "${RC}"

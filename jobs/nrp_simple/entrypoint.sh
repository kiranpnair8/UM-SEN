#!/bin/bash
set -euo pipefail
ROOT="${NRP_ROOT:?}"
RUN="${NRP_RUN_ID:?}"
[[ "${RUN}" =~ ^[a-z0-9_]+$ ]] || exit 2
mkdir -p "${ROOT}/runs/${RUN}/logs" "${ROOT}/runs/${RUN}/cache"
export PYTHONHASHSEED="${NRP_SEED:-42}" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUPY_CACHE_DIR="${ROOT}/runs/${RUN}/cache/cupy"
export MPLCONFIGDIR="${ROOT}/runs/${RUN}/cache/matplotlib"
export XDG_CACHE_HOME="${ROOT}/runs/${RUN}/cache"
python3.10 -u /opt/simple/runner.py "$@" > >(tee -a "${ROOT}/runs/${RUN}/logs/${POD_UID:-local}.log") 2>&1 &
PID=$!
trap 'if [[ "${1:-}" == train ]]; then kill -USR1 "${PID}" 2>/dev/null || true; else kill -TERM "${PID}" 2>/dev/null || true; fi' TERM INT
RC=0
wait "${PID}" || RC=$?
while kill -0 "${PID}" 2>/dev/null; do
  RC=0
  wait "${PID}" || RC=$?
done
exit "${RC}"

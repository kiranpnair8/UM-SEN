#!/bin/bash
#SBATCH --job-name=sage-online
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=logs/gradient_flow/controller_online_%j.log
#SBATCH --error=logs/gradient_flow/controller_online_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
date
hostname
nvidia-smi
COMMIT="$(git rev-parse HEAD)"
echo "Source commit: ${COMMIT}"
git status --short
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing dirty checkout" >&2; exit 2; }
[[ "$(git branch --show-current)" == "gradient-flow-stage1-validation" ]] || { echo "Wrong branch" >&2; exit 2; }
[[ "$(hostname -s)" == "gpu002" ]] || { echo "This launcher requires gpu002" >&2; exit 2; }
[[ "${EXPECTED_COMMIT:-${COMMIT}}" == "${COMMIT}" ]] || { echo "Unexpected commit" >&2; exit 2; }
command -v flock >/dev/null
mkdir -p logs/gradient_flow
exec 9>logs/gradient_flow/controller_online.lock
flock -n 9 || { echo "Another online writer is active" >&2; exit 2; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SNN_ENV:-/home/rizk_lab/shared/kiran/envs/snn}"
python --version
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${PWD}/.cupy/kernel_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
CONFIG=configs/gradient_flow/controller_online.yml
RUN_DIR="${RUN_DIR:-results/gradient_flow/controller_online_seed42}"
FIGURE_DIR="${FIGURE_DIR:-figures/gradient_flow/controller_online_seed42}"
python - "${CONFIG}" <<'PY'
import sys
from experiments.gradient_flow.common import ROOT, load_config
from experiments.gradient_flow.online_policy import OnlinePolicy
c = load_config(ROOT/sys.argv[1])
OnlinePolicy(c['layer'], 'response_aware', c['online'])
assert (c['time_step'], c['layer'], c['dim'], c['num_heads'], c['patch_size'], c['mlp_ratio']) == (4, 4, 384, 12, 4, 4)
assert c['seed'] == 42 and c['online']['epochs'] == 60
print('Reviewed online protocol:', c['online'])
print('Training prefix: 60 epochs; inherited cosine recipe:', c['epochs'], 'epochs +', c['cooldown_epochs'], 'cooldown')
PY
ACTIVE_PID=""
trap 'if [[ -n "${ACTIVE_PID}" ]]; then kill -USR1 "${ACTIVE_PID}" 2>/dev/null || true; fi' USR1
for CONDITION in fixed response_aware random_matched; do
  LOG="logs/gradient_flow/controller_online_${CONDITION}_${SLURM_JOB_ID:-interactive}.log"
  echo "Running/resuming ${CONDITION}; log=${LOG}"
  python -u -m experiments.gradient_flow.online_train --config "${CONFIG}" \
    --condition "${CONDITION}" --data-dir "${DATA_DIR:-data}" --output "${RUN_DIR}" \
    --device cuda --backend cupy --expected-commit "${COMMIT}" --resume > >(tee -a "${LOG}") 2>&1 &
  ACTIVE_PID=$!
  RC=0
  wait "${ACTIVE_PID}" || RC=$?
  # A trapped walltime warning may interrupt bash's wait before Python has
  # finished the current epoch and published its resumable checkpoint.
  while kill -0 "${ACTIVE_PID}" 2>/dev/null; do
    RC=0
    wait "${ACTIVE_PID}" || RC=$?
  done
  ACTIVE_PID=""
  if [[ "${RC}" == 75 ]]; then
    echo "Paused at an epoch boundary. Resubmit this SAME job to continue; no retuning."
    exit 75
  fi
  [[ "${RC}" == 0 ]] || { echo "${CONDITION} failed (${RC}); stopping without retry/tuning" >&2; exit "${RC}"; }
done
python -u -m analysis.gradient_flow.plot_controller_online --run-dir "${RUN_DIR}" --output-dir "${FIGURE_DIR}"
python - "${RUN_DIR}/summary.json" <<'PY'
import json
import sys
gate = json.load(open(sys.argv[1]))['gate']
print('Online feasibility:', gate)
if not gate['passed']:
    raise SystemExit(2)
PY
date

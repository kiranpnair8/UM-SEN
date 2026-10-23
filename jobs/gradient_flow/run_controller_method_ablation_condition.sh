#!/bin/bash
#SBATCH --job-name=sage-method
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=logs/gradient_flow/controller_method_%j.log
#SBATCH --error=logs/gradient_flow/controller_method_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
case "${VARIANT:-}" in deep_only|regularized|deadband_regularized) ;; *) echo "Set one of the three predeclared VARIANT names" >&2; exit 2 ;; esac
[[ "$(git branch --show-current)" == gradient-flow-stage1-validation ]] || exit 2
[[ -z "$(git status --porcelain)" ]] || { echo "Dirty checkout" >&2; exit 2; }
COMMIT="$(git rev-parse HEAD)"
date
hostname
nvidia-smi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SNN_ENV:-/home/rizk_lab/shared/kiran/envs/snn}"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${PWD}/.cupy/kernel_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
mkdir -p logs/gradient_flow
ACTIVE_PID=""
trap 'if [[ -n "${ACTIVE_PID}" ]]; then kill -USR1 "${ACTIVE_PID}" 2>/dev/null || true; fi' USR1
# Preparation is an explicit separate command. Jobs NEVER rewrite the common
# preregistration/deadband; only the selected variant's directory/lock is written.
python -u -m experiments.gradient_flow.method_ablation --variant "${VARIANT}" \
  --data-dir "${DATA_DIR:-data}" --device cuda --backend cupy --resume --expected-commit "${COMMIT}" \
  > >(tee -a "logs/gradient_flow/method_${VARIANT}_${SLURM_JOB_ID:-interactive}.log") 2>&1 &
ACTIVE_PID=$!
RC=0
wait "${ACTIVE_PID}" || RC=$?
while kill -0 "${ACTIVE_PID}" 2>/dev/null; do
  RC=0
  wait "${ACTIVE_PID}" || RC=$?
done
[[ "${RC}" != 75 ]] || echo "Paused after a completed epoch; resubmit the same variant to resume."
exit "${RC}"

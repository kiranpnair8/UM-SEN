#!/bin/bash
#SBATCH --job-name=sage-calibrate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=logs/gradient_flow/controller_calibration_%j.log
#SBATCH --error=logs/gradient_flow/controller_calibration_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
case "${VARIANT:-}" in conservative|very_conservative|anchored) ;; *) echo "Set VARIANT to conservative, very_conservative, or anchored" >&2; exit 2 ;; esac
[[ "$(git branch --show-current)" == gradient-flow-stage1-validation ]] || { echo "Wrong branch" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing dirty checkout" >&2; exit 2; }
COMMIT="$(git rev-parse HEAD)"
[[ "${EXPECTED_COMMIT:-${COMMIT}}" == "${COMMIT}" ]] || { echo "Unexpected commit" >&2; exit 2; }
date
hostname
nvidia-smi
echo "Calibration variant=${VARIANT}; implementation=${COMMIT}; job=${SLURM_JOB_ID:-interactive}"
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
# Python holds a per-variant lock for its entire lifetime, including resume.
# Different variants share only a brief preregistration lock, never a GPU-run lock.
python -u -m experiments.gradient_flow.controller_calibration \
  --variant "${VARIANT}" --data-dir "${DATA_DIR:-data}" --resume \
  --device cuda --backend cupy --expected-commit "${COMMIT}" \
  > >(tee -a "logs/gradient_flow/calibration_${VARIANT}_${SLURM_JOB_ID:-interactive}.log") 2>&1 &
ACTIVE_PID=$!
RC=0
wait "${ACTIVE_PID}" || RC=$?
while kill -0 "${ACTIVE_PID}" 2>/dev/null; do
  RC=0
  wait "${ACTIVE_PID}" || RC=$?
done
if [[ "${RC}" == 75 ]]; then
  echo "Paused at an epoch boundary; resubmit the same VARIANT to resume."
fi
exit "${RC}"

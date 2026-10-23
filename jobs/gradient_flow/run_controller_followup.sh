#!/bin/bash
# PREPARED ONLY: run AFTER explicit selection/review, never during method ablation.
#SBATCH --job-name=sage-followup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5
#SBATCH --signal=B:USR1@1800
#SBATCH --output=logs/gradient_flow/controller_followup_%A_%a.log
#SBATCH --error=logs/gradient_flow/controller_followup_%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -n "${FINAL_SAGE:-}" ]] || { echo "FINAL_SAGE must match the reviewed selection" >&2; exit 2; }
case "${DATASET:-cifar10}" in cifar10|cifar100) ;; *) exit 2 ;; esac
[[ "$(git branch --show-current)" == gradient-flow-stage1-validation ]] || exit 2
[[ -z "$(git status --porcelain)" ]] || exit 2
TASK="${SLURM_ARRAY_TASK_ID:?Use the six-task array}"
(( TASK >= 0 && TASK < 6 )) || exit 2
SEED=$((42+TASK/2))
CONDITION=fixed
(( TASK % 2 == 0 )) || CONDITION=final_sage
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SNN_ENV:-/home/rizk_lab/shared/kiran/envs/snn}"
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${PWD}/.cupy/kernel_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
date
hostname
nvidia-smi
ACTIVE_PID=""
trap 'if [[ -n "${ACTIVE_PID}" ]]; then kill -USR1 "${ACTIVE_PID}" 2>/dev/null || true; fi' USR1
python -u -m experiments.gradient_flow.controller_followup --dataset "${DATASET:-cifar10}" \
  --seed "${SEED}" --condition "${CONDITION}" --final-sage "${FINAL_SAGE}" --data-dir "${DATA_DIR:-data}" --resume &
ACTIVE_PID=$!
RC=0
wait "${ACTIVE_PID}" || RC=$?
while kill -0 "${ACTIVE_PID}" 2>/dev/null; do
  RC=0
  wait "${ACTIVE_PID}" || RC=$?
done
exit "${RC}"

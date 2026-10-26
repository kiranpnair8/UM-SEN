#!/bin/bash
# Isolated CIFAR100 worktree ONLY. Do not run/pull this in active CIFAR10 checkout.
#SBATCH --job-name=sage-cifar100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5
#SBATCH --signal=B:USR1@1800
#SBATCH --output=logs/gradient_flow/cifar100_followup_%A_%a.log
#SBATCH --error=logs/gradient_flow/cifar100_followup_%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -z "$(git status --porcelain)" ]] || { echo "Dirty CIFAR100 checkout" >&2; exit 2; }
# Detached worktrees are intentional: the running CIFAR10 HEAD must not move.
TASK="${SLURM_ARRAY_TASK_ID:?Use array task 0..5}"
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
echo "CIFAR100 commit=$(git rev-parse HEAD) seed=${SEED} condition=${CONDITION}"
date
hostname
nvidia-smi
DATA_ARGS=()
if [[ -n "${DATA_DIR:-}" ]]; then DATA_ARGS=(--data-dir "${DATA_DIR}"); fi
ACTIVE_PID=""
trap 'if [[ -n "${ACTIVE_PID}" ]]; then kill -USR1 "${ACTIVE_PID}" 2>/dev/null || true; fi' USR1
python -u -m experiments.gradient_flow.cifar100_followup \
  --seed "${SEED}" --condition "${CONDITION}" --resume "${DATA_ARGS[@]}" &
ACTIVE_PID=$!
RC=0
wait "${ACTIVE_PID}" || RC=$?
while kill -0 "${ACTIVE_PID}" 2>/dev/null; do
  RC=0
  wait "${ACTIVE_PID}" || RC=$?
done
exit "${RC}"

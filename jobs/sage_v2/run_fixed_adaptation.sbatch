#!/bin/bash
#SBATCH --job-name=sage-v2-fixed-adapt
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu005
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd -P)"
test -f "${REPO_ROOT}/spikformer/cifar10/train.py"
INIT="${REPO_ROOT}/results/umsen_official_recipe/baseline_seed42/model_best.pth.tar"
test -r "${INIT}"
LR="${LR:-1e-5}"
case "${LR}" in
  1e-5|5e-5) ;;
  *) echo "LR must be 1e-5 or 5e-5" >&2; exit 2 ;;
esac
EPOCHS=30
MODE=pilot
if [[ "${SMOKE_TEST:-0}" == 1 ]]; then
  EPOCHS=1
  MODE=smoke
fi
RUN_ID="lr${LR}_${MODE}_${SLURM_JOB_ID:-manual}"
LOG_DIR="${REPO_ROOT}/logs/sage_v2"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/fixed_adaptation_${RUN_ID}.log") 2>&1

SNN_ENV="${SNN_ENV:-/home/rizk_lab/shared/kiran/envs/snn}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SNN_ENV}"
export PYTHONPATH="${REPO_ROOT}/spikformer/cifar10:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${REPO_ROOT}/.cupy/kernel_cache}"

cd "${REPO_ROOT}/spikformer/cifar10"
python train.py \
    -c "${REPO_ROOT}/configs/sage_v2/fixed_cifar10_adaptation.yml" \
    --init-checkpoint "${INIT}" \
    -data-dir "${REPO_ROOT}/data" \
    --fixed-alpha 4 --sps-alpha 4 --seed 42 -T 4 --epochs "${EPOCHS}" \
    --lr "${LR}" --sched cosine --warmup-epochs 0 --cooldown-epochs 0 --min-lr 1e-7 \
    --output "${REPO_ROOT}/results/sage_v2/cifar10/fixed/T4/seed42" \
    --experiment "${RUN_ID}"

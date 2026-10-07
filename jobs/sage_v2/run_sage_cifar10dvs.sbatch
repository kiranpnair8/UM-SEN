#!/bin/bash
#SBATCH --job-name=sage-v2-dvs-t10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/sage_v2/sage_cifar10dvs_%j.slurm.log
#SBATCH --error=logs/sage_v2/sage_cifar10dvs_%j.slurm.log

# Generic GPU request follows this cluster's existing jobs. If V100 nodes are
# known, pass --nodelist=<V100-node> at submission; no unverified GPU label here.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd -P)"
test -f "${REPO_ROOT}/spikformer/cifar10dvs/train_sage.py"
T="${T:-10}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-106}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.001}"
SMOKE_TEST="${SMOKE_TEST:-0}"
case "${SMOKE_TEST}" in 0|1) ;; *) echo "SMOKE_TEST must be 0 or 1" >&2; exit 2 ;; esac
RUN_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/logs/sage_v2"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/sage_cifar10dvs_T${T}_seed${SEED}_${RUN_ID}.log") 2>&1
SNN_ENV="${SNN_ENV:-/home/rizk_lab/shared/kiran/envs/snn}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SNN_ENV}"
export PYTHONHASHSEED="${SEED}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${LOG_DIR}/cupy_kernel_cache}"
export PYTHONPATH="${REPO_ROOT}/spikformer/cifar10dvs:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
EXTRA=()
if [[ "${SMOKE_TEST}" == 1 ]]; then EXTRA+=(--smoke-test); fi
echo "Repository: ${REPO_ROOT}; GPU allocation: ${SLURM_JOB_GPUS:-unknown}"
python -u spikformer/cifar10dvs/train_sage.py \
    --config configs/sage_v2/sage_cifar10dvs_t10.yml \
    --data-path "${DATA_DIR:-${REPO_ROOT}/data/CIFAR10DVS}" \
    --T "${T}" --seed "${SEED}" --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" --lr "${LR}" --run-id "${RUN_ID}" \
    --output-dir "${REPO_ROOT}/results/sage_v2/cifar10dvs" "${EXTRA[@]}"

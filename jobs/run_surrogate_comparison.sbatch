#!/bin/bash
#SBATCH --job-name=surrogate-compare
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu004
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_ROOT="${REPO_ROOT}/results/surrogate_comparison"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/surrogate_comparison_gpu004_${SLURM_JOB_ID:-manual}.log"
SNN_ENV="/home/rizk_lab/shared/kiran/envs/snn"
CUPY_CACHE_DIR="${REPO_ROOT}/.cupy/kernel_cache"

# Single-run defaults. Override with sbatch --export, e.g.
# sbatch --export=ALL,DATASET=cifar10,METHOD=umsen,T=4,SEED=42 jobs/run_surrogate_comparison.sbatch
RUN_ALL="${RUN_ALL:-0}"
DATASET="${DATASET:-cifar10}"
METHOD="${METHOD:-fixed}"
T="${T:-4}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
CIFAR10_DATA_DIR="${CIFAR10_DATA_DIR:-${REPO_ROOT}/data}"
CIFAR100_DATA_DIR="${CIFAR100_DATA_DIR:-${REPO_ROOT}/data}"
IMAGENET200_DATA_DIR="${IMAGENET200_DATA_DIR:-${REPO_ROOT}/data/imagenet200}"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${CUPY_CACHE_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "Repository root: ${REPO_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Activating SNN environment: ${SNN_ENV}"
echo "CuPy cache directory: ${CUPY_CACHE_DIR}"
echo "RUN_ALL=${RUN_ALL}"
echo "DATASET=${DATASET}"
echo "METHOD=${METHOD}"
echo "T=${T}"
echo "SEED=${SEED}"
echo "EPOCHS=${EPOCHS}"

export CUPY_CACHE_DIR
export PYTHONPATH="${REPO_ROOT}/spikformer/cifar10:${REPO_ROOT}/spikformer/imagenet:${PYTHONPATH:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SNN_ENV}"
which python
python --version

cd "${REPO_ROOT}"

if [ "${RUN_ALL}" = "1" ]; then
  python scripts/run_surrogate_comparison.py \
    --all \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}" \
    --cifar10-data-dir "${CIFAR10_DATA_DIR}" \
    --cifar100-data-dir "${CIFAR100_DATA_DIR}" \
    --imagenet200-data-dir "${IMAGENET200_DATA_DIR}"
else
  case "${DATASET}" in
    cifar10)
      DATA_DIR="${CIFAR10_DATA_DIR}"
      ;;
    cifar100)
      DATA_DIR="${CIFAR100_DATA_DIR}"
      ;;
    imagenet200)
      DATA_DIR="${IMAGENET200_DATA_DIR}"
      ;;
    *)
      echo "Unknown DATASET=${DATASET}" >&2
      exit 2
      ;;
  esac
  python scripts/run_surrogate_comparison.py \
    --dataset "${DATASET}" \
    --method "${METHOD}" \
    --T "${T}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}" \
    --data-dir "${DATA_DIR}"
fi

echo "Surrogate comparison job finished."

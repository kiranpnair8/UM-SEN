#!/bin/bash
#SBATCH --job-name=umsen-official
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu005
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_DIR="${REPO_ROOT}/results/umsen_official_recipe"
DATA_DIR="${REPO_ROOT}/data"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/official_umsen_gpu005_${SLURM_JOB_ID:-manual}.log"
SNN_ENV="/home/rizk_lab/shared/kiran/envs/snn"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "Repository root: ${REPO_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "CIFAR-10 data directory: ${DATA_DIR}"
echo "Activating SNN environment: ${SNN_ENV}"

export PYTHONPATH="${REPO_ROOT}/spikformer/cifar10:${PYTHONPATH:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SNN_ENV}"
which python
python --version

cd "${REPO_ROOT}/spikformer/cifar10"

for SEED in 42 43 44; do
  echo "Running official UM-SEN seed ${SEED} on gpu005"
  python train_umsen.py \
    -c cifar10.yml \
    --umsen \
    --seed "${SEED}" \
    -data-dir "${DATA_DIR}" \
    --output "${OUTPUT_DIR}" \
    --experiment "umsen_seed${SEED}"
done

for SEED in 42 43 44; do
  test -f "${OUTPUT_DIR}/umsen_seed${SEED}/summary.csv"
  test -f "${OUTPUT_DIR}/umsen_seed${SEED}/args.yaml"
  test -f "${OUTPUT_DIR}/umsen_seed${SEED}/umsen_controller.json"
done

echo "Verified official-recipe UM-SEN outputs in ${OUTPUT_DIR}"

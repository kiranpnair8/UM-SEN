#!/bin/bash
#SBATCH --job-name=umsen-100ep-resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu005
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_DIR="${REPO_ROOT}/results/umsen_100ep"
DATA_DIR="${REPO_ROOT}/data"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/umsen_100ep_resume_${SLURM_JOB_ID:-manual}.log"
SNN_ENV="/home/rizk_lab/shared/kiran/envs/snn"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "Repository root: ${REPO_ROOT}"
echo "Resuming CIFAR-10 UM-SEN 100-epoch benchmark in: ${OUTPUT_DIR}"
echo "CIFAR-10 data directory: ${DATA_DIR}"
echo "Activating SNN environment: ${SNN_ENV}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SNN_ENV}"
which python
python --version

cd "${REPO_ROOT}"

python analysis/run_umsen_100ep.py \
  --configs baseline umsen \
  --seeds 42 43 44 \
  --epochs 100 \
  --output-dir "${OUTPUT_DIR}" \
  --data-dir "${DATA_DIR}"

test -f "${OUTPUT_DIR}/summary.json"

echo "Resume job finished for UM-SEN 100-epoch benchmark in ${OUTPUT_DIR}"

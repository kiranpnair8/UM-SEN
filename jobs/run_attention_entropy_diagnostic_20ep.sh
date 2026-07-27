#!/bin/bash
#SBATCH --job-name=spikformer-attn-entropy-20ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu005
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUTPUT_DIR="${REPO_ROOT}/results/attention_entropy_diagnostic_20ep"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/attention_entropy_diagnostic_20ep_${SLURM_JOB_ID:-manual}.log"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "Repository root: ${REPO_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate act_snn

cd "${REPO_ROOT}/spikformer/cifar10"

python attention_entropy_diagnostic.py \
  --epochs 20 \
  --seed 42 \
  --output-dir "${OUTPUT_DIR}" \
  --entropy-log-interval 1

test -f "${OUTPUT_DIR}/metrics.json"
test -f "${OUTPUT_DIR}/config.json"
compgen -G "${OUTPUT_DIR}/epoch_*.json" > /dev/null

echo "Verified diagnostic outputs in ${OUTPUT_DIR}"

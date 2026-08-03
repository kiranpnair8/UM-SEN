#!/bin/bash
#SBATCH --job-name=umsen-official
#SBATCH --partition=gpu
#SBATCH --nodes=1
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
LOG_FILE="${LOG_DIR}/umsen_official_recipe_launcher_${SLURM_JOB_ID:-manual}.log"
BASELINE_LOG="${LOG_DIR}/official_baseline_gpu004_${SLURM_JOB_ID:-manual}.log"
UMSEN_LOG="${LOG_DIR}/official_umsen_gpu005_${SLURM_JOB_ID:-manual}.log"
SNN_ENV="/home/rizk_lab/shared/kiran/envs/snn"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Logging to ${LOG_FILE}"
echo "Repository root: ${REPO_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "CIFAR-10 data directory: ${DATA_DIR}"
echo "Activating SNN environment: ${SNN_ENV}"
echo "Baseline log: ${BASELINE_LOG}"
echo "UM-SEN log: ${UMSEN_LOG}"

export PYTHONPATH="${REPO_ROOT}/spikformer/cifar10:${PYTHONPATH:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SNN_ENV}"
which python
python --version

run_baseline() {
  cd "${REPO_ROOT}/spikformer/cifar10"
  for SEED in 42 43 44; do
    echo "Running official baseline seed ${SEED} on gpu004"
    python train.py \
      -c cifar10.yml \
      --seed "${SEED}" \
      -data-dir "${DATA_DIR}" \
      --output "${OUTPUT_DIR}" \
      --experiment "baseline_seed${SEED}"
  done
}

run_umsen() {
  cd "${REPO_ROOT}/spikformer/cifar10"
  for SEED in 42 43 44; do
    echo "Running official UM-SEN seed ${SEED} on gpu005"
    python train_umsen.py \
      -c cifar10.yml \
      --seed "${SEED}" \
      -data-dir "${DATA_DIR}" \
      --output "${OUTPUT_DIR}" \
      --experiment "umsen_seed${SEED}" \
      --umsen
  done
}

export -f run_baseline
export -f run_umsen
export REPO_ROOT OUTPUT_DIR DATA_DIR SNN_ENV PYTHONPATH

srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --nodelist=gpu004 \
  bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${SNN_ENV}"; run_baseline' \
  > >(tee -a "${BASELINE_LOG}") 2>&1 &
BASELINE_PID=$!

srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --nodelist=gpu005 \
  bash -lc 'source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${SNN_ENV}"; run_umsen' \
  > >(tee -a "${UMSEN_LOG}") 2>&1 &
UMSEN_PID=$!

wait "${BASELINE_PID}"
wait "${UMSEN_PID}"

for SEED in 42 43 44; do
  test -f "${OUTPUT_DIR}/baseline_seed${SEED}/summary.csv"
  test -f "${OUTPUT_DIR}/baseline_seed${SEED}/args.yaml"
  test -f "${OUTPUT_DIR}/umsen_seed${SEED}/summary.csv"
  test -f "${OUTPUT_DIR}/umsen_seed${SEED}/args.yaml"
  test -f "${OUTPUT_DIR}/umsen_seed${SEED}/umsen_controller.json"
done

echo "Verified official-recipe baseline and UM-SEN outputs in ${OUTPUT_DIR}"

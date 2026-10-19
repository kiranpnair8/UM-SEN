#!/bin/bash
#SBATCH --job-name=sage-alpha-multibatch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/gradient_flow/alpha_multibatch_%j.log
#SBATCH --error=logs/gradient_flow/alpha_multibatch_%j.log

# Submit from the SAGE repository root AFTER mkdir -p logs/gradient_flow.
# Generic GPU allocation matches existing Lawrence jobs; optionally pass a known
# V100 node with sbatch --nodelist=... . No unverified account/QoS/module is added.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd -P)"
cd "${REPO_ROOT}"
test -f experiments/gradient_flow/multi_batch_intervene.py
hostname
date
nvidia-smi
git rev-parse HEAD
git status --short
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to run from a dirty SAGE repository." >&2
  exit 2
fi
SNN_ENV="${SNN_ENV:-/home/rizk_lab/shared/kiran/envs/snn}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SNN_ENV}"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${REPO_ROOT}/.cupy/kernel_cache}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
CHECKPOINT="${CHECKPOINT:-results/umsen_official_recipe/baseline_seed42/model_best.pth.tar}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RUN_DIR="${RUN_DIR:-results/gradient_flow/alpha_multibatch_seed42}"
FIGURE_DIR="${FIGURE_DIR:-figures/gradient_flow/alpha_multibatch_seed42}"
test -r "${CHECKPOINT}"
python -u -m experiments.gradient_flow.multi_batch_intervene \
  --config configs/gradient_flow/cifar10_diagnostics.yml \
  --checkpoint "${CHECKPOINT}" --data-dir "${DATA_DIR}" \
  --num-batches 25 --batch-size 32 --seed 42 \
  --alphas 2 3 4 5 6 --reference-alpha 4 --device cuda --backend cupy \
  --output "${RUN_DIR}" --resume
python -u -m analysis.gradient_flow.plot_alpha_multibatch \
  --run-dir "${RUN_DIR}" --output-dir "${FIGURE_DIR}"
date

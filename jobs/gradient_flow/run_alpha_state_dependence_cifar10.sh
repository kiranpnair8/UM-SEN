#!/bin/bash
#SBATCH --job-name=sage-alpha-states
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu006
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/gradient_flow/alpha_states_%j.log
#SBATCH --error=logs/gradient_flow/alpha_states_%j.log

set -euo pipefail
START_SECONDS=${SECONDS}
trap 'echo "Elapsed seconds: $((SECONDS - START_SECONDS))"' EXIT
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd -P)"
cd "${REPO_ROOT}"
test -f experiments/gradient_flow/checkpoint_stages.py
date
hostname
case "$(hostname -s)" in
  gpu006) ;;
  *) echo "ERROR: this experiment must execute on gpu006" >&2; exit 2 ;;
esac
nvidia-smi
git rev-parse HEAD
git status --short
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing dirty SAGE checkout" >&2
  exit 2
fi
command -v flock >/dev/null || { echo "flock is required to prevent concurrent study writers" >&2; exit 2; }
exec 9>logs/gradient_flow/alpha_states_gpu006.lock
flock -n 9 || { echo "Another state study is already running in this clone" >&2; exit 2; }
SNN_ENV="${SNN_ENV:-/home/rizk_lab/shared/kiran/envs/snn}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SNN_ENV}"
python --version
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${REPO_ROOT}/.cupy/kernel_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
RUN_DIR="${RUN_DIR:-results/gradient_flow/alpha_state_dependence_seed42}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
HISTORICAL_DIR="${HISTORICAL_DIR:-results/umsen_official_recipe/baseline_seed42}"
FIGURE_DIR="${FIGURE_DIR:-figures/gradient_flow/alpha_state_dependence_seed42}"
# Audit first. Missing args/provenance conflicts abort rather than guessing a recipe.
# Only missing target states authorize ONE fresh deterministic baseline trajectory.
python -u -m experiments.gradient_flow.checkpoint_stages \
  --historical-dir "${HISTORICAL_DIR}" --study-dir "${RUN_DIR}" \
  --data-dir "${DATA_DIR}" --regenerate-missing
python -u -m experiments.gradient_flow.state_intervene \
  --manifest "${RUN_DIR}/states.json" --data-dir "${DATA_DIR}" \
  --output "${RUN_DIR}/probes" --num-batches 25 --batch-size 32 \
  --backend cupy --device cuda --resume
python -u -m analysis.gradient_flow.plot_alpha_states \
  --run-dir "${RUN_DIR}/probes" --output-dir "${FIGURE_DIR}"
date

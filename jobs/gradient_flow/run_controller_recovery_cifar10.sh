#!/bin/bash
#SBATCH --job-name=sage-recovery
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/gradient_flow/controller_recovery_%j.log
#SBATCH --error=logs/gradient_flow/controller_recovery_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
date
hostname
nvidia-smi
git rev-parse HEAD
git status --short
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing dirty checkout" >&2; exit 2; }
[[ "$(hostname -s)" == "gpu002" ]] || { echo "This launcher requires gpu002" >&2; exit 2; }
command -v flock >/dev/null
mkdir -p logs/gradient_flow
exec 9>logs/gradient_flow/controller_recovery.lock
flock -n 9 || { echo "Another recovery writer is active" >&2; exit 2; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SNN_ENV:-/home/rizk_lab/shared/kiran/envs/snn}"
python --version
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${PWD}/.cupy/kernel_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
RUN_DIR="${RUN_DIR:-results/gradient_flow/controller_recovery_seed42}"
FIGURE_DIR="${FIGURE_DIR:-figures/gradient_flow/controller_recovery_seed42}"
python -u -m experiments.gradient_flow.controller_recovery \
  --config configs/gradient_flow/controller_feasibility.yml \
  --manifest "${MANIFEST:-results/gradient_flow/alpha_state_dependence_seed42/states.json}" \
  --data-dir "${DATA_DIR:-data}" --output "${RUN_DIR}" --backend cupy --device cuda --resume
python -u -m analysis.gradient_flow.plot_controller_recovery --run-dir "${RUN_DIR}" --output-dir "${FIGURE_DIR}"
python - "${RUN_DIR}/summary.json" <<'PY'
import json
import sys
gate = json.load(open(sys.argv[1]))["gate"]
print("Frozen-state feasibility gate:", gate)
if not gate["passed"]:
    raise SystemExit(2)
PY
if [[ "${RUN_ONLINE:-0}" != "0" ]]; then
  echo "Online training is NOT implemented/authorized yet. Review real frozen recovery first." >&2
  exit 2
fi
date

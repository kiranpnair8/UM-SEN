#!/bin/bash
#SBATCH --job-name=spikformer-attn-entropy-20ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu005
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/attention_entropy_diagnostic_20ep_%j.log
#SBATCH --error=logs/attention_entropy_diagnostic_20ep_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs
mkdir -p results/attention_entropy_diagnostic_20ep

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snn

python attention_entropy_diagnostic.py \
  --epochs 20 \
  --seed 42 \
  --output-dir results/attention_entropy_diagnostic_20ep \
  --entropy-log-interval 1

test -f results/attention_entropy_diagnostic_20ep/metrics.json
test -f results/attention_entropy_diagnostic_20ep/config.json
compgen -G "results/attention_entropy_diagnostic_20ep/epoch_*.json" > /dev/null

echo "Verified diagnostic outputs in ${SCRIPT_DIR}/results/attention_entropy_diagnostic_20ep"

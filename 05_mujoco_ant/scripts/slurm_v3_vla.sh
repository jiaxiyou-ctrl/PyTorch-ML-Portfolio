#!/bin/bash
#SBATCH --job-name=ant_vla_ppo
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_%j_vla.out
#SBATCH --error=logs/slurm_%j_vla.err
#
# V3 downloads CLIP weights on first run; ensure cache dir is on a writable
# filesystem (e.g. $SCRATCH or $HOME/.cache) if compute nodes have no internet.
# Submit from project root:  sbatch scripts/slurm_v3_vla.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

# module load ...  # set CUDA / conda as on your site

export PYTHONUNBUFFERED=1
# export HF_HOME="$SCRATCH/hf_cache"   # optional: large-model cache

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "PWD: $PWD"
echo "Starting V3 VLA PPO..."

python v3_vla_ppo/vla_train.py

echo "Done."

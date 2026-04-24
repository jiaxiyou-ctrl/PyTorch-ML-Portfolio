#!/bin/bash
#SBATCH --job-name=ant_pixel_ppo
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_%j_pixel.out
#SBATCH --error=logs/slurm_%j_pixel.err
#
# BluePebble / BlueCrystal (Bristol): edit --account, --partition, and module lines to match
# the cluster user guide. Submit from project root:  sbatch scripts/slurm_v2_pixel.sh

set -euo pipefail

# Project root = parent of scripts/
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

# Example: load toolchain (uncomment and adjust)
# module purge
# module load cuda/12.x
# module load miniforge3
# source activate your-env-name

export PYTHONUNBUFFERED=1

# Optional: prefer a specific GPU architecture
# export CUDA_VISIBLE_DEVICES=0

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "PWD: $PWD"
echo "Starting V2 pixel PPO..."

python v2_pixel_ppo/pixel_train.py

echo "Done."

#!/bin/bash
#SBATCH --job-name=pixel-ppo-ant
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_pixel_%j.log
#SBATCH --error=logs/slurm_pixel_%j.err
#
# Adjust modules and conda env for BluePebble / BlueCrystal. Submit from
# 05_mujoco_ant:  sbatch scripts/slurm_v2_pixel.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

# module load languages/python/3.10
# module load libs/cuda/11.8
# source activate pytorch_env

export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1

python v2_pixel_ppo/pixel_train.py \
  --steps 5000000 \
  --device cuda \
  --checkpoint-dir checkpoints_pixel/ \
  --save-interval 100000 \
  --log-dir logs/

echo "Pixel PPO training complete at $(date)"

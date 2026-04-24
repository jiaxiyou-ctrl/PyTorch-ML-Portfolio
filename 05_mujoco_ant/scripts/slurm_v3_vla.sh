#!/bin/bash
#SBATCH --job-name=vla-ppo-ant
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_vla_%j.log
#SBATCH --error=logs/slurm_vla_%j.err
#
# Adjust modules and conda env for your site. Submit from 05_mujoco_ant:
#   sbatch scripts/slurm_v3_vla.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

# module load languages/python/3.10
# module load libs/cuda/11.8
# source activate pytorch_env

export MUJOCO_GL=osmesa
export PYTHONUNBUFFERED=1
# export HF_HOME="$SCRATCH/hf_cache"

python v3_vla_ppo/vla_train.py \
  --steps 10000000 \
  --device cuda \
  --checkpoint-dir checkpoints_vla_v4/ \
  --save-interval 100000 \
  --log-dir logs/

echo "VLA PPO training complete at $(date)"

# HPC / GPU — meeting notes (template)

**With:** Jack Jones (Bristol HPC)  
**Context:** MuJoCo Ant RL project — training on **BluePebble** / **BlueCrystal** with GPU.

## Goals for this meeting

- Confirm how to request **GPU** jobs (queue/partition names, time limits, accounting/project codes).
- Confirm recommended **Python / PyTorch** setup (system modules vs conda/mamba, CUDA versions matching cluster drivers).
- Understand **data paths** (home vs scratch) and I/O: checkpoint writes, logs, and video outputs.
- Any **network** restrictions for `pip`/`huggingface` model downloads (CLIP) on compute nodes; plan for **offline** weights if needed.
- **Interactive vs batch**: when to use a GPU dev session for debugging vs `sbatch`.

## Project facts (this repo)

- **Three code paths:** V1 state PPO (`v1_state_ppo/`), V2 pixel PPO (`v2_pixel_ppo/`), V3 VLA (`v3_vla_ppo/`), plus `shared/`.
- **Training scripts:** `v1_state_ppo/train.py`, `v2_pixel_ppo/pixel_train.py`, `v3_vla_ppo/vla_train.py` (all assume cwd = project root `05_mujoco_ant/` for relative `checkpoints*/` paths).
- **Slurm examples:** `scripts/slurm_v2_pixel.sh`, `scripts/slurm_v3_vla.sh` (edit account/partition and paths before use).

## Questions to ask

1. Which partition and `#SBATCH -A` (or local equivalent) should this project use?
2. Is **multi-GPU** or `torchrun` used on these clusters for single-job scaling, or is one GPU per job the norm for student workloads?
3. **Walltime** recommendations for 1–5M environment steps (rough order of hours)?
4. Best practice for **$HOME** quota vs **scratch** for checkpoints and Conda envs.

## Action items (fill in after the meeting)

- [ ] …

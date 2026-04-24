# MuJoCo Ant Multi-Task Locomotion: From State-based RL to Vision-Language-Action

## Project Overview

This project trains a simulated quadruped robot (MuJoCo Ant-v5) to perform multiple locomotion tasks using reinforcement learning. The project progressed through three versions: **(V1)** state-based PPO using joint angles and velocities, **(V2)** pixel-based PPO learning directly from 84×84 RGB camera images, and **(V3)** a Vision-Language-Action (VLA) architecture that combines pixel observations with natural language task instructions (e.g., "walk forward", "turn left") using a CLIP-based language encoder and multimodal fusion network. The VLA agent successfully learned all 5 tasks with a best reward of **2644.5**.

## Version Evolution Table

| Version | Input | Architecture | Best Reward | Training Steps |
|---------|-------|-------------|-------------|----------------|
| V1 State PPO | Joint angles & velocities | MLP + PPO | ~626 (final mean reward from the V1 run summarized in `results/training_reward_curve.png`; see note below) | 1M |
| V2 Pixel PPO | 84×84 RGB images | CNN + PPO | ~300 | 1M |
| V3 VLA PPO | Pixels + Language instructions | CNN + CLIP + Fusion + PPO | 2644.5 | 5M+ |

**Note on V1:** The peak mean reward **~626** is taken from the original state-based PPO training log used to build `results/training_reward_curve.png` (the published curve was from a long run, typically on the order of 5M environment steps, while 1M steps is a standard short budget for quick experiments).

## Supported Tasks (V3)

- Walk forward
- Walk backward
- Turn left
- Turn right
- Stand still

## Project Structure

```text
05_mujoco_ant/
├── README.md
├── requirements.txt
├── .gitignore
├── .pylintrc
├── v1_state_ppo/           # V1: state-based PPO (train.py, MLP, buffers)
├── v2_pixel_ppo/           # V2: pixel PPO (pixel_train.py, CNN, augmentation)
├── v3_vla_ppo/             # V3: VLA PPO (vla_train.py, CLIP, multimodal fusion)
├── shared/                 # Shared: normalizers, domain randomization, tools
├── scripts/                # Recording, plotting, SLURM job templates
├── logs/                   # training_log_v1.txt …, timestamped train_*.log from runs
├── tests/                  # Smoke, mini-train, domain-randomization checks
├── docs/                   # Meeting / HPC notes (e.g. meeting_notes_jack.md)
├── checkpoints/            # V1 checkpoints (default; gitignored patterns apply)
├── checkpoints_pixel/     # V2
├── checkpoints_vla/         # V3 (older runs)
├── checkpoints_vla_v2/      # V3
├── checkpoints_vla_v4/     # V3 (current default in vla_train.py)
├── results/                 # Plots, GIF, exported videos
│   ├── training_reward_curve.png
│   ├── ant_walking.gif
│   ├── ant_walking.mp4
│   └── videos/
└── demo_videos/            # VLA task demo MP4s (e.g. walk_forward.mp4)
```

## Hardware Requirements

- **GPU:** Single NVIDIA GPU with ≥ 8GB VRAM (V100 / A100 / RTX 3090 class)
- **CPU:** ≥ 4 cores (MuJoCo simulation is CPU-bound)
- **RAM:** ~32GB
- **Storage:** ~20GB for checkpoints and logs
- **Estimated training time:** 12–48 hours per run (5–10M steps), depending on GPU and settings

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
# For V2/V3 (OpenCV) and MP4 recording utilities, also: pip install opencv-python imageio imageio-ffmpeg

# V1: State-based PPO
python v1_state_ppo/train.py --steps 1000000

# V2: Pixel-based PPO
python v2_pixel_ppo/pixel_train.py --steps 5000000 --device cuda

# V3: VLA PPO (requires GPU; CLIP weights downloaded on first run)
python v3_vla_ppo/vla_train.py --steps 10000000 --device cuda

# Resume from checkpoint (use a concrete .pt file under your checkpoint dir, e.g. vla_ppo_final.pt)
python v3_vla_ppo/vla_train.py --resume checkpoints_vla_v4/vla_ppo_final.pt
```

## Run on HPC Cluster

```bash
sbatch scripts/slurm_v3_vla.sh
squeue -u $USER          # check job status
scancel <job_id>         # cancel job
```

(Analogous `scripts/slurm_v2_pixel.sh` is available for pixel PPO. Edit `module` / conda lines for your site before submitting.)

## Results

- **Training curve (V1 state PPO):** see `results/training_reward_curve.png` for the mean-episode-reward curve used in the version table above.
- **Demo videos:** `demo_videos/` contains per-task VLA rollouts (e.g. `walk_forward.mp4`, `turn_left.mp4`).
- **Summary animation:** `results/ant_walking.gif` shows the Ant policy behaviour (V1 / summary visual).

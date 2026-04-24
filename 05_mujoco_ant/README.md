# 05 — MuJoCo Ant RL (PPO, pixel, VLA)

This directory contains **three** reinforcement-learning lineages for **Gymnasium `Ant-v5`**, all implemented in PyTorch: state-based PPO, pixel PPO, and a vision–language–action (VLA) PPO variant. They share small utilities under `shared/`.

**Working directory:** run commands with your current working directory set to this folder (`05_mujoco_ant/`), so paths like `checkpoints/` and `logs/` resolve correctly.

---

## Layout

| Path | Role |
|------|------|
| `v1_state_ppo/` | V1: MLP on proprio state; `train.py` |
| `v2_pixel_ppo/` | V2: pixel observations; `pixel_train.py`; imports `PixelActorCritic` from V1’s `networks` |
| `v3_vla_ppo/` | V3: pixels + language (CLIP), multimodal policy; `vla_train.py` |
| `shared/` | `obs_normalizer`, `reward_normalizer`, `domain_random`, `evaluate`, `explore_env` |
| `scripts/` | Recording, plotting, Slurm template scripts |
| `tests/` | Smoke, mini-train, domain-randomization checks |
| `logs/` | Text training logs (`training_log_v1.txt` …) |
| `checkpoints*`, `results/`, `demo_videos/` | **Untouched** artefact locations (see `.gitignore` for what is not committed) |

Sub-readmes: `v1_state_ppo/README.md`, `v2_pixel_ppo/README.md`, `v3_vla_ppo/README.md`. Meeting prep for HPC: `docs/meeting_notes_jack.md`.

---

## Install

```bash
pip install -r requirements.txt
```

V2/V3 use OpenCV; V3 uses Hugging Face `transformers` (CLIP). MuJoCo assets are pulled in via `gymnasium[mujoco]`.

---

## Train

All three entry points accept `--steps`, `--device` (`auto` / `cuda` / `cpu`), `--checkpoint-dir`, `--save-interval` (in **environment steps**), `--resume`, and `--log-dir`. Training logs go to `logs/train_YYYYMMDD_HHMMSS.log` as well as the console.

```bash
# V1 — state PPO
python v1_state_ppo/train.py --steps 5000000
python v1_state_ppo/train.py --resume checkpoints/ant_ppo_final.pt
python v1_state_ppo/train.py --domain-randomization

# V2 — pixel PPO
python v2_pixel_ppo/pixel_train.py --steps 5000000 --device auto

# V3 — VLA (default checkpoint dir: checkpoints_vla_v4)
python v3_vla_ppo/vla_train.py --steps 10000000 --device cuda
```

**Cluster:** see `scripts/slurm_v2_pixel.sh` and `scripts/slurm_v3_vla.sh` (edit modules, account, and partitions for your site first).

---

## Record video / demos

- **V1 (state) MP4** (Gymnasium `RecordVideo`):

  ```bash
  python scripts/record_v1_state.py --checkpoint checkpoints/ant_ppo_final.pt --episodes 3
  ```

- **V3 (VLA) task videos** (writes under `demo_videos/`):

  ```bash
  python scripts/record_demo.py
  ```

`record_v1_state.py` and `record_demo.py` are **different** (state policy vs VLA); both live under `scripts/`.

---

## Plots and logs

```bash
python scripts/plot_training_curve.py --log logs/training_log_v1.txt --output results/training_reward_curve.png
```

Legacy logs are stored as `logs/training_log_v1.txt` (formerly `training_log.txt`), `training_log_v2.txt`, et cetera.

---

## Tests

With `pytest` (optional) from this directory:

```bash
pytest tests/ -q
```

Or run individual scripts: `python tests/test_domain_random.py` (path setup is included). VLA tests download/use CLIP; expect a cold start the first time.

---

## V1 design notes (original course write-up)

The state agent uses an actor–critic MLP, GAE, running observation and reward normalisation, optional domain randomisation, and linear LR schedule. A typical result curve is in `results/training_reward_curve.png`.

**References:** Schulman et al., *PPO* and *GAE*; [Gymnasium Ant-v5](https://gymnasium.farama.org/environments/mujoco/ant/).

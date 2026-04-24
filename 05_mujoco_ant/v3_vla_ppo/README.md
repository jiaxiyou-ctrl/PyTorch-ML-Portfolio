# V3 — Vision–Language–Action (VLA) PPO

Multi-task Ant control with **pixels + language** (CLIP text encoder, multimodal actor–critic, language-conditioned rewards). Trains with `vla_train.py`.

**Checkpoints:** default `../checkpoints_vla_v4/` in `CONFIG` (other runs may use `checkpoints_vla/`, `checkpoints_vla_v2/`, etc.).

**Depends on V2:** `v2_pixel_ppo/augmentation.py` (data augmentation) is imported at training time.

**Demos:** `../scripts/record_demo.py` loads the latest checkpoint from `checkpoints_vla_v4` and writes task videos under `../demo_videos/`.

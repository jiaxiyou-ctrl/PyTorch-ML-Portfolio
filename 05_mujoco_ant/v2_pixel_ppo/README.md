# V2 — Pixel-based PPO

Trains PPO on **stacked RGB observations** from a MuJoCo render (CNN encoder + policy/value heads). Uses the same `PixelActorCritic` trunk definition as in V1’s `networks.py` (imported from `v1_state_ppo`).

**Entry point:** `pixel_train.py` (from project root).

**Checkpoints:** `../checkpoints_pixel/`.

**Shared with V3:** `augmentation.py` (e.g. random shift) is also used by the VLA agent.

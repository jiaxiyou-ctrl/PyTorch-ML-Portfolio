# V1 — State-based PPO (Ant-v5)

Trains a policy on **proprioceptive state** observations (the default Ant-v5 vector, 105-dim) with PPO, observation/reward normalisation, optional domain randomisation, and checkpoint resume.

**Entry point:** `train.py` (run from the project root `05_mujoco_ant/`, see main README).

**Checkpoints:** `../checkpoints/` (default `save_dir` in code).

**Related:** `shared/` for `ObsNormalizer`, `RewardNormalizer`, `DomainRandomizer`; `scripts/record_v1_state.py` to record MP4 rollouts of a trained V1 policy.

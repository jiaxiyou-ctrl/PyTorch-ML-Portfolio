"""Domain randomization for sim-to-real transfer.

Randomizes MuJoCo physics parameters (gravity, friction, body mass) at
the start of each episode so the learned policy generalizes to
environments with different dynamics.
"""

from typing import Dict, Tuple

import gymnasium as gym
import numpy as np


class DomainRandomizer:
    """Randomizes physics parameters of a MuJoCo environment.

    Default values cache is populated lazily on the first call to
    ``randomize`` so that the randomizer does not depend on the
    environment at construction time.

    Args:
        gravity_range:        Uniform range for the z-component of gravity.
        friction_scale_range: Multiplicative scale range for geom friction.
        mass_scale_range:     Multiplicative scale range for body masses.
        randomize_gravity:    Whether to randomize gravity.
        randomize_friction:   Whether to randomize friction.
        randomize_mass:       Whether to randomize mass.
    """

    def __init__(
        self,
        gravity_range: Tuple[float, float] = (-11.0, -8.0),
        friction_scale_range: Tuple[float, float] = (0.5, 1.5),
        mass_scale_range: Tuple[float, float] = (0.8, 1.2),
        randomize_gravity: bool = True,
        randomize_friction: bool = True,
        randomize_mass: bool = True,
    ) -> None:
        self.gravity_range = gravity_range
        self.friction_scale_range = friction_scale_range
        self.mass_scale_range = mass_scale_range
        self.randomize_gravity = randomize_gravity
        self.randomize_friction = randomize_friction
        self.randomize_mass = randomize_mass

        self._default_friction: np.ndarray | None = None
        self._default_mass: np.ndarray | None = None
        self._initialized = False

    def _save_default_values(self, model: object) -> None:
        """Cache the environment's original friction and mass arrays."""
        self._default_friction = model.geom_friction.copy()
        self._default_mass = model.body_mass.copy()
        self._initialized = True

    def randomize(self, env: gym.Env) -> None:
        """Apply random physics perturbations to *env*.

        Should be called after each ``env.reset()``.

        Args:
            env: A Gymnasium MuJoCo environment instance.
        """
        model = env.unwrapped.model

        if not self._initialized:
            self._save_default_values(model)

        if self.randomize_gravity:
            model.opt.gravity[2] = np.random.uniform(*self.gravity_range)

        if self.randomize_friction:
            scale = np.random.uniform(*self.friction_scale_range)
            model.geom_friction[:] = self._default_friction * scale

        if self.randomize_mass:
            scale = np.random.uniform(*self.mass_scale_range)
            model.body_mass[:] = self._default_mass * scale

    def get_current_params(self, env: gym.Env) -> Dict[str, float]:
        """Return the environment's current physics parameters.

        Args:
            env: A Gymnasium MuJoCo environment instance.

        Returns:
            Dictionary with ``gravity_z``, ``friction_0``, and
            ``mass_total``.
        """
        model = env.unwrapped.model
        return {
            "gravity_z": float(model.opt.gravity[2]),
            "friction_0": float(model.geom_friction[0, 0]),
            "mass_total": float(model.body_mass.sum()),
        }

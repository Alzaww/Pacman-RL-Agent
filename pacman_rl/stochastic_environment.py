"""Pacman environment with stochastic actions."""

import random

from pacman_rl.environment import Action, PacmanEnv


class StochasticPacmanEnv(PacmanEnv):
    """Environment in which Pacman may execute the wrong action."""

    def __init__(
        self,
        *args,
        action_error_probability: float = 0.0,
        action_seed: int | None = None,
        **kwargs,
    ):
        if not 0.0 <= action_error_probability <= 1.0:
            raise ValueError(
                "action_error_probability must be between 0 and 1."
            )

        super().__init__(*args, **kwargs)

        self.action_error_probability = action_error_probability
        self._action_rng = random.Random(action_seed)

    def step(self, action):
        """Execute an action, possibly replacing it with another action."""
        requested_action = Action(action)
        executed_action = requested_action

        if (
            self.action_error_probability > 0.0
            and self._action_rng.random()
            < self.action_error_probability
        ):
            alternative_actions = [
                candidate
                for candidate in Action
                if candidate != requested_action
            ]

            executed_action = self._action_rng.choice(
                alternative_actions
            )

        next_state, reward, done, info = super().step(
            executed_action
        )

        info = dict(info)
        info["requested_action"] = requested_action
        info["executed_action"] = executed_action
        info["action_changed"] = (
            executed_action != requested_action
        )

        return next_state, reward, done, info
"""Tabular Q-learning agent."""

import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent using an epsilon-greedy policy."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        if n_states <= 0:
            raise ValueError("n_states must be strictly positive.")

        if n_actions <= 0:
            raise ValueError("n_actions must be strictly positive.")

        if not 0 < alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")

        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1.")

        if not 0 <= epsilon_min <= epsilon <= 1:
            raise ValueError(
                "epsilon_min and epsilon must satisfy "
                "0 <= epsilon_min <= epsilon <= 1."
            )

        if not 0 < epsilon_decay <= 1:
            raise ValueError(
                "epsilon_decay must be between 0 and 1."
            )

        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros(
            (n_states, n_actions),
            dtype=np.float64,
        )

    def _validate_state(self, state: int) -> None:
        if not 0 <= state < self.n_states:
            raise ValueError(f"Invalid state index: {state}")

    def _validate_action(self, action: int) -> None:
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Invalid action index: {action}")

    def select_action(
        self,
        state: int,
        training: bool = True,
    ) -> int:
        """Select an action using an epsilon-greedy policy."""
        self._validate_state(state)

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        state_values = self.q_table[state]
        best_value = np.max(state_values)
        best_actions = np.flatnonzero(state_values == best_value)

        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """Update one state-action value and return the TD error."""
        self._validate_state(state)
        self._validate_action(action)
        self._validate_state(next_state)

        if done:
            td_target = float(reward)
        else:
            best_next_value = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * best_next_value

        td_error = td_target - self.q_table[state, action]

        self.q_table[state, action] += self.alpha * td_error

        return float(td_error)

    def decay_exploration(self) -> None:
        """Decrease epsilon without going below epsilon_min."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )
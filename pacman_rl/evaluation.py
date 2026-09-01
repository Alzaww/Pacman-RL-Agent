"""Evaluation utilities for learned Pacman policies."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from pacman_rl.agents.q_learning import QLearningAgent
from pacman_rl.environment import PacmanEnv
from pacman_rl.training import position_to_state


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics collected during policy evaluation."""

    episodes: int
    success_rate: float
    ghost_rate: float
    timeout_rate: float
    mean_return: float
    return_std: float
    mean_steps: float


def evaluate_q_learning(
    env: PacmanEnv,
    agent: QLearningAgent,
    episodes: int = 100,
) -> EvaluationResult:
    """Evaluate a Q-learning policy without exploration."""
    if episodes <= 0:
        raise ValueError("episodes must be strictly positive.")

    expected_states = env.rows * env.cols

    if agent.n_states != expected_states:
        raise ValueError(
            f"The agent has {agent.n_states} states, "
            f"but the environment requires {expected_states}."
        )

    if agent.n_actions != env.n_actions:
        raise ValueError(
            f"The agent has {agent.n_actions} actions, "
            f"but the environment requires {env.n_actions}."
        )

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    outcomes: Counter[str] = Counter()

    for _ in range(episodes):
        position = env.reset()
        done = False
        total_return = 0.0
        info: dict[str, str] = {}

        while not done:
            state = position_to_state(
                position,
                env.cols,
            )

            action = agent.select_action(
                state,
                training=False,
            )

            position, reward, done, info = env.step(action)
            total_return += reward

        episode_returns.append(total_return)
        episode_lengths.append(env.steps)
        outcomes[info["outcome"]] += 1

    return EvaluationResult(
        episodes=episodes,
        success_rate=outcomes["dot"] / episodes,
        ghost_rate=outcomes["ghost"] / episodes,
        timeout_rate=outcomes["timeout"] / episodes,
        mean_return=float(np.mean(episode_returns)),
        return_std=float(np.std(episode_returns)),
        mean_steps=float(np.mean(episode_lengths)),
    )
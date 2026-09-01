"""Training utilities for reinforcement learning agents."""

from dataclasses import dataclass

from pacman_rl.agents.q_learning import QLearningAgent
from pacman_rl.environment import PacmanEnv, Position


@dataclass
class TrainingHistory:
    """Measurements collected during Q-learning training."""

    episode_returns: list[float]
    episode_lengths: list[int]
    epsilon_values: list[float]
    outcomes: list[str]


def position_to_state(
    position: Position,
    n_columns: int,
) -> int:
    """Convert a grid position into a Q-table state index."""
    if n_columns <= 0:
        raise ValueError("n_columns must be strictly positive.")

    row, col = position

    if row < 0 or col < 0 or col >= n_columns:
        raise ValueError(f"Invalid grid position: {position}")

    return row * n_columns + col


def train_q_learning(
    env: PacmanEnv,
    agent: QLearningAgent,
    episodes: int,
) -> TrainingHistory:
    """Train a Q-learning agent in a Pacman environment."""
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

    history = TrainingHistory(
        episode_returns=[],
        episode_lengths=[],
        epsilon_values=[],
        outcomes=[],
    )

    for _ in range(episodes):
        position = env.reset()
        done = False
        total_return = 0.0
        info: dict[str, str] = {}

        history.epsilon_values.append(agent.epsilon)

        while not done:
            state = position_to_state(
                position,
                env.cols,
            )

            action = agent.select_action(
                state,
                training=True,
            )

            next_position, reward, done, info = env.step(action)

            next_state = position_to_state(
                next_position,
                env.cols,
            )

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            position = next_position
            total_return += reward

        history.episode_returns.append(total_return)
        history.episode_lengths.append(env.steps)
        history.outcomes.append(info["outcome"])

        agent.decay_exploration()

    return history
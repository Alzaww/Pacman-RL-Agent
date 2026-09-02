"""Training loop for the Deep Q-Network agent."""

from dataclasses import dataclass
import math

from pacman_rl.agents.dqn import DQNAgent
from pacman_rl.environment import PacmanEnv
from pacman_rl.training import position_to_state


@dataclass
class DQNTrainingHistory:
    """Metrics collected during DQN training."""

    episode_returns: list[float]
    episode_lengths: list[int]
    epsilon_values: list[float]
    mean_losses: list[float]
    outcomes: list[str]


def train_dqn(
    env: PacmanEnv,
    agent: DQNAgent,
    episodes: int,
) -> DQNTrainingHistory:
    """Train a DQN agent for several episodes."""
    if episodes <= 0:
        raise ValueError(
            "episodes must be strictly positive."
        )

    history = DQNTrainingHistory(
        episode_returns=[],
        episode_lengths=[],
        epsilon_values=[],
        mean_losses=[],
        outcomes=[],
    )

    for _ in range(episodes):
        position = env.reset()

        state = position_to_state(
            position,
            env.cols,
        )

        done = False
        episode_return = 0.0
        episode_length = 0
        episode_losses: list[float] = []
        info: dict = {}

        history.epsilon_values.append(
            agent.epsilon
        )

        while not done:
            action = agent.select_action(
                state=state,
                training=True,
            )

            (
                next_position,
                reward,
                done,
                info,
            ) = env.step(action)

            next_state = position_to_state(
                next_position,
                env.cols,
            )

            agent.remember(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            loss = agent.learn()

            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_return += reward
            episode_length += 1

        if episode_losses:
            mean_loss = (
                sum(episode_losses)
                / len(episode_losses)
            )
        else:
            mean_loss = math.nan

        history.episode_returns.append(
            episode_return
        )

        history.episode_lengths.append(
            episode_length
        )

        history.mean_losses.append(
            mean_loss
        )

        history.outcomes.append(
            str(info.get("outcome", "unknown"))
        )

        agent.decay_exploration()

    return history
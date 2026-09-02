"""Train and evaluate a DQN agent on Pacman."""

from pathlib import Path

import pandas as pd

from pacman_rl.agents.dqn import DQNAgent
from pacman_rl.dqn_training import train_dqn
from pacman_rl.environment import PacmanEnv
from pacman_rl.evaluation import evaluate_q_learning
from pacman_rl.grids import REFERENCE_GRID


def run_dqn_experiment(
    seed: int = 42,
    training_episodes: int = 1000,
    evaluation_episodes: int = 200,
):
    """Train and evaluate one DQN agent."""
    if training_episodes <= 0:
        raise ValueError(
            "training_episodes must be strictly positive."
        )

    if evaluation_episodes <= 0:
        raise ValueError(
            "evaluation_episodes must be strictly positive."
        )

    training_env = PacmanEnv(
        grid=REFERENCE_GRID,
        random_start=True,
        seed=seed,
    )

    agent = DQNAgent(
        n_states=(
            training_env.rows
            * training_env.cols
        ),
        n_actions=training_env.n_actions,
        hidden_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        replay_capacity=10_000,
        batch_size=64,
        target_update_interval=100,
        seed=seed,
    )

    history = train_dqn(
        env=training_env,
        agent=agent,
        episodes=training_episodes,
    )

    evaluation_env = PacmanEnv(
        grid=REFERENCE_GRID,
        random_start=True,
        seed=10_000 + seed,
    )

    metrics = evaluate_q_learning(
        env=evaluation_env,
        agent=agent,
        episodes=evaluation_episodes,
    )

    return agent, history, metrics


def main() -> None:
    """Run and save the reference DQN experiment."""
    seed = 42
    training_episodes = 1000
    evaluation_episodes = 200

    _, history, metrics = run_dqn_experiment(
        seed=seed,
        training_episodes=training_episodes,
        evaluation_episodes=evaluation_episodes,
    )

    output_directory = Path("results/data")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_results = pd.DataFrame(
        {
            "episode": range(
                1,
                training_episodes + 1,
            ),
            "episode_return": (
                history.episode_returns
            ),
            "episode_length": (
                history.episode_lengths
            ),
            "epsilon": (
                history.epsilon_values
            ),
            "mean_loss": (
                history.mean_losses
            ),
            "outcome": history.outcomes,
        }
    )

    summary_results = pd.DataFrame(
        [
            {
                "seed": seed,
                "training_episodes": (
                    training_episodes
                ),
                "evaluation_episodes": (
                    evaluation_episodes
                ),
                "success_rate": (
                    metrics.success_rate
                ),
                "ghost_rate": (
                    metrics.ghost_rate
                ),
                "timeout_rate": (
                    metrics.timeout_rate
                ),
                "mean_return": (
                    metrics.mean_return
                ),
                "return_std": (
                    metrics.return_std
                ),
                "mean_steps": (
                    metrics.mean_steps
                ),
                "mean_optimal_steps": (
                    metrics.mean_optimal_steps
                ),
                "mean_efficiency_ratio": (
                    metrics.mean_efficiency_ratio
                ),
                "optimal_path_rate": (
                    metrics.optimal_path_rate
                ),
            }
        ]
    )

    history_path = (
        output_directory
        / "dqn_training_history.csv"
    )

    summary_path = (
        output_directory
        / "dqn_reference_results.csv"
    )

    history_results.to_csv(
        history_path,
        index=False,
    )

    summary_results.to_csv(
        summary_path,
        index=False,
    )

    print("DQN experiment")
    print("--------------")
    print(
        "Mean training return over the last "
        f"100 episodes: "
        f"{sum(history.episode_returns[-100:]) / 100:.2f}"
    )
    print(
        f"Success rate: "
        f"{metrics.success_rate:.1%}"
    )
    print(
        f"Ghost rate: "
        f"{metrics.ghost_rate:.1%}"
    )
    print(
        f"Timeout rate: "
        f"{metrics.timeout_rate:.1%}"
    )
    print(
        f"Mean return: "
        f"{metrics.mean_return:.2f}"
    )
    print(
        f"Optimal path rate: "
        f"{metrics.optimal_path_rate:.1%}"
    )
    print()
    print(f"History saved to {history_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
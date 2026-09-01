"""Compare Q-learning learning rates across several seeds."""

from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd

from pacman_rl.agents.q_learning import QLearningAgent
from pacman_rl.environment import PacmanEnv
from pacman_rl.evaluation import evaluate_q_learning
from pacman_rl.training import train_q_learning


RESULT_COLUMNS = [
    "alpha",
    "seed",
    "training_episodes",
    "last_100_mean_return",
    "success_rate",
    "ghost_rate",
    "timeout_rate",
    "mean_return",
    "return_std",
    "mean_steps",
    "mean_optimal_steps",
    "mean_efficiency_ratio",
    "optimal_path_rate",
]


def run_alpha_experiments(
    alpha_values: Iterable[float],
    seeds: Iterable[int],
    training_episodes: int = 1000,
    evaluation_episodes: int = 200,
) -> pd.DataFrame:
    """Train and evaluate every alpha and seed combination."""
    alpha_values = list(alpha_values)
    seeds = list(seeds)

    if not alpha_values:
        raise ValueError("alpha_values cannot be empty.")

    if not seeds:
        raise ValueError("seeds cannot be empty.")

    if training_episodes <= 0:
        raise ValueError(
            "training_episodes must be strictly positive."
        )

    if evaluation_episodes <= 0:
        raise ValueError(
            "evaluation_episodes must be strictly positive."
        )

    results: list[dict[str, float | int]] = []

    for alpha in alpha_values:
        for seed in seeds:
            env = PacmanEnv(
                random_start=True,
                seed=seed,
            )

            agent = QLearningAgent(
                n_states=env.rows * env.cols,
                n_actions=env.n_actions,
                alpha=alpha,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.995,
                seed=seed,
            )

            history = train_q_learning(
                env=env,
                agent=agent,
                episodes=training_episodes,
            )

            metrics = evaluate_q_learning(
                env=env,
                agent=agent,
                episodes=evaluation_episodes,
            )

            recent_returns = history.episode_returns[-100:]

            results.append(
                {
                    "alpha": alpha,
                    "seed": seed,
                    "training_episodes": training_episodes,
                    "last_100_mean_return": float(
                        np.mean(recent_returns)
                    ),
                    "success_rate": metrics.success_rate,
                    "ghost_rate": metrics.ghost_rate,
                    "timeout_rate": metrics.timeout_rate,
                    "mean_return": metrics.mean_return,
                    "return_std": metrics.return_std,
                    "mean_steps": metrics.mean_steps,
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
            )

    return pd.DataFrame(
        results,
        columns=RESULT_COLUMNS,
    )


def main() -> None:
    """Run the complete alpha comparison and save the results."""
    alpha_values = [0.01, 0.1, 0.5, 0.9]
    seeds = [0, 1, 2, 3, 4]

    results = run_alpha_experiments(
        alpha_values=alpha_values,
        seeds=seeds,
        training_episodes=1000,
        evaluation_episodes=200,
    )

    summary = results.groupby("alpha").agg(
        success_rate_mean=("success_rate", "mean"),
        success_rate_std=("success_rate", "std"),
        optimal_path_rate_mean=("optimal_path_rate", "mean"),
        efficiency_mean=("mean_efficiency_ratio", "mean"),
        evaluation_return_mean=("mean_return", "mean"),
        training_return_mean=(
            "last_100_mean_return",
            "mean",
        ),
    )

    output_directory = Path("results/data")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "q_learning_alpha_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("Q-learning alpha comparison")
    print("---------------------------")
    print(summary.round(3))
    print()
    print(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
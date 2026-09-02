"""Evaluate Q-learning under stochastic action execution."""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from pacman_rl.agents.q_learning import QLearningAgent
from pacman_rl.environment import PacmanEnv
from pacman_rl.evaluation import evaluate_q_learning
from pacman_rl.grids import GridLike, REFERENCE_GRID
from pacman_rl.stochastic_environment import (
    StochasticPacmanEnv,
)
from pacman_rl.training import train_q_learning


RESULT_COLUMNS = [
    "action_error_probability",
    "seed",
    "training_episodes",
    "evaluation_episodes",
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


def run_action_noise_experiments(
    grid: GridLike,
    probabilities: Iterable[float],
    seeds: Iterable[int],
    training_episodes: int = 3000,
    evaluation_episodes: int = 100,
) -> pd.DataFrame:
    """Evaluate deterministic policies under action noise."""
    probabilities = list(probabilities)
    seeds = list(seeds)

    if not probabilities:
        raise ValueError("probabilities cannot be empty.")

    if not seeds:
        raise ValueError("seeds cannot be empty.")

    if any(
        probability < 0.0 or probability > 1.0
        for probability in probabilities
    ):
        raise ValueError(
            "Every probability must be between 0 and 1."
        )

    if training_episodes <= 0:
        raise ValueError(
            "training_episodes must be strictly positive."
        )

    if evaluation_episodes <= 0:
        raise ValueError(
            "evaluation_episodes must be strictly positive."
        )

    results: list[dict[str, float | int]] = []

    for seed in seeds:
        training_env = PacmanEnv(
            grid=grid,
            random_start=True,
            seed=seed,
        )

        agent = QLearningAgent(
            n_states=(
                training_env.rows
                * training_env.cols
            ),
            n_actions=training_env.n_actions,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.995,
            seed=seed,
        )

        train_q_learning(
            env=training_env,
            agent=agent,
            episodes=training_episodes,
        )

        for probability in probabilities:
            evaluation_env = StochasticPacmanEnv(
                grid=grid,
                random_start=True,
                seed=10_000 + seed,
                action_error_probability=probability,
                action_seed=20_000 + seed,
            )

            metrics = evaluate_q_learning(
                env=evaluation_env,
                agent=agent,
                episodes=evaluation_episodes,
            )

            results.append(
                {
                    "action_error_probability": (
                        probability
                    ),
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
                    "ghost_rate": metrics.ghost_rate,
                    "timeout_rate": (
                        metrics.timeout_rate
                    ),
                    "mean_return": (
                        metrics.mean_return
                    ),
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
    """Run and save the action-noise experiment."""
    results = run_action_noise_experiments(
        grid=REFERENCE_GRID,
        probabilities=[
            0.0,
            0.05,
            0.10,
            0.20,
        ],
        seeds=[0, 1, 2, 3, 4],
        training_episodes=3000,
        evaluation_episodes=200,
    )

    summary = results.groupby(
        "action_error_probability"
    ).agg(
        success_rate_mean=(
            "success_rate",
            "mean",
        ),
        success_rate_std=(
            "success_rate",
            "std",
        ),
        ghost_rate_mean=(
            "ghost_rate",
            "mean",
        ),
        timeout_rate_mean=(
            "timeout_rate",
            "mean",
        ),
        mean_return=(
            "mean_return",
            "mean",
        ),
        mean_steps=(
            "mean_steps",
            "mean",
        ),
        efficiency_mean=(
            "mean_efficiency_ratio",
            "mean",
        ),
        optimal_path_rate_mean=(
            "optimal_path_rate",
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
        / "q_learning_action_noise_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("Q-learning under action noise")
    print("-----------------------------")
    print(summary.round(3))
    print()
    print(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
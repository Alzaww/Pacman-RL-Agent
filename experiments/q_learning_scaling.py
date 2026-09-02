"""Measure Q-learning performance across several grid sizes."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from pacman_rl.agents.q_learning import QLearningAgent
from pacman_rl.environment import PacmanEnv
from pacman_rl.evaluation import evaluate_q_learning
from pacman_rl.grids import (
    GRID_GROUPS,
    GridLike,
)
from pacman_rl.training import train_q_learning


RESULT_COLUMNS = [
    "grid",
    "layout",
    "rows",
    "cols",
    "n_states",
    "seed",
    "training_episodes",
    "training_time_seconds",
    "episodes_per_second",
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


def run_scaling_experiments(
    grid_groups: Mapping[
        str,
        Iterable[GridLike],
    ],
    seeds: Iterable[int],
    training_episodes: int = 3000,
    evaluation_episodes: int = 200,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Train Q-learning on several layouts for each grid size."""
    grid_groups = {
        grid_name: list(grids)
        for grid_name, grids in grid_groups.items()
    }
    seeds = list(seeds)

    if not grid_groups:
        raise ValueError(
            "grid_groups cannot be empty."
        )

    if any(
        not grids
        for grids in grid_groups.values()
    ):
        raise ValueError(
            "Grid groups cannot contain an empty layout list."
        )

    if not seeds:
        raise ValueError(
            "seeds cannot be empty."
        )

    if training_episodes <= 0:
        raise ValueError(
            "training_episodes must be strictly positive."
        )

    if evaluation_episodes <= 0:
        raise ValueError(
            "evaluation_episodes must be strictly positive."
        )

    results: list[
        dict[str, float | int | str]
    ] = []

    total_runs = sum(
        len(grids) * len(seeds)
        for grids in grid_groups.values()
    )

    progress_bar = tqdm(
        total=total_runs,
        desc="Q-learning scaling",
        unit="run",
        disable=not show_progress,
    )

    try:
        for grid_name, grids in grid_groups.items():
            for layout_index, grid in enumerate(
                grids,
                start=1,
            ):
                for seed in seeds:
                    env = PacmanEnv(
                        grid=grid,
                        random_start=True,
                        seed=seed,
                    )

                    agent = QLearningAgent(
                        n_states=env.rows * env.cols,
                        n_actions=env.n_actions,
                        alpha=0.1,
                        gamma=0.99,
                        epsilon=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.995,
                        seed=seed,
                    )

                    start_time = perf_counter()

                    train_q_learning(
                        env=env,
                        agent=agent,
                        episodes=training_episodes,
                    )

                    training_time = (
                        perf_counter()
                        - start_time
                    )

                    metrics = evaluate_q_learning(
                        env=env,
                        agent=agent,
                        episodes=evaluation_episodes,
                    )

                    results.append(
                        {
                            "grid": grid_name,
                            "layout": layout_index,
                            "rows": env.rows,
                            "cols": env.cols,
                            "n_states": (
                                env.rows * env.cols
                            ),
                            "seed": seed,
                            "training_episodes": (
                                training_episodes
                            ),
                            "training_time_seconds": (
                                training_time
                            ),
                            "episodes_per_second": (
                                training_episodes
                                / training_time
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
                    )

                    progress_bar.update(1)
    finally:
        progress_bar.close()

    return pd.DataFrame(
        results,
        columns=RESULT_COLUMNS,
    )


def main() -> None:
    """Run the complete scaling experiment."""
    results = run_scaling_experiments(
        grid_groups=GRID_GROUPS,
        seeds=[0, 1, 2, 3, 4],
        training_episodes=3000,
        evaluation_episodes=200,
        show_progress=True,
    )

    summary = results.groupby(
        "grid",
        sort=False,
    ).agg(
        rows=("rows", "first"),
        cols=("cols", "first"),
        states=("n_states", "first"),
        layouts=("layout", "nunique"),
        runs=("seed", "size"),
        training_time_mean=(
            "training_time_seconds",
            "mean",
        ),
        training_time_std=(
            "training_time_seconds",
            "std",
        ),
        success_rate_mean=(
            "success_rate",
            "mean",
        ),
        ghost_rate_mean=(
            "ghost_rate",
            "mean",
        ),
        timeout_rate_mean=(
            "timeout_rate",
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

    output_directory = Path(
        "results/data"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "q_learning_scaling_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Q-learning scaling experiment")
    print("-----------------------------")
    print(summary.round(3))
    print()
    print(
        f"Detailed results saved to {output_path}"
    )


if __name__ == "__main__":
    main()
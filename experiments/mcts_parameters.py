"""Compare MCTS simulation budgets and exploration weights."""

import math
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pacman_rl.agents.mcts import MCTSAgent
from pacman_rl.environment import PacmanEnv
from pacman_rl.grids import (
    GridLike,
    REFERENCE_GRID,
)
from pacman_rl.mcts_evaluation import evaluate_mcts


RESULT_COLUMNS = [
    "experiment",
    "simulations",
    "exploration_weight",
    "seed",
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
    "total_decisions",
    "total_search_time",
    "mean_decision_time",
    "decisions_per_second",
]


def run_mcts_parameter_experiments(
    grid: GridLike,
    simulation_budgets: Iterable[int],
    exploration_weights: Iterable[float],
    seeds: Iterable[int],
    evaluation_episodes: int = 20,
    fixed_simulations: int = 200,
    fixed_exploration_weight: float = math.sqrt(2),
    show_progress: bool = False,
) -> pd.DataFrame:
    """Evaluate MCTS budgets and UCT exploration independently."""
    simulation_budgets = list(
        simulation_budgets
    )
    exploration_weights = list(
        exploration_weights
    )
    seeds = list(seeds)

    if not simulation_budgets:
        raise ValueError(
            "simulation_budgets cannot be empty."
        )

    if any(
        budget <= 0
        for budget in simulation_budgets
    ):
        raise ValueError(
            "Simulation budgets must be strictly positive."
        )

    if not exploration_weights:
        raise ValueError(
            "exploration_weights cannot be empty."
        )

    if any(
        weight < 0
        for weight in exploration_weights
    ):
        raise ValueError(
            "Exploration weights cannot be negative."
        )

    if not seeds:
        raise ValueError(
            "seeds cannot be empty."
        )

    if evaluation_episodes <= 0:
        raise ValueError(
            "evaluation_episodes must be strictly positive."
        )

    if fixed_simulations <= 0:
        raise ValueError(
            "fixed_simulations must be strictly positive."
        )

    if fixed_exploration_weight < 0:
        raise ValueError(
            "fixed_exploration_weight cannot be negative."
        )

    configurations = []

    for budget in simulation_budgets:
        configurations.append(
            {
                "experiment": "simulation_budget",
                "simulations": budget,
                "exploration_weight": (
                    fixed_exploration_weight
                ),
            }
        )

    for weight in exploration_weights:
        configurations.append(
            {
                "experiment": "exploration_weight",
                "simulations": fixed_simulations,
                "exploration_weight": weight,
            }
        )

    total_runs = (
        len(configurations)
        * len(seeds)
    )

    progress_bar = tqdm(
        total=total_runs,
        desc="MCTS parameters",
        unit="run",
        disable=not show_progress,
    )

    results: list[
        dict[str, float | int | str]
    ] = []

    try:
        for configuration in configurations:
            for seed in seeds:
                environment = PacmanEnv(
                    grid=grid,
                    random_start=True,
                    seed=seed,
                )

                agent = MCTSAgent(
                    simulations=configuration[
                        "simulations"
                    ],
                    exploration_weight=configuration[
                        "exploration_weight"
                    ],
                    seed=seed,
                )

                metrics = evaluate_mcts(
                    environment=environment,
                    agent=agent,
                    episodes=evaluation_episodes,
                )

                results.append(
                    {
                        "experiment": configuration[
                            "experiment"
                        ],
                        "simulations": configuration[
                            "simulations"
                        ],
                        "exploration_weight": (
                            configuration[
                                "exploration_weight"
                            ]
                        ),
                        "seed": seed,
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
                        "total_decisions": (
                            metrics.total_decisions
                        ),
                        "total_search_time": (
                            metrics.total_search_time
                        ),
                        "mean_decision_time": (
                            metrics.mean_decision_time
                        ),
                        "decisions_per_second": (
                            metrics.decisions_per_second
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
    """Run and save the MCTS parameter experiments."""
    results = run_mcts_parameter_experiments(
        grid=REFERENCE_GRID,
        simulation_budgets=[
            25,
            50,
            100,
            200,
            500,
        ],
        exploration_weights=[
            0.0,
            math.sqrt(2),
            5.0,
            10.0,
        ],
        seeds=[
            0,
            1,
            2,
            3,
            4,
        ],
        evaluation_episodes=20,
        fixed_simulations=200,
        fixed_exploration_weight=math.sqrt(2),
        show_progress=True,
    )

    budget_results = results[
        results["experiment"]
        == "simulation_budget"
    ]

    budget_summary = (
        budget_results
        .groupby("simulations")
        .agg(
            success_rate=(
                "success_rate",
                "mean",
            ),
            ghost_rate=(
                "ghost_rate",
                "mean",
            ),
            timeout_rate=(
                "timeout_rate",
                "mean",
            ),
            mean_return=(
                "mean_return",
                "mean",
            ),
            optimal_path_rate=(
                "optimal_path_rate",
                "mean",
            ),
            mean_decision_time=(
                "mean_decision_time",
                "mean",
            ),
        )
    )

    exploration_results = results[
        results["experiment"]
        == "exploration_weight"
    ]

    exploration_summary = (
        exploration_results
        .groupby("exploration_weight")
        .agg(
            success_rate=(
                "success_rate",
                "mean",
            ),
            ghost_rate=(
                "ghost_rate",
                "mean",
            ),
            timeout_rate=(
                "timeout_rate",
                "mean",
            ),
            mean_return=(
                "mean_return",
                "mean",
            ),
            optimal_path_rate=(
                "optimal_path_rate",
                "mean",
            ),
            mean_decision_time=(
                "mean_decision_time",
                "mean",
            ),
        )
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
        / "mcts_parameter_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print()
    print("MCTS simulation budget")
    print("----------------------")
    print(budget_summary.round(4))

    print()
    print("MCTS exploration weight")
    print("-----------------------")
    print(exploration_summary.round(4))

    print()
    print(
        f"Detailed results saved to {output_path}"
    )


if __name__ == "__main__":
    main()
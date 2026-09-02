"""Tune MCTS exploration and simulation budget."""

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
    "tree_value_scale",
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


def evaluate_configuration(
    grid: GridLike,
    simulations: int,
    exploration_weight: float,
    seed: int,
    evaluation_episodes: int,
    experiment: str,
) -> dict[str, float | int | str]:
    """Evaluate one MCTS parameter configuration."""
    environment = PacmanEnv(
        grid=grid,
        random_start=True,
        seed=seed,
    )

    agent = MCTSAgent(
        simulations=simulations,
        exploration_weight=exploration_weight,
        seed=seed,
    )

    metrics = evaluate_mcts(
        environment=environment,
        agent=agent,
        episodes=evaluation_episodes,
    )

    return {
        "experiment": experiment,
        "tree_value_scale": "normalized_0_1",
        "simulations": simulations,
        "exploration_weight": exploration_weight,
        "seed": seed,
        "evaluation_episodes": evaluation_episodes,
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


def select_exploration_weight(
    exploration_results: pd.DataFrame,
) -> float:
    """Select exploration weight by quality, then decision time."""
    summary = (
        exploration_results
        .groupby("exploration_weight")
        .agg(
            success_rate=(
                "success_rate",
                "mean",
            ),
            optimal_path_rate=(
                "optimal_path_rate",
                "mean",
            ),
            mean_return=(
                "mean_return",
                "mean",
            ),
            mean_decision_time=(
                "mean_decision_time",
                "mean",
            ),
        )
        .reset_index()
    )

    ranked_summary = summary.sort_values(
        by=[
            "success_rate",
            "optimal_path_rate",
            "mean_return",
            "mean_decision_time",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    )

    return float(
        ranked_summary.iloc[0][
            "exploration_weight"
        ]
    )


def run_mcts_parameter_experiments(
    grid: GridLike,
    simulation_budgets: Iterable[int],
    exploration_weights: Iterable[float],
    seeds: Iterable[int],
    evaluation_episodes: int = 20,
    fixed_simulations: int = 200,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, float]:
    """Tune exploration first, then simulation budget."""
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

    total_runs = (
        (
            len(exploration_weights)
            + len(simulation_budgets)
        )
        * len(seeds)
    )

    progress_bar = tqdm(
        total=total_runs,
        desc="Normalized MCTS tuning",
        unit="run",
        disable=not show_progress,
    )

    exploration_rows = []
    budget_rows = []

    try:
        # First stage: tune exploration weight.
        for exploration_weight in exploration_weights:
            for seed in seeds:
                exploration_rows.append(
                    evaluate_configuration(
                        grid=grid,
                        simulations=fixed_simulations,
                        exploration_weight=(
                            exploration_weight
                        ),
                        seed=seed,
                        evaluation_episodes=(
                            evaluation_episodes
                        ),
                        experiment=(
                            "exploration_weight"
                        ),
                    )
                )

                progress_bar.update(1)

        exploration_results = pd.DataFrame(
            exploration_rows,
            columns=RESULT_COLUMNS,
        )

        selected_weight = (
            select_exploration_weight(
                exploration_results
            )
        )

        # Second stage: tune simulation budget
        # with the selected exploration weight.
        for simulations in simulation_budgets:
            for seed in seeds:
                budget_rows.append(
                    evaluate_configuration(
                        grid=grid,
                        simulations=simulations,
                        exploration_weight=(
                            selected_weight
                        ),
                        seed=seed,
                        evaluation_episodes=(
                            evaluation_episodes
                        ),
                        experiment=(
                            "simulation_budget"
                        ),
                    )
                )

                progress_bar.update(1)
    finally:
        progress_bar.close()

    budget_results = pd.DataFrame(
        budget_rows,
        columns=RESULT_COLUMNS,
    )

    results = pd.concat(
        [
            exploration_results,
            budget_results,
        ],
        ignore_index=True,
    )

    return results, selected_weight


def main() -> None:
    """Run and save normalized MCTS parameter tuning."""
    results, selected_weight = (
        run_mcts_parameter_experiments(
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
                0.1,
                0.25,
                0.5,
                1.0,
                math.sqrt(2),
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
            show_progress=True,
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
    print("Normalized MCTS exploration weight")
    print("----------------------------------")
    print(exploration_summary.round(4))

    print()
    print(
        "Selected exploration weight:",
        round(selected_weight, 4),
    )

    print()
    print("MCTS simulation budget")
    print("----------------------")
    print(budget_summary.round(4))

    print()
    print(
        f"Detailed results saved to {output_path}"
    )


if __name__ == "__main__":
    main()
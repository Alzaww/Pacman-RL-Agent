import pytest

from experiments.mcts_parameters import (
    RESULT_COLUMNS,
    run_mcts_parameter_experiments,
    select_exploration_weight,
)
from pacman_rl.grids import REFERENCE_GRID


def test_parameter_experiment_tunes_exploration_then_budget():
    results, selected_weight = (
        run_mcts_parameter_experiments(
            grid=REFERENCE_GRID,
            simulation_budgets=[5, 10],
            exploration_weights=[0.0, 0.5],
            seeds=[42],
            evaluation_episodes=1,
            fixed_simulations=10,
        )
    )

    assert len(results) == 4
    assert list(results.columns) == RESULT_COLUMNS

    assert set(results["experiment"]) == {
        "exploration_weight",
        "simulation_budget",
    }

    assert set(
        results["tree_value_scale"]
    ) == {"normalized_0_1"}

    assert selected_weight in {
        0.0,
        0.5,
    }

    budget_results = results[
        results["experiment"]
        == "simulation_budget"
    ]

    assert set(
        budget_results["simulations"]
    ) == {5, 10}

    assert (
        budget_results["exploration_weight"]
        == selected_weight
    ).all()


def test_exploration_selection_prioritizes_success():
    results, _ = (
        run_mcts_parameter_experiments(
            grid=REFERENCE_GRID,
            simulation_budgets=[5],
            exploration_weights=[0.0, 0.5],
            seeds=[42],
            evaluation_episodes=1,
            fixed_simulations=5,
        )
    )

    exploration_results = results[
        results["experiment"]
        == "exploration_weight"
    ].copy()

    exploration_results.loc[
        exploration_results[
            "exploration_weight"
        ] == 0.0,
        "success_rate",
    ] = 0.0

    exploration_results.loc[
        exploration_results[
            "exploration_weight"
        ] == 0.5,
        "success_rate",
    ] = 1.0

    selected_weight = (
        select_exploration_weight(
            exploration_results
        )
    )

    assert selected_weight == 0.5


def test_parameter_experiment_records_runtime():
    results, _ = (
        run_mcts_parameter_experiments(
            grid=REFERENCE_GRID,
            simulation_budgets=[5],
            exploration_weights=[0.0],
            seeds=[42],
            evaluation_episodes=1,
            fixed_simulations=5,
        )
    )

    assert (
        results["total_decisions"] > 0
    ).all()

    assert (
        results["total_search_time"] > 0
    ).all()

    assert (
        results["mean_decision_time"] > 0
    ).all()

    assert (
        results["decisions_per_second"] > 0
    ).all()


@pytest.mark.parametrize(
    (
        "simulation_budgets",
        "exploration_weights",
        "seeds",
        "message",
    ),
    [
        ([], [0.5], [0], "simulation_budgets"),
        ([10], [], [0], "exploration_weights"),
        ([10], [0.5], [], "seeds"),
        ([0], [0.5], [0], "strictly positive"),
        ([10], [-1.0], [0], "cannot be negative"),
    ],
)
def test_parameter_experiment_rejects_invalid_inputs(
    simulation_budgets,
    exploration_weights,
    seeds,
    message,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        run_mcts_parameter_experiments(
            grid=REFERENCE_GRID,
            simulation_budgets=simulation_budgets,
            exploration_weights=exploration_weights,
            seeds=seeds,
            evaluation_episodes=1,
        )


def test_parameter_experiment_rejects_invalid_episode_count():
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        run_mcts_parameter_experiments(
            grid=REFERENCE_GRID,
            simulation_budgets=[10],
            exploration_weights=[0.5],
            seeds=[0],
            evaluation_episodes=0,
        )
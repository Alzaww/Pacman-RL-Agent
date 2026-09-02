import pytest

from experiments.q_learning_scaling import (
    RESULT_COLUMNS,
    run_scaling_experiments,
)
from pacman_rl.grids import GRID_GROUPS


def test_scaling_experiment_returns_one_row_per_run():
    grid_groups = {
        "4x5": GRID_GROUPS["4x5"][:2],
        "6x6": GRID_GROUPS["6x6"][:2],
    }

    results = run_scaling_experiments(
        grid_groups=grid_groups,
        seeds=[1, 2],
        training_episodes=100,
        evaluation_episodes=10,
    )

    assert len(results) == 8
    assert list(results.columns) == RESULT_COLUMNS
    assert set(results["grid"]) == {
        "4x5",
        "6x6",
    }
    assert set(results["layout"]) == {1, 2}
    assert set(results["seed"]) == {1, 2}


def test_scaling_experiment_records_grid_dimensions():
    grid_groups = {
        "4x5": GRID_GROUPS["4x5"][:1],
        "6x6": GRID_GROUPS["6x6"][:1],
    }

    results = run_scaling_experiments(
        grid_groups=grid_groups,
        seeds=[42],
        training_episodes=100,
        evaluation_episodes=10,
    )

    reference_result = results[
        results["grid"] == "4x5"
    ].iloc[0]

    larger_result = results[
        results["grid"] == "6x6"
    ].iloc[0]

    assert reference_result["layout"] == 1
    assert reference_result["rows"] == 4
    assert reference_result["cols"] == 5
    assert reference_result["n_states"] == 20

    assert larger_result["layout"] == 1
    assert larger_result["rows"] == 6
    assert larger_result["cols"] == 6
    assert larger_result["n_states"] == 36

    assert (
        results["training_time_seconds"] > 0
    ).all()

    assert (
        results["episodes_per_second"] > 0
    ).all()


def test_scaling_experiment_rejects_empty_groups():
    with pytest.raises(
        ValueError,
        match="grid_groups cannot be empty",
    ):
        run_scaling_experiments(
            grid_groups={},
            seeds=[42],
        )

    with pytest.raises(
        ValueError,
        match="empty layout list",
    ):
        run_scaling_experiments(
            grid_groups={"4x5": []},
            seeds=[42],
        )


def test_scaling_experiment_rejects_empty_seeds():
    with pytest.raises(
        ValueError,
        match="seeds cannot be empty",
    ):
        run_scaling_experiments(
            grid_groups={
                "4x5": GRID_GROUPS["4x5"][:1],
            },
            seeds=[],
        )


@pytest.mark.parametrize(
    ("training_episodes", "evaluation_episodes"),
    [
        (0, 10),
        (100, 0),
    ],
)
def test_scaling_experiment_rejects_invalid_episode_counts(
    training_episodes,
    evaluation_episodes,
):
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        run_scaling_experiments(
            grid_groups={
                "4x5": GRID_GROUPS["4x5"][:1],
            },
            seeds=[42],
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_episodes,
        )
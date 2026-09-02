import pytest

from pacman_rl.environment import Action
from pacman_rl.grids import REFERENCE_GRID
from pacman_rl.stochastic_environment import (
    StochasticPacmanEnv,
)


def test_zero_error_probability_keeps_requested_action():
    env = StochasticPacmanEnv(
        grid=REFERENCE_GRID,
        action_error_probability=0.0,
        action_seed=42,
    )

    env.reset()

    _, _, _, info = env.step(Action.LEFT)

    assert info["requested_action"] == Action.LEFT
    assert info["executed_action"] == Action.LEFT
    assert info["action_changed"] is False


def test_full_error_probability_changes_requested_action():
    env = StochasticPacmanEnv(
        grid=REFERENCE_GRID,
        action_error_probability=1.0,
        action_seed=42,
    )

    env.reset()

    _, _, _, info = env.step(Action.LEFT)

    assert info["requested_action"] == Action.LEFT
    assert info["executed_action"] != Action.LEFT
    assert info["action_changed"] is True


def test_stochastic_actions_are_reproducible():
    first_env = StochasticPacmanEnv(
        grid=REFERENCE_GRID,
        action_error_probability=1.0,
        action_seed=123,
    )

    second_env = StochasticPacmanEnv(
        grid=REFERENCE_GRID,
        action_error_probability=1.0,
        action_seed=123,
    )

    first_actions = []
    second_actions = []

    for _ in range(20):
        first_env.reset()
        second_env.reset()

        _, _, _, first_info = first_env.step(
            Action.LEFT
        )

        _, _, _, second_info = second_env.step(
            Action.LEFT
        )

        first_actions.append(
            first_info["executed_action"]
        )

        second_actions.append(
            second_info["executed_action"]
        )

    assert first_actions == second_actions


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1],
)
def test_rejects_invalid_error_probability(probability):
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        StochasticPacmanEnv(
            grid=REFERENCE_GRID,
            action_error_probability=probability,
        )
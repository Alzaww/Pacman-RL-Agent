import pytest

from pacman_rl.environment import ACTION_DELTAS, Action, PacmanEnv
from pacman_rl.grids import EMPTY


def test_actions_follow_assignment_order():
    assert list(Action) == [
        Action.UP,
        Action.LEFT,
        Action.RIGHT,
        Action.DOWN,
    ]

    assert ACTION_DELTAS[Action.UP] == (-1, 0)
    assert ACTION_DELTAS[Action.LEFT] == (0, -1)
    assert ACTION_DELTAS[Action.RIGHT] == (0, 1)
    assert ACTION_DELTAS[Action.DOWN] == (1, 0)


def test_environment_reads_reference_grid():
    env = PacmanEnv()

    assert env.rows == 4
    assert env.cols == 5
    assert env.n_actions == 4
    assert env.start_position == (0, 4)
    assert env.ghost_position == (2, 0)
    assert env.dot_position == (3, 2)


def test_pacman_is_removed_from_fixed_grid():
    env = PacmanEnv()

    row, col = env.start_position
    assert env.grid[row][col] == EMPTY


def test_reset_restores_initial_state():
    env = PacmanEnv()

    env.pacman_position = (1, 1)
    env.steps = 8
    env.done = True

    state = env.reset()

    assert state == (0, 4)
    assert env.pacman_position == (0, 4)
    assert env.steps == 0
    assert env.done is False
    assert env.last_action is None


def test_valid_movement():
    env = PacmanEnv()

    state, reward, done, info = env.step(Action.LEFT)

    assert state == (0, 3)
    assert reward == -1
    assert done is False
    assert info == {}


def test_boundary_prevents_movement():
    env = PacmanEnv()

    state, reward, done, _ = env.step(Action.UP)

    assert state == (0, 4)
    assert reward == -1
    assert done is False


def test_wall_prevents_movement():
    env = PacmanEnv()

    state, reward, done, _ = env.step(Action.DOWN)

    assert state == (0, 4)
    assert reward == -1
    assert done is False


def test_reaching_dot_ends_episode():
    env = PacmanEnv()

    actions = [
        Action.LEFT,
        Action.LEFT,
        Action.LEFT,
        Action.DOWN,
        Action.DOWN,
        Action.DOWN,
        Action.RIGHT,
    ]

    total_reward = 0

    for action in actions:
        _, reward, done, info = env.step(action)
        total_reward += reward

    assert env.pacman_position == env.dot_position
    assert total_reward == 4
    assert done is True
    assert info["outcome"] == "dot"


def test_reaching_ghost_ends_episode():
    grid = [
        ["P", "G"],
        [".", "D"],
    ]
    env = PacmanEnv(grid)

    state, reward, done, info = env.step(Action.RIGHT)

    assert state == (0, 1)
    assert reward == -10
    assert done is True
    assert info["outcome"] == "ghost"


def test_maximum_steps_ends_episode():
    env = PacmanEnv(max_steps=2)

    env.step(Action.UP)
    _, reward, done, info = env.step(Action.UP)

    assert reward == -1
    assert done is True
    assert info["outcome"] == "timeout"


def test_step_after_terminal_state_is_rejected():
    grid = [
        ["P", "G"],
        [".", "D"],
    ]
    env = PacmanEnv(grid)

    env.step(Action.RIGHT)

    with pytest.raises(RuntimeError, match="episode is finished"):
        env.step(Action.LEFT)


def test_invalid_action_is_rejected():
    env = PacmanEnv()

    with pytest.raises(ValueError, match="Invalid action"):
        env.step(99)
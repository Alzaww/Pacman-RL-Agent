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
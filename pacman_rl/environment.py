"""Core Pacman grid-world environment."""

import random
from enum import IntEnum
from typing import Any

from pacman_rl.grids import (
    DOT,
    EMPTY,
    GHOST,
    PACMAN,
    REFERENCE_GRID,
    WALL,
    GridLike,
    copy_grid,
    validate_grid,
)


Position = tuple[int, int]
StepResult = tuple[Position, int, bool, dict[str, Any]]


class Action(IntEnum):
    """Actions in the order specified by the assignment."""

    UP = 0
    LEFT = 1
    RIGHT = 2
    DOWN = 3


ACTION_DELTAS: dict[Action, Position] = {
    Action.UP: (-1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
}


class PacmanEnv:
    """Simplified Pacman environment with a fixed grid."""

    def __init__(
        self,
        grid: GridLike = REFERENCE_GRID,
        max_steps: int | None = None,
        random_start: bool = False,
        seed: int | None = None,
    ) -> None:
        validate_grid(grid)

        self.grid = copy_grid(grid)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.n_actions = len(Action)

        self.start_position = self._find_cell(PACMAN)
        self.ghost_position = self._find_cell(GHOST)
        self.dot_position = self._find_cell(DOT)

        # Pacman's position is dynamic and must not remain in the fixed grid.
        start_row, start_col = self.start_position
        self.grid[start_row][start_col] = EMPTY

        self.random_start = random_start
        self._rng = random.Random(seed)

        self.valid_start_positions = [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if self.grid[row][col] not in {WALL, GHOST, DOT}
        ]

        if max_steps is None:
            max_steps = 2 * self.rows * self.cols

        if max_steps <= 0:
            raise ValueError("max_steps must be strictly positive.")

        self.max_steps = max_steps
        self.pacman_position = self.start_position
        self.steps = 0
        self.done = False
        self.last_action: Action | None = None

    def _find_cell(self, symbol: str) -> Position:
        """Return the position of a symbol in the grid."""
        for row_index, row in enumerate(self.grid):
            for col_index, cell in enumerate(row):
                if cell == symbol:
                    return row_index, col_index

        raise ValueError(f"Cell {symbol!r} was not found in the grid.")

    def reset(self, seed: int | None = None) -> Position:
        """Reset the episode and return Pacman's initial position."""
        if seed is not None:
            self._rng.seed(seed)

        if self.random_start:
            self.pacman_position = self._rng.choice(
                self.valid_start_positions
            )
        else:
            self.pacman_position = self.start_position

        self.steps = 0
        self.done = False
        self.last_action = None

        return self.pacman_position

    def step(self, action: Action | int) -> StepResult:
        """Execute one action in the environment."""
        if self.done:
            raise RuntimeError(
                "The episode is finished. Call reset() first."
            )

        try:
            selected_action = Action(action)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid action: {action!r}") from error

        self.last_action = selected_action

        row, col = self.pacman_position
        row_delta, col_delta = ACTION_DELTAS[selected_action]

        next_row = row + row_delta
        next_col = col + col_delta

        inside_grid = (
            0 <= next_row < self.rows
            and 0 <= next_col < self.cols
        )

        if (
            not inside_grid
            or self.grid[next_row][next_col] == WALL
        ):
            next_row, next_col = row, col

        self.pacman_position = (next_row, next_col)
        self.steps += 1

        reward = -1
        info: dict[str, Any] = {}

        if self.pacman_position == self.dot_position:
            reward = 10
            self.done = True
            info["outcome"] = "dot"

        elif self.pacman_position == self.ghost_position:
            reward = -10
            self.done = True
            info["outcome"] = "ghost"

        elif self.steps >= self.max_steps:
            self.done = True
            info["outcome"] = "timeout"

        return self.pacman_position, reward, self.done, info
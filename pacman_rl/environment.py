"""Core Pacman grid-world environment."""

from enum import IntEnum

from pacman_rl.grids import (
    DOT,
    EMPTY,
    GHOST,
    PACMAN,
    REFERENCE_GRID,
    GridLike,
    copy_grid,
    validate_grid,
)


Position = tuple[int, int]


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

    def __init__(self, grid: GridLike = REFERENCE_GRID) -> None:
        validate_grid(grid)

        self.grid = copy_grid(grid)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.n_actions = len(Action)

        self.start_position = self._find_cell(PACMAN)
        self.ghost_position = self._find_cell(GHOST)
        self.dot_position = self._find_cell(DOT)

        # Pacman's position is dynamic and must not remain inside the fixed grid.
        start_row, start_col = self.start_position
        self.grid[start_row][start_col] = EMPTY

        self.pacman_position = self.start_position
        self.steps = 0
        self.done = False

    def _find_cell(self, symbol: str) -> Position:
        """Return the position of a symbol in the grid."""
        for row_index, row in enumerate(self.grid):
            for col_index, cell in enumerate(row):
                if cell == symbol:
                    return row_index, col_index

        raise ValueError(f"Cell {symbol!r} was not found in the grid.")

    def reset(self) -> Position:
        """Reset the episode and return Pacman's initial position."""
        self.pacman_position = self.start_position
        self.steps = 0
        self.done = False
        return self.pacman_position
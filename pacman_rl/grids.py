"""Grid definitions and validation utilities."""

from collections import Counter
from collections.abc import Sequence


EMPTY = "."
PACMAN = "P"
WALL = "B"
GHOST = "G"
DOT = "D"

VALID_CELLS = frozenset({EMPTY, PACMAN, WALL, GHOST, DOT})

Grid = tuple[tuple[str, ...], ...]
GridLike = Sequence[Sequence[str]]


REFERENCE_GRID: Grid = (
    (EMPTY, EMPTY, EMPTY, EMPTY, PACMAN),
    (EMPTY, EMPTY, WALL, WALL, WALL),
    (GHOST, EMPTY, WALL, EMPTY, EMPTY),
    (EMPTY, EMPTY, DOT, EMPTY, EMPTY),
)


def validate_grid(grid: GridLike) -> None:
    """Validate the dimensions and required entities of a Pacman grid."""
    if not grid:
        raise ValueError("The grid cannot be empty.")

    width = len(grid[0])
    if width == 0:
        raise ValueError("Grid rows cannot be empty.")

    if any(len(row) != width for row in grid):
        raise ValueError("All grid rows must have the same length.")

    cells = [cell for row in grid for cell in row]
    unknown_cells = set(cells) - VALID_CELLS

    if unknown_cells:
        raise ValueError(f"Unknown grid cells: {sorted(unknown_cells)}")

    counts = Counter(cells)

    for symbol, name in (
        (PACMAN, "Pacman"),
        (GHOST, "ghost"),
        (DOT, "PAC-DOT"),
    ):
        if counts[symbol] != 1:
            raise ValueError(
                f"The grid must contain exactly one {name}; "
                f"found {counts[symbol]}."
            )


def copy_grid(grid: GridLike) -> list[list[str]]:
    """Return a mutable copy of a grid."""
    return [list(row) for row in grid]
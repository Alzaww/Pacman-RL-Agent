import pytest

from pacman_rl.grids import REFERENCE_GRID, copy_grid, validate_grid


def test_reference_grid_is_valid():
    validate_grid(REFERENCE_GRID)


def test_copy_grid_returns_independent_copy():
    copied_grid = copy_grid(REFERENCE_GRID)
    copied_grid[0][0] = "B"

    assert REFERENCE_GRID[0][0] == "."
    assert copied_grid[0][0] == "B"


def test_rejects_non_rectangular_grid():
    grid = [
        [".", ".", "P"],
        ["G", "D"],
    ]

    with pytest.raises(ValueError, match="same length"):
        validate_grid(grid)


def test_rejects_unknown_cell():
    grid = [
        ["P", "X"],
        ["G", "D"],
    ]

    with pytest.raises(ValueError, match="Unknown grid cells"):
        validate_grid(grid)


def test_rejects_grid_without_ghost():
    grid = [
        ["P", "."],
        [".", "D"],
    ]

    with pytest.raises(ValueError, match="exactly one ghost"):
        validate_grid(grid)
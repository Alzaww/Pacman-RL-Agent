import pytest

from pacman_rl.evaluation import shortest_path_length
from pacman_rl.grids import (
    BENCHMARK_GRIDS,
    DOT,
    GHOST,
    PACMAN,
    REFERENCE_GRID,
    WALL,
    copy_grid,
    validate_grid,
)


def find_cell(grid, symbol):
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell == symbol:
                return row_index, col_index

    raise AssertionError(
        f"Cell {symbol!r} was not found."
    )


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

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        validate_grid(grid)


def test_rejects_unknown_cell():
    grid = [
        ["P", "X"],
        ["G", "D"],
    ]

    with pytest.raises(
        ValueError,
        match="Unknown grid cells",
    ):
        validate_grid(grid)


def test_rejects_grid_without_ghost():
    grid = [
        ["P", "."],
        [".", "D"],
    ]

    with pytest.raises(
        ValueError,
        match="exactly one ghost",
    ):
        validate_grid(grid)


@pytest.mark.parametrize(
    ("grid_name", "expected_shape"),
    [
        ("4x5", (4, 5)),
        ("6x6", (6, 6)),
        ("8x8", (8, 8)),
        ("10x10", (10, 10)),
    ],
)
def test_benchmark_grid_dimensions(
    grid_name,
    expected_shape,
):
    grid = BENCHMARK_GRIDS[grid_name]

    validate_grid(grid)

    rows = len(grid)
    cols = len(grid[0])

    assert (rows, cols) == expected_shape


@pytest.mark.parametrize(
    "grid_name",
    [
        "4x5",
        "6x6",
        "8x8",
        "10x10",
    ],
)
def test_every_valid_start_can_reach_dot(
    grid_name,
):
    grid = BENCHMARK_GRIDS[grid_name]
    goal = find_cell(grid, DOT)

    valid_starts = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, cell in enumerate(row)
        if cell not in {WALL, GHOST, DOT}
    ]

    assert find_cell(grid, PACMAN) in valid_starts

    for start in valid_starts:
        distance = shortest_path_length(
            grid=grid,
            start=start,
            goal=goal,
        )

        assert distance is not None
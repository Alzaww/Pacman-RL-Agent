import pytest

from pacman_rl.evaluation import shortest_path_length
from pacman_rl.grids import (
    BENCHMARK_GRIDS,
    DOT,
    GHOST,
    GRID_GROUPS,
    PACMAN,
    REFERENCE_GRID,
    WALL,
    copy_grid,
    validate_grid,
)


EXPECTED_SHAPES = {
    "4x5": (4, 5),
    "6x6": (6, 6),
    "8x8": (8, 8),
    "10x10": (10, 10),
}

GRID_CASES = [
    (grid_name, grid_index)
    for grid_name, grids in GRID_GROUPS.items()
    for grid_index in range(len(grids))
]


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
    EXPECTED_SHAPES.items(),
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
    EXPECTED_SHAPES,
)
def test_each_size_has_ten_distinct_grids(
    grid_name,
):
    grids = GRID_GROUPS[grid_name]

    assert len(grids) == 10
    assert len(set(grids)) == 10


@pytest.mark.parametrize(
    "grid_name",
    EXPECTED_SHAPES,
)
def test_original_grid_is_first_in_each_group(
    grid_name,
):
    assert (
        GRID_GROUPS[grid_name][0]
        == BENCHMARK_GRIDS[grid_name]
    )


@pytest.mark.parametrize(
    ("grid_name", "grid_index"),
    GRID_CASES,
)
def test_group_grid_is_valid_and_has_expected_size(
    grid_name,
    grid_index,
):
    grid = GRID_GROUPS[grid_name][grid_index]
    expected_shape = EXPECTED_SHAPES[grid_name]

    validate_grid(grid)

    rows = len(grid)
    cols = len(grid[0])

    assert (rows, cols) == expected_shape


@pytest.mark.parametrize(
    ("grid_name", "grid_index"),
    GRID_CASES,
)
def test_every_valid_start_can_reach_dot(
    grid_name,
    grid_index,
):
    grid = GRID_GROUPS[grid_name][grid_index]
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
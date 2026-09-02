"""Grid definitions and validation utilities."""

import random
from collections import Counter, deque
from collections.abc import Sequence


EMPTY = "."
PACMAN = "P"
WALL = "B"
GHOST = "G"
DOT = "D"

VALID_CELLS = frozenset({
    EMPTY,
    PACMAN,
    WALL,
    GHOST,
    DOT,
})

Grid = tuple[tuple[str, ...], ...]
GridLike = Sequence[Sequence[str]]
GridGroup = tuple[Grid, ...]
Position = tuple[int, int]


def _make_grid(*rows: str) -> Grid:
    """Create an immutable grid from readable string rows."""
    return tuple(
        tuple(row)
        for row in rows
    )


REFERENCE_GRID: Grid = _make_grid(
    "....P",
    "..BBB",
    "G.B..",
    "..D..",
)

GRID_6X6: Grid = _make_grid(
    ".....P",
    ".BB...",
    "..BB..",
    "G...B.",
    "..B...",
    "....D.",
)

GRID_8X8: Grid = _make_grid(
    ".......P",
    ".BB..B..",
    "..BB....",
    "G...B.B.",
    "..B.....",
    ".B..B...",
    ".....B..",
    "......D.",
)

GRID_10X10: Grid = _make_grid(
    ".........P",
    ".BB..B....",
    "..BB...B..",
    "G...B.B...",
    "..B.....B.",
    ".B..B.....",
    ".....B....",
    "......B...",
    "........B.",
    ".........D",
)


BENCHMARK_GRIDS: dict[str, Grid] = {
    "4x5": REFERENCE_GRID,
    "6x6": GRID_6X6,
    "8x8": GRID_8X8,
    "10x10": GRID_10X10,
}


def _connected_cells(
    rows: int,
    cols: int,
    blocked: set[Position],
) -> set[Position]:
    """Return cells reachable without crossing blocked positions."""
    available_cells = {
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if (row, col) not in blocked
    }

    if not available_cells:
        return set()

    start = next(iter(available_cells))
    visited = {start}
    queue = deque([start])

    while queue:
        row, col = queue.popleft()

        for row_change, col_change in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            neighbour = (
                row + row_change,
                col + col_change,
            )

            if (
                neighbour in available_cells
                and neighbour not in visited
            ):
                visited.add(neighbour)
                queue.append(neighbour)

    return visited


def _is_connected(
    rows: int,
    cols: int,
    blocked: set[Position],
) -> bool:
    """Check that every non-blocked cell belongs to one component."""
    available_count = (
        rows * cols
        - len(blocked)
    )

    return len(
        _connected_cells(
            rows=rows,
            cols=cols,
            blocked=blocked,
        )
    ) == available_count


def _generate_grid(
    rows: int,
    cols: int,
    seed: int,
    wall_ratio: float = 0.15,
) -> Grid:
    """Generate one deterministic, fully connected grid layout."""
    random_generator = random.Random(seed)

    cells = [
        (row, col)
        for row in range(rows)
        for col in range(cols)
    ]

    wall_candidates = cells.copy()
    random_generator.shuffle(wall_candidates)

    target_wall_count = max(
        1,
        round(rows * cols * wall_ratio),
    )

    walls: set[Position] = set()

    for candidate in wall_candidates:
        if len(walls) == target_wall_count:
            break

        candidate_walls = walls | {candidate}

        if _is_connected(
            rows=rows,
            cols=cols,
            blocked=candidate_walls,
        ):
            walls.add(candidate)

    ghost_candidates = [
        cell
        for cell in cells
        if cell not in walls
    ]
    random_generator.shuffle(ghost_candidates)

    ghost_position = None

    for candidate in ghost_candidates:
        if _is_connected(
            rows=rows,
            cols=cols,
            blocked=walls | {candidate},
        ):
            ghost_position = candidate
            break

    if ghost_position is None:
        raise RuntimeError(
            "Could not place the ghost without disconnecting the grid."
        )

    entity_candidates = [
        cell
        for cell in cells
        if cell not in walls
        and cell != ghost_position
    ]
    random_generator.shuffle(entity_candidates)

    pacman_position = entity_candidates[0]
    dot_position = entity_candidates[1]

    mutable_grid = [
        [EMPTY for _ in range(cols)]
        for _ in range(rows)
    ]

    for row, col in walls:
        mutable_grid[row][col] = WALL

    ghost_row, ghost_col = ghost_position
    pacman_row, pacman_col = pacman_position
    dot_row, dot_col = dot_position

    mutable_grid[ghost_row][ghost_col] = GHOST
    mutable_grid[pacman_row][pacman_col] = PACMAN
    mutable_grid[dot_row][dot_col] = DOT

    return tuple(
        tuple(row)
        for row in mutable_grid
    )


def _build_grid_group(
    reference_grid: Grid,
    rows: int,
    cols: int,
    first_seed: int,
) -> GridGroup:
    """Combine the original grid with nine reproducible layouts."""
    generated_grids = tuple(
        _generate_grid(
            rows=rows,
            cols=cols,
            seed=first_seed + offset,
        )
        for offset in range(9)
    )

    return (
        reference_grid,
        *generated_grids,
    )


GRID_GROUPS: dict[str, GridGroup] = {
    "4x5": _build_grid_group(
        reference_grid=REFERENCE_GRID,
        rows=4,
        cols=5,
        first_seed=4500,
    ),
    "6x6": _build_grid_group(
        reference_grid=GRID_6X6,
        rows=6,
        cols=6,
        first_seed=6600,
    ),
    "8x8": _build_grid_group(
        reference_grid=GRID_8X8,
        rows=8,
        cols=8,
        first_seed=8800,
    ),
    "10x10": _build_grid_group(
        reference_grid=GRID_10X10,
        rows=10,
        cols=10,
        first_seed=10100,
    ),
}


def validate_grid(grid: GridLike) -> None:
    """Validate the dimensions and required entities of a Pacman grid."""
    if not grid:
        raise ValueError("The grid cannot be empty.")

    width = len(grid[0])

    if width == 0:
        raise ValueError("Grid rows cannot be empty.")

    if any(len(row) != width for row in grid):
        raise ValueError(
            "All grid rows must have the same length."
        )

    cells = [
        cell
        for row in grid
        for cell in row
    ]

    unknown_cells = set(cells) - VALID_CELLS

    if unknown_cells:
        raise ValueError(
            f"Unknown grid cells: {sorted(unknown_cells)}"
        )

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


def copy_grid(
    grid: GridLike,
) -> list[list[str]]:
    """Return a mutable copy of a grid."""
    return [
        list(row)
        for row in grid
    ]
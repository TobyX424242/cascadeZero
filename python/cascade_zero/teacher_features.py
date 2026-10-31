from __future__ import annotations

from typing import Iterable

import numpy as np

BOARD_SIDE = 8
BOARD_CELLS = BOARD_SIDE * BOARD_SIDE
TEACHER_FEATURE_PLANES = 32
FEATURE_LAYOUT_HEIGHT_BUCKETS_V2 = "height_buckets_v2"

PLACE_ACTIONS = BOARD_CELLS
MOVE_ACTIONS = BOARD_CELLS * 4
EAT_ACTIONS = BOARD_CELLS * 4
CASCADE_ACTIONS = BOARD_CELLS * 4
TOTAL_ACTIONS = PLACE_ACTIONS + MOVE_ACTIONS + EAT_ACTIONS + CASCADE_ACTIONS

__all__ = [
    "TEACHER_FEATURE_PLANES",
    "FEATURE_LAYOUT_HEIGHT_BUCKETS_V2",
    "TOTAL_ACTIONS",
    "encode_teacher_feature_planes",
    "flip_action_index",
    "resolve_feature_layout_version",
]

INITIAL_STACK_HEIGHT = 3
PLACEMENTS_PER_PLAYER = 4

MAX_STACK_HEIGHT = 12.0
MAX_PIECES_PER_PLAYER = 12.0
MAX_STACKS_PER_PLAYER = 12.0
MAX_REPETITIONS = 3.0
MAX_PLAY_TURN = 300.0
MAX_PLY = 400.0

LOCAL_SUPPORT_SCALE = 5.5
LOCAL_ATTACK_SCALE = 6.0
LOCAL_VULNERABILITY_SCALE = 6.0

DIRECTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def resolve_feature_layout_version(feature_layout_version: str | None) -> str:
    if feature_layout_version is None:
        return FEATURE_LAYOUT_HEIGHT_BUCKETS_V2
    normalized = str(feature_layout_version).strip().lower()
    if not normalized:
        return FEATURE_LAYOUT_HEIGHT_BUCKETS_V2
    if normalized == FEATURE_LAYOUT_HEIGHT_BUCKETS_V2:
        return normalized
    raise ValueError(f"unsupported teacher feature layout version: {feature_layout_version}")


def encode_teacher_feature_planes(
    root_state: dict,
    feature_layout_version: str = FEATURE_LAYOUT_HEIGHT_BUCKETS_V2,
) -> np.ndarray:
    feature_layout_version = resolve_feature_layout_version(feature_layout_version)
    board = _decode_board(root_state["board"]["cells"])
    turn = root_state["turn"]
    opponent = _opponent(turn)
    phase = root_state["phase"]

    planes = np.zeros((TEACHER_FEATURE_PLANES, BOARD_SIDE, BOARD_SIDE), dtype=np.float32)

    for index, stack in enumerate(board):
        row = index // BOARD_SIDE
        col = index % BOARD_SIDE
        perspective_index = _to_perspective_index(index, turn)
        perspective_row = perspective_index // BOARD_SIDE
        perspective_col = perspective_index % BOARD_SIDE

        planes[20, perspective_row, perspective_col] = _center_score_normalized(row, col)
        planes[21, perspective_row, perspective_col] = _edge_penalty(row, col)

        if phase == "placement" and _is_legal_placement(board, turn, row, col):
            planes[22, perspective_row, perspective_col] = 1.0

        if stack is None:
            continue

        height = int(stack["height"])
        support = _normalize_nonnegative(
            _local_support_score(board, row, col, stack["owner"]),
            LOCAL_SUPPORT_SCALE,
        )
        attack = _normalize_nonnegative(
            _local_attack_score(board, row, col, stack["owner"], stack["height"]),
            LOCAL_ATTACK_SCALE,
        )
        vulnerability = _normalize_nonnegative(
            _local_vulnerability_score(board, row, col, stack["owner"], stack["height"]),
            LOCAL_VULNERABILITY_SCALE,
        )

        if stack["owner"] == turn:
            _set_height_bucket_plane(
                planes,
                perspective_row,
                perspective_col,
                height,
                own_piece=True,
            )
            planes[10, perspective_row, perspective_col] = 1.0
            planes[12, perspective_row, perspective_col] = support
            planes[14, perspective_row, perspective_col] = attack
            planes[16, perspective_row, perspective_col] = vulnerability
        else:
            _set_height_bucket_plane(
                planes,
                perspective_row,
                perspective_col,
                height,
                own_piece=False,
            )
            planes[11, perspective_row, perspective_col] = 1.0
            planes[13, perspective_row, perspective_col] = support
            planes[15, perspective_row, perspective_col] = attack
            planes[17, perspective_row, perspective_col] = vulnerability

    planes[23].fill(1.0 if phase == "placement" else 0.0)
    planes[24].fill(1.0 if phase == "play" else 0.0)
    planes[25].fill(min(root_state.get("repetition_count", 0) / MAX_REPETITIONS, 1.0))
    if phase == "play":
        _populate_cascade_coverage_plane(planes[26], board, turn, turn)
        _populate_cascade_coverage_plane(planes[27], board, opponent, turn)
    else:
        planes[26].fill(min(root_state.get("play_turn", 0) / MAX_PLAY_TURN, 1.0))
        planes[27].fill(min(root_state.get("ply", 0) / MAX_PLY, 1.0))
    planes[28].fill(min(_piece_count(board, turn) / MAX_PIECES_PER_PLAYER, 1.0))
    planes[29].fill(min(_piece_count(board, opponent) / MAX_PIECES_PER_PLAYER, 1.0))
    planes[30].fill(min(_stack_count(board, turn) / MAX_STACKS_PER_PLAYER, 1.0))
    planes[31].fill(min(_stack_count(board, opponent) / MAX_STACKS_PER_PLAYER, 1.0))

    return planes


def _set_height_bucket_plane(
    planes: np.ndarray,
    row: int,
    col: int,
    height: int,
    own_piece: bool,
) -> None:
    if height <= 0:
        return

    if height >= 6:
        plane = 18 if own_piece else 19
        planes[plane, row, col] = 1.0
        return

    capped_height = min(height, 5)
    height_idx = capped_height - 1
    plane = height_idx if own_piece else 5 + height_idx
    planes[plane, row, col] = 1.0


def _populate_cascade_coverage_plane(
    plane: np.ndarray,
    board: list[dict | None],
    player: str,
    perspective_player: str,
) -> None:
    for index, stack in enumerate(board):
        if stack is None or stack["owner"] != player:
            continue

        height = int(stack["height"])
        if height < 2:
            continue

        row = index // BOARD_SIDE
        col = index % BOARD_SIDE
        for direction in DIRECTIONS:
            current_row = row
            current_col = col
            for _ in range(height):
                next_position = _checked_step(current_row, current_col, direction)
                if next_position is None:
                    break
                current_row, current_col = next_position
                perspective_index = _to_perspective_index(
                    _index(current_row, current_col),
                    perspective_player,
                )
                perspective_row = perspective_index // BOARD_SIDE
                perspective_col = perspective_index % BOARD_SIDE
                plane[perspective_row, perspective_col] = 1.0


def _decode_board(cells: Iterable[dict]) -> list[dict | None]:
    board: list[dict | None] = []
    for cell in cells:
        if cell["state"] == "empty":
            board.append(None)
        else:
            stack = cell["stack"]
            board.append({"owner": stack["owner"], "height": int(stack["height"])})
    return board


def _opponent(player: str) -> str:
    return "blue" if player == "red" else "red"


def _to_perspective_index(index: int, turn: str) -> int:
    row = index // BOARD_SIDE
    col = index % BOARD_SIDE
    if turn == "blue":
        row = BOARD_SIDE - 1 - row
        col = BOARD_SIDE - 1 - col
    return row * BOARD_SIDE + col


def _center_score_normalized(row: int, col: int) -> float:
    row_distance = abs(row - 3.5)
    col_distance = abs(col - 3.5)
    return max((4.0 - 0.55 * (row_distance + col_distance)) / 4.0, 0.0)


def _edge_penalty(row: int, col: int) -> float:
    distance = min(row, BOARD_SIDE - 1 - row, col, BOARD_SIDE - 1 - col)
    if distance == 0:
        return 1.0
    if distance == 1:
        return 0.45
    if distance == 2:
        return 0.12
    return 0.0


def _normalize_nonnegative(value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return float(np.clip(value / scale, 0.0, 1.0))


def _index(row: int, col: int) -> int:
    return row * BOARD_SIDE + col


def _checked_step(row: int, col: int, direction: tuple[int, int]) -> tuple[int, int] | None:
    next_row = row + direction[0]
    next_col = col + direction[1]
    if next_row < 0 or next_row >= BOARD_SIDE or next_col < 0 or next_col >= BOARD_SIDE:
        return None
    return next_row, next_col


def _stack_at(board: list[dict | None], row: int, col: int) -> dict | None:
    return board[_index(row, col)]


def _local_support_score(board: list[dict | None], row: int, col: int, player: str) -> float:
    total = 0.0
    for direction in DIRECTIONS:
        next_position = _checked_step(row, col, direction)
        if next_position is None:
            continue
        neighbor = _stack_at(board, next_position[0], next_position[1])
        if neighbor is None or neighbor["owner"] != player:
            continue
        total += 0.35 + neighbor["height"] * 0.08
    return total


def _local_attack_score(
    board: list[dict | None],
    row: int,
    col: int,
    player: str,
    height: int,
) -> float:
    total = 0.0
    for direction in DIRECTIONS:
        next_position = _checked_step(row, col, direction)
        if next_position is None:
            continue
        neighbor = _stack_at(board, next_position[0], next_position[1])
        if neighbor is None or neighbor["owner"] == player:
            continue
        if height >= neighbor["height"]:
            total += 1.0 + (height - neighbor["height"]) * 0.2
    return total


def _local_vulnerability_score(
    board: list[dict | None],
    row: int,
    col: int,
    player: str,
    height: int,
) -> float:
    total = 0.0
    for direction in DIRECTIONS:
        next_position = _checked_step(row, col, direction)
        if next_position is None:
            continue
        neighbor = _stack_at(board, next_position[0], next_position[1])
        if neighbor is None or neighbor["owner"] == player:
            continue
        if neighbor["height"] >= height:
            total += 1.0 + (neighbor["height"] - height) * 0.2
    return total


def _piece_count(board: list[dict | None], player: str) -> float:
    return float(sum(stack["height"] for stack in board if stack is not None and stack["owner"] == player))


def _stack_count(board: list[dict | None], player: str) -> float:
    return float(sum(1 for stack in board if stack is not None and stack["owner"] == player))


def _derive_placement_counts(board: list[dict | None]) -> tuple[int, int]:
    red_pieces = int(_piece_count(board, "red"))
    blue_pieces = int(_piece_count(board, "blue"))
    return red_pieces // INITIAL_STACK_HEIGHT, blue_pieces // INITIAL_STACK_HEIGHT


def _is_legal_placement(board: list[dict | None], player: str, row: int, col: int) -> bool:
    red_placements, blue_placements = _derive_placement_counts(board)
    if (player == "red" and red_placements >= PLACEMENTS_PER_PLAYER) or (
        player == "blue" and blue_placements >= PLACEMENTS_PER_PLAYER
    ):
        return False
    if _stack_at(board, row, col) is not None:
        return False
    if red_placements + blue_placements == 0:
        return True

    for direction in DIRECTIONS:
        next_position = _checked_step(row, col, direction)
        if next_position is None:
            continue
        neighbor = _stack_at(board, next_position[0], next_position[1])
        if neighbor is not None and neighbor["owner"] == _opponent(player):
            return False
    return True


def flip_action_index(index: int) -> int:
    if index < PLACE_ACTIONS:
        row = index // BOARD_SIDE
        col = index % BOARD_SIDE
        return (BOARD_SIDE - 1 - row) * BOARD_SIDE + (BOARD_SIDE - 1 - col)

    offset = index - PLACE_ACTIONS
    if offset < MOVE_ACTIONS:
        base = PLACE_ACTIONS
    elif offset < MOVE_ACTIONS + EAT_ACTIONS:
        base = PLACE_ACTIONS + MOVE_ACTIONS
        offset -= MOVE_ACTIONS
    else:
        base = PLACE_ACTIONS + MOVE_ACTIONS + EAT_ACTIONS
        offset -= (MOVE_ACTIONS + EAT_ACTIONS)

    pos_index = offset // 4
    dir_index = offset % 4

    row = pos_index // BOARD_SIDE
    col = pos_index % BOARD_SIDE
    new_pos_index = (BOARD_SIDE - 1 - row) * BOARD_SIDE + (BOARD_SIDE - 1 - col)

    # Directions: 0: Up, 1: Down, 2: Left, 3: Right
    # Up(0) <-> Down(1)
    # Left(2) <-> Right(3)
    new_dir_index = dir_index ^ 1

    return base + new_pos_index * 4 + new_dir_index

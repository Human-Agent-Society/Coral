#!/usr/bin/env python3
"""Small deterministic implementation of standard 4x4 2048."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable


Board = tuple[tuple[int, ...], ...]
MOVES = ("UP", "DOWN", "LEFT", "RIGHT")


def _merge_left(row: Iterable[int]) -> tuple[tuple[int, ...], int]:
    values = [value for value in row if value]
    merged: list[int] = []
    gained = 0
    index = 0
    while index < len(values):
        if index + 1 < len(values) and values[index] == values[index + 1]:
            value = values[index] * 2
            merged.append(value)
            gained += value
            index += 2
        else:
            merged.append(values[index])
            index += 1
    return tuple(merged + [0] * (4 - len(merged))), gained


def _transpose(board: Board) -> Board:
    return tuple(tuple(board[row][col] for row in range(4)) for col in range(4))


def _reverse(board: Board) -> Board:
    return tuple(tuple(reversed(row)) for row in board)


def apply_move(board: Board, move: str) -> tuple[Board, int, bool]:
    work = board
    if move == "RIGHT":
        work = _reverse(work)
    elif move == "UP":
        work = _transpose(work)
    elif move == "DOWN":
        work = _reverse(_transpose(work))
    elif move != "LEFT":
        raise ValueError(f"unknown move: {move!r}")

    rows = []
    gained = 0
    for row in work:
        merged, points = _merge_left(row)
        rows.append(merged)
        gained += points
    result: Board = tuple(rows)

    if move == "RIGHT":
        result = _reverse(result)
    elif move == "UP":
        result = _transpose(result)
    elif move == "DOWN":
        result = _transpose(_reverse(result))
    return result, gained, result != board


def legal_moves(board: Board) -> tuple[str, ...]:
    return tuple(move for move in MOVES if apply_move(board, move)[2])


def _spawn(board: Board, rng: random.Random) -> Board:
    empty = [(row, col) for row in range(4) for col in range(4) if board[row][col] == 0]
    if not empty:
        return board
    row, col = empty[rng.randrange(len(empty))]
    value = 2 if rng.random() < 0.9 else 4
    mutable = [list(line) for line in board]
    mutable[row][col] = value
    return tuple(tuple(line) for line in mutable)


@dataclass(frozen=True)
class GameResult:
    seed: int
    score: int
    max_tile: int
    moves: int
    error: str | None


def play(
    seed: int,
    choose_move: Callable[[Board], str],
    max_moves: int = 10000,
) -> GameResult:
    rng = random.Random(seed)
    board: Board = ((0, 0, 0, 0),) * 4
    board = _spawn(_spawn(board, rng), rng)
    score = 0
    error = None
    moves = 0
    while moves < max_moves and legal_moves(board):
        try:
            move = choose_move(board)
        except Exception as exc:
            error = f"policy error: {type(exc).__name__}: {exc}"
            break
        if move not in MOVES:
            error = f"invalid move: {move!r}"
            break
        board_after, gained, changed = apply_move(board, move)
        if not changed:
            error = f"illegal move: {move}"
            break
        board = _spawn(board_after, rng)
        score += gained
        moves += 1
    return GameResult(
        seed=seed,
        score=score,
        max_tile=max(max(row) for row in board),
        moves=moves,
        error=error,
    )

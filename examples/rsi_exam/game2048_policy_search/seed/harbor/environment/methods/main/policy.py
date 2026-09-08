"""Weak baseline: keep moving toward the upper-left corner."""

from game2048 import apply_move


def choose_move(board):
    for move in ("UP", "LEFT", "RIGHT", "DOWN"):
        if apply_move(board, move)[2]:
            return move
    return "UP"

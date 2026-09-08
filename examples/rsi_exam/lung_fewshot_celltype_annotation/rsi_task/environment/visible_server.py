"""Label-isolated macro-F1 service for the visible development query."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


SOCKET_PATH = Path(os.environ.get("VISIBLE_EVAL_SOCKET", "/run/cell-visible/evaluator.sock"))
LABELS_PATH = Path(os.environ.get("VISIBLE_LABELS_PATH", "/opt/cell-visible/visible_labels.csv"))
CLASSES_PATH = Path(os.environ.get("CLASSES_PATH", "/opt/cell-visible/classes.txt"))
QUERY_BUDGET = int(os.environ.get("VISIBLE_QUERY_BUDGET", "128"))
SCORE_DECIMALS = int(os.environ.get("VISIBLE_SCORE_DECIMALS", "4"))
MIN_CHANGED_PREDICTIONS = int(os.environ.get("VISIBLE_MIN_CHANGED_PREDICTIONS", "40"))
MAX_REQUEST_BYTES = 4 * 1024 * 1024


def load_contract() -> tuple[pd.Series, list[str]]:
    frame = pd.read_csv(LABELS_PATH, dtype=str)
    if list(frame.columns) != ["cell_id", "label"]:
        raise RuntimeError("visible labels must have columns cell_id,label")
    if frame["cell_id"].duplicated().any() or frame.isna().any().any():
        raise RuntimeError("visible labels contain duplicate IDs or missing values")
    classes = [line.strip() for line in CLASSES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(classes) != 15 or set(frame["label"]) - set(classes):
        raise RuntimeError("visible labels do not match the class vocabulary")
    return frame.set_index("cell_id")["label"], classes


def align_request(request: object, truth: pd.Series, classes: list[str]) -> pd.Series:
    if not isinstance(request, dict) or set(request) != {"cell_ids", "pred_labels"}:
        raise ValueError("request must contain only cell_ids and pred_labels")
    cell_ids = request["cell_ids"]
    pred_labels = request["pred_labels"]
    if not isinstance(cell_ids, list) or not isinstance(pred_labels, list):
        raise ValueError("cell_ids and pred_labels must be lists")
    if len(cell_ids) != len(truth) or len(pred_labels) != len(truth):
        raise ValueError("prediction count does not match the visible query")
    cell_ids = [str(value) for value in cell_ids]
    pred_labels = [str(value) for value in pred_labels]
    if len(cell_ids) != len(set(cell_ids)) or set(cell_ids) != set(truth.index):
        raise ValueError("cell IDs must match the visible query exactly")
    if set(pred_labels) - set(classes):
        raise ValueError("predictions contain labels outside the class vocabulary")
    return pd.Series(pred_labels, index=cell_ids).reindex(truth.index)


def score_request(request: object, truth: pd.Series, classes: list[str]) -> float:
    aligned = align_request(request, truth, classes)
    return float(f1_score(truth.to_numpy(), aligned.to_numpy(), labels=classes, average="macro", zero_division=0))


def encode_predictions(aligned: pd.Series, classes: list[str]) -> np.ndarray:
    class_code = {label: index for index, label in enumerate(classes)}
    return np.fromiter((class_code[label] for label in aligned), dtype=np.uint8, count=len(aligned))


def reject_near_duplicate(codes: np.ndarray, history: list[np.ndarray], minimum_changed: int) -> None:
    if minimum_changed <= 0:
        return
    for previous in history:
        if int(np.count_nonzero(previous != codes)) < minimum_changed:
            raise ValueError(
                f"prediction vector must differ from every prior scored vector in at least {minimum_changed} cells"
            )


def response_line(value: dict[str, object]) -> bytes:
    return (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")


def main() -> None:
    truth, classes = load_contract()
    if QUERY_BUDGET <= 0 or not 0 <= SCORE_DECIMALS <= 9:
        raise RuntimeError("invalid visible-evaluator disclosure settings")
    if not 0 <= MIN_CHANGED_PREDICTIONS <= len(truth):
        raise RuntimeError("invalid minimum changed-predictions setting")
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOCKET_PATH.unlink(missing_ok=True)
    used = 0
    history: list[np.ndarray] = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(SOCKET_PATH))
        os.chmod(SOCKET_PATH, 0o666)
        server.listen(16)
        while True:
            connection, _ = server.accept()
            with connection:
                stream = connection.makefile("rwb")
                payload = stream.readline(MAX_REQUEST_BYTES + 1)
                if len(payload) > MAX_REQUEST_BYTES:
                    stream.write(response_line({"ok": False, "error": "request too large"}))
                    stream.flush()
                    continue
                used += 1
                if used > QUERY_BUDGET:
                    stream.write(response_line({"ok": False, "error": "visible evaluation budget exhausted"}))
                    stream.flush()
                    continue
                try:
                    request = json.loads(payload)
                    aligned = align_request(request, truth, classes)
                    codes = encode_predictions(aligned, classes)
                    reject_near_duplicate(codes, history, MIN_CHANGED_PREDICTIONS)
                    metric = float(
                        f1_score(
                            truth.to_numpy(), aligned.to_numpy(), labels=classes, average="macro", zero_division=0
                        )
                    )
                    history.append(codes)
                    response = {
                        "ok": True,
                        "macro_f1": round(metric, SCORE_DECIMALS),
                        "queries_used": used,
                        "queries_remaining": QUERY_BUDGET - used,
                    }
                except Exception as exc:  # noqa: BLE001 - malformed probes consume budget
                    response = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "queries_used": used,
                        "queries_remaining": QUERY_BUDGET - used,
                    }
                stream.write(response_line(response))
                stream.flush()


if __name__ == "__main__":
    main()

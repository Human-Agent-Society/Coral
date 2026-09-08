"""Fast structural checks that do not consume visible-evaluator queries."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import anndata as ad


APP = Path("/app")
LABEL = "ann_finest_level"


def main() -> None:
    data = APP / "data"
    labeled = ad.read_h5ad(data / "visible_labeled.h5ad", backed="r")
    unlabeled = ad.read_h5ad(data / "visible_unlabeled.h5ad", backed="r")
    query = ad.read_h5ad(data / "visible_query.h5ad", backed="r")
    classes = [line.strip() for line in (data / "classes.txt").read_text().splitlines() if line.strip()]
    assert labeled.shape == (75, 27402), labeled.shape
    assert unlabeled.shape == (22740, 27402), unlabeled.shape
    assert query.shape == (1200, 27402), query.shape
    assert len(classes) == len(set(classes)) == 15
    assert LABEL in labeled.obs and LABEL not in unlabeled.obs and LABEL not in query.obs
    counts = labeled.obs[LABEL].astype(str).value_counts()
    assert set(counts.index) == set(classes) and (counts == 5).all()
    assert labeled.var_names.equals(unlabeled.var_names) and labeled.var_names.equals(query.var_names)
    model = data / "reference_model.pkl"
    assert model.stat().st_size == 774940
    assert hashlib.sha256(model.read_bytes()).hexdigest() == "ca34a816b693613eed7591efc1ed1c5581d10a310942ed5cf25d7e98884fae66"
    subprocess.run(
        [sys.executable, str(APP / "methods" / "main" / "solver.py"), "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print("selfcheck: ok")


    if Path("/app/budget.py").exists():
        subprocess.run([sys.executable, "/app/budget.py"], check=False)

if __name__ == "__main__":
    main()

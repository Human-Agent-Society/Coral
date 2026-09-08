"""Contract-only self-check for the anonymous visible cloud package."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import math
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
PUBLIC_ROOT = HERE / "public" / "observations"
SOLVER_PATH = HERE / "methods" / "main" / "solver.py"
SCHEMA_VERSION = "multiview-cloud-v4"
MANIFEST_KEYS = {"schema_version", "split", "cases"}
CASE_ENTRY_KEYS = {"case_id", "file", "num_views"}
CASE_KEYS = {"case_id", "views"}
VIEW_KEYS = {"view_id", "points", "confidence"}
CASE_RE = re.compile(r"visible_case_[0-9]{3}\Z")
VIEW_RE = re.compile(r"view_[0-9]{3}\Z")
sys.dont_write_bytecode = True


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_points(value: Any, label: str) -> list[list[float]]:
    _require(
        isinstance(value, list) and len(value) >= 4,
        f"{label} must contain at least four points",
    )
    for index, point in enumerate(value):
        _require(
            isinstance(point, list) and len(point) == 3,
            f"{label}[{index}] must be a 3-vector",
        )
        _require(
            all(_finite_number(item) for item in point),
            f"{label}[{index}] must be finite",
        )
    return value


def _load_public() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(
        (PUBLIC_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    _require(
        isinstance(manifest, dict) and set(manifest) == MANIFEST_KEYS,
        f"manifest keys must be exactly {sorted(MANIFEST_KEYS)}",
    )
    _require(
        manifest["schema_version"] == SCHEMA_VERSION,
        "unexpected schema_version",
    )
    _require(manifest["split"] == "public", "visible package split must be public")
    cases = manifest["cases"]
    _require(
        isinstance(cases, list) and cases,
        "manifest cases must be a non-empty list",
    )
    expected_case_ids = [
        f"visible_case_{index:03d}" for index in range(1, len(cases) + 1)
    ]
    loaded: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(cases):
        label = f"cases[{index}]"
        _require(
            isinstance(entry, dict) and set(entry) == CASE_ENTRY_KEYS,
            f"{label} keys must be exactly {sorted(CASE_ENTRY_KEYS)}",
        )
        case_id = entry["case_id"]
        _require(
            isinstance(case_id, str) and CASE_RE.fullmatch(case_id) is not None,
            f"{label}.case_id must be opaque visible_case_NNN text",
        )
        _require(case_id == expected_case_ids[index], "visible case IDs must be sequential")
        _require(entry["file"] == f"{case_id}.json", f"{label}.file must match case_id")
        _require(
            isinstance(entry["num_views"], int)
            and not isinstance(entry["num_views"], bool)
            and entry["num_views"] > 0,
            f"{label}.num_views must be positive",
        )
        case_path = PUBLIC_ROOT / entry["file"]
        _require(case_path.parent == PUBLIC_ROOT, f"{label}.file must not traverse directories")
        case = json.loads(case_path.read_text(encoding="utf-8"))
        _require(
            isinstance(case, dict) and set(case) == CASE_KEYS,
            f"{case_id} keys must be exactly {sorted(CASE_KEYS)}",
        )
        _require(case["case_id"] == case_id, f"{case_id} identity mismatch")
        views = case["views"]
        _require(
            isinstance(views, list) and len(views) == entry["num_views"],
            f"{case_id} view count mismatch",
        )
        expected_view_ids = [
            f"view_{view_index:03d}" for view_index in range(1, len(views) + 1)
        ]
        for view_index, view in enumerate(views):
            view_label = f"{case_id}.views[{view_index}]"
            _require(
                isinstance(view, dict) and set(view) == VIEW_KEYS,
                f"{view_label} keys must be exactly {sorted(VIEW_KEYS)}",
            )
            view_id = view["view_id"]
            _require(
                isinstance(view_id, str) and VIEW_RE.fullmatch(view_id) is not None,
                f"{view_label}.view_id must be opaque view_NNN text",
            )
            _require(
                view_id == expected_view_ids[view_index],
                f"{case_id} view IDs must be sequential",
            )
            points = _validate_points(view["points"], f"{case_id}.{view_id}.points")
            confidence = view["confidence"]
            _require(
                isinstance(confidence, list) and len(confidence) == len(points),
                f"{case_id}.{view_id}.confidence length mismatch",
            )
            _require(
                all(_finite_number(value) and 0.0 <= float(value) <= 1.0 for value in confidence),
                f"{case_id}.{view_id}.confidence must be finite values in [0,1]",
            )
        loaded[case_id] = case
    return manifest, loaded


def _validate_imports(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    allowed = set(sys.stdlib_module_names) | {"__future__", "numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            _require(node.level == 0, "solver may not use relative imports")
            names = [(node.module or "").split(".", 1)[0]]
        else:
            continue
        for name in names:
            _require(
                name in allowed,
                f"solver import is not available in the task image: {name}",
            )


def _load_solver(path: Path) -> ModuleType:
    _validate_imports(path)
    spec = importlib.util.spec_from_file_location("submitted_cloud_solver", path)
    _require(spec is not None and spec.loader is not None, f"cannot load solver at {path}")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    _require(
        hasattr(module, "predict") and callable(module.predict),
        "solver must define callable predict(export_dir)",
    )
    parameters = list(inspect.signature(module.predict).parameters.values())
    _require(
        len(parameters) == 1 and parameters[0].name == "export_dir",
        "predict signature must be exactly predict(export_dir)",
    )
    _require(
        parameters[0].kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD),
        "export_dir must be a positional parameter",
    )
    return module


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_predictions(predictions: Any, cases: dict[str, dict[str, Any]]) -> int:
    _require(isinstance(predictions, dict), "predict must return a dictionary")
    _require(
        set(predictions) == set(cases),
        "prediction case keys must exactly match the manifest",
    )
    point_count = 0
    for case_id, cloud in predictions.items():
        points = _validate_points(cloud, f"{case_id} output cloud")
        _require(len(points) <= 2048, f"{case_id} output exceeds the 2048-point cap")
        distinct = {tuple(float(value) for value in point) for point in points}
        _require(len(distinct) >= 4, f"{case_id} output must contain at least four distinct points")
        point_count += len(points)
    return point_count


def main() -> None:
    manifest, cases = _load_public()
    solver = _load_solver(SOLVER_PATH)
    before = _tree_digest(PUBLIC_ROOT)
    first = solver.predict(str(PUBLIC_ROOT))
    point_count = _validate_predictions(first, cases)
    second = solver.predict(str(PUBLIC_ROOT))
    _validate_predictions(second, cases)
    _require(first == second, "predict must be deterministic for identical input")
    _require(_tree_digest(PUBLIC_ROOT) == before, "predict modified the public input package")
    print(json.dumps({
        "status": "ok",
        "schema_version": manifest["schema_version"],
        "public_cases": len(cases),
        "validated_points": point_count,
        "quality_measurement": False,
        "feedback_mode": "contract_only",
    }, sort_keys=True))
    if Path("/app/budget.py").exists():
        import subprocess
        subprocess.run([sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()

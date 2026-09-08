from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(HERE))

from protocol import Case, make_inputs, make_metadata, reference_decode  # noqa: E402


def load_solver(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.paged_gqa_decode


def exercise(solver, case: Case) -> None:
    metadata = make_metadata(case, device="cpu")
    args = make_inputs(case, seed=case.seed + 999, metadata=metadata, device="cpu")
    snapshots = [
        value.clone() if isinstance(value, torch.Tensor) else value
        for value in args
    ]
    expected = reference_decode(*args)
    actual = solver(*args)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.data_ptr() not in {
        value.data_ptr() for value in args if isinstance(value, torch.Tensor)
    }
    assert torch.allclose(actual.float(), expected.float(), rtol=3e-2, atol=3e-2)
    for current, snapshot in zip(args, snapshots):
        if isinstance(current, torch.Tensor):
            assert torch.equal(current, snapshot)
        else:
            assert current == snapshot


def main() -> None:
    cases = [
        Case(
            case_id="cpu_full_no_rope",
            seed=41,
            dtype_name="float16",
            batch=3,
            query_heads=8,
            kv_heads=2,
            head_dim=64,
            page_size=1,
            page_counts=(1, 5, 3),
            last_page_len=(1, 1, 1),
            logits_soft_cap=4.0,
            pos_encoding_mode=0,
            window_left=-1,
            rope_scale=1.0,
            rope_theta=10_000.0,
        ),
        Case(
            case_id="cpu_window_rope",
            seed=73,
            dtype_name="bfloat16",
            batch=3,
            query_heads=8,
            kv_heads=2,
            head_dim=64,
            page_size=8,
            page_counts=(2, 1, 3),
            last_page_len=(5, 3, 7),
            logits_soft_cap=7.5,
            pos_encoding_mode=1,
            window_left=6,
            rope_scale=2.0,
            rope_theta=500_000.0,
        ),
    ]
    paths = [
        TASK / "environment" / "methods" / "main" / "solver.py",
        HERE / "production_baseline" / "solver.py",
        TASK / "solution" / "baseline" / "solver.py",
    ]
    for solver_index, path in enumerate(paths):
        solver = load_solver(path, f"vectorized_baseline_{solver_index}")
        for case in cases:
            exercise(solver, case)
    print("test_vectorized_baseline: PASS")


if __name__ == "__main__":
    main()

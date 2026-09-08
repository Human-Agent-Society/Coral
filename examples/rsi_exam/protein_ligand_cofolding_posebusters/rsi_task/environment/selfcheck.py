#!/usr/bin/env python3
"""Run the editable method on the canonical visible suite.

The default result is the only comparable visible score: every visible case,
the same scientific metric, parser-work bounds, fresh prediction process,
timeouts, invalid-output treatment, and aggregation as the sealed verifier.
The sealed verifier additionally enforces filesystem/process visibility
isolation.  ``--case-index`` and ``--smoke`` are diagnostics only and are
labelled non-comparable in output.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metric
import source_contract


CASE_TIMEOUT_SEC = 3600.0
SCORE_TIMEOUT_SEC = 300.0
TOTAL_TIMEOUT_SEC = 43200.0
MAX_PREDICTION_JSON_BYTES = 24 * 1024 * 1024
PR_SET_CHILD_SUBREAPER = 36
EXPECTED_VISIBLE_CASES = 20


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--case-index",
        type=int,
        help="run one zero-based visible case (diagnostic; not a comparable score)",
    )
    group.add_argument(
        "--smoke",
        action="store_true",
        help="run the first visible case (diagnostic; not a comparable score)",
    )
    # Internal fresh-process entry point.  Keeping it in this file avoids a
    # second public prediction harness that could drift from the self-check.
    parser.add_argument("--_predict", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_prediction-out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_source-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_item-in", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_score-case", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_prediction-in", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_score-out", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _load_visible() -> tuple[Path, list[dict]]:
    base = HERE / "data" / "visible"
    manifest = base / "items.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"visible manifest is missing: {manifest}; task data must be installed locally"
        )
    items = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(items, list)
        or len(items) != EXPECTED_VISIBLE_CASES
        or not all(isinstance(item, dict) for item in items)
    ):
        raise ValueError(
            f"visible manifest must contain exactly {EXPECTED_VISIBLE_CASES} objects"
        )
    return base, items


def _stage_agent_item(item: dict, base: Path, root: Path) -> Path:
    """Stage the same anonymous three-field input shape used by the verifier."""
    input_dir = root / "input"
    msa_dir = input_dir / "msa"
    msa_dir.mkdir(parents=True)
    base_real = base.resolve(strict=True)
    source_msa = (base / item["msa_dir"]).resolve(strict=True)
    source_msa.relative_to(base_real)
    for chain in item["protein_chains"]:
        chain_id = str(chain["chain_id"])
        payload = (source_msa / f"{chain_id}.a3m").read_bytes().replace(b"\x00", b"")
        if not payload:
            raise ValueError("visible MSA is empty")
        target = msa_dir / f"{chain_id}.a3m"
        target.write_bytes(payload)
        target.chmod(0o444)
    msa_dir.chmod(0o555)
    payload = {
        "protein_chains": item["protein_chains"],
        "ligand_smiles": item["ligand_smiles"],
        "msa_dir": str(msa_dir),
    }
    item_path = input_dir / "item.json"
    item_path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    item_path.chmod(0o444)
    input_dir.chmod(0o555)
    return item_path


def _load_agent_item(item_path: Path) -> dict:
    item = json.loads(item_path.read_text(encoding="utf-8"))
    if not isinstance(item, dict) or set(item) != {
        "protein_chains",
        "ligand_smiles",
        "msa_dir",
    }:
        raise ValueError("internal prediction item has the wrong schema")
    if not isinstance(item["protein_chains"], list) or not item["protein_chains"]:
        raise ValueError("internal prediction item has no protein chains")
    if not isinstance(item["ligand_smiles"], str) or not item["ligand_smiles"]:
        raise ValueError("internal prediction item has no ligand SMILES")
    if not isinstance(item["msa_dir"], str) or not Path(item["msa_dir"]).is_dir():
        raise ValueError("internal prediction item has no staged MSA directory")
    return item


def _predict_child(output_path: Path, source_dir: Path, item_path: Path) -> int:
    """Predict one staged visible case in a newly started Python process."""
    source_contract.validate_source_tree(source_dir)
    item = _load_agent_item(item_path)
    sys.path.insert(0, str(source_dir))
    # Submitted imports must not observe self-check control arguments.  The
    # sealed child likewise replaces argv with the effective solver path.
    sys.argv = [str(source_dir / "solver.py")]
    from solver import predict_complex

    prediction = predict_complex(item)
    encoded = json.dumps(prediction, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_PREDICTION_JSON_BYTES:
        raise ValueError("prediction JSON exceeds the verifier output bound")
    output_path.write_bytes(encoded)
    return 0


def _become_child_subreaper() -> None:
    """Adopt orphaned grandchildren so every case can be fully torn down."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise RuntimeError(f"cannot enable child subreaper: errno={error}")


def _direct_child_pids() -> list[int]:
    parent_pid = os.getpid()
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text(encoding="utf-8")
            after_name = text.rsplit(")", 1)[1].split()
            ppid = int(after_name[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == parent_pid:
            result.append(int(entry.name))
    return result


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _safe_cache_path_env(name: str) -> str | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        candidate = Path(raw).expanduser().resolve(strict=False)
    except OSError:
        return None
    for root in (Path("/opt"), Path("/usr")):
        try:
            candidate.relative_to(root)
            return str(candidate)
        except ValueError:
            continue
    return None


def _child_environment(root: Path) -> dict[str, str]:
    """Match the sealed worker environment without forwarding agent secrets."""
    home = root / "home"
    temporary = root / "tmp"
    home.mkdir(parents=True)
    temporary.mkdir()
    python_bin = str(Path(sys.executable).resolve().parent)
    env = {
        "PATH": ":".join(
            dict.fromkeys(
                (
                    python_bin,
                    "/opt/conda/bin",
                    "/usr/local/cuda/bin",
                    "/usr/local/bin",
                    "/usr/bin",
                    "/bin",
                )
            )
        ),
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "TMP": str(temporary),
        "TEMP": str(temporary),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "CUDA_MODULE_LOADING",
        "CUBLAS_WORKSPACE_CONFIG",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "LD_LIBRARY_PATH",
    ):
        value = os.environ.get(name)
        if value:
            env[name] = value
    for name in (
        "BOLTZ_CACHE",
        "TORCH_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "CHAI_DOWNLOADS_DIR",
    ):
        value = _safe_cache_path_env(name)
        if value is not None:
            env[name] = value
    return env


def _kill_and_reap_worker(
    process: subprocess.Popen[bytes], *, grace_sec: float = 0.3
) -> None:
    """Mirror the verifier: clear the worker group and adopted descendants."""
    _signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline and process.poll() is None:
        time.sleep(0.01)
    _signal_group(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=max(0.1, grace_sec))
    except subprocess.TimeoutExpired:
        _signal_group(process.pid, signal.SIGKILL)
        process.wait(timeout=1.0)

    reap_deadline = time.monotonic() + 2.0
    while True:
        adopted = _direct_child_pids()
        if not adopted:
            break
        for pid in adopted:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in adopted:
            try:
                os.waitpid(pid, 0)
            except (ChildProcessError, ProcessLookupError):
                pass
        if time.monotonic() >= reap_deadline and _direct_child_pids():
            raise RuntimeError("fresh worker descendants could not be fully reaped")


def _run_fresh(
    command: list[str],
    timeout: float,
    label: str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a child group and tear down all descendants on every exit path."""
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    timed_out = False
    try:
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = None
    finally:
        _kill_and_reap_worker(process)
    if timed_out:
        raise TimeoutError(f"{label} exceeded its {timeout:.0f}s limit")
    if return_code != 0:
        raise RuntimeError(f"fresh {label} process exited with code {return_code}")


def _predict_fresh(
    timeout: float,
    output_path: Path,
    source_dir: Path,
    item_path: Path,
) -> dict:
    """Run one prediction with the verifier's fresh-process/timeout semantics."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_predict",
        "--_prediction-out",
        str(output_path),
        "--_source-dir",
        str(source_dir),
        "--_item-in",
        str(item_path),
    ]
    runtime = output_path.parent / "prediction_runtime"
    runtime.mkdir()
    _run_fresh(
        command,
        timeout,
        "prediction",
        cwd=runtime,
        env=_child_environment(runtime),
    )
    if not output_path.is_file():
        raise RuntimeError("fresh prediction process produced no output")
    if output_path.stat().st_size > MAX_PREDICTION_JSON_BYTES:
        raise ValueError("prediction JSON exceeds the verifier output bound")
    prediction = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(prediction, dict):
        raise TypeError("prediction must be a JSON object")
    return prediction


def _score_child(case_index: int, prediction_path: Path, output_path: Path) -> int:
    """Score one completed prediction without importing submitted source."""
    base, items = _load_visible()
    if case_index < 0 or case_index >= len(items):
        raise ValueError("internal visible case index is out of range")
    if prediction_path.stat().st_size > MAX_PREDICTION_JSON_BYTES:
        raise ValueError("prediction JSON exceeds the verifier output bound")
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    item = items[case_index]
    with tempfile.TemporaryDirectory(prefix="visible_metric_") as work:
        score = metric.score_pose(
            prediction,
            crystal_ligand_path=base / item["crystal_ligand_sdf"],
            crystal_protein_path=base / item["crystal_protein_pdb"],
            expected_chains=item["protein_chains"],
            expected_ligand_smiles=item["ligand_smiles"],
            work_dir=work,
        )
    output_path.write_text(
        json.dumps(
            {
                "passed": score.passed,
                "pb_valid": score.pb_valid,
                "rmsd_within_2a": score.rmsd_within_2a,
                "checks": dict(score.checks),
                "matched_ca_count": score.matched_ca_count,
                "protein_alignment_rmsd": score.protein_alignment_rmsd,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return 0


def _score_fresh(
    case_index: int,
    prediction_path: Path,
    timeout: float,
    output_path: Path,
) -> metric.PoseScore:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_score-case",
        str(case_index),
        "--_prediction-in",
        str(prediction_path),
        "--_score-out",
        str(output_path),
    ]
    runtime = output_path.parent / "scoring_runtime"
    runtime.mkdir()
    _run_fresh(
        command,
        timeout,
        "scoring",
        cwd=runtime,
        env=_child_environment(runtime),
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    return metric.PoseScore(**payload)


def _snapshot_source(source: Path, destination: Path) -> None:
    """Freeze one validated source snapshot before the visible suite starts."""
    relative_files = source_contract.validate_source_tree(source)
    destination.mkdir(mode=0o700)
    for relative_text in relative_files:
        relative = Path(relative_text)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (source / relative).read_bytes()
        target.write_bytes(payload)
    # Revalidate the bytes that will seed every per-case source copy.
    source_contract.validate_source_tree(destination)


def _copy_snapshot(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)
    source_contract.validate_source_tree(destination)


def main() -> int:
    args = _arguments()
    prediction_mode = any(
        (
            args._predict,
            args._prediction_out is not None,
            args._source_dir is not None,
            args._item_in is not None,
        )
    )
    scoring_mode = any(
        value is not None
        for value in (args._score_case, args._prediction_in, args._score_out)
    )
    if prediction_mode and scoring_mode:
        raise SystemExit("internal prediction and scoring modes are exclusive")
    if prediction_mode and (
        not args._predict
        or any(
            value is None
            for value in (args._prediction_out, args._source_dir, args._item_in)
        )
    ):
        raise SystemExit("internal prediction arguments must be supplied together")
    if scoring_mode and any(
        value is None
        for value in (args._score_case, args._prediction_in, args._score_out)
    ):
        raise SystemExit("internal scoring arguments must be supplied together")
    if args._predict:
        if args.smoke or args.case_index is not None:
            raise SystemExit("internal prediction mode cannot select a public subset")
        return _predict_child(args._prediction_out, args._source_dir, args._item_in)
    if args._score_case is not None:
        if args.smoke or args.case_index is not None:
            raise SystemExit("internal scoring mode cannot select a public subset")
        return _score_child(
            args._score_case,
            args._prediction_in,
            args._score_out,
        )

    _become_child_subreaper()
    source_parent = tempfile.TemporaryDirectory(prefix="visible_source_snapshot_")
    source_snapshot = Path(source_parent.name) / "source"
    _snapshot_source(HERE / "methods" / "main", source_snapshot)
    base, all_items = _load_visible()

    diagnostic = args.smoke or args.case_index is not None
    if args.smoke:
        selected = [(0, all_items[0])]
    elif args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(all_items):
            raise SystemExit(
                f"--case-index must be in [0, {len(all_items) - 1}]"
            )
        selected = [(args.case_index, all_items[args.case_index])]
    else:
        selected = list(enumerate(all_items))

    if diagnostic:
        print("DIAGNOSTIC SUBSET: this result is not comparable to the canonical visible score")

    scores: list[metric.PoseScore] = []
    passed = 0
    deadline = time.monotonic() + TOTAL_TIMEOUT_SEC
    for index, item in selected:
        target = str(item.get("target_id", f"visible-{index}"))
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"self-check exceeded the {TOTAL_TIMEOUT_SEC:.0f}s total limit"
                )
            with tempfile.TemporaryDirectory(prefix="visible_case_") as work:
                work_path = Path(work)
                case_source = work_path / "submission"
                _copy_snapshot(source_snapshot, case_source)
                item_path = _stage_agent_item(item, base, work_path)
                prediction_path = work_path / "prediction.json"
                prediction = _predict_fresh(
                    timeout=min(CASE_TIMEOUT_SEC, remaining),
                    output_path=prediction_path,
                    source_dir=case_source,
                    item_path=item_path,
                )
                # Parsing the JSON here enforces the same typed handoff before
                # the trusted scorer starts; the scorer reads the frozen file.
                if not isinstance(prediction, dict):
                    raise TypeError("prediction must be a JSON object")
                # Match the sealed verifier exactly: TOTAL_TIMEOUT_SEC is
                # checked before a case starts; once prediction completes,
                # trusted scoring retains its independent bounded allowance.
                score = _score_fresh(
                    index,
                    prediction_path,
                    timeout=SCORE_TIMEOUT_SEC,
                    output_path=work_path / "score.json",
                )
            scores.append(score)
            passed += int(score.passed)
            print(
                f"[{index:02d}] {target}: {'PASS' if score.passed else 'FAIL'} "
                f"(PB-valid={score.pb_valid}, RMSD<=2A={score.rmsd_within_2a}, "
                f"matched_CA={score.matched_ca_count}, "
                f"protein_align_RMSD={score.protein_alignment_rmsd:.3f}A)"
            )
        except Exception as exc:
            # Invalid predictions and scoring errors are case failures, exactly
            # as in the sealed verifier.  Continue so the full suite is useful.
            scores.append(
                metric.PoseScore(
                    passed=False,
                    pb_valid=False,
                    rmsd_within_2a=False,
                    checks={},
                    matched_ca_count=0,
                    protein_alignment_rmsd=float("inf"),
                )
            )
            print(f"[{index:02d}] {target}: FAIL ({type(exc).__name__}: {exc})")

    rate = metric.success_rate(scores)
    label = "diagnostic subset" if diagnostic else "canonical visible"
    print(f"\n{label} success: {passed}/{len(selected)} = {rate:.6f}")
    if len(scores) != len(selected) or sum(int(score.passed) for score in scores) != passed:
        raise RuntimeError("internal visible aggregation mismatch")

    # Budget reminder. /app/budget.py is mounted by the harness; when it is not mounted the
    # whole block is skipped, so running this task standalone is unaffected. check=False so a
    # failing budget.py can never take selfcheck down with it. Placed at the end of main() so
    # the reminder prints after this run's score.
    import os as _os, subprocess as _sp, sys as _sys
    if _os.path.exists("/app/budget.py"):
        _sys.stdout.flush()          # without this the reminder prints before the score in a pipe
        _sp.run([_sys.executable, "/app/budget.py"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

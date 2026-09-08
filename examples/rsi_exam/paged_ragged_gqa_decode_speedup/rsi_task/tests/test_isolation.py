from __future__ import annotations

import hashlib
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
TASK = HERE.parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    repo_environment = TASK / "environment"
    if repo_environment.is_dir():
        for name in ("protocol.py", "submission_guard.py", "restricted_runner.py", "child_eval.py"):
            assert digest(HERE / name) == digest(repo_environment / name), f"agent/verifier drift: {name}"
        calibration_entry = (
            TASK / "solution" / "calibration" / "build_anchor_manifest.py"
        ).read_text()
        assert 'TASK / "tests" / "build_anchor_manifest.py"' in calibration_entry
        assert 'run_name="__main__"' in calibration_entry
        assert digest(HERE / "production_baseline" / "solver.py") == digest(
            repo_environment / "methods" / "main" / "solver.py"
        ), "verifier production baseline drifted from the inherited starter"
        assert digest(HERE / "production_baseline" / "solver.py") == digest(
            TASK / "solution" / "baseline" / "solver.py"
        ), "calibration baseline drifted from the scoring baseline"
        docker = (repo_environment / "Dockerfile").read_text().lower()
        assert "copy tests" not in docker and "copy solution" not in docker
        assert "flashinfer" not in docker
        assert "copy problems" in docker and "copy methods" in docker
        assert "ln -s /app/methods/experiment_log.md /app/experiment_log.md" in docker
        experiment_log = (repo_environment / "methods" / "experiment_log.md").read_text()
        assert "raw speedup" in experiment_log and "kept/reverted" in experiment_log
        visible_files = [TASK / "instruction.md"]
        visible_files.extend(path for path in repo_environment.rglob("*") if path.is_file())
        leak_patterns = (
            r"flashinfer", r"6104afe", r"sota[_ -]?anchor", r"upper[_ -]?bound[_ -]?anchor",
            r"reward_of", r"piecewise", r"interpolated between anchors", r"scoring band",
        )
        for path in visible_files:
            if path.suffix == ".pyc":
                continue
            text = path.read_text(errors="ignore").lower()
            for pattern in leak_patterns:
                assert re.search(pattern, text) is None, f"agent-visible leak {pattern!r} in {path}"
        instruction = (TASK / "instruction.md").read_text()
        headings = re.findall(r"^## .+$", instruction, flags=re.MULTILINE)
        assert headings == [
            "## Hard Constraints", "## What You Have", "## What You Submit",
            "## How It Is Judged", "## Common Pitfalls",
        ]

    child_path = Path("/runner/child_eval.py") if Path("/runner/child_eval.py").is_file() else HERE / "child_eval.py"
    child_text = child_path.read_text()
    assert child_text.index("timer = TrustedCudaTimer") < child_text.index("solver = load_solver")
    assert child_text.index("challenge_path.unlink()") < child_text.index("solver = load_solver")
    assert child_text.index("measure_one = timer.measure_one") < child_text.index(
        "solver = load_solver"
    )
    assert child_text.index("prime_clock()") < child_text.index(
        'outputs = run_group(group)'
    )
    assert '"streamed_fresh_input_groups": True' in child_text
    assert '"cpu_exact_mutation_snapshots": True' in child_text
    assert '"cuda_snapshot_bytes": 0' in child_text
    assert '"contiguous_calls_inside_each_timed_event": True' in child_text
    assert '"synchronize_after_each_timed_event": True' in child_text
    assert '"validation_inside_timing": False' in child_text
    assert '"timed_event_count": repeats' in child_text
    assert '"timed_calls_per_event": timed_inner_calls' in child_text
    assert '"timed_call_count": repeats * timed_inner_calls' in child_text
    assert '"event_latency_divided_by_timed_calls": True' in child_text
    assert '"fixed_steady_state_warmup": warmup == 128' in child_text
    assert '"adaptive_warmup_or_posthoc_selection": False' in child_text
    assert '"clock_primer_rounds": timer._clock_primer_rounds' in child_text
    assert '"symmetric_trim_each_side": trim_each_side' in child_text
    assert '"cpu_affinity_before_submission_import"' in child_text
    assert '"cpu_affinity_after_all_candidate_calls"' in child_text
    assert '"sched_affinity_verified"' in child_text
    runner_text = (HERE / "restricted_runner.py").read_text()
    assert '"PYTHONPATH": "/runner"' in runner_text
    assert "env = {" in runner_text
    tests_docker = (HERE / "Dockerfile").read_text()
    assert "COPY production_baseline /tests/production_baseline" in tests_docker
    grade_text = (HERE / "grade.py").read_text()
    assert "bracketed_raw_speedup" in grade_text
    assert "geometric_mean_speedup" in grade_text
    assert "baseline-before, candidate, baseline-after per case" in grade_text
    assert "ANCHOR_STATUS" not in grade_text
    print("test_isolation: PASS")


if __name__ == "__main__":
    main()

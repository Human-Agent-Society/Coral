from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).resolve().parent
GRADE_PATH = TESTS_DIR / "grade.py"
CHILD_PATH = TESTS_DIR / "child_predict.py"
SOURCE_CONTRACT_PATH = TESTS_DIR / "source_contract.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


grade = _load(GRADE_PATH, "protein_secure_grade")
source_contract = _load(SOURCE_CONTRACT_PATH, "protein_source_contract")


def _case(tmp_path: Path):
    msa = tmp_path / "hidden" / "msa" / "SECRET_TARGET"
    crystals = tmp_path / "hidden" / "crystals"
    msa.mkdir(parents=True)
    crystals.mkdir()
    (msa / "A.a3m").write_bytes(b">query\nACD\x00EF\n")
    (crystals / "secret_ligand.sdf").write_text("ligand", encoding="utf-8")
    (crystals / "secret_protein.pdb").write_text("protein", encoding="utf-8")
    item = {
        "target_id": "SECRET_TARGET",
        "protein_chains": [{"chain_id": "A", "sequence": "ACDEF"}],
        "ligand_smiles": "CCO",
        "msa_dir": "msa/SECRET_TARGET",
        "crystal_ligand_sdf": "crystals/secret_ligand.sdf",
        "crystal_protein_pdb": "crystals/secret_protein.pdb",
    }
    return item


def test_normalized_reward_is_bounded_and_anchor_file_fails_closed(tmp_path, monkeypatch):
    anchors = grade.Anchors(0.2, 1.0)
    assert grade.normalized_reward(0.0, anchors) == 0.0
    assert grade.normalized_reward(0.2, anchors) == 0.0
    assert grade.normalized_reward(0.4, anchors) == pytest.approx(0.25)
    assert grade.normalized_reward(1.0, anchors) == pytest.approx(1.0)

    anchor_file = tmp_path / "anchors.json"
    monkeypatch.setattr(grade, "ANCHOR_FILE", anchor_file)
    anchor_file.write_text('{"BASELINE":1.0,"UPPER_BOUND":1.0}', encoding="utf-8")
    with pytest.raises(grade.ConfigurationError, match="must satisfy"):
        grade.load_anchors()
    anchor_file.write_text('{"BASELINE":0.2}', encoding="utf-8")
    with pytest.raises(grade.ConfigurationError, match="unexpected"):
        grade.load_anchors()


def test_case_staging_omits_identity_and_crystals_and_copies_only_current_msa(
    tmp_path, monkeypatch
):
    item = _case(tmp_path)
    hidden = tmp_path / "hidden"
    other = hidden / "msa" / "OTHER_TARGET"
    other.mkdir()
    (other / "A.a3m").write_text(">query\nOTHER\n", encoding="utf-8")
    monkeypatch.setattr(grade, "HELDOUT_DIR", hidden)
    parsed = grade._parse_case(item)

    staged = tmp_path / "staged-msa"
    grade._copy_case_msas(parsed, staged)
    assert sorted(path.name for path in staged.iterdir()) == ["A.a3m"]
    assert (staged / "A.a3m").read_bytes() == b">query\nACDEF\n"
    serialized = json.dumps(parsed.child_item)
    assert "SECRET_TARGET" not in serialized
    assert "crystal" not in serialized
    assert set(parsed.child_item) == {"protein_chains", "ligand_smiles", "msa_dir"}


def test_submission_copy_is_source_only_and_rejects_links(tmp_path):
    source = tmp_path / "submission"
    source.mkdir()
    (source / "solver.py").write_text(
        "def predict_complex(item): return item\n", encoding="utf-8"
    )
    (source / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    destination = tmp_path / "copy"
    solver = grade._copy_submission(source, destination)
    assert solver.is_file()
    assert (destination / "helper.py").is_file()

    non_source = tmp_path / "non-source-submission"
    non_source.mkdir()
    (non_source / "solver.py").write_text(
        "def predict_complex(item): return item\n", encoding="utf-8"
    )
    (non_source / "precomputed_pose.sdf").write_text(
        "must not copy", encoding="utf-8"
    )
    with pytest.raises(grade.SubmissionError, match="only Python"):
        grade._copy_submission(non_source, tmp_path / "non-source-copy")

    linked_source = tmp_path / "linked-submission"
    linked_source.mkdir()
    original = linked_source / "real.py"
    original.write_text("VALUE = 1\n", encoding="utf-8")
    (linked_source / "solver.py").symlink_to(original)
    with pytest.raises(grade.SubmissionError, match="linked or special"):
        grade._copy_submission(linked_source, tmp_path / "linked-copy")


@pytest.mark.parametrize(
    "payload, message",
    [
        ("import zlib\ndef predict_complex(item): return item\n", "data codec"),
        (
            "POSE = " + repr("A" * (source_contract.MAX_LITERAL_BYTES + 1))
            + "\ndef predict_complex(item): return item\n",
            "static literal",
        ),
        (
            "def predict_complex(item):\n    return eval('item')\n",
            "dynamic code",
        ),
    ],
)
def test_submission_rejects_embedded_lookup_table_mechanisms(
    tmp_path, payload, message
):
    source = tmp_path / "submission"
    source.mkdir()
    (source / "solver.py").write_text(payload, encoding="utf-8")
    with pytest.raises(grade.SubmissionError, match=message):
        grade._copy_submission(source, tmp_path / "copy")


@pytest.mark.parametrize("reserved", ["metric.py", "source_contract.py"])
def test_submission_cannot_shadow_public_selfcheck_contract(tmp_path, reserved):
    source = tmp_path / "submission"
    source.mkdir()
    (source / "solver.py").write_text(
        "def predict_complex(item): return item\n", encoding="utf-8"
    )
    (source / reserved).write_text("VALUE = 'shadow'\n", encoding="utf-8")
    with pytest.raises(grade.SubmissionError, match="reserved verifier filename"):
        grade._copy_submission(source, tmp_path / "copy")


def test_prediction_inode_rejects_symlink_hardlink_fifo_and_oversize(tmp_path, monkeypatch):
    regular = tmp_path / "regular.json"
    regular.write_text('{}', encoding="utf-8")
    grade._validate_prediction_inode(regular)

    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(regular)
    with pytest.raises(grade.PredictionError, match="non-symlink"):
        grade._validate_prediction_inode(symlink)

    hardlink = tmp_path / "hardlink.json"
    os.link(regular, hardlink)
    with pytest.raises(grade.PredictionError, match="hard-linked"):
        grade._validate_prediction_inode(hardlink)

    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(grade.PredictionError, match="regular"):
        grade._validate_prediction_inode(fifo)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * 17)
    monkeypatch.setattr(grade, "MAX_PREDICTION_JSON_BYTES", 16)
    with pytest.raises(grade.PredictionError, match="invalid size"):
        grade._validate_prediction_inode(oversized)


def _pdb_atom(serial, residue_number, x="   1.000", atom=" CA ", record="ATOM"):
    return (
        f"{record:<6s}{serial:5d} {atom} ALA A{residue_number:4d}    "
        f"{x}{'   2.000'}{'   3.000'}  1.00 20.00           C"
    )


def _sdf(atom_count=1):
    atom_lines = [
        f"{float(index):10.4f}{0.0:10.4f}{0.0:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0"
        for index in range(atom_count)
    ]
    return "\n".join(
        ["ligand", "  test", "", f"{atom_count:3d}{0:3d}  0  0  0  0  0  0  0  0  1 V2000"]
        + atom_lines
        + ["M  END", "$$$$", ""]
    )


def test_prediction_work_bounds_reject_nonfinite_non_ca_and_graph_bombs(tmp_path):
    metric_module = grade._load_trusted_metric()
    crystal = tmp_path / "crystal.sdf"
    crystal.write_text(_sdf(1), encoding="utf-8")
    case = grade.CaseAssets(
        child_item={},
        expected_chains=({"chain_id": "A", "sequence": "ACD"},),
        msa_source_dir=tmp_path,
        crystal_ligand=crystal,
        crystal_protein=tmp_path / "unused.pdb",
    )
    finite = {
        "protein_pdb": "\n".join(_pdb_atom(i, i) for i in range(1, 4)),
        "ligand_sdf": _sdf(1),
    }
    grade._validate_prediction_work_bounds(finite, case, metric_module)

    nonfinite = dict(finite)
    nonfinite["protein_pdb"] += "\n" + _pdb_atom(4, 4, x="     nan", atom=" CB ")
    with pytest.raises(grade.PredictionError, match="work bounds"):
        grade._validate_prediction_work_bounds(nonfinite, case, metric_module)

    residue_bomb = dict(finite)
    residue_bomb["protein_pdb"] = "\n".join(
        _pdb_atom(index, index) for index in range(1, 40)
    )
    with pytest.raises(grade.PredictionError, match="work bounds"):
        grade._validate_prediction_work_bounds(residue_bomb, case, metric_module)

    hetatm_bomb = dict(finite)
    hetatm_bomb["protein_pdb"] = "\n".join(
        _pdb_atom(index, index, record="HETATM") for index in range(1, 40)
    )
    with pytest.raises(grade.PredictionError, match="work bounds"):
        grade._validate_prediction_work_bounds(hetatm_bomb, case, metric_module)

    repeated_key_bomb = dict(finite)
    repeated_key_bomb["protein_pdb"] = "\n".join(
        line
        for index in range(1, 40)
        for line in (_pdb_atom(index, 1), "TER")
    )
    with pytest.raises(grade.PredictionError, match="work bounds"):
        grade._validate_prediction_work_bounds(
            repeated_key_bomb, case, metric_module
        )

    ligand_bomb = dict(finite)
    ligand_bomb["ligand_sdf"] = _sdf(65)
    with pytest.raises(grade.PredictionError, match="work bounds"):
        grade._validate_prediction_work_bounds(ligand_bomb, case, metric_module)


def test_trusted_score_worker_is_time_bounded_and_fail_closed(tmp_path, monkeypatch):
    fake = tmp_path / "score-worker.py"
    fake.write_text("import time\ntime.sleep(10)\n", encoding="utf-8")
    scratch = grade.CaseScratch(
        root=tmp_path,
        solver=tmp_path / "solver.py",
        item_json=tmp_path / "item.json",
        prediction_json=tmp_path / "prediction.json",
        work_dir=tmp_path,
    )
    case = grade.CaseAssets(
        child_item={"ligand_smiles": "C"},
        expected_chains=({"chain_id": "A", "sequence": "ACD"},),
        msa_source_dir=tmp_path,
        crystal_ligand=tmp_path / "crystal.sdf",
        crystal_protein=tmp_path / "crystal.pdb",
    )
    monkeypatch.setattr(grade, "SCORE_SCRIPT", fake)
    monkeypatch.setattr(grade, "SCORE_TIMEOUT_SEC", 0.05)
    with pytest.raises(grade.PredictionError, match="bounded trusted scoring"):
        grade._run_trusted_score_worker(scratch, case)


def test_child_contract_executes_solver_outside_trusted_parent(tmp_path):
    launcher = tmp_path / "child_predict.py"
    shutil.copy2(CHILD_PATH, launcher)
    (tmp_path / "grade.py").write_text(
        "def prepare_private_shm_namespace(uid, gid): return 123\n"
        "def drop_child_privileges(uid, gid): return None\n"
        "def restrict_child_filesystem(scratch, *, private_shm_device=None): return None\n"
        "def restrict_child_persistent_ipc(): return None\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission"
    submission.mkdir()
    marker = tmp_path / "imported.marker"
    (submission / "solver.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('child only')\n"
        "def predict_complex(item):\n"
        "    assert set(item) == {'protein_chains', 'ligand_smiles', 'msa_dir'}\n"
        "    return {'protein_pdb': 'ATOM', 'ligand_sdf': 'M  END'}\n",
        encoding="utf-8",
    )
    msa = tmp_path / "msa"
    msa.mkdir()
    item = tmp_path / "item.json"
    item.write_text(
        json.dumps(
            {
                "protein_chains": [{"chain_id": "A", "sequence": "ACD"}],
                "ligand_smiles": "CCO",
                "msa_dir": str(msa),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prediction.json"
    assert not marker.exists()
    proc = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--scratch",
            str(tmp_path),
            "--solver",
            str(submission / "solver.py"),
            "--item",
            str(item),
            "--output",
            str(output),
            "--runner-uid",
            "10001",
            "--runner-gid",
            "10001",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert marker.read_text(encoding="utf-8") == "child only"
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "protein_pdb": "ATOM",
        "ligand_sdf": "M  END",
    }


def test_child_uses_distinct_exit_code_when_sandbox_setup_fails(tmp_path, monkeypatch):
    child = _load(CHILD_PATH, "protein_child_setup_failure")
    solver = tmp_path / "solver.py"
    item = tmp_path / "item.json"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    solver.write_text("def predict_complex(item): return {}\n", encoding="utf-8")
    item.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        child.trusted_sandbox,
        "restrict_child_filesystem",
        lambda _scratch, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("no Landlock")
        ),
    )
    monkeypatch.setattr(
        child.trusted_sandbox, "prepare_private_shm_namespace", lambda _u, _g: 123
    )
    monkeypatch.setattr(
        child.trusted_sandbox, "drop_child_privileges", lambda _u, _g: None
    )
    assert child.main(
        [
            "--scratch",
            str(tmp_path),
            "--solver",
            str(solver),
            "--item",
            str(item),
                "--output",
                str(output_dir / "prediction.json"),
                "--runner-uid",
                "10001",
                "--runner-gid",
                "10001",
            ]
    ) == grade.SANDBOX_SETUP_EXIT


@pytest.mark.skipif(sys.platform != "linux", reason="Landlock is Linux-only")
def test_landlock_allows_scratch_but_denies_trusted_reads_and_external_writes(
    tmp_path, monkeypatch
):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    secret = tmp_path / "heldout-secret"
    secret.write_text("secret", encoding="utf-8")
    external = tmp_path / "outside-write"
    child = tmp_path / "landlock-child.py"
    child.write_text(
        "from pathlib import Path\n"
        "import os, sys\n"
        "scratch, secret, external = map(Path, sys.argv[1:])\n"
        "fd = os.open('/dev/null', os.O_RDWR)\n"
        "os.close(fd)\n"
        "comm = Path(f'/proc/self/task/{os.getpid()}/comm')\n"
        "comm.write_text('pb-runner\\n')\n"
        "(scratch / 'allowed').write_text('ok')\n"
        "try:\n"
        "    secret.read_text()\n"
        "except PermissionError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit(10)\n"
        "try:\n"
        "    external.write_text('bad')\n"
        "except PermissionError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit(11)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(grade, "CHILD_SCRIPT", child)
    system_python = Path("/usr/bin/python3")
    if not system_python.is_file():
        pytest.skip("system Python under a Landlock-readable runtime root is unavailable")
    proc = subprocess.run(
        [str(system_python), str(child), str(scratch), str(secret), str(external)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=lambda: grade.restrict_child_filesystem(scratch),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (scratch / "allowed").read_text(encoding="utf-8") == "ok"
    assert not external.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="seccomp is Linux-only")
def test_two_workers_cannot_persist_state_through_sysv_ipc(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    child = tmp_path / "ipc-child.py"
    child.write_text(
        "import ctypes, sys\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "segment = libc.shmget(0x50424346, 64, 0o1000 | 0o600)\n"
        "if segment != -1 or ctypes.get_errno() != 1:\n"
        "    raise SystemExit(20)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(grade, "CHILD_SCRIPT", child)
    system_python = Path("/usr/bin/python3")
    if not system_python.is_file():
        pytest.skip("system Python under a Landlock-readable runtime root is unavailable")

    def isolate():
        grade.restrict_child_filesystem(scratch)
        grade.restrict_child_persistent_ipc()

    # Both independently sandboxed workers use the same UID and fixed key.  A
    # missing seccomp boundary would let the first create cross-case state.
    for _ in range(2):
        proc = subprocess.run(
            [str(system_python), str(child)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=isolate,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="seccomp is Linux-only")
def test_seccomp_resolves_and_denies_namespace_mount_entry_points(tmp_path):
    import ctypes
    import ctypes.util

    library_name = ctypes.util.find_library("seccomp")
    if not library_name:
        pytest.skip("libseccomp is unavailable")
    seccomp = ctypes.CDLL(library_name)
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    required = set(grade.NAMESPACE_MOUNT_SYSCALLS) | {"clone", "clone3"}
    numbers = {
        name: grade._resolve_seccomp_syscall_number(seccomp, name)
        for name in required
    }
    assert all(number >= 0 for number in numbers.values()), numbers
    assert numbers["open_tree_attr"] == 467

    child = tmp_path / "namespace-child.py"
    child.write_text(
        "import ctypes, errno, json, os, signal, sys, threading\n"
        "numbers = json.loads(sys.argv[1])\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "# Ordinary runtime thread creation must remain available.\n"
        "thread = threading.Thread(target=lambda: None)\n"
        "thread.start(); thread.join()\n"
        "# Exercise every unconditional namespace/mount entry point.  Invalid\n"
        "# pointer arguments make a missed rule harmless; seccomp runs first,\n"
        "# so every covered syscall must still return EPERM.\n"
        "for name in sorted(set(numbers) - {'clone', 'clone3'}):\n"
        "    ctypes.set_errno(0)\n"
        "    result = libc.syscall(numbers[name], -1, -1, -1, -1, -1, -1)\n"
        "    if result != -1 or ctypes.get_errno() != errno.EPERM:\n"
        "        raise SystemExit('unblocked %s: result=%s errno=%s' % (name, result, ctypes.get_errno()))\n"
        "# Each masked clone rule is exercised with CLONE_SIGHAND but without\n"
        "# CLONE_VM.  If a rule were missing the kernel would reject EINVAL,\n"
        "# without ever creating a child or namespace.\n"
        "for flag in " + repr(list(grade.CLONE_NAMESPACE_FLAGS)) + ":\n"
        "    ctypes.set_errno(0)\n"
        "    result = libc.syscall(numbers['clone'], flag | 0x00000800 | signal.SIGCHLD, 0, 0, 0, 0)\n"
        "    if result != -1 or ctypes.get_errno() != errno.EPERM:\n"
        "        raise SystemExit('unblocked clone flag %#x: result=%s errno=%s' % (flag, result, ctypes.get_errno()))\n"
        "ctypes.set_errno(0)\n"
        "if libc.syscall(numbers['clone3'], 0, 0) != -1 or ctypes.get_errno() != errno.ENOSYS:\n"
        "    raise SystemExit(31)\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(child),
            json.dumps(numbers, sort_keys=True),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=grade.restrict_child_persistent_ipc,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="seccomp is Linux-only")
def test_descendant_cannot_escape_worker_group_and_mutate_after_teardown(
    tmp_path, monkeypatch
):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    marker = scratch / "late-marker"
    child = tmp_path / "process-child.py"
    child.write_text(
        "import os, time\n"
        "pid = os.fork()\n"
        "if pid:\n"
        "    raise SystemExit(0)\n"
        "try:\n"
        "    os.setsid()\n"
        "except PermissionError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit(21)\n"
        "time.sleep(0.8)\n"
        f"open({str(marker)!r}, 'w').write('survived')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(grade, "CHILD_SCRIPT", child)
    system_python = Path("/usr/bin/python3")
    if not system_python.is_file():
        pytest.skip("system Python under a Landlock-readable runtime root is unavailable")

    def isolate():
        grade.restrict_child_filesystem(scratch)
        grade.restrict_child_persistent_ipc()

    proc = subprocess.Popen(
        [str(system_python), str(child)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        preexec_fn=isolate,
    )
    proc.wait(timeout=2.0)
    grade._kill_and_reap_worker(proc)
    time.sleep(0.9)
    assert not marker.exists()


def test_debug_output_is_aggregate_only(tmp_path, monkeypatch):
    monkeypatch.setattr(grade, "VERIFIER_LOG_DIR", tmp_path)
    result = {
        "reward": 0.5,
        "metric": 0.6,
        "n_cases": 10,
        "passed_cases": 6,
        "pb_valid_cases": 7,
        "rmsd_within_2a_cases": 6,
        "correctness": True,
        "error_code": "none",
        "protocol": "protein-cofold-sealed-runner-v2",
        "split_protocol": "posebusters-visible20-hidden42-isolated-final",
    }
    grade._write_outputs(result)
    debug = json.loads((tmp_path / "grade_debug.json").read_text(encoding="utf-8"))
    serialized = json.dumps(debug)
    assert "target" not in serialized.lower()
    assert "path" not in serialized.lower()
    assert "errors" not in serialized.lower()
    assert not any(isinstance(value, list) for value in debug.values())


def test_child_environment_scrubs_secrets_and_accepts_only_baked_cache_paths(
    tmp_path, monkeypatch
):
    scratch = grade.CaseScratch(
        root=tmp_path,
        solver=tmp_path / "submission" / "solver.py",
        item_json=tmp_path / "input" / "item.json",
        prediction_json=tmp_path / "output" / "prediction.json",
        work_dir=tmp_path / "work",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("BASELINE", "0.2")
    monkeypatch.setenv("UPPER_BOUND", "1.0")
    monkeypatch.setenv("PYTHONPATH", "/app/methods/main")
    monkeypatch.setenv("BOLTZ_CACHE", "/opt/boltz-cache")
    monkeypatch.setenv("HF_HOME", "/tests/heldout/not-allowed")
    env = grade._child_environment(scratch)
    assert "OPENAI_API_KEY" not in env
    assert "HF_TOKEN" not in env
    assert "BASELINE" not in env
    assert "UPPER_BOUND" not in env
    assert "PYTHONPATH" not in env
    assert env["BOLTZ_CACHE"] == "/opt/boltz-cache"
    assert "HF_HOME" not in env


def test_per_case_prediction_failure_stays_in_denominator_without_prefix_output(
    tmp_path, monkeypatch
):
    cases = [object(), object(), object()]
    outcomes = iter(
        [
            grade.CaseOutcome(True, True, True),
            grade.PredictionError("hidden case detail must not escape"),
            grade.CaseOutcome(False, True, False),
        ]
    )

    def score_one(*_args, **_kwargs):
        value = next(outcomes)
        if isinstance(value, BaseException):
            raise value
        return value

    class Metric:
        @staticmethod
        def success_rate(scores):
            return sum(int(score.passed) for score in scores) / len(scores)

    monkeypatch.setattr(grade, "_runner_identity", lambda: (12345, 12345))
    monkeypatch.setattr(grade, "_become_child_subreaper", lambda: None)
    monkeypatch.setattr(grade, "_validate_trusted_launcher_permissions", lambda: None)
    monkeypatch.setattr(grade, "_score_one_case", score_one)
    monkeypatch.setattr(grade, "RUNNER_ROOT", tmp_path / "runner-root")
    monkeypatch.setattr(grade, "TOTAL_TIMEOUT_SEC", 60.0)
    result = grade._score_all_cases(cases, Metric(), tmp_path / "frozen-source")
    assert result == {
        "metric": pytest.approx(1 / 3),
        "n_cases": 3,
        "passed_cases": 1,
        "pb_valid_cases": 2,
        "rmsd_within_2a_cases": 1,
        "invalid_cases": 1,
    }
    assert "detail" not in json.dumps(result)


def test_effective_submission_is_snapshotted_once_and_hashed(tmp_path, monkeypatch):
    source = tmp_path / "live"
    source.mkdir()
    (source / "solver.py").write_text(
        "def predict_complex(item): return item\n", encoding="utf-8"
    )
    (source / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(grade, "SUBMISSION_DIR", source)
    snapshot, first_hash = grade._make_submission_snapshot()
    try:
        assert len(first_hash) == 64
        assert grade._hash_effective_submission(snapshot) == first_hash
        (source / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
        assert (snapshot / "helper.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        assert grade._hash_effective_submission(snapshot) == first_hash
    finally:
        grade._remove_submission_snapshot(snapshot)

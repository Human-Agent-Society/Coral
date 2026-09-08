from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]


def _load_selfcheck(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _mini_selfcheck(tmp_path: Path, solver_source: str) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    for name in ("selfcheck.py", "metric.py", "source_contract.py"):
        shutil.copy2(TASK_ROOT / "environment" / name, app / name)
    method = app / "methods" / "main"
    method.mkdir(parents=True)
    (method / "solver.py").write_text(solver_source, encoding="utf-8")
    visible = app / "data" / "visible"
    (visible / "msa" / "case").mkdir(parents=True)
    (visible / "msa" / "case" / "A.a3m").write_bytes(b">query\nACD\x00EF\n")
    item = {
        "protein_chains": [{"chain_id": "A", "sequence": "ACDEFGHIKL"}],
        "ligand_smiles": "CC",
        "msa_dir": "msa/case",
        "crystal_ligand_sdf": "unused.sdf",
        "crystal_protein_pdb": "unused.pdb",
    }
    (visible / "items.json").write_text(
        json.dumps([item for _ in range(20)]),
        encoding="utf-8",
    )
    return app / "selfcheck.py"


def _agent_item_path(selfcheck: Path, tmp_path: Path) -> Path:
    item = tmp_path / "agent_item.json"
    item.write_text(
        json.dumps(
            {
                "protein_chains": [
                    {"chain_id": "A", "sequence": "ACDEFGHIKL"}
                ],
                "ligand_smiles": "CC",
                "msa_dir": str(selfcheck.parent / "data" / "visible" / "msa" / "case"),
            }
        ),
        encoding="utf-8",
    )
    return item


def test_internal_prediction_entrypoint_writes_typed_json(tmp_path):
    selfcheck = _mini_selfcheck(
        tmp_path,
        "import sys\n"
        "def predict_complex(item):\n"
        "    assert len(sys.argv) == 1 and sys.argv[0].endswith('/solver.py')\n"
        "    return {'protein_pdb': 'PDB', 'ligand_sdf': 'SDF'}\n",
    )
    output = tmp_path / "prediction.json"
    item = _agent_item_path(selfcheck, tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            str(selfcheck),
            "--_predict",
            "--_prediction-out",
            str(output),
            "--_source-dir",
            str(selfcheck.parent / "methods" / "main"),
            "--_item-in",
            str(item),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "protein_pdb": "PDB",
        "ligand_sdf": "SDF",
    }


def test_internal_scorer_does_not_import_submitted_source(tmp_path):
    selfcheck = _mini_selfcheck(
        tmp_path,
        "raise RuntimeError('the trusted scorer imported submitted source')\n",
    )
    app = selfcheck.parent
    (app / "metric.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass(frozen=True)\n"
        "class PoseScore:\n"
        "    passed: bool\n"
        "    pb_valid: bool\n"
        "    rmsd_within_2a: bool\n"
        "    checks: dict\n"
        "    matched_ca_count: int\n"
        "    protein_alignment_rmsd: float\n"
        "def score_pose(prediction, **kwargs):\n"
        "    return PoseScore(True, True, True, {'ok': True}, 10, 0.25)\n",
        encoding="utf-8",
    )
    prediction = tmp_path / "prediction.json"
    prediction.write_text(
        json.dumps({"protein_pdb": "PDB", "ligand_sdf": "SDF"}),
        encoding="utf-8",
    )
    output = tmp_path / "score.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(selfcheck),
            "--_score-case",
            "0",
            "--_prediction-in",
            str(prediction),
            "--_score-out",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["checks"] == {"ok": True}


def test_fresh_prediction_timeout_kills_the_process_group(tmp_path):
    selfcheck = _mini_selfcheck(
        tmp_path,
        "import time\n"
        "def predict_complex(item):\n"
        "    time.sleep(30)\n"
        "    return {'protein_pdb': 'PDB', 'ligand_sdf': 'SDF'}\n",
    )
    item = _agent_item_path(selfcheck, tmp_path)
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import importlib.util, pathlib, sys\n"
        "spec = importlib.util.spec_from_file_location('fresh_selfcheck', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "try:\n"
        "    module._become_child_subreaper()\n"
        "    module._predict_fresh(\n"
        "        0.1, pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]),\n"
        "        pathlib.Path(sys.argv[4])\n"
        "    )\n"
        "except TimeoutError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    proc = subprocess.run(
        [
            sys.executable,
            str(driver),
            str(selfcheck),
            str(tmp_path / "never.json"),
            str(selfcheck.parent / "methods" / "main"),
            str(item),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - started < 2


def test_successful_prediction_reaps_background_descendants(tmp_path):
    marker = tmp_path / "escaped.marker"
    selfcheck = _mini_selfcheck(
        tmp_path,
        "import subprocess, sys\n"
        "def predict_complex(item):\n"
        "    subprocess.Popen(\n"
        f"        [sys.executable, '-c', \"import pathlib,time; time.sleep(0.4); pathlib.Path({str(marker)!r}).write_text('escaped')\"],\n"
        "        start_new_session=True,\n"
        "    )\n"
        "    return {'protein_pdb': 'PDB', 'ligand_sdf': 'SDF'}\n",
    )
    item = _agent_item_path(selfcheck, tmp_path)
    driver = tmp_path / "cleanup_driver.py"
    driver.write_text(
        "import importlib.util, pathlib, sys\n"
        "spec = importlib.util.spec_from_file_location('fresh_selfcheck', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module._become_child_subreaper()\n"
        "module._predict_fresh(\n"
        "    2.0, pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]),\n"
        "    pathlib.Path(sys.argv[4])\n"
        ")\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(driver),
            str(selfcheck),
            str(tmp_path / "prediction.json"),
            str(selfcheck.parent / "methods" / "main"),
            str(item),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    time.sleep(0.6)
    assert not marker.exists()


def test_visible_prediction_item_uses_neutral_staged_msa_path(tmp_path):
    selfcheck = _mini_selfcheck(
        tmp_path,
        "def predict_complex(item):\n"
        "    return {'protein_pdb': 'PDB', 'ligand_sdf': 'SDF'}\n",
    )
    module = _load_selfcheck(selfcheck, "neutral_input_selfcheck")
    base, items = module._load_visible()
    item = dict(items[0])
    item["target_id"] = "VISIBLE_ID_MUST_NOT_ENTER_ITEM"
    staged = module._stage_agent_item(item, base, tmp_path / "case_work")
    payload = json.loads(staged.read_text(encoding="utf-8"))
    assert set(payload) == {"protein_chains", "ligand_smiles", "msa_dir"}
    assert Path(payload["msa_dir"]).parts[-2:] == ("input", "msa")
    assert "VISIBLE_ID_MUST_NOT_ENTER_ITEM" not in staged.read_text(encoding="utf-8")
    assert b"\x00" not in (Path(payload["msa_dir"]) / "A.a3m").read_bytes()


def test_selfcheck_child_environment_scrubs_agent_secrets(tmp_path, monkeypatch):
    selfcheck = _mini_selfcheck(
        tmp_path,
        "def predict_complex(item):\n"
        "    return {'protein_pdb': 'PDB', 'ligand_sdf': 'SDF'}\n",
    )
    module = _load_selfcheck(selfcheck, "scrubbed_env_selfcheck")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("CODEX_AUTH_JSON", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://agent-egress:3128")
    monkeypatch.setenv("PYTHONPATH", "/agent/private")
    monkeypatch.setenv("BOLTZ_CACHE", "/opt/boltz_cache")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "unsafe-cache"))
    env = module._child_environment(tmp_path / "runtime")
    assert "OPENAI_API_KEY" not in env
    assert "CODEX_AUTH_JSON" not in env
    assert "HTTPS_PROXY" not in env
    assert "PYTHONPATH" not in env
    assert env["BOLTZ_CACHE"] == "/opt/boltz_cache"
    assert "HF_HOME" not in env
    assert env["HOME"].endswith("runtime/home")
    assert env["TMPDIR"].endswith("runtime/tmp")


def test_source_snapshot_is_frozen_and_reused_by_copy(tmp_path):
    selfcheck = _mini_selfcheck(
        tmp_path,
        "VALUE = 1\n"
        "def predict_complex(item):\n"
        "    return {'protein_pdb': str(VALUE), 'ligand_sdf': 'SDF'}\n",
    )
    module_path = tmp_path / "snapshot_driver.py"
    frozen = tmp_path / "frozen"
    copied = tmp_path / "copied"
    live = selfcheck.parent / "methods" / "main"
    module_path.write_text(
        "import importlib.util, pathlib, sys\n"
        "spec = importlib.util.spec_from_file_location('fresh_selfcheck', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "module._snapshot_source(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))\n"
        "(pathlib.Path(sys.argv[2]) / 'solver.py').write_text('VALUE = 2\\n')\n"
        "module._copy_snapshot(pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(module_path), str(selfcheck), str(live), str(frozen), str(copied)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (frozen / "solver.py").read_text(encoding="utf-8").startswith("VALUE = 1")
    assert (copied / "solver.py").read_text(encoding="utf-8").startswith("VALUE = 1")


def test_main_freezes_once_and_copies_that_snapshot_for_every_case(tmp_path, monkeypatch):
    selfcheck = _mini_selfcheck(
        tmp_path,
        "def predict_complex(item):\n"
        "    return {'protein_pdb': 'PDB', 'ligand_sdf': 'SDF'}\n",
    )
    module = _load_selfcheck(selfcheck, "snapshot_count_selfcheck")
    args = types.SimpleNamespace(
        smoke=False,
        case_index=None,
        _predict=False,
        _prediction_out=None,
        _source_dir=None,
        _item_in=None,
        _score_case=None,
        _prediction_in=None,
        _score_out=None,
    )
    monkeypatch.setattr(module, "_arguments", lambda: args)
    monkeypatch.setattr(module, "_become_child_subreaper", lambda: None)
    items = [{"target_id": "a"}, {"target_id": "b"}]
    monkeypatch.setattr(module, "_load_visible", lambda: (tmp_path, items))
    calls = {"snapshot": 0, "copy": 0}

    def snapshot(_source, _destination):
        calls["snapshot"] += 1

    def copy(_source, _destination):
        calls["copy"] += 1

    monkeypatch.setattr(module, "_snapshot_source", snapshot)
    monkeypatch.setattr(module, "_copy_snapshot", copy)
    monkeypatch.setattr(module, "_stage_agent_item", lambda *_args: tmp_path / "item.json")
    monkeypatch.setattr(
        module,
        "_predict_fresh",
        lambda *_args, **_kwargs: {"protein_pdb": "PDB", "ligand_sdf": "SDF"},
    )
    score = types.SimpleNamespace(
        passed=True,
        pb_valid=True,
        rmsd_within_2a=True,
        checks={"ok": True},
        matched_ca_count=10,
        protein_alignment_rmsd=0.0,
    )
    monkeypatch.setattr(module, "_score_fresh", lambda *_args, **_kwargs: score)
    assert module.main() == 0
    assert calls == {"snapshot": 1, "copy": 2}


def test_started_case_keeps_sealed_scoring_allowance(tmp_path, monkeypatch):
    selfcheck = _mini_selfcheck(
        tmp_path,
        "def predict_complex(item):\n"
        "    return {'protein_pdb': 'PDB', 'ligand_sdf': 'SDF'}\n",
    )
    module = _load_selfcheck(selfcheck, "score_budget_selfcheck")
    args = types.SimpleNamespace(
        smoke=False,
        case_index=None,
        _predict=False,
        _prediction_out=None,
        _source_dir=None,
        _item_in=None,
        _score_case=None,
        _prediction_in=None,
        _score_out=None,
    )
    monkeypatch.setattr(module, "_arguments", lambda: args)
    monkeypatch.setattr(module, "_become_child_subreaper", lambda: None)
    monkeypatch.setattr(module, "_snapshot_source", lambda *_args: None)
    monkeypatch.setattr(module, "_copy_snapshot", lambda *_args: None)
    monkeypatch.setattr(module, "_stage_agent_item", lambda *_args: tmp_path / "item.json")
    monkeypatch.setattr(
        module,
        "_load_visible",
        lambda: (tmp_path, [{"target_id": "a"}]),
    )
    monkeypatch.setattr(
        module,
        "_predict_fresh",
        lambda *_args, **_kwargs: {"protein_pdb": "PDB", "ligand_sdf": "SDF"},
    )
    score = types.SimpleNamespace(
        passed=True,
        pb_valid=True,
        rmsd_within_2a=True,
        checks={"ok": True},
        matched_ca_count=10,
        protein_alignment_rmsd=0.0,
    )
    observed: list[float] = []

    def score_fresh(*_args, **kwargs):
        observed.append(kwargs["timeout"])
        return score

    monkeypatch.setattr(module, "_score_fresh", score_fresh)
    moments = iter((0.0, module.TOTAL_TIMEOUT_SEC - 0.01))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(moments))
    assert module.main() == 0
    assert observed == [module.SCORE_TIMEOUT_SEC]

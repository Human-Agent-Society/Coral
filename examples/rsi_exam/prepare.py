# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=0.34,<2", "pyyaml>=6"]
# ///
"""Import one or all public RSI-Exam tasks as standalone CORAL examples."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_grader/src"))
from rsi_exam_grader.contract import artifact_sources  # noqa: E402

DATASET = "RSI-Exam/RSI-Exam"
REVISION = "66f54935eaa576e27dae446f74c2ce17875c14da"


class TaskDumper(yaml.SafeDumper):
    """Keep multi-line task instructions reviewable in the generated YAML."""


TaskDumper.add_representer(
    str,
    lambda dumper, value: dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None
    ),
)


def public_tasks() -> list[str]:
    return json.loads((HERE / "tasks.json").read_text())


def build_task(
    source: Path, output: Path, *, revision: str = REVISION, files: list[dict] | None = None
) -> None:
    """Copy the trusted task separately from its public environment and submission."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = tomllib.loads((source / "task.toml").read_text())
    sources = artifact_sources(config)
    if config.get("verifier", {}).get("environment_mode") != "separate":
        raise ValueError("Expected RSI-Exam's separate verifier environment")
    for required in [
        "instruction.md",
        "environment/Dockerfile",
        "tests/Dockerfile",
        "tests/test.sh",
    ]:
        if not (source / required).is_file():
            raise ValueError(f"Incomplete task: missing {source / required}")
    # Imported symlinks could cross the public/private boundary when copied.
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Source task must not contain symlinks")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rsi-import-", dir=output.parent) as temp:
        staging = Path(temp) / "task"
        staging.mkdir()
        shutil.copytree(
            HERE / "_grader",
            staging / "grader",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".venv", "dist"),
        )
        trusted = staging / "rsi_task"
        trusted.mkdir()
        seed = staging / "seed"
        public = seed / "harbor"
        public.mkdir(parents=True)
        for name in ["task.toml", "instruction.md"]:
            shutil.copy2(source / name, trusted / name)
            shutil.copy2(source / name, public / name)
        for name in ["environment", "tests"]:
            shutil.copytree(source / name, trusted / name)
        shutil.copytree(source / "environment", public / "environment")
        shutil.copy2(source / "instruction.md", seed / "instruction.md")
        package = HERE / "_grader/src/rsi_exam_grader"
        shutil.copy2(package / "replay.py", seed / "rsi_runtime.py")
        shutil.copy2(package / "contract.py", seed / "rsi_contract.py")
        # Most baselines are COPY'd directly from environment/methods. For
        # Dockerfile-generated artifacts, rsi_runtime.py bootstrap extracts them.
        for artifact in sources:
            relative = Path(artifact).relative_to("/app")
            baseline = source / "environment" / relative
            target = seed / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if baseline.is_dir():
                shutil.copytree(baseline, target)
            elif baseline.is_file():
                shutil.copy2(baseline, target)
        (seed / ".gitignore").write_text(".rsi_runs/\n.venv/\n__pycache__/\n*.pyc\n")
        license_path = source.parent / "LICENSE"
        if not license_path.exists():
            license_path = HERE / "UPSTREAM_LICENSE"
        shutil.copy2(license_path, staging / "UPSTREAM_LICENSE")
        shutil.copy2(license_path, seed / "UPSTREAM_LICENSE")
        hashes = {
            str(p.relative_to(trusted)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(trusted.rglob("*"))
            if p.is_file()
        }
        downloads = []
        ignored = []
        for entry in files or []:
            if entry["type"] != "file":
                continue
            relative = Path(entry["path"]).relative_to(source.name)
            if relative.parts[0] not in {"environment", "tests"}:
                continue
            if (trusted / relative).exists():
                continue
            record = {"path": relative.as_posix(), "size": entry["size"]}
            if entry.get("lfs"):
                record["sha256"] = entry["lfs"]["oid"]
            else:
                record["git_blob_sha1"] = entry["oid"]
            downloads.append(record)
            ignored.append("/rsi_task/" + relative.as_posix())
            if relative.parts[0] == "environment":
                ignored.append("/seed/harbor/" + relative.as_posix())
        if downloads:
            (trusted / "assets.json").write_text(json.dumps(downloads, indent=2) + "\n")
            escaped = [
                "".join("\\" + c if c in "\\[]*?!#" else c for c in path) for path in ignored
            ]
            (staging / ".gitignore").write_text("\n".join(escaped) + "\n")
        (staging / "UPSTREAM.json").write_text(
            json.dumps(
                {
                    "dataset": DATASET,
                    "revision": revision,
                    "task": source.name,
                    "source": f"https://huggingface.co/datasets/{DATASET}/tree/{revision}/{source.name}",
                    "sha256": hashes,
                    "downloads": downloads,
                },
                indent=2,
            )
            + "\n"
        )

        verifier = config["verifier"]
        environment = config["environment"]
        build_budget = float(environment.get("build_timeout_sec", 1800))
        build_budget += float(
            verifier.get("environment", {}).get("build_timeout_sec", build_budget)
        )
        timeout = int(build_budget + float(verifier.get("timeout_sec", 3600)) + 900)
        description = (
            "Optimize the RSI-Exam task below. The CORAL worktree mirrors /app for "
            "submission paths only: edit methods/ (and other declared artifacts). "
            "Public support code and Docker build inputs are in harbor/environment/. "
            "Read instruction.md for the upstream contract.\n\n"
            "Start with `uv run rsi_runtime.py bootstrap` to extract any missing "
            "image-generated baseline artifacts. Run visible experiments with "
            "`uv run rsi_runtime.py run 'python /app/selfcheck.py'` (or the command "
            "in instruction.md). This executes inside the original agent container "
            "and copies declared submission artifacts back to the worktree.\n\n"
            "Use visible feedback for iteration, then submit the final method with "
            "`coral eval -m 'final method'`. The default run stops after one real "
            "evaluation. Only declared artifacts cross to the original separate "
            "verifier. Changing rsi_runtime.py, harbor/, or local support files "
            "does not change the trusted grader. --tune does not run hidden tests.\n\n"
            + (source / "instruction.md").read_text()
        )
        task = {
            "task": {
                "name": f"rsi-exam-{source.name}",
                "description": description,
                "tips": "Reward is the upstream reward (maximize), including raw GPU speedups. "
                "Do not inspect hidden tests. Follow the upstream artifact constraints.",
            },
            "grader": {
                "entrypoint": "rsi_exam_grader.grader:Grader",
                "setup": ["uv pip install -e ./grader"],
                "private": ["rsi_task"],
                "timeout": timeout,
                "direction": "maximize",
                "args": {"reward_key": "reward"},
                "parallel": {"max_workers": 1},
            },
            "agents": {
                "count": 1,
                "runtime": "claude_code",
                "model": "sonnet",
                "timeout": int(config.get("agent", {}).get("timeout_sec", 43200)),
                "research": False,
            },
            "workspace": {"repo_path": str(output.resolve() / "seed"), "results_dir": "./results"},
            "run": {"session": "local", "stop": {"max_real_attempts": 1}},
        }
        (staging / "task.yaml").write_text(
            yaml.dump(task, Dumper=TaskDumper, sort_keys=False, width=100)
        )
        shutil.move(staging, output)


def hydrate_task(task_dir: Path, *, source: Path | None = None) -> None:
    """Fetch omitted data assets, preserving checked-in code and edited baselines."""
    manifest = json.loads((task_dir / "UPSTREAM.json").read_text())
    records = manifest.get("downloads", [])
    for record in records:
        path = Path(record["path"])
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.parts[0] not in {"environment", "tests"}
        ):
            raise ValueError(f"Invalid asset path: {path}")
    pending = []
    for record in records:
        targets = [task_dir / "rsi_task" / record["path"]]
        if record["path"].startswith("environment/"):
            targets.append(task_dir / "seed/harbor" / record["path"])
        if any(not p.exists() for p in targets):
            pending.append((record, targets))
    if not pending:
        return

    with tempfile.TemporaryDirectory(prefix="rsi-assets-") as temp:

        def fetch(item):
            record, targets = item
            remote = f"{manifest['task']}/{record['path']}"
            if source is not None:
                downloaded = source / remote
            else:
                from huggingface_hub import hf_hub_download

                downloaded = Path(
                    hf_hub_download(
                        repo_id=manifest["dataset"],
                        repo_type="dataset",
                        revision=manifest["revision"],
                        filename=remote,
                        local_dir=temp,
                    )
                )
            data = downloaded.read_bytes()
            if "sha256" in record:
                digest = hashlib.sha256(data).hexdigest()
                expected = record["sha256"]
            else:
                digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                expected = record["git_blob_sha1"]
            if len(data) != record["size"] or digest != expected:
                raise ValueError(f"Asset checksum mismatch: {remote}")
            for target in targets:
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Copy atomically so an interrupted download cannot leave a
                # partial file that a future invocation mistakes for complete.
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(data)
                temporary.chmod(0o644)
                temporary.replace(target)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, pending))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", choices=public_tasks())
    parser.add_argument("--list", action="store_true", help="List the 35 pinned public tasks")
    parser.add_argument(
        "--all", action="store_true", help="Prepare all public tasks (downloads can be large)"
    )
    parser.add_argument(
        "--source", type=Path, help="Offline dataset root containing task directories"
    )
    parser.add_argument("--output", type=Path, default=HERE, help="Parent of task directories")
    args = parser.parse_args()
    if args.list:
        print("\n".join(public_tasks()))
        return
    if bool(args.task) == args.all:
        parser.error("choose one task or --all")
    names = public_tasks() if args.all else [args.task]
    existing = [name for name in names if (args.output / name).exists()]
    for name in existing:
        hydrate_task(args.output / name, source=args.source)
        print(f"Prepared {name}: coral validate {args.output / name}")
    names = [name for name in names if name not in existing]
    if not names:
        return
    if args.source:
        root = args.source.resolve()
        revision = "local-source (see sha256 manifest)"
        for name in names:
            build_task(root / name, args.output / name, revision=revision)
    else:
        from huggingface_hub import snapshot_download

        # Download directly to temporary storage, not HF's persistent cache of
        # hidden data. The imported grader.private directory is the durable copy.
        with tempfile.TemporaryDirectory(prefix="rsi-download-") as temp:
            snapshot_download(
                repo_id=DATASET,
                repo_type="dataset",
                revision=REVISION,
                local_dir=temp,
                allow_patterns=["LICENSE", *[f"{name}/*" for name in names]],
            )
            for name in names:
                build_task(Path(temp) / name, args.output / name)
    for name in names:
        print(f"Imported {name}: coral validate {args.output / name}")


if __name__ == "__main__":
    main()

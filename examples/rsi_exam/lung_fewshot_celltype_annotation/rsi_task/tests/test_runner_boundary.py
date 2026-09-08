from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fail_closed  # noqa: E402
import secure_runtime as runtime  # noqa: E402


class RunnerBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def valid_tree(self) -> Path:
        root = self.tmp / "methods" / "main"
        root.mkdir(parents=True)
        (root / "solver.py").write_text("print('ok')\n", encoding="utf-8")
        (root / "config.json").write_text("{}\n", encoding="utf-8")
        return root

    def test_fd_bundle_accepts_bounded_regular_source(self) -> None:
        bundle = runtime.read_source_bundle(self.valid_tree())
        files = {entry.relative: entry.payload for entry in bundle.entries if not entry.is_directory}
        self.assertEqual(set(files), {Path("solver.py"), Path("config.json")})
        self.assertEqual(files[Path("solver.py")], b"print('ok')\n")
        self.assertTrue(all(entry.identity.ctime_ns > 0 for entry in bundle.entries))

    def test_fd_bundle_rejects_symlink(self) -> None:
        root = self.valid_tree()
        (root / "alias.py").symlink_to(root / "solver.py")
        with self.assertRaisesRegex(runtime.SubmissionError, "symlinks"):
            runtime.read_source_bundle(root)

    def test_fd_bundle_rejects_hardlink(self) -> None:
        root = self.valid_tree()
        os.link(root / "solver.py", root / "second.py")
        with self.assertRaisesRegex(runtime.SubmissionError, "hardlinks"):
            runtime.read_source_bundle(root)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO test requires POSIX")
    def test_fd_bundle_rejects_fifo_without_blocking(self) -> None:
        root = self.valid_tree()
        os.mkfifo(root / "pipe.py")
        with self.assertRaisesRegex(runtime.SubmissionError, "special"):
            runtime.read_source_bundle(root)

    def test_fd_bundle_rejects_generated_cache(self) -> None:
        root = self.valid_tree()
        (root / "__pycache__").mkdir()
        with self.assertRaisesRegex(runtime.SubmissionError, "cache"):
            runtime.read_source_bundle(root)

    def test_fd_bundle_enforces_total_bytes(self) -> None:
        root = self.valid_tree()
        with mock.patch.object(runtime, "MAX_SUBMISSION_BYTES", 4):
            with self.assertRaisesRegex(runtime.SubmissionError, "one MiB"):
                runtime.read_source_bundle(root)

    def test_identity_detects_same_size_rewrite_with_restored_mtime(self) -> None:
        path = self.tmp / "solver.py"
        path.write_bytes(b"aaaa")
        before_stat = path.stat()
        before = runtime.identity(before_stat)
        time.sleep(0.02)
        path.write_bytes(b"bbbb")
        os.utime(path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
        after = runtime.identity(path.stat())
        self.assertEqual(before.size, after.size)
        self.assertEqual(before.mtime_ns, after.mtime_ns)
        self.assertNotEqual(before.ctime_ns, after.ctime_ns)
        self.assertNotEqual(before, after)

    def test_scratch_audit_accepts_bounded_regular_tree(self) -> None:
        root = self.tmp / "scratch"
        root.mkdir()
        output = root / "predictions.csv"
        output.write_text("cell_id,pred_label\na,b\n", encoding="utf-8")
        entries, total = runtime.audit_scratch(root, os.getuid(), os.getgid())
        self.assertEqual(entries, 1)
        self.assertEqual(total, output.stat().st_size)

    def test_scratch_audit_rejects_symlink(self) -> None:
        root = self.tmp / "scratch"
        root.mkdir()
        target = self.tmp / "outside"
        target.write_text("sentinel", encoding="utf-8")
        target.chmod(0o640)
        before_mode = target.stat().st_mode & 0o777
        (root / "predictions.csv").symlink_to(target)
        with self.assertRaisesRegex(runtime.SubmissionError, "symlinks"):
            runtime.audit_scratch(root, os.getuid(), os.getgid())
        runtime._remove_tree(root)
        self.assertFalse(root.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "sentinel")
        self.assertEqual(target.stat().st_mode & 0o777, before_mode)

    def test_stable_prediction_reader_rejects_hardlink(self) -> None:
        path = self.tmp / "predictions.csv"
        path.write_text("cell_id,pred_label\na,b\n", encoding="utf-8")
        os.link(path, self.tmp / "alias.csv")
        with self.assertRaisesRegex(runtime.SubmissionError, "metadata"):
            runtime._stable_child_file(path, os.getuid(), os.getgid(), 4096)

    def test_active_diagnostic_limit_rejects_next_byte(self) -> None:
        buffer = bytearray(b"1234")
        with mock.patch.object(runtime, "MAX_DIAGNOSTIC_BYTES", 4):
            with self.assertRaisesRegex(runtime.SubmissionError, "active byte limit"):
                runtime._extend_diagnostic(buffer, b"5", "stdout")
        self.assertEqual(buffer, b"1234")

    def test_second_diagnostic_pipe_failure_closes_first_pipe(self) -> None:
        original_pipe2 = os.pipe2
        opened: list[int] = []

        def fail_second_pipe(flags: int) -> tuple[int, int]:
            if opened:
                raise OSError("injected second pipe failure")
            pair = original_pipe2(flags)
            opened.extend(pair)
            return pair

        with (
            mock.patch.object(runtime, "_uid_pids", return_value=[]),
            mock.patch.object(runtime, "_uid_ipc_objects", return_value=[]),
            mock.patch.object(runtime.os, "pipe2", side_effect=fail_second_pipe),
        ):
            with self.assertRaisesRegex(runtime.GraderError, "initialize submission diagnostics"):
                runtime.run_submission(self.tmp / "solver.py", self.tmp, os.getuid(), os.getgid())

        self.assertEqual(len(opened), 2)
        for descriptor in opened:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_diagnostic_selector_failure_closes_both_pipes(self) -> None:
        original_pipe2 = os.pipe2
        opened: list[int] = []

        def capture_pipe(flags: int) -> tuple[int, int]:
            pair = original_pipe2(flags)
            opened.extend(pair)
            return pair

        with (
            mock.patch.object(runtime, "_uid_pids", return_value=[]),
            mock.patch.object(runtime, "_uid_ipc_objects", return_value=[]),
            mock.patch.object(runtime.os, "pipe2", side_effect=capture_pipe),
            mock.patch.object(
                runtime.selectors,
                "DefaultSelector",
                side_effect=OSError("injected selector construction failure"),
            ),
        ):
            with self.assertRaisesRegex(runtime.GraderError, "initialize submission diagnostics"):
                runtime.run_submission(self.tmp / "solver.py", self.tmp, os.getuid(), os.getgid())

        self.assertEqual(len(opened), 4)
        for descriptor in opened:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_visible_asset_validator_rejects_lfs_pointer_before_schema_import(self) -> None:
        validator_path = HERE.parent / "environment" / "build_validate_assets.py"
        specification = importlib.util.spec_from_file_location("lung_visible_build_validator", validator_path)
        assert specification is not None and specification.loader is not None
        validator = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(validator)
        pointer = self.tmp / "asset.h5ad"
        pointer.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
            "size 1234\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "unsmudged Git-LFS pointer"):
            validator.validate_materialized({pointer: (1234, "0" * 64)})

    def test_visible_dockerfiles_run_the_asset_gate(self) -> None:
        environment = HERE.parent / "environment"
        agent = (environment / "Dockerfile").read_text(encoding="utf-8")
        evaluator = (environment / "Dockerfile.evaluator").read_text(encoding="utf-8")
        self.assertIn("COPY build_validate_assets.py /tmp/build_validate_assets.py", agent)
        self.assertIn("python /tmp/build_validate_assets.py agent", agent)
        self.assertIn("COPY build_validate_assets.py /tmp/build_validate_assets.py", evaluator)
        self.assertIn("python /tmp/build_validate_assets.py evaluator", evaluator)

    def test_fail_closed_removes_symlink_without_touching_target(self) -> None:
        sentinel = self.tmp / "sentinel"
        sentinel.write_text("unchanged", encoding="utf-8")
        sentinel.chmod(0o640)
        before_mode = sentinel.stat().st_mode & 0o777
        link = self.tmp / "reward.json"
        link.symlink_to(sentinel)
        fail_closed._remove_path(link)
        self.assertFalse(link.exists())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")
        self.assertEqual(sentinel.stat().st_mode & 0o777, before_mode)

    def test_dockerfile_uses_exact_runtime_copies(self) -> None:
        dockerfile = (HERE / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("COPY public/ /tests/public/", dockerfile)
        self.assertNotIn("COPY heldout/ /tests/heldout/", dockerfile)
        self.assertIn("COPY public/visible_labeled.h5ad /tests/inputs/labeled.h5ad", dockerfile)
        self.assertIn("COPY heldout/labels.csv /tests/sealed/truth.csv", dockerfile)
        self.assertIn("python /tmp/build_validate_assets.py", dockerfile)

    def test_runtime_readmes_are_author_side_only(self) -> None:
        self.assertFalse((HERE / "public" / "README.md").exists())
        self.assertFalse((HERE / "heldout" / "README.md").exists())

    def test_entrypoint_initializes_zero_before_grader(self) -> None:
        script = (HERE / "test.sh").read_text(encoding="utf-8")
        self.assertLess(script.index("fail_closed.py --initialize"), script.index("python /tests/grade.py"))
        self.assertIn("validate_outputs.py", script)
        self.assertIn('rm -rf -- "${LOG_DIR}"', script)

    def test_critical_boundary_values_are_not_host_overridable(self) -> None:
        grade = (HERE / "grade.py").read_text(encoding="utf-8")
        secure = (HERE / "secure_runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("os.environ.get(", grade)
        self.assertNotIn("os.environ.get(", secure)
        for token in (
            "st_ctime_ns", "O_NOFOLLOW", "O_NONBLOCK", "PR_SET_DUMPABLE",
            "PR_SET_CHILD_SUBREAPER", "RLIMIT_AS", "RLIMIT_NPROC", "sysvipc",
        ):
            self.assertIn(token, secure)
        for name in ("OMP_THREAD_LIMIT", "NUMBA_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            self.assertIn(f'"{name}": "1"', secure)



if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Synthetic adversarial/unit tests for the Lean workflow verifier.

These tests never open the held-out fixture and never call a hosted model or a real Lean binary.
They exercise the checked-in verifier with one synthetic theorem and parent-owned fakes.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
TASK = HERE.parent
ENVIRONMENT = TASK / "environment"
sys.path.insert(0, str(HERE))

import evaluator  # noqa: E402
import lean_compiler  # noqa: E402
import model_proxy  # noqa: E402


GOOD_PROOF = "by\n  trivial"
OTHER_VALID_PROOF = "by\n  exact True.intro"
FORMAL_STATEMENT = "theorem synthetic_target : True :="
EXAMPLE = {
    "id": "sealed-stable-id",
    "problem": "Prove True.",
    "formal_statement": FORMAL_STATEMENT,
    "difficulty": "Easy",
    "subtopic": "synthetic",
    "stratum": "library_abstract",
}


class FakeModelProxy:
    """Drop-in parent model with the production budget-accounting attributes."""

    def __init__(self, max_calls, max_tokens):
        self.max_calls = int(max_calls)
        self.max_tokens = int(max_tokens)
        self.num_calls = 0
        self.total_tokens = 0
        self.calls_exceeded = False
        self.tokens_exceeded = False

    def __call__(self, messages, max_tokens=1024, stop=None):
        self.num_calls += 1
        if self.num_calls > self.max_calls:
            self.calls_exceeded = True
            raise evaluator.BudgetExceeded("synthetic model call budget exceeded")
        self.total_tokens += int(max_tokens)
        if self.total_tokens > self.max_tokens:
            self.tokens_exceeded = True
            raise evaluator.BudgetExceeded("synthetic model token budget exceeded")
        return "Return a small Lean proof."

    def healthcheck(self):
        return {"model": model_proxy.PINNED_MODEL,
                "revision": model_proxy.PINNED_REVISION,
                "endpoint_verified": True, "completion_verified": True}


class CompileScript:
    """Parent-owned fake compiler that still applies the real proof text policy."""

    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes) if outcomes is not None else None
        self.calls = []

    def __call__(self, formal_statement, proof, timeout_sec=75):
        self.calls.append((formal_statement, proof, timeout_sec))
        invalid = lean_compiler.validate_proof(proof)
        if invalid:
            return {"ok": False, "diagnostics": invalid}
        if self.outcomes is None:
            ok = proof == GOOD_PROOF
        else:
            ok = bool(self.outcomes.pop(0))
        return {"ok": ok, "diagnostics": "synthetic compile result"}


class VerifierTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lean-verifier-unit-")
        self.root = Path(self.temporary.name)
        self.root.chmod(0o755)
        self.submission = self.root / "submission"
        self.submission.mkdir(mode=0o755)
        self.original_budget = dict(evaluator.BUDGET)

    def tearDown(self):
        evaluator.BUDGET.clear()
        evaluator.BUDGET.update(self.original_budget)
        self.temporary.cleanup()

    def write_solver(self, source):
        path = self.submission / "solver.py"
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
        path.chmod(0o644)
        return path

    def run_question(self, compiler=None):
        compiler = compiler or CompileScript()
        with mock.patch.object(evaluator, "ModelProxy", FakeModelProxy), \
                mock.patch.object(evaluator, "compile_proof", compiler):
            result = evaluator.run_question(dict(EXAMPLE), self.submission)
        return result, compiler


class StaticGuardTests(VerifierTestCase):
    def test_invalid_utf8_and_special_files_are_invalid_not_infrastructure_failures(self):
        solver = self.submission / "solver.py"
        solver.write_bytes(b"\xff\xfe\x00")
        result = evaluator.evaluate([dict(EXAMPLE)], self.submission, model_identity={})
        self.assertTrue(result["invalid"])
        self.assertIn("unreadable", result["error"])

        solver.write_text(
            "def answer_batch(examples, llm, lean, budget):\n    return ['by trivial']\n",
            encoding="utf-8",
        )
        fifo = self.submission / "payload.fifo"
        os.mkfifo(fifo)
        result = evaluator.evaluate([dict(EXAMPLE)], self.submission, model_identity={})
        self.assertTrue(result["invalid"])
        self.assertIn("only regular files", result["error"])

    def test_rejects_explicit_dangerous_imports_and_import_builtin(self):
        sources = {
            "socket": "import socket",
            "subprocess-from": "from subprocess import Popen",
            "os": "import os",
            "ctypes": "from ctypes import CDLL",
            "_ctypes": "import _ctypes",
            "pyseccomp": "import pyseccomp",
            "dynamic-import-builtin": "module = __import__('o' + 's')",
        }
        for label, statement in sources.items():
            with self.subTest(label=label):
                self.write_solver(f"""
                {statement}
                def answer_batch(examples, llm, lean, budget):
                    return [{GOOD_PROOF!r}]
                """)
                self.assertTrue(evaluator.static_guard(self.submission))

    def test_rejects_direct_builtin_file_and_code_execution_calls(self):
        # This assertion intentionally catches a name-normalization error where the guard lists
        # "builtins.open"/"builtins.eval" but ast.Name nodes resolve to "open"/"eval".
        calls = (
            "open('/etc/passwd')",
            "eval('1 + 1')",
            "exec('value = 1')",
            "compile('1 + 1', '<x>', 'eval')",
            "breakpoint()",
        )
        for call in calls:
            with self.subTest(call=call):
                self.write_solver(f"""
                def answer_batch(examples, llm, lean, budget):
                    {call}
                    return [{GOOD_PROOF!r}]
                """)
                error = evaluator.static_guard(self.submission)
                self.assertIn("forbidden call", error)

    def test_rejects_symlink_non_python_and_sealed_literals(self):
        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            return [{GOOD_PROOF!r}]
        """)
        link = self.submission / "helper.py"
        link.symlink_to(self.submission / "solver.py")
        self.assertIn("symlink is not allowed", evaluator.static_guard(self.submission))
        link.unlink()
        extra = self.submission / "payload.txt"
        extra.write_text("x", encoding="utf-8")
        self.assertIn("only Python source", evaluator.static_guard(self.submission))
        extra.unlink()
        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            marker = '/te' + 'sts/' + 'held' + 'out'
            return [{GOOD_PROOF!r}]
        """)
        # Fragmented strings are intentionally left to the runtime boundary.
        self.assertEqual(evaluator.static_guard(self.submission), "")
        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            marker = '/tests/heldout'
            return [{GOOD_PROOF!r}]
        """)
        self.assertIn("sealed-data reference", evaluator.static_guard(self.submission))

    def test_rejects_direct_problem_id_lookup_but_hidden_id_is_not_transmitted(self):
        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            return [examples[0]['id']]
        """)
        self.assertIn("problem-ID lookup", evaluator.static_guard(self.submission))
        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            values = list(examples[0].values())
            return [{OTHER_VALID_PROOF!r} if 'sealed-stable-id' in values else {GOOD_PROOF!r}]
        """)
        result, _ = self.run_question()
        self.assertTrue(result["correct"], result)


class RuntimeIsolationTests(VerifierTestCase):
    def test_privilege_drop_fails_closed_if_identity_remains_root(self):
        for module in (evaluator, lean_compiler):
            with self.subTest(module=module.__name__), \
                    mock.patch.object(module.os, "geteuid", return_value=0), \
                    mock.patch.object(module.os, "getegid", return_value=0), \
                    mock.patch.object(module.os, "setgroups"), \
                    mock.patch.object(module.os, "setgid"), \
                    mock.patch.object(module.os, "setuid"):
                with self.assertRaises(RuntimeError):
                    module._drop_privileges()

    def test_documented_hashlib_import_is_available(self):
        self.write_solver(f"""
        import hashlib
        def answer_batch(examples, llm, lean, budget):
            digest = hashlib.sha256(b'allowed').hexdigest()
            return [{GOOD_PROOF!r} if len(digest) == 64 else {OTHER_VALID_PROOF!r}]
        """)
        self.assertEqual(evaluator.static_guard(self.submission), "")
        result, _ = self.run_question()
        self.assertTrue(result["correct"], result)

    def test_constructed_sealed_read_is_blocked(self):
        self.write_solver(f"""
        import builtins
        def answer_batch(examples, llm, lean, budget):
            try:
                reader = getattr(builtins, 'op' + 'en')
                reader('/pr' + 'oc/self/status').read()
                blocked = False
            except Exception:
                blocked = True
            return [{GOOD_PROOF!r} if blocked else {OTHER_VALID_PROOF!r}]
        """)
        self.assertEqual(evaluator.static_guard(self.submission), "")
        result, _ = self.run_question()
        self.assertTrue(result["correct"], result)

    def test_constructed_network_access_is_blocked(self):
        self.write_solver(f"""
        import builtins
        def answer_batch(examples, llm, lean, budget):
            try:
                loader = getattr(builtins, '__im' + 'port__')
                loader('sock' + 'et').socket()
                blocked = False
            except Exception:
                blocked = True
            return [{GOOD_PROOF!r} if blocked else {OTHER_VALID_PROOF!r}]
        """)
        self.assertEqual(evaluator.static_guard(self.submission), "")
        result, _ = self.run_question()
        self.assertTrue(result["correct"], result)

    def test_constructed_process_creation_is_blocked(self):
        self.write_solver(f"""
        import sys
        def answer_batch(examples, llm, lean, budget):
            try:
                module = sys.modules['o' + 's']
                getattr(module, 'fo' + 'rk')()
                blocked = False
            except Exception:
                blocked = True
            return [{GOOD_PROOF!r} if blocked else {OTHER_VALID_PROOF!r}]
        """)
        self.assertEqual(evaluator.static_guard(self.submission), "")
        result, _ = self.run_question()
        self.assertTrue(result["correct"], result)

    def test_isolation_modules_are_not_recoverable_from_sys_modules(self):
        self.write_solver(f"""
        import sys
        def answer_batch(examples, llm, lean, budget):
            leaked = any(name in sys.modules for name in (
                'ct' + 'ypes', '_ct' + 'ypes', 'py' + 'seccomp'
            ))
            return [{OTHER_VALID_PROOF!r} if leaked else {GOOD_PROOF!r}]
        """)
        self.assertEqual(evaluator.static_guard(self.submission), "")
        result, _ = self.run_question()
        self.assertTrue(result["correct"], result)

    @unittest.skipUnless(os.environ.get("LEAN_REQUIRE_SECCOMP") == "1",
                         "requires the production kernel-isolation boundary")
    def test_production_runtime_preflights_compiler_and_kernel_isolation(self):
        compiler = lean_compiler.compiler_preflight()
        isolation = evaluator.isolation_preflight()
        self.assertTrue(compiler["compiler_verified"])
        self.assertTrue(compiler["mathlib_import_verified"])
        self.assertIn("version 4.9.0", compiler["lean_version"])
        self.assertEqual(compiler["mathlib_commit"], lean_compiler.PINNED_MATHLIB_COMMIT)
        self.assertEqual(isolation["isolation_verified"], True)
        self.assertEqual(isolation["isolation_backend"], "seccomp+landlock")
        self.assertNotEqual(isolation["isolated_uid"], 0)
        self.assertNotEqual(isolation["isolated_gid"], 0)

    @unittest.skipUnless(os.environ.get("LEAN_REQUIRE_SECCOMP") == "1",
                         "requires the production kernel-isolation boundary")
    def test_recovered_ffi_is_still_confined_by_seccomp_and_landlock(self):
        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            try:
                cdll = next(cls for cls in object.__subclasses__()
                            if getattr(cls, '__module__', '') == 'ct' + 'ypes'
                            and getattr(cls, '__name__', '') == 'CD' + 'LL')
                globals_ = cdll.__init__.__globals__
                libc = cdll(None, use_errno=True)
                libc.syscall.restype = globals_['c_' + 'long']
                def syscall(number, *args):
                    globals_['set_' + 'errno'](0)
                    value = int(libc.syscall(number, *args))
                    return value, globals_['get_' + 'errno']()

                # Prove the recovered FFI path works on an explicitly readable file.
                readable = globals_['create_' + 'string_buffer'](b'/etc/hostname')
                globals_['set_' + 'errno'](0)
                readable_fd = int(libc.open(readable, 0))
                if readable_fd < 0:
                    return [{OTHER_VALID_PROOF!r}]
                libc.close(readable_fd)

                socket_result = syscall(41, 2, 1, 0)       # socket(AF_INET, SOCK_STREAM, 0)
                shm_result = syscall(29, 0x5A17BEEF, 4096, 0)  # shmget without IPC_CREAT
                uring_result = syscall(425, 1, 0)          # io_uring_setup with NULL params

                sealed = globals_['create_' + 'string_buffer'](b'/te' + b'sts/grade.py')
                globals_['set_' + 'errno'](0)
                sealed_fd = int(libc.open(sealed, 0))
                sealed_result = (sealed_fd, globals_['get_' + 'errno']())
                if socket_result[0] >= 0:
                    libc.close(socket_result[0])
                if uring_result[0] >= 0:
                    libc.close(uring_result[0])
                if sealed_fd >= 0:
                    libc.close(sealed_fd)
                confined = (socket_result == (-1, 1) and shm_result == (-1, 1)
                             and uring_result == (-1, 1) and sealed_result == (-1, 13))
                return [{GOOD_PROOF!r} if confined else {OTHER_VALID_PROOF!r}]
            except Exception:
                return [{OTHER_VALID_PROOF!r}]
        """)
        self.assertEqual(evaluator.static_guard(self.submission), "")
        result, _ = self.run_question()
        self.assertTrue(result["correct"], result)

    def test_child_environment_has_no_parent_model_credentials(self):
        self.write_solver(f"""
        import sys
        def answer_batch(examples, llm, lean, budget):
            module = sys.modules['o' + 's']
            key = '_'.join(['LEAN', 'MODEL', 'API', 'KEY'])
            return [{OTHER_VALID_PROOF!r} if module.environ.get(key) else {GOOD_PROOF!r}]
        """)
        with mock.patch.dict(os.environ, {"LEAN_MODEL_API_KEY": "parent-secret"}):
            result, _ = self.run_question()
        self.assertTrue(result["correct"], result)

    def test_question_sandboxes_do_not_share_tmp_home_or_cwd_state(self):
        marker = self.root / "absolute-cross-question-marker"
        self.write_solver(f"""
        import io
        import sys
        def answer_batch(examples, llm, lean, budget):
            os_module = sys.modules['o' + 's']
            paths = [
                {str(marker)!r},
                'relative-cross-question-marker',
                os_module.path.join(os_module.environ['HOME'], 'home-cross-question-marker'),
            ]
            seen = False
            for path in paths:
                try:
                    handle = getattr(io, 'op' + 'en')(path, 'a+', encoding='utf-8')
                    handle.seek(0)
                    seen = seen or bool(handle.read())
                    handle.write('state')
                    handle.close()
                except Exception:
                    pass
            try:
                seen = seen or {marker.name!r} in os_module.listdir('/tmp')
            except Exception:
                pass
            return [{OTHER_VALID_PROOF!r} if seen else {GOOD_PROOF!r}]
        """)
        first, _ = self.run_question()
        second, _ = self.run_question()
        self.assertTrue(first["correct"], first)
        self.assertTrue(second["correct"], second)
        self.assertFalse(marker.exists())

    def test_model_infrastructure_failure_cannot_be_swallowed_by_submission(self):
        class BrokenModelProxy(FakeModelProxy):
            def __call__(self, messages, max_tokens=1024, stop=None):
                raise evaluator.ModelInfrastructureError("synthetic endpoint outage")

        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            try:
                llm([{{'role': 'user', 'content': 'prove it'}}], max_tokens=8)
            except Exception:
                pass
            return [{GOOD_PROOF!r}]
        """)
        with mock.patch.object(evaluator, "ModelProxy", BrokenModelProxy), \
                mock.patch.object(evaluator, "compile_proof", CompileScript()):
            with self.assertRaises(evaluator.ModelInfrastructureError):
                evaluator.run_question(dict(EXAMPLE), self.submission)


class ProtocolIntegrityTests(VerifierTestCase):
    @staticmethod
    def raw_solver(frame):
        return f"""
        import sys
        def answer_batch(examples, llm, lean, budget):
            module = sys.modules['o' + 's']
            module.write(1, {frame!r})
            module._exit(0)
        """

    def test_submission_authored_validity_and_nan_extra_field_are_rejected(self):
        frames = (
            (json.dumps({"type": "result", "proof": GOOD_PROOF, "correct": True}) + "\n").encode(),
            ("{\"type\":\"result\",\"proof\":" + json.dumps(GOOD_PROOF)
             + ",\"score\":NaN}\n").encode(),
        )
        for frame in frames:
            with self.subTest(frame=frame):
                self.write_solver(self.raw_solver(frame))
                result, _ = self.run_question()
                self.assertFalse(result["correct"], result)
                self.assertEqual(result["status"], "error")

    def test_duplicate_keys_are_rejected_instead_of_last_value_winning(self):
        frame = (
            "{\"type\":\"result\",\"proof\":" + json.dumps(OTHER_VALID_PROOF)
            + ",\"proof\":" + json.dumps(GOOD_PROOF) + "}\n"
        ).encode()
        self.write_solver(self.raw_solver(frame))
        result, _ = self.run_question()
        self.assertFalse(result["correct"], result)
        self.assertEqual(result["status"], "error")

    def test_lone_unicode_surrogate_is_rejected_as_a_submission_error(self):
        frame = b'{"type":"result","proof":"\\ud800"}\n'
        self.write_solver(self.raw_solver(frame))
        result, _ = self.run_question()
        self.assertFalse(result["correct"], result)
        self.assertEqual(result["status"], "error")
        self.assertIn("Unicode", result["error"])

    def test_second_result_or_trailing_stdout_is_rejected(self):
        one = json.dumps({"type": "result", "proof": GOOD_PROOF}, separators=(",", ":"))
        frame = (one + "\n" + one + "\n").encode()
        self.write_solver(self.raw_solver(frame))
        result, _ = self.run_question()
        self.assertFalse(result["correct"], result)
        self.assertEqual(result["status"], "error")

    def test_python_print_is_diagnostic_not_protocol(self):
        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            print('diagnostic only')
            return [{GOOD_PROOF!r}]
        """)
        result, _ = self.run_question()
        self.assertTrue(result["correct"], result)


class ProofAndBudgetTests(VerifierTestCase):
    def test_malformed_return_and_malformed_proof_are_wrong(self):
        self.write_solver("""
        def answer_batch(examples, llm, lean, budget):
            return ('not', 'a', 'one-element-list')
        """)
        result, _ = self.run_question()
        self.assertFalse(result["correct"])
        self.assertEqual(result["status"], "error")

        for proof in ("not a Lean proof", "by\n  sorry", "by\n  admit",
                      "by\n  run_tac pure ()", "by\n  unsafe exact True.intro",
                      "by\n  trivial\naxiom forged : False"):
            with self.subTest(proof=proof):
                self.write_solver(f"""
                def answer_batch(examples, llm, lean, budget):
                    return [{proof!r}]
                """)
                result, _ = self.run_question()
                self.assertFalse(result["correct"], result)

    def test_underlying_sorry_axiom_is_rejected(self):
        # Surface `sorry` is not the only escape hatch: Lean exposes the underlying sorryAx.
        self.assertTrue(
            lean_compiler.validate_proof("by\n  exact sorryAx True true"),
            "sorryAx must be rejected before invoking Lean",
        )

    def test_child_cannot_raise_parent_llm_budget(self):
        evaluator.BUDGET["max_llm_calls_per_question"] = 1
        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            budget['max_llm_calls_per_question'] = 999
            for _ in range(2):
                try:
                    llm([{{'role': 'user', 'content': 'x'}}], max_tokens=1)
                except Exception:
                    pass
            return [{GOOD_PROOF!r}]
        """)
        result, compiler = self.run_question()
        self.assertTrue(result["budget_exceeded"])
        self.assertFalse(result["correct"])
        self.assertEqual(compiler.calls, [])

    def test_child_cannot_raise_parent_lean_budget(self):
        evaluator.BUDGET["max_lean_checks_per_question"] = 1
        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            budget['max_lean_checks_per_question'] = 999
            for _ in range(2):
                try:
                    lean({GOOD_PROOF!r})
                except Exception:
                    pass
            return [{GOOD_PROOF!r}]
        """)
        result, compiler = self.run_question()
        self.assertTrue(result["budget_exceeded"])
        self.assertFalse(result["correct"])
        self.assertEqual(len(compiler.calls), 1)

    def test_python_wall_timeout_kills_question(self):
        evaluator.BUDGET["max_wall_time_per_question_sec"] = 1
        self.write_solver("""
        def answer_batch(examples, llm, lean, budget):
            while True:
                pass
        """)
        started = time.monotonic()
        result, _ = self.run_question()
        self.assertLess(time.monotonic() - started, 4.0)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["budget_exceeded"])

    def test_advisory_compiler_timeout_is_inside_question_wall_budget(self):
        evaluator.BUDGET["max_wall_time_per_question_sec"] = 1

        def slow_compile(formal_statement, proof, timeout_sec=75):
            time.sleep(5)
            return {"ok": True, "diagnostics": "late"}

        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            lean({GOOD_PROOF!r})
            return [{GOOD_PROOF!r}]
        """)
        started = time.monotonic()
        result, _ = self.run_question(slow_compile)
        self.assertLess(time.monotonic() - started, 4.0)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["budget_exceeded"])

    def test_final_compiler_timeout_is_inside_question_wall_budget(self):
        evaluator.BUDGET["max_wall_time_per_question_sec"] = 1

        def slow_compile(formal_statement, proof, timeout_sec=75):
            time.sleep(5)
            return {"ok": True, "diagnostics": "late"}

        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            return [{GOOD_PROOF!r}]
        """)
        started = time.monotonic()
        result, _ = self.run_question(slow_compile)
        self.assertLess(time.monotonic() - started, 4.0)
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["budget_exceeded"])
        self.assertFalse(result["correct"])

    def test_final_compile_is_independent_of_advisory_lean_ok(self):
        compiler = CompileScript([True, False])
        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            advisory = lean({GOOD_PROOF!r})
            return [{GOOD_PROOF!r}] if advisory.get('ok') else [{OTHER_VALID_PROOF!r}]
        """)
        result, compiler = self.run_question(compiler)
        self.assertFalse(result["correct"], result)
        self.assertEqual(len(compiler.calls), 2)
        self.assertEqual([call[:2] for call in compiler.calls],
                         [(FORMAL_STATEMENT, GOOD_PROOF), (FORMAL_STATEMENT, GOOD_PROOF)])

    def test_returned_proof_is_finally_compiled_even_without_lean_rpc(self):
        compiler = CompileScript([True])
        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            return [{GOOD_PROOF!r}]
        """)
        result, compiler = self.run_question(compiler)
        self.assertTrue(result["correct"], result)
        self.assertEqual(len(compiler.calls), 1)

    def test_lean_infrastructure_failure_propagates(self):
        self.write_solver(f"""
        def answer_batch(examples, llm, lean, budget):
            return [{GOOD_PROOF!r}]
        """)
        with mock.patch.object(evaluator, "ModelProxy", FakeModelProxy), \
                mock.patch.object(
                    evaluator, "compile_proof",
                    side_effect=lean_compiler.LeanInfrastructureError("runtime missing"),
                ):
            with self.assertRaises(lean_compiler.LeanInfrastructureError):
                evaluator.run_question(dict(EXAMPLE), self.submission)


class VisibleHiddenParityTests(unittest.TestCase):
    def test_environment_and_tests_share_same_implementation(self):
        for name in ("child_runner.py", "lean_compiler.py", "model_proxy.py"):
            with self.subTest(name=name):
                self.assertEqual((ENVIRONMENT / name).read_bytes(), (HERE / name).read_bytes())

        # Ignore the harmless final newline while requiring identical executable syntax.
        environment_tree = ast.dump(
            ast.parse((ENVIRONMENT / "evaluator.py").read_text(encoding="utf-8")),
            include_attributes=False,
        )
        verifier_tree = ast.dump(
            ast.parse((HERE / "evaluator.py").read_text(encoding="utf-8")),
            include_attributes=False,
        )
        self.assertEqual(environment_tree, verifier_tree)

    def test_selfcheck_and_grade_both_call_shared_evaluator(self):
        selfcheck = (ENVIRONMENT / "selfcheck.py").read_text(encoding="utf-8")
        grade = (HERE / "grade.py").read_text(encoding="utf-8")
        import_line = "from evaluator import SUBMISSION_DIR, evaluate, load_jsonl"
        self.assertIn(import_line, selfcheck)
        self.assertIn(
            "from evaluator import SUBMISSION_DIR, evaluate, load_jsonl, runtime_preflight",
            grade,
        )
        self.assertNotIn("--n", selfcheck)
        self.assertNotIn("max_examples", selfcheck)


class ModelAndGradeFailClosedTests(unittest.TestCase):
    @staticmethod
    def configured_proxy():
        values = {
            "LEAN_GPT54_MODEL_NAME": model_proxy.PINNED_MODEL,
            "LEAN_GPT54_MODEL_BASE_URL": "http://127.0.0.1:1/v1",
            "LEAN_GPT54_MODEL_API_KEY": "synthetic",
            "LEAN_GPT54_MODEL_REVISION": model_proxy.PINNED_REVISION,
        }
        with mock.patch.dict(os.environ, values, clear=True):
            return model_proxy.ModelProxy(3, 12000)

    def test_fixed_model_credentials_are_required_and_identity_ignores_host_overrides(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(model_proxy.ModelInfrastructureError):
                model_proxy.ModelProxy(3, 12000)
        complete = {
            "LEAN_GPT54_MODEL_NAME": "host-override",
            "LEAN_GPT54_MODEL_BASE_URL": "http://127.0.0.1:1/v1",
            "LEAN_GPT54_MODEL_API_KEY": "synthetic",
            "LEAN_GPT54_MODEL_REVISION": "wrong-revision",
        }
        with mock.patch.dict(os.environ, complete, clear=True):
            proxy = model_proxy.ModelProxy(3, 12000)
            self.assertEqual(proxy.model, model_proxy.PINNED_MODEL)
            self.assertEqual(proxy.revision, model_proxy.PINNED_REVISION)
        for key, value in (("LEAN_MODEL_TEMPERATURE", "0.1"),
                           ("LEAN_MODEL_TEMPERATURE", "nan"),
                           ("LEAN_MODEL_SEED", "1")):
            with self.subTest(key=key, value=value), \
                    mock.patch.dict(os.environ, {**complete, key: value}, clear=True):
                proxy = model_proxy.ModelProxy(3, 12000)
                self.assertEqual((proxy.temperature, proxy.seed), (0.0, 0))

    def test_runtime_budget_configuration_is_pinned(self):
        with mock.patch.dict(
                evaluator.BUDGET, {"max_llm_calls_per_question": 2}, clear=False):
            with self.assertRaisesRegex(
                    model_proxy.ModelInfrastructureError, "runtime budgets"):
                evaluator.model_preflight()

    def test_submission_bad_request_is_per_case_but_service_failure_is_infrastructure(self):
        class SyntheticAPIError(Exception):
            def __init__(self, status_code):
                super().__init__(f"status {status_code}")
                self.status_code = status_code

        class Completions:
            def __init__(self, status_code):
                self.status_code = status_code

            def create(self, **kwargs):
                raise SyntheticAPIError(self.status_code)

        class Client:
            def __init__(self, status_code):
                self.chat = type("Chat", (), {"completions": Completions(status_code)})()

        messages = [{"role": "user", "content": "synthetic"}]
        per_case = self.configured_proxy()
        per_case._client = Client(400)
        with self.assertRaises(model_proxy.ModelRequestError):
            per_case(messages, max_tokens=8)
        infrastructure = self.configured_proxy()
        infrastructure._client = Client(503)
        with self.assertRaises(model_proxy.ModelInfrastructureError):
            infrastructure(messages, max_tokens=8)

    def test_completion_probe_turns_global_bad_request_into_infrastructure_failure(self):
        class SyntheticAPIError(Exception):
            status_code = 400

        class Models:
            @staticmethod
            def list():
                item = type("Model", (), {"id": model_proxy.PINNED_MODEL})()
                return type("Listing", (), {"data": [item]})()

        class Completions:
            @staticmethod
            def create(**kwargs):
                raise SyntheticAPIError("global serving contract is broken")

        proxy = self.configured_proxy()
        proxy._client = type("Client", (), {
            "models": Models(),
            "chat": type("Chat", (), {"completions": Completions()})(),
        })()
        with self.assertRaises(model_proxy.ModelInfrastructureError):
            proxy.healthcheck()

    def test_missing_token_usage_is_an_infrastructure_failure(self):
        class Response:
            model = model_proxy.PINNED_REVISION
            choices = [type("Choice", (), {
                "message": type("Message", (), {"content": "by trivial"})()
            })()]
            usage = None

        class Completions:
            @staticmethod
            def create(**kwargs):
                return Response()

        proxy = self.configured_proxy()
        proxy._client = type("Client", (), {
            "chat": type("Chat", (), {"completions": Completions()})()
        })()
        with self.assertRaisesRegex(
                model_proxy.ModelInfrastructureError, "token usage"):
            proxy([{"role": "user", "content": "synthetic"}], max_tokens=8)

    def test_parent_wall_alarm_is_not_mislabeled_as_model_outage(self):
        class Completions:
            @staticmethod
            def create(**kwargs):
                raise TimeoutError("per-question wall limit")

        proxy = self.configured_proxy()
        proxy._client = type("Client", (), {
            "chat": type("Chat", (), {"completions": Completions()})()
        })()
        with self.assertRaisesRegex(TimeoutError, "wall limit"):
            proxy([{"role": "user", "content": "synthetic"}], max_tokens=8)

    def test_builtin_timeout_kills_and_reaps_lean_process(self):
        class FakeProcess:
            pid = 424242

            def __init__(self):
                self.calls = 0

            def communicate(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("outer wall alarm")
                return b"", None

        process = FakeProcess()
        with mock.patch.object(lean_compiler.Path, "is_file", return_value=True), \
                mock.patch.object(lean_compiler.Path, "is_dir", return_value=True), \
                mock.patch.object(lean_compiler.Path, "read_text", return_value=""), \
                mock.patch.object(lean_compiler.subprocess, "Popen", return_value=process), \
                mock.patch.object(lean_compiler.os, "killpg") as killpg:
            result = lean_compiler.compile_proof(FORMAL_STATEMENT, GOOD_PROOF, timeout_sec=1)
        self.assertFalse(result["ok"])
        self.assertIn("timeout", result["diagnostics"].lower())
        killpg.assert_called_once_with(process.pid, lean_compiler.signal.SIGKILL)
        self.assertEqual(process.calls, 2)

    def test_missing_lean_runtime_is_an_infrastructure_failure(self):
        with mock.patch.object(lean_compiler.Path, "is_dir", return_value=False):
            with self.assertRaises(lean_compiler.LeanInfrastructureError):
                lean_compiler.compile_proof(FORMAL_STATEMENT, GOOD_PROOF)

    def test_host_cannot_override_child_isolation_mode(self):
        with mock.patch.dict(os.environ, {"LEAN_REQUIRE_SECCOMP": "1"}):
            with tempfile.TemporaryDirectory() as tmp:
                env = evaluator._child_env(Path(tmp), Path(tmp))
            expected = "1" if evaluator.REQUIRE_KERNEL_ISOLATION else "0"
            self.assertEqual(env["LEAN_REQUIRE_SECCOMP"], expected)

    def test_reward_json_is_numeric_only_and_debug_is_explicit(self):
        import grade
        with tempfile.TemporaryDirectory(prefix="lean-grade-unit-") as tmp:
            old_reward_dir = grade.REWARD_DIR
            grade.REWARD_DIR = Path(tmp)
            old_umask = os.umask(0o027)
            try:
                detail = {
                    "accuracy_pct": 60.0,
                    "n_correct": 24,
                    "n_total": 40,
                    "verifier_success": True,
                    "protocol_version": "lean-proof-workflow-v2.5-gpt54-fixed-statement",
                    "artifact_sha256": "a" * 64,
                }
                grade.write_result(0.5, detail)
                reward = json.loads((Path(tmp) / "reward.json").read_text())
                debug = json.loads((Path(tmp) / "grade_debug.json").read_text())
                output_modes = {
                    name: (Path(tmp) / name).stat().st_mode & 0o777
                    for name in ("reward.txt", "reward.json", "grade_debug.json",
                                 "score_details.json")
                }
            finally:
                os.umask(old_umask)
                grade.REWARD_DIR = old_reward_dir
        self.assertEqual(reward, {"reward": 0.5})
        self.assertIs(type(reward["reward"]), float)
        self.assertTrue(debug["correctness"])
        self.assertEqual(debug["errors"], [])
        self.assertEqual(debug["artifact_sha256"], "a" * 64)
        self.assertEqual(set(output_modes.values()), {0o644})

    def test_hidden_debug_sanitization_removes_proof_and_goal_text(self):
        import grade
        result = {"per_problem": [{
            "id": "heldout-001", "correct": False, "status": "ok",
            "error": "SECRET_FORMAL",
            "proof_sha256": "a" * 64, "proof_excerpt": "secret proof",
            "diagnostics": "child trace", "final_diagnostics": "hidden Lean goal",
            "calls": 1, "tokens": 10, "lean_checks": 0, "wall_sec": 1.0,
            "budget_exceeded": False, "stratum": "library_abstract",
            "difficulty": "easy",
        }]}
        grade.sanitize_hidden_result(result)
        self.assertEqual(result["per_problem"], [])
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("SECRET_FORMAL", serialized)
        self.assertNotIn("secret proof", serialized)
        self.assertNotIn("hidden Lean goal", serialized)

    def test_raw_reward_mapping(self):
        import grade
        self.assertEqual(grade.to_reward(0.0), 0.0)
        self.assertEqual(grade.to_reward(60.0), 0.6)
        self.assertEqual(grade.to_reward(100.0), 1.0)

    def test_model_preflight_failure_does_not_read_or_seal_hidden(self):
        import grade
        with tempfile.TemporaryDirectory(prefix="lean-grade-preflight-") as tmp:
            root = Path(tmp)
            heldout = root / "heldout"
            logs = root / "logs"
            heldout.mkdir()
            hidden = heldout / "test.jsonl"
            payload = b'{"synthetic":"must remain unread"}\n'
            hidden.write_bytes(payload)
            with mock.patch.multiple(grade, HELDOUT=heldout, REWARD_DIR=logs), \
                    mock.patch.object(
                        grade, "runtime_preflight",
                        side_effect=model_proxy.ModelInfrastructureError("endpoint unavailable"),
                    ):
                with self.assertRaises(model_proxy.ModelInfrastructureError):
                    grade.main()
            self.assertEqual(hidden.read_bytes(), payload)
            debug = json.loads((logs / "grade_debug.json").read_text())
            self.assertFalse(debug["correctness"])
            self.assertTrue(debug["errors"])

    def test_invalid_submission_is_a_successful_zero_score_grade(self):
        import grade
        with tempfile.TemporaryDirectory(prefix="lean-grade-invalid-") as tmp:
            root = Path(tmp)
            heldout = root / "heldout"
            logs = root / "logs"
            heldout.mkdir()
            hidden = heldout / "test.jsonl"
            hidden.write_text("synthetic sealed payload\n", encoding="utf-8")
            strata = ("numeric_algebra", "olympiad_theorem", "library_abstract")
            rows = [
                {**EXAMPLE, "id": f"heldout-{index:03d}",
                 "stratum": strata[(index - 1) % len(strata)]}
                for index in range(1, 41)
            ]
            invalid_result = {
                "accuracy_pct": 0.0,
                "n_correct": 0,
                "n_total": 40,
                "invalid": True,
                "error": "only Python source is allowed: payload.txt",
                "per_problem": [],
            }
            expected_sha = grade.hashlib.sha256(hidden.read_bytes()).hexdigest()
            with mock.patch.multiple(
                    grade, HELDOUT=heldout, REWARD_DIR=logs,
                    EXPECTED_HELDOUT_SHA256=expected_sha), \
                    mock.patch.object(grade, "runtime_preflight", return_value={
                        "model": model_proxy.PINNED_MODEL,
                        "revision": model_proxy.PINNED_REVISION,
                        "endpoint_verified": True,
                    }), \
                    mock.patch.object(grade, "load_jsonl", return_value=rows), \
                    mock.patch.object(grade, "evaluate", return_value=invalid_result):
                grade.main()
            reward = json.loads((logs / "reward.json").read_text())
            debug = json.loads((logs / "grade_debug.json").read_text())
            self.assertEqual(reward, {"reward": 0.0})
            self.assertTrue(debug["correctness"])
            self.assertEqual(debug["errors"], [])
            self.assertFalse(debug["submission_valid"])
            self.assertTrue(debug["verifier_success"])

    def test_hidden_hash_mismatch_fails_before_seal_or_evaluate(self):
        import grade
        with tempfile.TemporaryDirectory(prefix="lean-grade-drift-") as tmp:
            root = Path(tmp)
            heldout = root / "heldout"
            logs = root / "logs"
            heldout.mkdir()
            hidden = heldout / "test.jsonl"
            payload = b'{"drifted":"payload"}\n'
            hidden.write_bytes(payload)
            with mock.patch.multiple(grade, HELDOUT=heldout, REWARD_DIR=logs), \
                    mock.patch.object(grade, "runtime_preflight", return_value={
                        "model": model_proxy.PINNED_MODEL,
                        "revision": model_proxy.PINNED_REVISION,
                        "endpoint_verified": True,
                    }), \
                    mock.patch.object(grade, "load_jsonl") as load_rows, \
                    mock.patch.object(grade, "evaluate") as evaluate_rows, \
                    mock.patch.object(grade, "seal") as seal_hidden:
                with self.assertRaisesRegex(ValueError, "frozen manifest"):
                    grade.main()
            self.assertEqual(hidden.read_bytes(), payload)
            load_rows.assert_not_called()
            evaluate_rows.assert_not_called()
            seal_hidden.assert_not_called()
            debug = json.loads((logs / "grade_debug.json").read_text())
            self.assertFalse(debug["correctness"])
            self.assertTrue(debug["errors"])


if __name__ == "__main__":
    unittest.main()

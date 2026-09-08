from __future__ import annotations

from pathlib import Path
import tempfile

from submission_guard import violations


HERE = Path(__file__).resolve().parent
TASK = HERE.parent


def check_rejected(source: str, expected: str) -> None:
    with tempfile.TemporaryDirectory(prefix="paged-gqa-guard-") as directory:
        root = Path(directory)
        (root / "solver.py").write_text(source)
        hits = violations(root)
        assert hits and any(expected in hit for hit in hits), (source, hits)


def main() -> None:
    starter = TASK / "environment" / "methods" / "main"
    if starter.is_dir():
        assert violations(starter) == [], violations(starter)
    else:
        with tempfile.TemporaryDirectory(prefix="paged-gqa-valid-") as directory:
            valid = Path(directory)
            (valid / "solver.py").write_text("import torch\ndef paged_gqa_decode(*args): return args[0]\n")
            assert violations(valid) == [], violations(valid)
    production_baseline = HERE / "production_baseline"
    assert violations(production_baseline) == [], violations(production_baseline)
    check_rejected("import os\n", "import 'os'")
    check_rejected("from pathlib import Path\n", "import 'pathlib'")
    check_rejected("def f():\n    return open('/tests/x')\n", "forbidden literal")
    check_rejected("import torch\ntorch.cuda.synchronize = lambda: None\n", "attribute mutation")
    check_rejected("import torch\ndef f(): return torch.cuda.Event()\n", "attribute 'Event'")
    check_rejected("imp = __import__\ndef f(): return imp('protocol')\n", "builtin/dunder reference")
    check_rejected("reader = open\ndef f(): return reader('/tmp/x')\n", "builtin/dunder reference")
    check_rejected("g = getattr\ndef f(x): return g(x, 'shape')\n", "builtin/dunder reference")
    check_rejected("from torch.cuda import Stream\ndef f(): return Stream()\n", "accelerator API")
    check_rejected("import protocol\ndef f(): return protocol\n", "import 'protocol'")
    check_rejected("import flashinfer\n", "import 'flashinfer'")
    print("test_security: PASS")


if __name__ == "__main__":
    main()

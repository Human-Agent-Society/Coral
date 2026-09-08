"""Exclusive_Process launcher for the reviewed v24 author calibration.

The original parent process calls ``torch.cuda.empty_cache()`` after each
short-lived measurement child.  That lazily creates a persistent parent CUDA
context and prevents the next child from starting when the GPU is in
Exclusive_Process mode.  The parent performs only CPU-side validation between
children, so suppressing that no-op cleanup leaves every measured child and
all timing/correctness logic unchanged while keeping exactly one CUDA context.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BASE = Path("/calibration/author_calibrate_base.py")


def main() -> None:
    spec = importlib.util.spec_from_file_location("paged_v24_author_calibrate_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reviewed v24 author calibration")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.torch.cuda.empty_cache = lambda: None
    module.main()


if __name__ == "__main__":
    main()

from __future__ import annotations

import math

from speedup_math import bracketed_raw_speedup, geometric_mean_speedup


def main() -> None:
    baseline, speedup = bracketed_raw_speedup(4.0, 2.0, 9.0)
    assert baseline == 6.0
    assert speedup == 3.0
    assert geometric_mean_speedup([1.0, 4.0]) == 2.0
    assert math.isclose(geometric_mean_speedup([0.5, 1.0, 2.0]), 1.0, abs_tol=1e-12)
    for values in ((0.0, 1.0, 1.0), (1.0, float("nan"), 1.0)):
        try:
            bracketed_raw_speedup(*values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid bracket accepted: {values}")
    try:
        geometric_mean_speedup([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty aggregate accepted")
    print("test_speedup: PASS")


if __name__ == "__main__":
    main()

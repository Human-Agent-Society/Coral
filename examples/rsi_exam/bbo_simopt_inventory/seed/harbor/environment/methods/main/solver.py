import numpy as np

class Optimizer:
    """Shipped floor baseline: uniform random search."""
    batch = 1
    def __init__(self, dim, lower, upper, budget, seed, rng):
        self.dim = int(dim)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.rng = rng
    def ask(self, n):
        return self.lower + (self.upper - self.lower) * self.rng.random((int(n), self.dim))
    def tell(self, X, y):
        pass

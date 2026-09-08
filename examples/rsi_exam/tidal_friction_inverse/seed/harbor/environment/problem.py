import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

N = 24
G = 1.0            # non-dimensional gravity
OMEGA = 1.15       # non-dimensional tidal angular frequency (M2-like)
_H2 = 1.0 / (N - 1) ** 2

# Fixed 5-point connectivity of the grid, precomputed once so each forward solve
# only refills coefficient values instead of rebuilding the graph.
_IDX = np.arange(N * N).reshape(N, N)
_INTERIOR = _IDX[1:, :].ravel()                 # rows 1..N-1 solve the PDE
_BOUNDARY = _IDX[0, :]                           # row 0 is the forced open boundary
_edges = []
for yy in range(1, N):
    for xx in range(N):
        i = _IDX[yy, xx]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny_, nx_ = yy + dy, xx + dx
            if 0 <= ny_ < N and 0 <= nx_ < N:
                _edges.append((i, _IDX[ny_, nx_]))
_EDGES = np.array(_edges)                        # (E, 2): (node, neighbor)


def grid_xy():
    y, x = np.mgrid[0:N, 0:N]
    return np.column_stack((x.ravel() / (N - 1), y.ravel() / (N - 1)))


def solve_field(logf, depth, forcing, omega=OMEGA):
    """Frequency-domain linearized shallow water for one tidal constituent.

    The complex tidal elevation ``eta`` solves ``i*omega*eta - div(kappa grad eta)=0``
    with ``kappa = G*depth / (i*omega + r)`` and friction field ``r = exp(logf)``.
    Row ``y=0`` is the forced open boundary (Dirichlet = ``forcing``); the other three
    sides are no-flux. Returns the complex ``eta`` field with shape ``(N, N)``.
    """
    r = np.exp(np.asarray(logf, float)).reshape(-1)
    depth = np.asarray(depth, float).reshape(-1)
    kappa = (G * depth) / (1j * omega + r)                       # (N*N,) complex
    a, b = _EDGES[:, 0], _EDGES[:, 1]
    kf = (2.0 * kappa[a] * kappa[b] / (kappa[a] + kappa[b])) / _H2   # harmonic face value

    diag = np.zeros(N * N, dtype=complex)
    np.add.at(diag, a, kf)
    diag[_INTERIOR] += 1j * omega
    diag[_BOUNDARY] = 1.0

    rows = np.concatenate([np.arange(N * N), a])
    cols = np.concatenate([np.arange(N * N), b])
    vals = np.concatenate([diag, -kf])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N * N, N * N), dtype=complex)

    rhs = np.zeros(N * N, dtype=complex)
    rhs[_BOUNDARY] = np.asarray(forcing, complex).reshape(-1)
    return spla.spsolve(A, rhs).reshape(N, N)


def simulate(logf, gauges, depth, forcing, omega=OMEGA):
    """Complex tidal elevation sampled at gauge ``(x, y)`` in ``[0,1]^2`` (bilinear)."""
    eta = solve_field(logf, depth, forcing, omega)
    g = np.asarray(gauges, float)
    fx = g[:, 0] * (N - 1); fy = g[:, 1] * (N - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, N - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, N - 2)
    tx = fx - x0; ty = fy - y0
    return ((1 - tx) * (1 - ty) * eta[y0, x0] + tx * (1 - ty) * eta[y0, x0 + 1]
            + (1 - tx) * ty * eta[y0 + 1, x0] + tx * ty * eta[y0 + 1, x0 + 1])


def unpack_case(case):
    keys = ("depth", "forcing", "gauges", "obs_heads")
    return {k: np.asarray(case[k]) for k in keys}

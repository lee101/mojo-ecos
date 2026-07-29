"""Upstream-compatible Python interface for the covered ECOS subset."""

from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Mapping
from math import isfinite
from operator import index

import numpy as np
from scipy import sparse

from ._lib import addr, lib

_DUMMY_FLOAT = np.zeros(1, dtype=np.float64)
_DUMMY_MATRIX = np.zeros((1, 1), dtype=np.float64)
_DUMMY_Q = np.zeros(1, dtype=np.int64)
_WORKSPACES = threading.local()
_MAX_EXACT_FLOAT64_INTEGER = 2**53
_MIN_ABI_INTEGER = -(2**63)
_MAX_ABI_INTEGER = 2**63 - 1


class _Workspace:
    __slots__ = (
        "matrix",
        "pivots",
        "rhs",
        "direction",
        "ds_aff",
        "dz_aff",
        "rd",
        "rp",
        "rg",
        "correction",
        "stats",
    )

    def __init__(self, n: int, m: int, p: int):
        dim = n + p + m
        reduced_dim = n + p
        self.matrix = np.empty(reduced_dim * reduced_dim + 16 * m, dtype=np.float64)
        self.pivots = np.empty(reduced_dim, dtype=np.int64)
        self.rhs = np.empty(dim, dtype=np.float64)
        self.direction = np.empty(dim, dtype=np.float64)
        self.ds_aff = np.empty(m, dtype=np.float64)
        self.dz_aff = np.empty(m, dtype=np.float64)
        self.rd = np.empty(n, dtype=np.float64)
        self.rp = np.empty(max(p, 1), dtype=np.float64)
        self.rg = np.empty(m, dtype=np.float64)
        self.correction = np.empty(m, dtype=np.float64)
        self.stats = np.empty(8, dtype=np.float64)


def _workspace(n: int, m: int, p: int) -> _Workspace:
    key = (n, m, p)
    cached = getattr(_WORKSPACES, "cached", None)
    if cached is None or cached[0] != key:
        cached = (key, _Workspace(n, m, p))
        _WORKSPACES.cached = cached
    return cached[1]


def _float64_array(value, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind == "c":
        raise TypeError(f"{name} must contain real values")
    if source.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numeric values")
    if source.dtype.kind == "f" and source.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError(f"{name} cannot be narrowed to float64")
    if source.dtype.kind in "iu" and source.size:
        if np.any(source > _MAX_EXACT_FLOAT64_INTEGER) or np.any(
            source < -_MAX_EXACT_FLOAT64_INTEGER
        ):
            raise ValueError(f"{name} contains an integer not exactly representable as float64")
    try:
        return np.ascontiguousarray(source, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must contain real numeric values") from error


def _vector(value, name: str, size: int | None = None) -> np.ndarray:
    array = _float64_array(value, name)
    if array.ndim != 1:
        raise TypeError(f"{name} must be a dense one-dimensional array")
    if size is not None and array.size != size:
        raise TypeError(f"{name} has length {array.size}; expected {size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _matrix(value, name: str, columns: int) -> np.ndarray:
    if not sparse.issparse(value):
        raise TypeError(f"{name} is required to be a sparse matrix")
    if not sparse.isspmatrix_csc(value):
        warnings.warn(f"Converting {name} to a CSC matrix; may take a while.", stacklevel=3)
        value = value.tocsc()
    if value.shape[1] != columns:
        raise TypeError("Columns of A and G don't match")
    dense = _float64_array(value.toarray(), name)
    if not np.all(np.isfinite(dense)):
        raise ValueError(f"{name} contains a non-finite value")
    return dense


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if not _MIN_ABI_INTEGER <= result <= _MAX_ABI_INTEGER:
        raise OverflowError(f"{name} is outside the signed 64-bit ABI range")
    return result


def solve(c, G, h, dims, A=None, b=None, **kwargs):
    """Solve a dense LP/SOCP in ECOS standard form.

    The signature and returned mapping match :func:`ecos.solve` for the
    nonnegative (``l``) and second-order (``q``) cone subset.
    """
    started = time.perf_counter()
    c_array = _vector(c, "c")
    n = int(c_array.size)
    if n == 0:
        raise ValueError("c must not be empty")

    if not isinstance(dims, Mapping):
        raise TypeError("dims must be a mapping")
    l = _integer(dims.get("l", 0), 'dims["l"]')
    try:
        q_values = [
            _integer(value, f'dims["q"][{position}]')
            for position, value in enumerate(dims.get("q", []))
        ]
    except TypeError as error:
        if 'dims["q"]' in str(error):
            raise
        raise TypeError('dims["q"] must be an iterable of integers') from error
    exponential = _integer(dims.get("e", 0), 'dims["e"]')
    if exponential:
        raise NotImplementedError("exponential cones are not covered")
    if l < 0 or any(value < 2 for value in q_values):
        raise ValueError("cone dimensions must be nonnegative and SOC sizes at least 2")

    if (G is None) != (h is None):
        raise TypeError("G and h must be supplied together")
    if (A is None) != (b is None):
        raise TypeError("A and b must be supplied together")

    if G is None:
        m = 0
        g_array = _DUMMY_MATRIX
        h_array = _DUMMY_FLOAT
    else:
        g_array = _matrix(G, "G", n)
        m = int(g_array.shape[0])
        h_array = _vector(h, "h", m)
    if l + sum(q_values) != m:
        raise ValueError(
            f"cone dimensions describe {l + sum(q_values)} rows, but G has {m}"
        )
    if m == 0:
        raise NotImplementedError("at least one conic inequality is required")

    if A is None:
        p = 0
        a_array = _DUMMY_MATRIX
        b_array = _DUMMY_FLOAT
    else:
        a_array = _matrix(A, "A", n)
        p = int(a_array.shape[0])
        b_array = _vector(b, "b", p)

    verbose = bool(kwargs.pop("verbose", True))
    max_iters = _integer(kwargs.pop("max_iters", 100), "max_iters")
    abstol = float(kwargs.pop("abstol", 1e-8))
    reltol = float(kwargs.pop("reltol", 1e-8))
    feastol = float(kwargs.pop("feastol", 1e-8))
    if kwargs:
        name = next(iter(kwargs))
        raise TypeError(f"invalid keyword argument: {name}")
    if max_iters < 0:
        raise ValueError("max_iters must be nonnegative")
    if not all(isfinite(value) and value > 0 for value in (abstol, reltol, feastol)):
        raise ValueError("tolerances must be positive and finite")

    q_array = np.asarray(q_values, dtype=np.int64) if q_values else _DUMMY_Q
    x = np.empty(n, dtype=np.float64)
    y_work = np.empty(max(p, 1), dtype=np.float64)
    z = np.empty(m, dtype=np.float64)
    s = np.empty(m, dtype=np.float64)
    workspace = _workspace(n, m, p)
    matrix = workspace.matrix
    pivots = workspace.pivots
    rhs = workspace.rhs
    direction = workspace.direction
    ds_aff = workspace.ds_aff
    dz_aff = workspace.dz_aff
    rd = workspace.rd
    rp = workspace.rp
    rg = workspace.rg
    correction = workspace.correction
    stats = workspace.stats
    setup_done = time.perf_counter()

    status = int(
        lib().mecos_solve(
            addr(c_array),
            addr(g_array),
            addr(h_array),
            addr(a_array),
            addr(b_array),
            addr(q_array),
            addr(pivots),
            addr(x),
            addr(y_work),
            addr(z),
            addr(s),
            addr(matrix),
            addr(rhs),
            addr(direction),
            addr(ds_aff),
            addr(dz_aff),
            addr(rd),
            addr(rp),
            addr(rg),
            addr(correction),
            addr(stats),
            n,
            m,
            p,
            l,
            len(q_values),
            max_iters,
            abstol,
            reltol,
            feastol,
        )
    )
    finished = time.perf_counter()

    if status == 0:
        infostring = "Optimal solution found"
        numerr = 0
    elif status == -7:
        infostring = "Maximum number of iterations reached"
        numerr = 0
    else:
        infostring = "Numerical failure"
        numerr = 1

    pcost, dcost, gap = float(stats[4]), float(stats[5]), float(stats[3])
    info = {
        "exitFlag": status,
        "pcost": pcost,
        "dcost": dcost,
        "pres": float(stats[1]),
        "dres": float(stats[2]),
        "pinf": 0.0,
        "dinf": 0.0,
        "pinfres": np.nan,
        "dinfres": np.nan,
        "gap": gap,
        "relgap": gap / max(abs(dcost), 1e-30),
        "r0": 1e-8,
        "iter": int(stats[0]),
        "mi_iter": -1,
        "infostring": infostring,
        "timing": {
            "runtime": finished - started,
            "tsetup": setup_done - started,
            "tsolve": finished - setup_done,
        },
        "numerr": numerr,
    }
    if verbose:
        print(
            f"mojo-ecos: {infostring}; iter={info['iter']}, "
            f"pcost={pcost:.9g}, pres={info['pres']:.2e}, dres={info['dres']:.2e}"
        )
    y = y_work if p else np.empty(0, dtype=np.float64)
    return {"x": x, "y": y, "z": z, "s": s, "info": info}

"""End-to-end mojo-ecos versus upstream ECOS benchmarks."""

from __future__ import annotations

import gc
import importlib.metadata
import os
import platform
import statistics
import sys
import time

import ecos
import numpy as np
from scipy import sparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_ecos  # noqa: E402


def box_lp(n, seed=1):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    G = sparse.csc_matrix(np.vstack((np.eye(n), -np.eye(n))))
    h = np.full(2 * n, 2.0)
    return c, G, h, {"l": 2 * n, "q": []}


def dense_lp(n, extra, seed=2):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    random_G = rng.normal(size=(extra, n))
    G = sparse.csc_matrix(
        np.vstack((random_G, np.eye(n), -np.eye(n)))
    )
    h = np.r_[rng.uniform(0.5, 2.0, size=extra), np.full(2 * n, 2.0)]
    return c, G, h, {"l": extra + 2 * n, "q": []}


def portfolio(n):
    weights = np.linspace(0.5, 2.0, n)
    c = np.r_[1.0, np.zeros(n)]
    G_linear = np.c_[np.zeros((n, 1)), -np.eye(n)]
    G_soc = np.zeros((n + 1, n + 1))
    G_soc[0, 0] = -1.0
    G_soc[1:, 1:] = -np.diag(weights)
    G = sparse.csc_matrix(np.vstack((G_linear, G_soc)))
    h = np.zeros(2 * n + 1)
    A = sparse.csc_matrix(np.c_[np.zeros((1, 1)), np.ones((1, n))])
    b = np.array([1.0])
    return c, G, h, {"l": n, "q": [n + 1]}, A, b


def elapsed(fn, repeats):
    fn()
    samples = []
    gc.disable()
    try:
        for _ in range(repeats):
            started = time.perf_counter_ns()
            result = fn()
            samples.append((time.perf_counter_ns() - started) * 1e-6)
    finally:
        gc.enable()
    return statistics.median(samples), result


def solve_upstream(args):
    return ecos.solve(*args, verbose=False)


def solve_mojo(args):
    return mojo_ecos.solve(*args, verbose=False)


def cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    cases = [
        ("box LP, n=30 / m=60", box_lp(30), 15),
        ("box LP, n=80 / m=160", box_lp(80), 7),
        ("dense LP, n=30 / m=90", dense_lp(30, 30), 7),
        ("portfolio SOCP, n=20", portfolio(20), 10),
        ("portfolio SOCP, n=40", portfolio(40), 5),
    ]

    rows = []
    for name, args, repeats in cases:
        mojo_ms, mojo_result = elapsed(lambda: solve_mojo(args), repeats)
        ecos_ms, ecos_result = elapsed(lambda: solve_upstream(args), repeats)
        if mojo_result["info"]["exitFlag"] != 0 or ecos_result["info"]["exitFlag"] != 0:
            raise RuntimeError(f"{name}: a solver did not report optimal")
        if not np.isclose(
            mojo_result["info"]["pcost"],
            ecos_result["info"]["pcost"],
            rtol=2e-6,
            atol=2e-7,
        ):
            raise RuntimeError(f"{name}: objective mismatch")
        rows.append((name, mojo_ms, ecos_ms, ecos_ms / mojo_ms))

    print(f"Machine: {cpu_model()}, {os.cpu_count()} logical CPUs")
    print(f"Platform: {platform.platform()}")
    print(f"Reference: ECOS {importlib.metadata.version('ecos')}")
    print()
    print("| Problem | mojo-ecos | upstream ECOS | Mojo speedup |")
    print("|---|---:|---:|---:|")
    for name, mojo_ms, ecos_ms, ratio in rows:
        print(f"| {name} | {mojo_ms:.3f} ms | {ecos_ms:.3f} ms | {ratio:.2f}x |")


if __name__ == "__main__":
    main()

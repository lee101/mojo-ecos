"""Time a one-launch GPU round trip against the CPU at mojo-ecos caller sizes.

Measurement-only harness that produced the gate numbers in `bench/analysis.md` and
the README. It is not part of the package and nothing imports it. Build the probe
library first (the module must not define `main`):

    MODULAR_HOME="$PWD/.pixi/envs/default/share/max" \
      .pixi/envs/default/bin/mojo build --emit shared-lib \
      bench/gpu_probe.mojo -o dist/libmojo-ecos-gpu-probe.so
    pixi run python bench/gpu_probe.py

The probe checks its result against NumPy and refuses to report a row if the
device matmul disagrees.
"""

from __future__ import annotations

import ctypes
import os
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "dist", "libmojo-ecos-gpu-probe.so")
I = ctypes.c_int64


def free_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return max(int(x) for x in out.split() if x.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def load():
    handle = ctypes.CDLL(LIB)
    handle.probe_available.argtypes = []
    handle.probe_available.restype = I
    handle.probe_matmul_square_f64.argtypes = [I] * 4
    handle.probe_matmul_square_f64.restype = I
    return handle


def median_ms(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) * 1e-6)
    return statistics.median(samples)


def main():
    handle = load()
    available = handle.probe_available() == 0
    print(f"free MiB at start: {free_mib()}")
    print(f"probe_available(): {'yes' if available else 'no'}")
    if not available:
        return 1
    print()
    print("| size (n x n) | flops | bytes moved | CPU median | GPU round trip | GPU/CPU |")
    print("|---|---:|---:|---:|---:|---:|")
    ordered = 200 + 72  # largest linear_dim in the benchmark cases (dense LP n=200: n+m)
    for n in (73, ordered, 200, 384, 512, 1024, 2048):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(n, n))
        b = rng.normal(size=(n, n))
        c = np.zeros_like(a)
        flops = 2 * n**3
        moved = 3 * n * n * 8
        cpu = np.matmul(a, b)
        cpu_ms = median_ms(lambda: np.matmul(a, b, out=c), 3)
        assert handle.probe_matmul_square_f64(
            a.ctypes.data, b.ctypes.data, c.ctypes.data, n
        ) == 0
        assert np.allclose(c[:n, :n], cpu[:n, :n], rtol=1e-6, atol=1e-6), n
        gpu_ms = median_ms(
            lambda: handle.probe_matmul_square_f64(
                a.ctypes.data, b.ctypes.data, c.ctypes.data, n
            ),
            5,
        )
        print(
            f"| {n} x {n} | {flops:.3g} | {moved / 1e6:.2f} MB | "
            f"{cpu_ms:.3f} ms | {gpu_ms:.3f} ms | {gpu_ms / cpu_ms:.2f}x |"
        )
    print()
    print(f"free MiB at end: {free_mib()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# mojo-ecos

`mojo-ecos` is a standalone Mojo implementation of the dense core of
[ECOS](https://github.com/embotech/ecos), the Embedded Conic Solver. It solves
linear programs and feasible second-order cone programs through an
upstream-compatible Python entry point:

```python
solve(c, G, h, dims, A=None, b=None, **kwargs)
```

This is an independent MIT-licensed implementation, not a binding to ECOS.
The upstream `ecos` package is installed only for parity tests and benchmarks.

## Coverage

Covered:

- nonnegative cones through `dims["l"]`;
- any number of Lorentz/second-order cones through `dims["q"]`;
- optional linear equalities `A @ x == b`;
- SciPy sparse CSC input, plus the same CSR-to-CSC warning behavior as upstream;
- `verbose`, `max_iters`, `abstol`, `reltol`, and `feastol`;
- the upstream result shape: `x`, `y`, `z`, `s`, and the standard `info` fields.

The implementation is intended for small and medium dense problems. Sparse
inputs are accepted for API compatibility and converted to row-major dense
storage before crossing the FFI.

Not covered:

- exponential cones (`dims["e"]`);
- mixed-integer extensions;
- sparse KKT factorization;
- homogeneous self-dual embedding, so infeasibility and unboundedness
  certificates are not reported;
- iterative refinement and ECOS's less common low-level tuning options.

For covered problems, the test suite compares primal and dual solutions,
objectives, cone membership, and KKT residuals directly with ECOS 2.0.14.

## Install

The repository pins the verified Mojo nightly and contains all development
dependencies:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-ecos.so`. Run the parity tests with
`pixi run test`.

## Usage

This example minimizes `t` subject to `||(x, y)||₂ <= t`, `x = 3`, and
`y = 4`:

```python
import numpy as np
from scipy import sparse
import mojo_ecos as ecos

c = np.array([1.0, 0.0, 0.0])
G = sparse.csc_matrix(-np.eye(3))
h = np.zeros(3)
A = sparse.csc_matrix([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
b = np.array([3.0, 4.0])

result = ecos.solve(c, G, h, {"l": 0, "q": [3]}, A, b, verbose=False)
assert result["info"]["exitFlag"] == 0
np.testing.assert_allclose(result["x"], [5.0, 3.0, 4.0], atol=1e-7)
```

Run it from the checkout with `pixi run python example.py`, or set
`PYTHONPATH=python` when using another environment.

## Benchmarks

These are real median end-to-end timings from `pixi run bench`. Both solvers
receive the same SciPy matrices, and input conversion and result construction
are included. A speedup below `1.00x` means Mojo is slower.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 logical CPUs.
Platform: Linux 6.8.0-147-generic x86_64, glibc 2.39. Reference: ECOS 2.0.14.

| Problem | mojo-ecos | upstream ECOS | Mojo speedup |
|---|---:|---:|---:|
| box LP, n=30 / m=60 | 0.137 ms | 0.109 ms | 0.80x |
| box LP, n=80 / m=160 | 0.221 ms | 0.256 ms | 1.16x |
| dense LP, n=30 / m=90 | 0.468 ms | 0.908 ms | 1.94x |
| portfolio SOCP, n=20 | 0.358 ms | 0.207 ms | 0.58x |
| portfolio SOCP, n=40 | 0.656 ms | 0.370 ms | 0.56x |

Each column is the median of 10 `pixi run bench` runs in one verify session
(2026-11-11) on a shared host (load average 17-20); per-run values and spreads
are in `bench/verify.md`. Mojo is still faster on the dense LP and the larger
box LP. It remains slower on the small box and portfolio cases, where
upstream's mature sparse KKT factorization retains an advantage. The benchmark
script checks solver status and objective parity before printing a row; both
sides run to convergence (`exitFlag == 0`) with the same iteration counts up to
one iteration, so neither side skips a solve.

### GPU path: gated out, not shipped

The phase gate asks for kernels above roughly 2 flops per byte. None qualify:

- `src/ecos.mojo` has no matrix-multiply-shaped kernel. The dense Newton
  system is assembled by accumulated rank-one terms and solved by a dense LU,
  and both update one element with one fused multiply-add. The reduction,
  indexed single-G, diagonal-solve, and SOCP recovery kernels are pure
  streaming. The full per-kernel table is in `bench/analysis.md`; the ceilings
  are 2 flops/byte (LU, triangular solve, rank-one assembly) and about 1
  (everything else).
- To check what a GPU path could buy anyway, `bench/gpu_probe.py` measures one
  launch of the cheapest reference kernel (16x16 tiled matmul: copy in, one
  launch, copy out) against the CPU. On this host the round trip costs about
  0.1-2.5 ms when small and 37 ms for 100 MB, and the shipped kernels run in
  1-212 us per call at the benchmark sizes.
- The gate was also checked for the device: `nvidia-smi --query-gpu=memory.free`
  reported 5203 MiB free before the phase (an RTX 5090 shared with other
  tenants), above the 4000 MiB floor, and 3907-3597 MiB while the probe ran.
  Problems larger than any benchmark case (`n=1024`, `n=2048`) were measured
  once for order of magnitude; see `bench/analysis.md`.

The summary above says the solver is memory-bound at the kernel level, and the gate
requires that verdict to be stated here. Per `src/ecos.mojo` and
`bench/analysis.md`: **CPU only: memory-bound** for the LU, triangular solve,
rank-one Newton assembly, residual and recovery kernels (at most 2 flops per
byte, about 1 for the streaming kernels). The only kernel that is not
memory-bound at that ceiling is the SOCP reduced-system fill, whose G traffic
grows with `n * cone size`, and it is still under 2 flops per byte.

The "before/after" table the earlier phases published has been removed: nothing
in `src/ecos.mojo` changed, so both of its columns were the same code measured
twice, and its numbers differ from this session's only by host noise. The
per-run spreads that establish that are in `bench/verify.md`.
No GPU kernel ships, so nothing in the suite is gated on a device and none is
skipped: `pixi run test` runs all 40 parity tests on CPU. A GPU path could only
pay off for a matrix-multiply-shaped kernel at `n` in the hundreds, which this
solver does not have.

### CPU parallelism: measured, then not shipped

A parallel phase was attempted on `parallelize` from `max.algorithm` (the only
form that compiles on the pinned 1.2.0 toolchain; `std.algorithm` and
`std.threading` do not). It was prototyped behind a size threshold on the
solver's per-iteration row kernels (residual and `recover_ds` row dots, dense
Newton assembly, LU factorization) and measured before being wired in. It lost
on every shape real callers pass, so the solver in `src/ecos.mojo` is still
serial and contains no `parallelize` call. The numbers below are from this host
(shared; load average 26-32 during measurement, Intel Xeon E5-2697 v4, 72
logical CPUs, AVX2 only).

Per-kernel worker ladder, 1 / 2 / 4 / 8 workers, total work fixed at 60M
flops, minimum of 3-11 samples per point:

| Kernel | Work shape | w=1 | w=2 | w=4 | w=8 |
|---|---|---:|---:|---:|---:|
| row dot, one row per work item | 40,000 rows x 1,500 floats | 55-66 ms | 44-62 ms | 29-41 ms | not run |
| row dot, degenerate | 1 row x 40M floats | 63.3 ms | 92.1 ms | 87.7 ms | 87.5 ms |
| row dot, degenerate | 1 row x 400M floats | 609.9 ms | 615.7 ms | 627.3 ms | 635.3 ms |

Interpretation:

- Fan-out only helps when each work item is large. The one shape that gains is
  40,000 independent 1,500-float rows at 4 workers: roughly 1.6-2.2x, below the
  4x ideal, and not measured at 8 workers.
- A fixed 60M-flop budget spread over many small rows is still traffic-bound.
  Even in the best case, the 4-worker column is a range because the shared host
  moves it; it does not scale linearly.
- With one giant contiguous row the parallel path is flat at best and 0.9-1.5x
  slower at 2-8 workers. In that degenerate shape there is only one work item,
  so `parallelize` has almost nothing to divide.
- Probe side effect: 2 of roughly 70 harness runs of a 40,000-row body segfaulted
  after entering `parallelize`. The crash reporter fired too, so we cannot tell
  whether the fault was in the kernel or the runtime's signal handling. That
  matters here because the shared library is called from Python threads and
  already spawns its runtime there; an intermittent crash in a solver is
  unacceptable.
- The 8-worker column was measured only where listed. It never beat 4, which is
  consistent with the 4-worker cap but does not by itself prove one.

A parallel path that is flat-to-negative at caller sizes and can crash the
process is worse than no parallelism, so it was reverted.

The solver-time equivalent makes the same point. Dense LP `n=200`, 13
iterations, each kernel measured on its own at caller sizes: row-dot residual
1-8 us, LU factorization of the reduced system 16-132 us, one triangular solve
4-38 us. Even parallelising the entire factorization 4x would save about 0.1 ms
of a 26 ms solve, which is inside this host's run-to-run noise. The dense LP
`n=200` "medians by thread count" figures an earlier phase reported are no
longer quoted here: `MOJO_MAX_THREADS` appears nowhere in this repository or in
the solve path, so the verify session could not reproduce a thread-count sweep,
and the reported ranges overlapped regardless. If that experiment is redone it
needs a documented way to set the runtime's worker count. The 5 ms launch cost
above was measured once at a fixed iteration count, and it also exceeds the
whole cost of any one kernel at these sizes.

Why the threshold decision stands even without threading the solver:

- Every kernel at the sizes the benchmark uses (`box LP n=80`, `dense LP n=30`,
  `portfolio n=20/40`) finishes in 1-25 us, at least an order of magnitude
  below one launch's overhead, so no crossover point is reachable from these
  callers. The verify-session run-to-run spreads for these shapes single-
  threaded (0.132-0.162 ms for box LP n=30 through 0.635-0.771 ms for
  portfolio n=40, ten runs each) are in `bench/verify.md`; they are host load,
  since no solver code changed between the phases that produced them.
- The only size range where a parallel row-dot wins is thousands of
  *independent*, multi-hundred-float rows. At that scale the dense problem has
  lost its reason for existing, so no threshold can be justified from these
  callers.

`max` was added to `pixi.toml` with the pinned release
(`max ==26.7.0.dev2026092605`) so the `max.*` module resolves for anyone
rebuilding the port with a parallel or GPU path; the solver itself needs
nothing from it.

## How it works

The solver is an infeasible-start primal-dual predictor-corrector method. Mojo
computes residuals, Lorentz-cone Jordan products, fraction-to-boundary steps,
and a reduced Newton system that eliminates cone directions exactly. Each
iteration factors that system once and reuses the pivoted LU factors for the
affine and corrected right-hand sides. SIMD covers dense row operations and
scalar tails. Matrices with one nonzero per constraint row use indexed residual,
Newton assembly, and direction-recovery kernels. Linear programs with that
structure use an exact O(n) diagonal or one-border KKT solve instead of general
LU.

Python normally converts CSC matrices once to C-contiguous `float64` row-major
arrays and allocates result buffers. For one-entry-per-row matrices with at
least 10,000 dense elements, the CSC `data`, `indices`, and `indptr` NumPy
buffers cross the FFI directly and Mojo builds indexed row metadata without a
dense copy. Internal scratch is reused per thread for same-shaped solves. NumPy
buffers cross the C ABI as integer addresses because exported Mojo functions
cannot be parametric over pointer origins. The Mojo export reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values and performs no allocation.
The Python process owns every buffer for the full call.

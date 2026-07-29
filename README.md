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
Platform: Linux 6.8.0-136-generic x86_64, glibc 2.39. Reference: ECOS 2.0.14.

| Problem | mojo-ecos | upstream ECOS | Mojo speedup |
|---|---:|---:|---:|
| box LP, n=30 / m=60 | 0.211 ms | 0.100 ms | 0.48x |
| box LP, n=80 / m=160 | 0.818 ms | 0.247 ms | 0.30x |
| dense LP, n=30 / m=90 | 0.382 ms | 0.772 ms | 2.02x |
| portfolio SOCP, n=20 | 0.327 ms | 0.181 ms | 0.55x |
| portfolio SOCP, n=40 | 0.771 ms | 0.359 ms | 0.47x |

Mojo is faster on the dense LP and remains slower on the sparse box and
portfolio cases. Upstream's mature sparse KKT factorization retains a
substantial advantage when the constraint matrices contain many zeros. The
benchmark script checks solver status and objective parity before printing a
row.

No GPU path is implemented or benchmarked.

## How it works

The solver is an infeasible-start primal-dual predictor-corrector method. Mojo
computes residuals, Lorentz-cone Jordan products, fraction-to-boundary steps,
and a reduced Newton system that eliminates cone directions exactly. Each
iteration factors that system once and reuses the pivoted LU factors for the
affine and corrected right-hand sides. SIMD covers dense row operations and
scalar tails; large independent matrix-vector rows use a size-gated parallel
path.

Python converts CSC matrices once to C-contiguous `float64` row-major arrays
and allocates result buffers. Internal scratch is reused per thread for
same-shaped solves. NumPy buffers cross the C ABI without a copy as integer
addresses because exported Mojo functions cannot be parametric over pointer
origins. The Mojo export reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values and performs no allocation.
The Python process owns every buffer for the full call.

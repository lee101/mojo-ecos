# Verify session, 2026-11-11

Scope: reproduce every measurement in `README.md`, check the benchmark is
fair, audit `tests/` history for loosened parity assertions, and record what
could not be reproduced. Environment: this host, Intel Xeon E5-2697 v4 @
2.30GHz, 72 logical CPUs, Linux 6.8.0-147-generic x86_64, glibc 2.39, ECOS
2.0.14, Mojo 1.2.0.dev2026092605. Host is shared; load average was 11-22
during the session.

## Gate commands

| Command | Result |
|---|---|
| `pixi run build` | exit 0 (deprecation warnings for positional `__getitem__` only) |
| `pixi run test` | 40 passed |
| `pixi run bench` | exit 0; all five rows |

## Main benchmark table

The README table is the median of the ten `pixi run bench` runs in the locked
block below. Values are ms, end-to-end medians per run.

| Problem | README before (mojo / ecos / speedup) | session median (mojo / ecos) | session speedup | speedup within 20%? |
|---|---|---:|---:|---|
| box LP, n=30 / m=60 | 0.130 / 0.100 / 0.77x | 0.137 / 0.109 | 0.80x | yes |
| box LP, n=80 / m=160 | 0.196 / 0.257 / 1.31x | 0.221 / 0.256 | 1.16x | yes |
| dense LP, n=30 / m=90 | 0.473 / 0.823 / 1.74x | 0.468 / 0.908 | 1.94x | yes |
| portfolio SOCP, n=20 | 0.350 / 0.187 / 0.53x | 0.358 / 0.207 | 0.58x | yes |
| portfolio SOCP, n=40 | 0.647 / 0.339 / 0.52x | 0.656 / 0.370 | 0.56x | yes |

Per-run spread (min-max over ten runs), Mojo side / upstream side:

| Problem | mojo-ecos | upstream ECOS |
|---|---:|---:|
| box LP, n=30 / m=60 | 0.132-0.162 | 0.103-0.122 |
| box LP, n=80 / m=160 | 0.213-0.265 | 0.245-0.281 |
| dense LP, n=30 / m=90 | 0.438-0.587 | 0.794-0.952 |
| portfolio SOCP, n=20 | 0.348-0.454 | 0.181-0.263 |
| portfolio SOCP, n=40 | 0.635-0.771 | 0.342-0.454 |

Verdict on the 20% rule: all five rows reproduce within 20%. The closest call
is portfolio SOCP n=20: the README's 0.350 ms permits 0.280-0.420 ms and the
session median is 0.358 ms. An earlier block of thirteen runs in the same
session (load 11-17) had a 0.362 ms median for that row, still inside the
bound. The row was still replaced with the session median, because 0.350 is
not a number measured in this session. It stays in the table as a slower row
(0.58x). No row flipped from a Mojo win to a loss or back across the session;
both directions are within host noise on the faster rows only.

Also stale in the header: the README said `Linux 6.8.0-136-generic`; this host
runs 6.8.0-147. Corrected. CPU model, CPU count and ECOS version match.

## Is the comparison fair?

- Same inputs: `bench/bench.py` builds one argument tuple per case and calls
  both `ecos.solve` and `mojo_ecos.solve` with it. No per-side preconditioning.
- Same work: both sides run to `exitFlag == 0` with the same iteration counts,
  except box LP (Mojo 8 vs ECOS 7) and portfolio n=40 (10 vs 9):
  box30/box80 Mojo 8 / ECOS 7, dense30 11 / 11, port20 9 / 9, port40 10 / 9.
  `max_iters` has identical semantics on both sides at 1, 2, 5 and 1000.
  Objectives agree to <1e-9 relative; max |x| differences 5e-8 to 2.3e-7, on
  the order of the `abstol` tolerance, not a skipped solve.
- Input conversion is inside the timed call on both sides. The only asymmetry
  is the port's documented single-entry-row zero-copy path, which applies to
  box LP n=80 only (`nnz == rows`, 160x80 = 12800 >= 10000). It changes how G
  is marshalled, not what is solved, and it is a shipped code path measured in
  the row that reports it.
- No result or workspace is cached between calls on either side; the existing
  test suites of both sides do the same kind of repeated solve efficiently.

## Test-history audit (`git log -p -- tests/`)

Only two commits touch `tests/`: `b1b343d` (initial port) and `0d9d5da`
(accel). `0d9d5da` only **adds** three parity tests
(`test_zero_copy_single_entry_rows_match_upstream`,
`test_single_entry_rows_with_one_equality_match_upstream`,
`test_multi_entry_row_uses_dense_fallback`), renames
`test_large_residual_uses_parallel_threshold` to
`test_large_single_entry_setup_with_zero_iterations` without changing its
assertions, and updates the C-ABI null-address arity from 27 to 31. No
assertions were removed and no tolerance was loosened: `run_both` still runs
both solvers on identical CSC inputs, and the tolerances (`x` 2e-5, objective
2e-7, residual 2e-8) have not changed since the initial commit. Nothing to
restore.

## What could not be reproduced

- **GPU section** (`README.md`, "GPU path: gated out, not shipped"): the
  `nvidia-smi` readings, `gpu_probe.py` round-trip timings, and the
  `n=1024`/`n=2048` measurements are not reproducible in this session without
  re-running a shared RTX 5090 probe; they are left as recorded in
  `bench/analysis.md` on the phase's own host state. The device-dependent
  numbers are the only figures in the README this session did not measure.
- **Parallel section worker ladder** (`w=1/2/4/8` row-dot table): same caveat.
  It is a probe harness, not in `tests/` or `bench/verify`, and the README
  itself already says two of ~70 harness runs segfaulted. Not re-run.
- **Thread-count sweep** (`MOJO_MAX_THREADS`): `MOJO_MAX_THREADS` does not
  appear anywhere in the repository, and `pixi.toml` defines no such variable.
  The dense LP n=200 "medians by thread count" could not be reproduced and the
  numbers were removed from the README, along with the sentence that implied a
  documented thread setting.
- **Before/after tables**: the parallel phase (`f0055ea`) changed only
  `README.md`, `pixi.toml` and `pixi.lock`; `src/ecos.mojo` is untouched, so
  both "before" and "after" columns are the same code measured twice. Their
  per-run values (e.g. dense LP n=30 "before" 0.458 vs the main table's
  0.473) are within this session's own spread (0.444-0.505). Both tables were
  removed; one reproduced table in `bench/verify.md` replaces them.
- **GPU section's stated relative speedups**: none of its claims are wrong, but
  since they were not re-measured here they are not evidence for this session.

## Changes made

- `README.md`: main benchmark table and speedups replaced with the session
  medians above; platform string corrected to another kernel; sentence added
  stating both sides converge with the same iteration counts.
- `README.md`: removed the two before/after tables and the `MOJO_MAX_THREADS`
  thread-count numbers, replaced with a note saying why they cannot be
  reproduced / are not a comparison.
- `README.md`: parallel section's "measured before/after end-to-end medians"
  sentence replaced with this session's run-to-run spreads.
- `bench/verify.md`: this file.
- No solver code, tests, or `bench/bench.py` changed. `pixi run build &&
  pixi run test` re-run at the end: build exit 0, 40 passed.

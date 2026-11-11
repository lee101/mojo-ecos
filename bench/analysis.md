# GPU gate analysis

Created during the GPU phase because the workspace did not contain `bench/analysis.md`
(checked the working tree and full git history: the path is not in any commit).

## Gate items

| Gate | Reading | Result |
| --- | --- | --- |
| `nvidia-smi --query-gpu=memory.free` | gate checks: 5203, 5170, 4655, 4411 MiB (RTX 5090, shared). Readings during probe runs: 3907, 3597, 3651 MiB, i.e. under the floor while the probe was allocating and while other tenants were busy | passed the gate at check time; needs re-checking per run |
| Device allocation cap | probe allocates three `n x n` float64 buffers; largest run 2048 (~100 MB total) | under 2 GB |
| Toolchain API | `max.gpu.host.DeviceContext`, `DeviceBuffer`, `enqueue_function` launch in `max` 26.7.0.dev2026092605 | present, launches and returns results (see notes below) |
| Intensity ceiling of a shipped kernel | see below | at most ~2 flops/byte |

## Arithmetic intensity of the shipped solver (`src/ecos.mojo`)

`perf stat` is unavailable on this host (`perf_event_paranoid = 4`), so these counts
come from the kernel source and the buffers each kernel touches.

| Kernel | Work per element | Traffic per element | flops/byte ceiling |
| --- | ---: | ---: | ---: |
| `gaussian_factor` (LU, no pivoting) | 2 per trailing column | 2 read + 2 write per update | 2 (plus the `-1` in `a - b*c`) |
| `gaussian_solve_factored` | 2 per trailing column | 2 read + 1 write | 2 |
| `fill_newton` / `fill_newton_linear` rank-one part | 2 per matrix element | 3 per matrix element | 2 |
| `residuals` / `recover_ds` / `recover_dz_*` (general G) | 2 per element | 2 per element | ~1 |
| `solve_diagonal_kkt` / single-G indexed path | O(n) scalars | O(n) scalars | ~1 |
| SOCP reduced fill (`fill_newton_reduced_soc`, non single-G) | O(n) per cone column | O(n * size) G traffic | under 2 |

Nothing in the shipped solver is matrix-multiply-like: G is general but the Newton
system is assembled by accumulated rank-one terms, and the solve is a dense LU that
is memory-bound at the 2 flops/byte ceiling.

## What a GPU round trip would have to beat

Measured with `bench/gpu_probe.py` / `bench/gpu_probe.mojo` (measurement-only probe, a
reference 16x16-tile matmul kernel: copy in, one launch, copy out - the same shape a
GPU path for this solver would have). CPU column is NumPy `matmul(out=)`.

| size | flops | CPU median | GPU round trip | GPU/CPU |
| --- | ---: | ---: | ---: | ---: |
| 73 x 73 (largest linear system in a benchmark case) | 7.8e5 | 0.044-0.049 ms | 0.111-2.326 ms | 2.3x-53x slower |
| 272 x 272 (dense LP n=200: n+m) | 4.0e7 | 0.420-0.313 ms | 0.357-2.351 ms | 0.85x-7.5x slower |
| 200 x 200 | 1.6e7 | 0.163-0.142 ms | 0.233-2.341 ms | 1.43x-16.5x slower |
| 384 x 384 | 1.1e8 | 12.0-18.1 ms | 0.719-0.733 ms | 12x-25x faster |
| 512 x 512 | 2.7e8 | 7.2-12.1 ms | 1.008-2.468 ms | 7x-12x faster |
| 1024 x 1024 | 2.2e9 | 27.0-9.2 ms | 3.819-5.683 ms | 3.5x-7x faster |
| 2048 x 2048 | 1.7e10 | 67.0-51.0 ms | 38.6-36.9 ms | 1.3x-1.7x faster |

Ranges are the two runs recorded in this session. The shared GPU was at 48-100%
utilisation from other tenants, which is why the small round trips range from 0.1 ms
to 2.5 ms; the large ones are dominated by the kernel itself. The CPU column is also
noisy for the same reason (the 1024 and 2048 rows get faster in the second run), so
only the order of magnitude is meaningful.

## GPU API facts that differ from `MOJO_NOTES.md` section 5

Verified by compiling against `max ==26.7.0.dev2026092605`:

- `block_idx` / `thread_idx` are values, not callables: `block_idx(0)` is a parse
  error. Use `block_idx.x`, `block_idx.y`, `thread_idx.x`.
- `DeviceBuffer` is in `max.gpu.host`, not `max.gpu`.
- The working launch form is
  `ctx.enqueue_function[kernel](buf0, buf1, Int64(n), grid_dim=(gx, gy), block_dim=(tx, ty))`
  where `kernel(a: FPtr, b: FPtr, dst: FPtr, n_value: Int64)`. `out` is a reserved
  name and cannot be a parameter name.
- Inside a kernel, take `var p = buf.unsafe_ptr()` (returns `Pointer[Float64,
  origin_of(buf)]`) and use `p[row * n + col]`; `buf[i]` and `buf.unsafe_store` do
  not exist.
- `nvidia-smi` on this host reports about 32.6 GB total but only a few GB free; the
  four `memory.free` readings above are the authoritative gate numbers.

## Conclusion

- The round trip costs roughly 0.1-2.5 ms of launch/transfer latency (small) and
  about 37 ms for 100 MB of traffic (large).
- A GPU path would only win when one launch replaces a CPU kernel costing more than
  that latency. Every kernel in `src/ecos.mojo` is at or below the 2 flops/byte
  ceiling, so there is no such kernel to move, and no threshold at which a caller
  would benefit.

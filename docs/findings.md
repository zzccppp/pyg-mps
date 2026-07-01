# Findings

## 2026-07-01: Initial `pyg-lib` MPS operator gap

Host/direct-shell baseline:

- macOS `26.5.1`, arm64, Apple MPS visible.
- `torch==2.12.1`
- `torch.backends.mps.is_available() == True`
- `torch.mps.device_count() == 1`
- `torch-geometric==2.8.0`
- `pyg-lib==0.7.0+pt212`

Working on MPS:

- PyTorch tensor creation, indexing, matmul backward, `index_add_`, sparse COO
  matmul.
- PyG `Data.to("mps")`.
- PyG `GCNConv` forward/backward.

Initial native MPS gaps after installing the upstream `pyg-lib` wheel:

- `pyg::knn`
- `pyg::radius`
- `pyg::nearest`
- `pyg::fps`
- `pyg::grid_cluster`
- `pyg::scatter_sum`
- `pyg::spline_basis`
- `pyg::spline_weighting`

These gaps appear both through PyG high-level APIs and through direct
`pyg_lib.ops` calls. The native `pyg-lib` probe initially reported:

- `ok`: 7
- `unsupported`: 10
- `failed`: 0

With `PYTORCH_ENABLE_MPS_FALLBACK=1`, the fallback probe reports:

- `ok`: 17
- `unsupported`: 0
- `failed`: 0

## 2026-07-01: Local source build baseline

The uv environment now points at local editable source builds:

- `torch-geometric==2.9.0` from `src/pytorch_geometric`
- `pyg-lib==0.8.0` from `src/pyg-lib`
- `torch-scatter==2.1.2` from `src/pytorch_scatter`
- `torch-sparse==0.6.18` from `src/pytorch_sparse`
- `torch-cluster==1.6.3` from `src/pytorch_cluster`
- `torch-spline-conv==1.2.2` from `src/pytorch_spline_conv`
- `scipy==1.18.0` for sparse/cluster Python imports

Local CPU validation after the scatter-family probe expansion:

- `./scripts/uv_stage.sh meta`: all scatter Meta dispatch probes pass.
- `./scripts/uv_stage.sh probe-cpu local-cpu-after-scatter-family`: 21 ok,
  0 failed.
- `./scripts/uv_stage.sh extensions-cpu`: minimal CPU tests pass for
  `torch_scatter`, `torch_sparse`, `torch_cluster`, and `torch_spline_conv`.

Direct-shell MPS validation for legacy optional extensions:

- `torch_scatter.scatter_add`: works on MPS.
- `torch_cluster.knn_graph`: CPU-only kernel is reached and asserts that input
  tensors must be CPU tensors.
- `torch_cluster.grid_cluster`: CPU-only kernel is reached and asserts that
  input tensors must be CPU tensors.
- `torch_sparse.SparseTensor.matmul`: fails when `torch_sparse.ind2ptr` reaches
  a CPU-only path with MPS indices.
- `torch_spline_conv.spline_basis`: CPU-only kernel is reached and asserts
  that `pseudo` must be a CPU tensor.

The first source patch is tracked in:

- `patches/pyg-lib/0001-add-scatter-meta-dispatch.patch`

It adds Meta dispatch registrations for:

- `pyg::scatter_sum`
- `pyg::scatter_mul`
- `pyg::scatter_mean`
- `pyg::scatter_min`
- `pyg::scatter_max`

The second local patch is tracked in:

- `patches/pyg-lib/0002-add-scatter-family-mps-dispatch.patch`

It adds initial MPS dispatch registrations for:

- `pyg::scatter_sum`
- `pyg::scatter_mul`
- `pyg::scatter_min`
- `pyg::scatter_max`

`pyg::scatter_mean` is implemented as a composite operation over
`scatter_sum`, so the MPS `scatter_sum` dispatch is expected to unlock it too.

The implementation is intentionally conservative:

- `scatter_sum` reuses PyTorch's native MPS `scatter_add_` tensor operation.
- `scatter_mul`, `scatter_min`, and `scatter_max` use PyTorch's native MPS
  `scatter_reduce_` path first.
- `scatter_min` and `scatter_max` compute values on MPS but compute `arg_out`
  through a CPU helper before copying the integer arg tensor back to MPS. This
  avoids PyTorch MPS's current lack of `torch.int64` `scatter_reduce_` support.
- A dedicated Metal kernel can follow if profiling or runtime behavior shows
  that the conservative path is insufficient.

Direct-shell MPS validation after the first MPS patch:

- `pyg_lib_scatter_sum`: `ok`, output device `mps:0`, shape `[4, 3]`.
- Native summary improved from `7 ok / 10 unsupported / 0 failed` to
  `8 ok / 9 unsupported / 0 failed`.
- In that pre-expansion probe revision, fallback remained
  `17 ok / 0 unsupported / 0 failed`.

The expanded local probe now also tests:

- `pyg_lib_scatter_mul`
- `pyg_lib_scatter_mean`
- `pyg_lib_scatter_min`
- `pyg_lib_scatter_max`

These additional MPS scatter-family registrations compile and pass CPU/Meta
regression checks.

Direct-shell MPS validation after the first scatter-family expansion:

- `pyg_lib_scatter_sum`: `ok`
- `pyg_lib_scatter_mul`: `ok`
- `pyg_lib_scatter_mean`: `ok`
- `pyg_lib_scatter_min`: failed with `RuntimeError: not supported for
  torch.int64`
- `pyg_lib_scatter_max`: failed with `RuntimeError: not supported for
  torch.int64`
- Native summary reached `10 ok / 9 unsupported / 2 failed`.

The failure was isolated to the `arg_out.scatter_reduce_(..., "amin")` path for
integer arg indices, not to the floating-point min/max value reduction. The
current patch keeps min/max values on MPS and computes `arg_out` with CPU tensor
ops before returning it on MPS.

Direct-shell MPS validation after the CPU-assisted argindex fix:

- `pyg_lib_scatter_sum`: `ok`, output device `mps:0`
- `pyg_lib_scatter_mul`: `ok`, output device `mps:0`
- `pyg_lib_scatter_mean`: `ok`, output device `mps:0`
- `pyg_lib_scatter_min`: `ok`, output and arg devices `mps:0`
- `pyg_lib_scatter_max`: `ok`, output and arg devices `mps:0`
- Native summary reached `12 ok / 9 unsupported / 0 failed`.
- Fallback summary reached `21 ok / 0 unsupported / 0 failed`.

The third local patch is tracked in:

- `patches/pyg-lib/0003-add-cpu-assisted-mps-shims.patch`

It adds CPU-assisted MPS dispatch shims for:

- `pyg::knn`
- `pyg::radius`
- `pyg::nearest`
- `pyg::fps`
- `pyg::grid_cluster`
- `pyg::spline_basis`
- `pyg::spline_basis_backward`
- `pyg::spline_weighting`
- `pyg::spline_weighting_backward_x`
- `pyg::spline_weighting_backward_weight`
- `pyg::spline_weighting_backward_basis`

These shims copy inputs to CPU, reuse the existing CPU kernels, then copy
outputs back to the caller's MPS device. This is a compatibility milestone, not
a performance endpoint. The patch builds, passes local CPU/Meta regression
checks, applies cleanly after the previous two patches, and is verified from a
direct macOS shell.

Direct-shell MPS validation after the CPU-assisted point-cloud and spline
shims:

- `pyg-macos-native`: `21 ok / 0 unsupported / 0 failed`.
- `pyg-macos-fallback`: `21 ok / 0 unsupported / 0 failed`.
- PyG high-level `knn_graph`: `ok`, output edge tensor on `mps:0`.
- PyG high-level `SplineConv`: `ok` on MPS-facing tensors.
- Direct `pyg-lib` point-cloud ops `knn`, `radius`, `nearest`, `fps`, and
  `grid_cluster`: `ok`, outputs on `mps:0`.
- Direct `pyg-lib` spline ops `spline_basis` and `spline_weighting`: `ok`,
  outputs on `mps:0`.
- Direct `pyg-lib` scatter-family ops `scatter_sum`, `scatter_mul`,
  `scatter_mean`, `scatter_min`, and `scatter_max`: `ok`, outputs on `mps:0`;
  `scatter_min/max` also return arg tensors on `mps:0`.

This confirms that the patched `pyg-lib` path no longer needs global PyTorch
MPS CPU fallback for the probed operators. The remaining MPS gaps are now in
the legacy optional extension packages when they are called directly.

Direct-shell MPS validation for legacy optional extension packages after the
`pyg-lib` shim milestone:

- `torch_scatter.scatter_add`: `ok` on MPS.
- `torch_cluster.knn_graph`: `unsupported`; reaches a CPU-only kernel that
  requires CPU tensors.
- `torch_cluster.grid_cluster`: `unsupported`; reaches a CPU-only kernel that
  requires CPU tensors.
- `torch_sparse.SparseTensor.matmul`: `unsupported`; reaches CPU-only
  `ind2ptr` with MPS indices.
- `torch_spline_conv.spline`: `unsupported`; reaches a CPU-only spline basis
  kernel that requires CPU tensors.

These are not PyG high-level API failures. `torch_geometric.nn.knn_graph`
calls `torch.ops.pyg.knn`, and `SplineConv` calls `pyg_lib.ops.spline_basis`
and `pyg_lib.ops.spline_weighting`.

In `src/pyg-lib`, the operators are defined in:

- `pyg_lib/csrc/ops/knn.cpp`
- `pyg_lib/csrc/ops/radius.cpp`
- `pyg_lib/csrc/ops/nearest.cpp`
- `pyg_lib/csrc/ops/fps.cpp`
- `pyg_lib/csrc/ops/cluster.cpp`
- `pyg_lib/csrc/ops/scatter.cpp`
- `pyg_lib/csrc/ops/spline.cpp`

Existing backend registrations are CPU and CUDA:

- `pyg_lib/csrc/ops/cpu/knn_kernel.cpp`
- `pyg_lib/csrc/ops/cuda/knn_kernel.cu`
- `pyg_lib/csrc/ops/cpu/radius_kernel.cpp`
- `pyg_lib/csrc/ops/cuda/radius_kernel.cu`
- `pyg_lib/csrc/ops/cpu/nearest_kernel.cpp`
- `pyg_lib/csrc/ops/cuda/nearest_kernel.cu`
- `pyg_lib/csrc/ops/cpu/fps_kernel.cpp`
- `pyg_lib/csrc/ops/cuda/fps_kernel.cu`
- `pyg_lib/csrc/ops/cpu/cluster_kernel.cpp`
- `pyg_lib/csrc/ops/cuda/cluster_kernel.cu`
- `pyg_lib/csrc/ops/cpu/scatter_kernel.cpp`
- `pyg_lib/csrc/ops/cuda/scatter_kernel.cu`
- `pyg_lib/csrc/ops/cpu/spline_kernel.cpp`
- `pyg_lib/csrc/ops/cuda/spline_kernel.cu`

The point-cloud and spline operators now have CPU-assisted MPS compatibility
shims under `pyg_lib/csrc/ops/mps/`, registered with
`TORCH_LIBRARY_IMPL(pyg, MPS, m)`. Dedicated Metal kernels remain the
performance-oriented follow-up.

Fallback remains useful as a compatibility switch while developing or testing
unpatched operators:

```bash
./scripts/uv_stage.sh pyg-lib-fallback
```

This starts Python with `PYTORCH_ENABLE_MPS_FALLBACK=1`, allowing missing MPS
ops to fall back to CPU when PyTorch supports fallback for that op.

Reproduce the current MPS compatibility milestone from a direct macOS shell:

```bash
./scripts/uv_stage.sh install-local-pyg-lib
./scripts/uv_stage.sh verify-mps
./scripts/uv_stage.sh extensions-mps
./scripts/uv_stage.sh summary
```

After the latest probe changes, native missing MPS kernels should appear as
`unsupported`, not generic `failed`.

## 2026-07-01: Native on-device int32 arg path for `scatter_min`/`scatter_max`

The earlier `scatter_min`/`scatter_max` shim reduced values on MPS but computed
the arg indices on CPU, because MPS raises `RuntimeError: not supported for
torch.int64` for `scatter_reduce_`. That CPU round-trip ran on every call, which
is a real cost in a message-passing training loop.

Direct-shell experiment on `torch==2.12.1` isolates the constraint to the dtype,
not the reduction:

- `scatter_reduce_(..., "amin")` with `torch.int64` on MPS: fails.
- `scatter_reduce_(..., "amin")` with `torch.int32` on MPS: works, with either
  int32 or int64 index tensors.
- `scatter_reduce_(..., "amax")` with `torch.float32` on MPS: works.

The MPS kernel now derives the arg index **entirely on-device** using int32:

- flag each source position whose value equals its group's reduced value
  (exact float equality is valid because `amin`/`amax` select an actual source
  element rather than computing a new value),
- take the smallest such position per group with an int32
  `scatter_reduce_(..., "amin")`,
- widen the int32 arg to int64 before returning, matching pyg-lib's contract.

A `TORCH_CHECK` guards the int32 range (`src.size(dim) < 2^31`), which holds for
any realistic graph. The patch keeps zero `.cpu()` calls on the min/max path.

Verified from a direct macOS shell:

- `pyg_lib.ops.scatter_min`/`scatter_max`: values and arg indices both return on
  `mps:0`; arg dtype is `int64`.
- Numerical parity vs the CPU `pyg-lib` kernel and vs `torch_scatter`
  (`tests/test_scatter_parity.py`, 26 cases): exact match, including tie
  first-occurrence semantics and empty-group `arg == src.size(dim)`.

The patch is regenerated into
`patches/pyg-lib/0002-add-scatter-family-mps-dispatch.patch`.

## 2026-07-01: Honest native vs CPU-assisted probe classification

The `21 ok` headline conflated true GPU execution with operators that merely
accept MPS tensors and run on CPU internally. An `ok` on `mps:0` only reflects
the final copy-back, not where the compute happened. The probe now tags every
successful MPS case with an `impl` field and the summary reports the split:

- `pyg-macos-native`: `21 ok = 12 native + 9 cpu-assisted`.
- Native (12): torch core ops, PyG `Data.to` and `GCNConv`, and the full
  `pyg-lib` scatter family (`scatter_sum`, `scatter_mul`, `scatter_mean`,
  `scatter_min`, `scatter_max`).
- CPU-assisted (9): PyG `knn_graph` and `SplineConv`, and direct `pyg-lib`
  `knn`, `radius`, `nearest`, `fps`, `grid_cluster`, `spline_basis`, and
  `spline_weighting`. These remain CPU-assisted by design; they are
  preprocessing ops that profiling has not shown to be hot.

## 2026-07-01: Fused single-pass Metal kernel for `scatter_min`/`scatter_max`

Profiling showed the int32 tensor-op arg path cost ~3x `scatter_sum` on MPS at
1M edges because it ran five generic kernels (scatter_reduce, gather, eq, where,
scatter_reduce). This was replaced with a hand-written Metal kernel.

Host: Apple M4 Pro (Apple9 GPU family), Metal 4. This GPU supports 64-bit
atomics, which is what makes a genuine single-pass fused argmax possible.

Kernel design (`pyg_lib/csrc/ops/mps/scatter_metal.mm`):

- Pack an order-preserving `uint` transform of the float value in the high 32
  bits and the bit-complemented source position in the low 32 bits of a 64-bit
  word, then `atomic_max` per element into the target cell. A larger value
  wins; on a tie, the complemented position makes the smallest source index win
  (first-occurrence, matching the CPU kernel). `min` stores `~transform` so the
  smallest value yields the largest key.
- A second cheap kernel unpacks the 64-bit keys into the value and int64 arg
  tensors. Empty cells (key high bits == 0) become value 0 and arg
  `src.size(dim)`, matching upstream.

Integration:

- Registered for MPS via `TORCH_LIBRARY_IMPL(pyg, MPS, m)` in the `.mm`; the
  tensor-op `scatter_min`/`scatter_max` registration was removed from
  `mps/scatter_kernel.cpp` to avoid a double registration.
- The kernel covers the hot path (2-D float32 `src`, `dim == 0`,
  column-broadcast index from a 1-D edge index). All other shapes/dtypes fall
  back to the portable int32 tensor path, which is retained in the `.mm`.
- CMake builds `mps/*.mm` as Objective-C++ on APPLE and links `Metal` and
  `Foundation`. Pipelines are compiled once (thread-safe function-local static);
  if 64-bit atomics or the Metal toolchain are unavailable, the kernel reports
  itself unavailable and callers use the tensor fallback.
- Tracked as `patches/pyg-lib/0004-add-fused-metal-scatter-kernel.patch` (plus
  the regenerated `0002`).

Correctness (`tests/test_scatter_parity.py`, 30 cases): exact value and arg
parity vs the CPU kernel and `torch_scatter`, including a heavy-contention
stress test (60k edges into 32 nodes, integer values forcing many ties) that
would expose an atomic race or bad tie-break, plus a fallback-routing test.

Performance (`benchmarks/REPORT.md`): the fused kernel is 4-30x faster than the
five-op path and 10-36x faster than CPU; at 1M edges it drops `scatter_max` from
177 ms to 5.9 ms. Native/CPU-assisted probe counts are unchanged (still 12/9);
`scatter_min`/`scatter_max` remain native but are now Metal-backed.

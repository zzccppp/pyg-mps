# Roadmap

## Project Goal

Make PyTorch Geometric practical on macOS Apple Silicon by providing:

- a `uv`-first install workflow,
- repeatable MPS compatibility probes,
- a documented fallback mode for missing native kernels,
- a verified compatibility layer for high-value `pyg-lib` operators,
- and, over time, dedicated native/Metal implementations where profiling shows
  the CPU-assisted path is not enough.

## Milestone 1: Install and Evidence Baseline

Status: complete for the current local development baseline.

Deliverables:

- `./scripts/uv_stage.sh bootstrap`
- `./scripts/uv_stage.sh verify-mps`
- JSON probe reports under `reports/`
- human-readable summaries under `docs/`
- a minimal GCN example that runs on MPS without optional custom kernels
- local editable builds for PyG and all legacy optional extensions
- initial `pyg-lib` Meta dispatch patch for scatter operators
- verified patched `pyg-lib` MPS path for the current 21-case probe suite

Success criteria:

- PyTorch MPS is detected from a direct macOS shell.
- PyG core imports and runs `GCNConv` forward/backward on MPS.
- `pyg-lib` installs through `uv`.
- Native MPS gaps are classified as `unsupported`, not mixed into generic
  failures.
- Local CPU and Meta probes pass after building from source.
- The direct-shell no-fallback MPS probe reports `21 ok / 0 unsupported /
  0 failed` for the patched `pyg-lib`/PyG path.

## Milestone 2: Compatibility Distribution

Status: active. The patched local source stack now passes the direct-shell MPS
probe with and without PyTorch's global MPS fallback.

Deliverables:

- one-command install plus native/fallback verification,
- wrapper guidance for running PyG workloads with or without
  `PYTORCH_ENABLE_MPS_FALLBACK=1`,
- test coverage for common PyG workloads that rely on patched `pyg-lib`
  operators,
- clear documentation that CPU-assisted shims are compatibility support, not
  final performance kernels.

Success criteria:

- Representative PyG models run end-to-end on macOS from a clean `uv`
  environment.
- Reports clearly identify which ops are fully MPS-backed, CPU-assisted, or
  still unsupported.
- The same verification command works on a fresh Apple Silicon machine.

## Milestone 3: Native MPS Kernel Prototypes

Status: compatibility milestone complete for the current probe suite. The full
scatter family is verified on real MPS and is now **fully native**:
`scatter_min` and `scatter_max` reduce values with MPS `scatter_reduce` and
derive arg indices entirely on-device in int32 (working around Metal's missing
int64 `scatter_reduce`), with numerical parity tests against CPU and
`torch_scatter`. Point-cloud and spline operators remain CPU-assisted MPS
dispatch shims by design. The probe now reports the native vs CPU-assisted
split explicitly: `21 ok = 12 native + 9 cpu-assisted`.

Candidate first kernels:

- `pyg::scatter_sum`: verified. The first implementation uses PyTorch's native
  MPS `scatter_add_` as the backend implementation.
- `pyg::scatter_mul`: verified through PyTorch's MPS `scatter_reduce_` path.
- `pyg::scatter_mean`: verified through the composite implementation over
  `scatter_sum`.
- `pyg::scatter_min`, `pyg::scatter_max`: value reduction uses MPS
  `scatter_reduce_`; argindex computation runs on-device in int32 because MPS
  does not support the int64 reduction path, then widens to int64. No CPU
  round-trip. Verified on real MPS with parity tests.
- `pyg::knn`, `pyg::radius`, `pyg::nearest`, `pyg::fps`, `pyg::grid_cluster`:
  CPU-assisted MPS shims are verified on real MPS.
- `pyg::spline_basis` and `pyg::spline_weighting`: CPU-assisted MPS shims
  are verified on real MPS, including the high-level `SplineConv` path.

Implementation direction:

- Use Meta kernels for shape-only validation and fake tensor compatibility.
- Add `pyg_lib/csrc/ops/mps/*_kernel.cpp` compatibility shims first; reserve
  `.mm`/Metal implementations for profiling-driven performance work.
- Register kernels with `TORCH_LIBRARY_IMPL(pyg, MPS, m)`.
- Match CPU semantics first, then optimize.
- Add parity tests against CPU outputs.
- Replace CPU-assisted shims with Metal kernels selectively based on profiling
  and model impact.

## Milestone 4: Packaging

Status: planned.

Deliverables:

- reproducible macOS wheel build notes,
- `uv` install documentation,
- version matrix for Python, PyTorch, macOS, and Apple Silicon chips.

Success criteria:

- A fresh macOS arm64 machine can install the stack and run the verification
  suite with one command.

# Resume Notes

## Project Title

PyG macOS MPS Port

## Short Description

Built a macOS/Apple Silicon compatibility layer and diagnostic workflow for
PyTorch Geometric, using `uv` to automate installation, MPS validation, and
operator-level failure classification.

## Resume Bullets

- Built a `uv`-first PyTorch Geometric macOS workspace that installs PyTorch,
  PyG, and `pyg-lib` from source and validates Apple MPS execution with
  reproducible staged probes.
- Diagnosed PyG/MPS failures at the PyTorch dispatcher level, separating ops
  with native MPS kernels from those blocked by Metal's lack of int64 integer
  reductions, and registered the missing backends via `TORCH_LIBRARY_IMPL(pyg,
  MPS/Meta, m)`.
- Implemented native-MPS scatter reductions for `pyg-lib`; engineered an
  int32-based **on-device** argmin/argmax to bypass MPS's missing int64
  `scatter_reduce`, eliminating a per-call CPU round-trip in the message-passing
  hot path.
- Built a numerical parity harness (MPS vs CPU kernel vs `torch_scatter`)
  covering dtypes, tie-breaking, empty groups, and multi-dimensional inputs, and
  a probe that classifies each operator as native / CPU-assisted / unsupported
  to drive kernel-porting priorities.
- Implemented CPU-assisted MPS dispatch shims for rarely-hot point-cloud and
  spline operators (`knn`, `radius`, `fps`, `spline_basis`, ...) as an explicit,
  documented engineering trade-off rather than a performance endpoint.
- Documented remaining legacy optional-extension gaps in `torch_cluster`,
  `torch_sparse`, and `torch_spline_conv` when called directly with MPS tensors.

Note: of the 21 passing MPS probe cases, 12 execute natively on the GPU and 9
are CPU-assisted. Prefer the native/CPU-assisted framing over a bare "21/21"
number, which an interviewer who knows PyTorch will (correctly) probe.

## Interview Talking Points

- PyTorch custom ops are dispatched by backend keys such as CPU, CUDA, and MPS.
- PyG core can run on MPS when it uses standard PyTorch tensor operations.
- `pyg-lib` wheels may install successfully on macOS while still lacking MPS
  kernels for custom operators.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is useful for compatibility, but native MPS
  kernels are needed for performance and clean backend support.
- Metal/MPS has no int64 `scatter_reduce`; argmin/argmax for `scatter_min/max`
  is therefore computed on-device in int32 (safe because graph indices are far
  below 2^31) and widened to int64 on the way out. This is the key trick that
  turns the weakest op into a fully native one without a CPU round-trip.
- A hardcoded per-op CPU shim is functionally identical to the global MPS
  fallback; the honest way to report it is as a distinct "CPU-assisted" class,
  not as a native MPS win.
- CPU-assisted dispatch shims are useful for API compatibility and project
  adoption, while profiling should decide which ops deserve dedicated Metal
  kernels first.

# Resume Notes

## Project Title

PyG macOS MPS Port

## Short Description

Built a macOS/Apple Silicon compatibility layer and diagnostic workflow for
PyTorch Geometric, using `uv` to automate installation, MPS validation, and
operator-level failure classification.

## Resume Bullets

- Built a `uv`-first PyTorch Geometric macOS workspace that installs PyTorch,
  PyG, and `pyg-lib`, then validates Apple MPS execution with reproducible
  staged probes.
- Diagnosed PyG/MPS compatibility at the PyTorch custom-operator dispatch
  level, separating working PyG core message passing from missing native
  `pyg-lib` MPS kernels.
- Implemented and verified a patched `pyg-lib` path that passes a 21-case
  direct-shell MPS probe suite, including PyG `knn_graph`, `SplineConv`,
  point-cloud ops, spline ops, and scatter-family ops.
- Created a roadmap for native MPS kernel support via `TORCH_LIBRARY_IMPL(pyg,
  MPS, m)` registrations and parity tests against CPU implementations.
- Built local editable PyG extension wheels from source, added initial
  `pyg-lib` Meta dispatch support for scatter operators, and verified MPS
  dispatch for the full scatter family with CPU-assisted argindex support for
  `scatter_min`/`scatter_max`.
- Prototyped CPU-assisted MPS dispatch shims for point-cloud and spline
  operators to preserve MPS-facing tensors while reusing proven CPU kernels.
- Documented remaining legacy optional-extension gaps in `torch_cluster`,
  `torch_sparse`, and `torch_spline_conv` when called directly with MPS tensors.

## Interview Talking Points

- PyTorch custom ops are dispatched by backend keys such as CPU, CUDA, and MPS.
- PyG core can run on MPS when it uses standard PyTorch tensor operations.
- `pyg-lib` wheels may install successfully on macOS while still lacking MPS
  kernels for custom operators.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is useful for compatibility, but native MPS
  kernels are needed for performance and clean backend support.
- CPU-assisted dispatch shims are useful for API compatibility and project
  adoption, while profiling should decide which ops deserve dedicated Metal
  kernels first.

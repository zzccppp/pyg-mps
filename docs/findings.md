# Findings

## 2026-07-01: `pyg-lib` MPS operator gap

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

Current native MPS gaps after installing `pyg-lib`:

- `pyg::knn`
- `pyg::spline_basis`

These are not PyG high-level API failures. `torch_geometric.nn.knn_graph`
calls `torch.ops.pyg.knn`, and `SplineConv` calls `pyg_lib.ops.spline_basis`
and `pyg_lib.ops.spline_weighting`.

In `src/pyg-lib`, the operators are defined in:

- `pyg_lib/csrc/ops/knn.cpp`
- `pyg_lib/csrc/ops/spline.cpp`

Existing backend registrations are CPU and CUDA:

- `pyg_lib/csrc/ops/cpu/knn_kernel.cpp`
- `pyg_lib/csrc/ops/cuda/knn_kernel.cu`
- `pyg_lib/csrc/ops/cpu/spline_kernel.cpp`
- `pyg_lib/csrc/ops/cuda/spline_kernel.cu`

No MPS registration exists yet for these operators. The likely native-support
path is to add MPS dispatch implementations in `pyg-lib`, for example under
`pyg_lib/csrc/ops/mps/`, and register them with `TORCH_LIBRARY_IMPL(pyg, MPS,
m)`.

Short-term workaround to test pipeline viability:

```bash
./scripts/uv_stage.sh pyg-lib-fallback
```

This starts Python with `PYTORCH_ENABLE_MPS_FALLBACK=1`, allowing missing MPS
ops to fall back to CPU when PyTorch supports fallback for that op.

Next direct-shell probe:

```bash
./scripts/uv_stage.sh pyg-lib
./scripts/uv_stage.sh summary
```

After the latest probe changes, native missing MPS kernels should appear as
`unsupported`, not generic `failed`.

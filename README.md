# PyG MPS Investigation

This workspace is for isolating when PyTorch MPS support remains healthy and
where PyG optional dependencies fall back to CPU or fail on MPS tensors.

Environment and dependency management should go through `uv`.

## Quick Start

Create the clean Python 3.12 environment:

```bash
./scripts/uv_stage.sh init
```

Install and probe one stage at a time:

```bash
./scripts/uv_stage.sh torch
./scripts/uv_stage.sh pyg-core
./scripts/uv_stage.sh torch-scatter
./scripts/uv_stage.sh torch-sparse
./scripts/uv_stage.sh torch-cluster
./scripts/uv_stage.sh torch-spline-conv
```

Run a probe without installing anything:

```bash
./scripts/uv_stage.sh probe manual
```

Summarize all JSON probe reports:

```bash
./scripts/uv_stage.sh summary
```

Probe the same stage with PyTorch's CPU fallback enabled for missing MPS ops:

```bash
./scripts/uv_stage.sh probe-fallback pyg-lib-fallback
```

## MPS Sandbox Caveat

MPS availability must be validated from a direct outer shell, not from a
Codex-nested sandbox shell.

On this host, the same Python environment can report different MPS status:

- `CODEX_SANDBOX=None`: MPS is visible.
- `CODEX_SANDBOX=seatbelt`: MPS may become a false negative.

When `CODEX_SANDBOX=seatbelt`, `torch.backends.mps.is_available()` can return
`False`, `torch.mps.device_count()` can return `0`, and the first real MPS
operation can fail with a misleading macOS-version error. Treat such reports as
sandbox-contaminated, not as evidence that PyTorch, PyG, or the host GPU lacks
MPS support.

## Suggested Stages

1. `torch`
   - Create a clean virtual environment.
   - Install only PyTorch.
   - Run the probe.

2. `pyg-core`
   - Install `torch-geometric` without optional native extensions.
   - Run the same probe.

3. `pyg-lib`
   - Add `pyg-lib` if available for the local PyTorch and Python version.
   - Run the same probe.

4. `torch-scatter`
   - Add `torch-scatter`.
   - Run the same probe.

5. `torch-sparse`
   - Add `torch-sparse`.
   - Run the same probe.

6. `torch-cluster`
   - Add `torch-cluster`.
   - Run the same probe.

7. `torch-spline-conv`
   - Add `torch-spline-conv`.
   - Run the same probe.

## Source Layout

Keep source clones under `src/`:

- `src/pytorch_geometric`
- `src/torch_cluster`
- `src/torch_spline_conv`
- `src/torch_scatter`
- `src/torch_sparse`
- `src/pyg_lib`

Keep machine-readable probe results under `reports/`.

## Notes

- Run each stage in a fresh environment when possible.
- Record exact Python, PyTorch, macOS, and Apple Silicon details.
- Treat import success, CPU success, and MPS tensor success as separate results.
- Treat `skipped` as an expected missing optional dependency for the current
  stage; investigate only `failed` cases.
- Treat `unsupported` as an installed operator that is missing a native MPS
  implementation. This is the main category for MPS enablement work.

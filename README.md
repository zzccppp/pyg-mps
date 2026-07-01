# PyG macOS MPS Port

This project is a macOS-focused compatibility and porting workspace for
PyTorch Geometric. It aims to make PyG easier to install with `uv`, verify what
already runs on Apple's MPS backend, and identify the native PyG kernels that
need MPS implementations.

The current local source build has reached a compatibility milestone: PyTorch
MPS, PyG core message passing, and the probed `pyg-lib` scatter, point-cloud,
and spline operators run with MPS-facing tensors on Apple Silicon. The current
patches prioritize correctness and installability; some operators use
CPU-assisted MPS dispatch shims and remain candidates for dedicated Metal
kernels.

## Quick Start

Use a direct macOS terminal for MPS validation. Codex sandbox shells can make
MPS appear unavailable even when it works on the host.

Install the current PyG/MPS stack and run both native and fallback probes:

```bash
./scripts/uv_stage.sh bootstrap
```

Run the minimal GCN example:

```bash
./scripts/uv_stage.sh example
```

Summarize all JSON probe reports:

```bash
./scripts/uv_stage.sh summary
```

## Current Results

Direct-shell MPS results on this host:

- `torch==2.12.1`: MPS available, device count 1.
- Upstream wheel baseline: `torch-geometric==2.8.0` runs `Data.to("mps")` and
  `GCNConv` forward/backward, while `pyg-lib==0.7.0+pt212` installs but lacks
  MPS kernels for several custom operators.
- Local editable source stack: `torch-geometric==2.9.0`, `pyg-lib==0.8.0`,
  `torch-scatter==2.1.2`, `torch-sparse==0.6.18`, `torch-cluster==1.6.3`,
  and `torch-spline-conv==1.2.2`.
- Local patched `pyg-lib==0.8.0`: the no-fallback `pyg-macos-native` probe now
  reports `21 ok / 0 unsupported / 0 failed` on MPS.
- The verified path includes PyG high-level `knn_graph` and `SplineConv`, plus
  direct `pyg-lib` point-cloud, spline, and scatter-family operators returning
  tensors on `mps:0`.
- `PYTORCH_ENABLE_MPS_FALLBACK=1`: the same probe also reports
  `21 ok / 0 unsupported / 0 failed`, but fallback is no longer required for
  the probed patched `pyg-lib` path.
- Legacy optional extension packages still have independent MPS gaps:
  `torch_cluster`, `torch_sparse`, and `torch_spline_conv` can still reach
  CPU-only kernels when called directly with MPS tensors.

See [docs/findings.md](docs/findings.md) for the evidence trail and
[docs/roadmap.md](docs/roadmap.md) for the porting plan.

## Staged Commands

Create the clean Python 3.12 environment:

```bash
./scripts/uv_stage.sh init
```

Install and probe one stage at a time:

```bash
./scripts/uv_stage.sh torch
./scripts/uv_stage.sh pyg-core
./scripts/uv_stage.sh pyg-lib
./scripts/uv_stage.sh pyg-lib-fallback
```

Build from local source checkouts:

```bash
./scripts/uv_stage.sh sync-sources
./scripts/uv_stage.sh install-local
./scripts/uv_stage.sh install-local-optionals
./scripts/uv_stage.sh meta
./scripts/uv_stage.sh extensions-cpu
```

`install-local` installs editable `torch-geometric` and `pyg-lib` from `src/`.
The `pyg-lib` build automatically applies patches from `patches/pyg-lib/`
before compiling.

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
- `src/pytorch_cluster`
- `src/pytorch_spline_conv`
- `src/pytorch_scatter`
- `src/pytorch_sparse`
- `src/pyg-lib`

Keep machine-readable probe results under `reports/`.
Keep source patches that should survive a fresh checkout under `patches/`.

## Notes

- Run each stage in a fresh environment when possible.
- Record exact Python, PyTorch, macOS, and Apple Silicon details.
- Treat import success, CPU success, and MPS tensor success as separate results.
- Treat `skipped` as an expected missing optional dependency for the current
  stage; investigate only `failed` cases.
- Treat `unsupported` as an installed operator that is missing a native MPS
  implementation. This is the main category for MPS enablement work.

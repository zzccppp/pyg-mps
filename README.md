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

## Install in your own project

The patched `pyg-lib` (with the native + fused Metal MPS scatter kernels) lives
on a fork branch and installs into any fresh `uv` project. This recipe is tested
end-to-end on Apple Silicon:

```bash
# 1. Clone the fork. METIS has a nested GKlib submodule, so --recursive is
#    required; only these two submodules are needed for the macOS build.
git clone -b macos-mps-scatter https://github.com/zzccppp/pyg-lib.git
git -C pyg-lib submodule update --init --recursive \
    third_party/METIS third_party/parallel-hashmap

# 2. Fresh uv environment with torch + build tools.
uv venv --python 3.12
uv pip install torch setuptools wheel ninja

# 3. Build + install pyg-lib. --no-build-isolation is required because
#    pyg-lib's setup.py imports torch at build time.
uv pip install --no-build-isolation ./pyg-lib
```

Then the fused Metal scatter kernel runs on MPS:

```python
import torch
from pyg_lib import ops

src = torch.randn(50_000, 32, device="mps")
index = torch.randint(0, 500, (50_000,), device="mps")
value, argmax = ops.scatter_max(src, index, dim=0, dim_size=500)  # both on mps:0
```

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
  reports `21 ok / 0 unsupported / 0 failed` on MPS. Crucially, the probe splits
  those 21 `ok` cases by *how* they execute: **12 native** (GPU execution via
  standard PyTorch or native `pyg-lib` MPS kernels) and **9 cpu-assisted** (the
  operator accepts MPS tensors but copies to CPU and back). A bare `ok` on
  `mps:0` does not by itself prove GPU execution, so the reports make the
  distinction explicit.
- Native MPS kernels: the full `pyg-lib` scatter family (`scatter_sum`,
  `scatter_mul`, `scatter_mean`, `scatter_min`, `scatter_max`). `scatter_min`
  and `scatter_max` reduce values with MPS `scatter_reduce` and derive the arg
  indices **entirely on-device in int32**, working around Metal's lack of an
  int64 `scatter_reduce` kernel; the int32 arg is widened to int64 before
  returning. Numerical parity against the CPU kernel and `torch_scatter` is
  covered by `tests/test_scatter_parity.py`.
- CPU-assisted shims (intentional, for rarely-hot preprocessing ops): PyG
  `knn_graph` and `SplineConv`, plus direct `pyg-lib` `knn`, `radius`,
  `nearest`, `fps`, `grid_cluster`, `spline_basis`, and `spline_weighting`.
  These are candidates for dedicated Metal kernels only if profiling shows they
  matter.
- `PYTORCH_ENABLE_MPS_FALLBACK=1`: the same probe also reports
  `21 ok / 0 unsupported / 0 failed`, but fallback is no longer required for
  the probed patched `pyg-lib` path.
- Legacy optional extension packages still have independent MPS gaps:
  `torch_cluster`, `torch_sparse`, and `torch_spline_conv` can still reach
  CPU-only kernels when called directly with MPS tensors.

See [docs/findings.md](docs/findings.md) for the evidence trail and
[docs/roadmap.md](docs/roadmap.md) for the porting plan.

## Benchmarks

[benchmarks/REPORT.md](benchmarks/REPORT.md) quantifies the MPS scatter kernels.
Headlines on this host (macOS 26.5.1, Apple M4 Pro, `torch==2.12.1`):

- `scatter_min`/`scatter_max` use a **fused single-pass Metal kernel**
  (`pyg_lib/csrc/ops/mps/scatter_metal.mm`) that computes value and arg together
  via a 64-bit atomic. It is **4–30× faster than the previous tensor-op path**
  and **10–36× faster than CPU**, dropping `scatter_max` at 1M edges from 177 ms
  to 5.9 ms.
- `scatter_sum`/`scatter_mean` use PyTorch's native `scatter_add_` and only edge
  past CPU beyond ~500k edges — a purpose-built kernel for the hot path is where
  the real win is.

Regenerate with `./scripts/uv_stage.sh benchmark` then `./scripts/uv_stage.sh report`.

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

Run the scatter-family numerical parity suite (MPS vs CPU vs `torch_scatter`):

```bash
./scripts/uv_stage.sh parity
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

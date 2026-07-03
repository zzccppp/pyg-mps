# PyG macOS/MPS Port — Project State

_Last updated: 2026-07-02_

## What this project is

A port of **PyTorch Geometric** to **macOS / Apple Silicon (MPS)**: a `uv`-first
install, native + hand-written Metal kernels for the hot `pyg-lib` operators, and
a binary-wheel distribution. Goal: make PyG usable and fast on Apple GPUs, and
serve as a deep-dive PyTorch/Metal portfolio project.

## Location & environment

- **Project moved to `~/Developer/PyG_MacOS`** (out of `~/Documents`, which is
  iCloud-synced — iCloud was corrupting `.venv` and creating `" 2"` conflict
  files). **Do not put this project back under `~/Documents`, `~/Desktop`, or any
  iCloud-synced folder.**
- Host: Apple **M4 Pro** (Apple9 GPU family, Metal 4, supports 64-bit atomics),
  macOS 26.x, PyTorch **2.12.1**, Python **3.12**.

### Recreate the dev environment
```bash
cd ~/Developer/PyG_MacOS
export UV_CACHE_DIR="$PWD/.uv-cache"
uv venv --python 3.12 .venv
uv pip install torch numpy pytest
# If a rebuild ever fails after moving/renaming: rm -rf src/pyg-lib/build
uv pip install --no-build-isolation src/pyg-lib   # non-editable! (see gotchas)
.venv/bin/python -m pytest tests/ -q               # expect 82 passed, 2 skipped
```

## GitHub repos

| Repo | Branch | Purpose |
|---|---|---|
| **`zzccppp/pyg-lib`** (fork of `pyg-team/pyg-lib`) | `macos-mps-scatter` (default) | The installable patched pyg-lib with all MPS/Metal kernels |
| **`zzccppp/pyg-mps`** | `main` | The workspace: patches, scripts, tests, benchmarks, docs |

- Local `src/pyg-lib` is on branch `macos-mps-scatter` with remotes
  `fork` (git@github.com:zzccppp/pyg-lib.git, SSH) and `origin` (upstream).
- Workspace remote: `origin` = git@github.com:zzccppp/pyg-mps.git.
- `gh` is authenticated as **zzccppp** (SSH). Pushes/releases work directly.

## Distribution (DONE)

- **Binary wheels** install with no compiler/submodules:
  ```bash
  uv pip install torch==2.12.1
  uv pip install https://github.com/zzccppp/pyg-lib/releases/download/v0.8.1-mps/pyg_lib-0.8.1+pt212-cp312-cp312-macosx_11_0_arm64.whl
  ```
- **Releases**: `v0.8.0-mps` (scatter only), `v0.8.1-mps` (adds COO/CSR
  segment+gather), `v0.8.2-mps` (adds softmax_csr, fused sampled_op,
  grouped/segment_matmul).
- **CI**: `.github/workflows/build-macos-mps-wheels.yml` on the fork builds a
  `python {3.10–3.13} × torch {2.11,2.12}` wheel matrix on macOS runners and
  attaches wheels to the GitHub Release on any `v*` tag. Validated green (8/8).
- Wheels are torch-minor-version-specific; version suffix via
  `PYG_LIB_VERSION_SUFFIX=+ptXYZ` (e.g. `+pt212`).

## Operator MPS coverage

| Operator(s) | Status | Implementation |
|---|---|---|
| `scatter_sum`, `scatter_mul`, `scatter_mean` | **native** | PyTorch `scatter_add_`/`scatter_reduce_` |
| `scatter_min`, `scatter_max` | **native (Metal)** | **Fused single-pass Metal kernel** (`mps/scatter_metal.mm`): 64-bit atomic packs order-preserving value + `~index`; value & arg in one pass; arbitrary dim/rank; f32/f16/bf16. 4–30× vs tensor path, 10–36× vs CPU. |
| `segment_{sum,mean,min,max}_coo`, `gather_coo` | **native** | Delegate to scatter (COO segment = scatter along `dim-1`); `gather_coo`=`index_select` (`mps/segment_coo_kernel.cpp`) |
| `segment_{sum,mean,min,max}_csr`, `gather_csr` | **native (Metal)** | **Dedicated atomic-free per-row Metal kernel** (`mps/segment_csr_metal.mm`); `gather_csr`=`repeat_interleave`+`index_select` |
| `index_sort` | native | already worked on MPS |
| knn/radius/nearest/fps/grid_cluster, spline_* | CPU-assisted | shims in `mps/point_cloud_kernel.cpp`, `mps/spline_kernel.cpp` |
| `softmax_csr` (+ backward) | **native (composite)** | On-device composite over the CSR kernels (`mps/softmax_csr_kernel.cpp`): `segment_max_csr`→`gather_csr`→exp→`segment_sum_csr`→`gather_csr`→divide. Mirrors CPU max→sub→exp→sum→div (singleton→1.0, empty→no output); hot path `dim==0`/1-D `ptr`/f32-f16-bf16, else CPU fallback. ~11.5× vs naive-tensor MPS, ~12.6× vs CPU at 1M edges. |
| `sampled_add/sub/mul/div` | **native (Metal)** | **Fused gather+arith Metal kernel** (`mps/sampled_metal.mm`): one thread per output cell resolves `left[li[m]]`/`right[ri[m]]` on the fly + applies the op in a single pass (no materialized gathered operands). Hot path 2-D contiguous / f32-f16-bf16 / 1-D int64 index; else native `index_select` composite. Backward composes `sampled_op`+`index_select_backward`. ~2.5× vs composite, ~4.3× vs CPU at 1M edges. |
| **`spmm_csr`** (NEW op) | **native (Metal)** | **Fused GNN neighbor-aggregation kernel** (`mps/spmm_metal.mm`): `out[i]=⊕_e w[e]·x[col[e]]` over each node's CSR edge range, `⊕`∈{sum,mean,max} + optional edge weight; atomic-free one-thread-per-row, no `[E,F]` message tensor. Replaces PyG's gather+`scatter_add` aggregation. **47× vs MPS `scatter_add`, 13× vs CPU at 1M edges.** New op (not in upstream pyg-lib); Python autograd (transpose SpMM) lives in the bench project. |
| **`spmm_max_csr`** (NEW op) | **native (Metal)** | Max-reducing SpMM returning `(out, arg)` where `arg[i,f]`=winning source node `col[e*]` (first on ties, sentinel `x.size(0)` for empty rows) so a GraphSAGE-max layer's backward routes grad to that neighbor (`mps/spmm_metal.mm`). Atomic-free per-row Metal (f32/f16/bf16) + CPU. Enables fused GraphSAGE-max on MPS. Backward `spmm_max_csr_bw` (fused **atomic float-add** scatter, `grad_x[arg]+=grad_out`; ~1.7× vs native `scatter_add`) makes the max path fully fused on-device. |
| `grouped_matmul` / `segment_matmul` | **native (MPS GEMM)** | Dispatch to Apple's GEMM via `at::matmul`/`mm`/`bmm` (`mps/matmul_kernel.cpp`), no shader. `grouped_matmul` loops per pair; `segment_matmul` uses a batched `bmm` fast path when all segments share a positive row count (zero-copy `[G,m,K]×[G,K,N]` views), else a per-segment `mm` loop (empty segments skipped). Backward composes automatically. Batched `bmm` ~flat vs a linearly-scaling loop (~11× at 256 uniform segments). |

All implemented ops have **exact value+arg parity** vs the CPU kernel (incl.
empty segments, ties, f32/f16/bf16): `tests/test_scatter_parity.py` (58),
`tests/test_segment_parity.py` (24), `tests/test_softmax_parity.py` (7,
forward+backward+row-sum+singleton+fallback), `tests/test_sampled_parity.py`
(53: 4 ops × index-modes × dtypes fwd, per-op backward, int32 fallback),
`tests/test_matmul_parity.py` (10: grouped + segment fwd/backward, uniform/
ragged/empty segments). Benchmarks + charts in `benchmarks/`
(softmax: `scripts/benchmark_softmax.py`; sampled: `scripts/benchmark_sampled.py`;
matmul: `scripts/benchmark_matmul.py`).

## Key source files (in `src/pyg-lib/pyg_lib/csrc/ops/`)

- `mps/scatter_metal.mm` — fused Metal scatter_min/max (+ int32 tensor fallback).
- `mps/scatter_kernel.cpp` — native scatter_sum/mul MPS.
- `mps/meta/scatter_kernel.cpp` — Meta (fake-tensor) dispatch.
- `mps/segment_coo_kernel.cpp` — COO segment/gather + gather_csr.
- `mps/segment_csr_metal.mm` — CSR segment Metal kernel.
- `mps/point_cloud_kernel.cpp`, `mps/spline_kernel.cpp` — CPU-assisted shims.
- Workspace patches mirror these in `patches/pyg-lib/0001..0006`.

## Gotchas / conventions

- **Non-editable install** (`uv pip install --no-build-isolation src/pyg-lib`,
  no `-e`). The editable finder was flaky under iCloud; non-editable copies the
  package and is robust. Each source edit → reinstall (~50 s rebuild).
- After moving/renaming the repo, **delete `src/pyg-lib/build`** (cmake caches
  absolute paths) before rebuilding.
- Fresh clone of the fork needs `git submodule update --init --recursive
  third_party/METIS third_party/parallel-hashmap` (METIS has a nested `GKlib`).
- Metal fast paths cover the hot cases (2-D-ish, `dim=0` for scatter, 1-D
  indptr for CSR, f32/f16/bf16); everything else falls back correctly.
- MPS timing: warmup + `torch.mps.synchronize()` around each call (see
  `scripts/benchmark_scatter.py`).
- Commit style: Conventional Commits. Commit to **both** the fork (canonical
  source) and the workspace (regenerate the matching `patches/pyg-lib/000N`).

## Cheat-sheet commands
```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
uv pip install --no-build-isolation src/pyg-lib     # rebuild after kernel edits
.venv/bin/python -m pytest tests/ -q                # parity
.venv/bin/python scripts/benchmark_scatter.py       # benchmark
gh run list --repo zzccppp/pyg-lib --limit 3        # CI status
```

## Next steps (priority order)
1. ~~`softmax_csr` — native composite over the CSR reduction.~~ **DONE**
   (`mps/softmax_csr_kernel.cpp`, patch `0007`, fork commit `cee9e29`).
2. ~~`sampled_add/sub/mul/div` — fused gather+arith Metal kernel.~~ **DONE**
   (`mps/sampled_metal.mm`, patch `0008`, fork commit `12df957`).
3. ~~`grouped_matmul` / `segment_matmul` — batched MPS matmul.~~ **DONE**
   (`mps/matmul_kernel.cpp`, patch `0009`, fork commit `9dce9ae`).
4. ~~Bump version, tag `v0.8.2-mps` (CI auto-builds wheels).~~ **DONE**
   (fork `0928cf1`, tag `v0.8.2-mps` pushed; CI matrix building).
5. Optionally update `docs/findings.md` / `docs/roadmap.md` with the segment +
   softmax/sampled/matmul work.
6. Remaining `pyg-lib` ops still CPU-assisted (knn/radius/fps/cluster/spline)
   are candidates for future dedicated Metal kernels.

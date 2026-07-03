#!/usr/bin/env python3
"""Benchmark the fused ``spmm_csr`` GNN-aggregation kernel on Apple Silicon.

GNN neighbor aggregation is ``out[i] = REDUCE_e weight[e] * x[col[e]]``. PyG runs
it on MPS as gather + ``scatter_add``, whose MPS ``scatter_add`` is pathologically
slow under the index collisions every graph produces. ``spmm_csr`` fuses the
whole thing into one atomic-free per-row pass. This benchmark compares, at
message-passing scale:

1. ``scatter_add`` path (what PyG uses today on MPS),
2. gather + ``segment_sum_csr`` (our earlier CSR reduction, still materializes
   the ``[E, F]`` message),
3. ``spmm_csr`` (fully fused, no message tensor),
4. the same aggregation on CPU (reference).

Timing is MPS-aware (warmup + ``torch.mps.synchronize()``). Results are written
as JSON for ``scripts/plot_benchmarks.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from pyg_lib import ops

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EDGE_COUNTS = [100_000, 500_000, 1_000_000, 2_000_000]
FANOUT = 10
FEATURE_DIM = 128


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def time_fn(fn: Callable[[], Any], device: str, warmup: int,
            iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples: list[float] = []
    for _ in range(iters):
        s = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - s) * 1e3)
    return {"ms_median": statistics.median(samples),
            "ms_mean": statistics.fmean(samples),
            "ms_stdev": statistics.pstdev(samples)}


def make_csr(edges: int, nodes: int, device: str, dtype: torch.dtype):
    g = torch.Generator().manual_seed(edges)
    x = torch.randn(nodes, FEATURE_DIM, generator=g, dtype=torch.float32)
    tgt = torch.randint(0, nodes, (edges,), generator=g).sort().values
    col = torch.randint(0, nodes, (edges,), generator=g).long()
    deg = torch.bincount(tgt, minlength=nodes)
    indptr = torch.cat([deg.new_zeros(1), deg.cumsum(0)]).long()
    w = torch.randn(edges, generator=g, dtype=torch.float32)
    row = torch.repeat_interleave(torch.arange(nodes), deg)
    return (x.to(device=device, dtype=dtype), indptr.to(device), col.to(device),
            w.to(device=device, dtype=dtype), row.to(device), nodes)


def build_cases(device: str):
    cases: dict[str, Callable] = {}

    def scatter_add_path(x, indptr, col, w, row, n):
        msg = x.index_select(0, col) * w.unsqueeze(-1)
        out = torch.zeros(n, x.size(1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, row.unsqueeze(-1).expand(-1, x.size(1)), msg)
        return out

    def gather_segment(x, indptr, col, w, row, n):
        msg = x.index_select(0, col) * w.unsqueeze(-1)
        return ops.segment_sum_csr(msg, indptr)

    def spmm(x, indptr, col, w, row, n):
        return ops.spmm_csr(x, indptr, col, w, "sum")

    cases["scatter_add (PyG path)"] = scatter_add_path
    if device == "mps":
        cases["gather+segment_csr"] = gather_segment
    cases["spmm_csr (fused)"] = spmm
    return cases


def run(edge_counts, warmup, iters, dtype) -> dict[str, Any]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    results: list[dict[str, Any]] = []
    for edges in edge_counts:
        nodes = max(1, edges // FANOUT)
        for device in devices:
            x, indptr, col, w, row, n = make_csr(edges, nodes, device, dtype)
            for label, fn in build_cases(device).items():
                # scatter_add is the only case worth timing on CPU too
                if device == "cpu" and label != "scatter_add (PyG path)":
                    continue
                t = time_fn(lambda: fn(x, indptr, col, w, row, n), device,
                            warmup, iters)
                results.append({"op": label, "device": device, "edges": edges,
                                "nodes": nodes, **t})
                logger.info("%-26s %-4s edges=%-9d %8.3f ms", label, device,
                            edges, t["ms_median"])
    return {"meta": {"torch_version": torch.__version__,
                     "platform": platform.platform(),
                     "machine": platform.machine(), "dtype": str(dtype),
                     "feature_dim": FEATURE_DIM, "fanout": FANOUT,
                     "warmup": warmup, "iters": iters, "devices": devices},
            "results": results}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="benchmarks/spmm_results.json")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--edges", type=int, nargs="*", default=DEFAULT_EDGE_COUNTS)
    args = p.parse_args()
    report = run(args.edges, args.warmup, args.iters, torch.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

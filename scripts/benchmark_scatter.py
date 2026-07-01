#!/usr/bin/env python3
"""Benchmark the pyg-lib MPS scatter family on Apple Silicon.

Three questions drive these benchmarks:

1. Does the native MPS scatter kernel beat CPU as graphs grow? (scatter_sum,
   scatter_mean, scatter_max on MPS vs CPU.)
2. What does the on-device int32 arg path buy over the earlier approach that
   reduced values on MPS but computed arg indices via a CPU round-trip?
   (scatter_max: ``native_mps`` vs ``mps_cpu_arg``.)
3. How large is the win in absolute wall-clock terms at message-passing scale?

Timing is MPS-aware: every measured region is bracketed by
``torch.mps.synchronize()`` because MPS dispatch is asynchronous, and each
configuration is warmed up before timing to exclude first-call kernel
compilation.

Results are written as JSON for ``scripts/plot_benchmarks.py`` to render.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
from pyg_lib import ops

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# (edges, feature_dim) configurations. nodes are derived as edges // FANOUT so
# each output row aggregates ~FANOUT messages, mimicking a real neighborhood.
DEFAULT_EDGE_COUNTS = [10_000, 50_000, 100_000, 500_000, 1_000_000]
FANOUT = 10
FEATURE_DIM = 64


def synchronize(device: str) -> None:
    """Block until queued device work has finished, so timing is accurate."""
    if device == "mps":
        torch.mps.synchronize()


def time_op(
    fn: Callable[[], Any], device: str, warmup: int, iters: int
) -> dict[str, float]:
    """Return median/stdev milliseconds for ``fn`` with warmup and sync."""
    for _ in range(warmup):
        fn()
    synchronize(device)

    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1e3)

    return {
        "ms_median": statistics.median(samples),
        "ms_mean": statistics.fmean(samples),
        "ms_stdev": statistics.pstdev(samples),
    }


def make_inputs(
    edges: int, nodes: int, device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (src, index) pair emulating edge-to-node message aggregation."""
    generator = torch.Generator().manual_seed(edges)
    src = torch.randn(edges, FEATURE_DIM, generator=generator, dtype=torch.float32)
    index = torch.randint(0, nodes, (edges,), generator=generator, dtype=torch.long)
    return src.to(device=device, dtype=dtype), index.to(device)


def scatter_max_mps_cpu_arg(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for the *earlier* approach: MPS value reduction, CPU arg.

    This reproduces the pre-optimization dispatch path (values reduced on MPS,
    arg indices recovered through a CPU round-trip in int64) so the benchmark
    can isolate exactly what moving the arg on-device in int32 saved.
    """
    device = src.device
    out = torch.full(
        (dim_size, src.size(1)), float("-inf"), device=device, dtype=src.dtype
    )
    idx = index.unsqueeze(-1).expand_as(src)
    out.scatter_reduce_(0, idx, src, "amax", include_self=True)

    gathered = out.gather(0, idx)
    src_cpu = src.cpu()
    idx_cpu = idx.cpu()
    gathered_cpu = gathered.cpu()
    sentinel = src.size(0)
    positions = (
        torch.arange(src.size(0), dtype=torch.long)
        .view(-1, 1)
        .expand_as(idx_cpu)
    )
    candidates = torch.where(
        src_cpu.eq(gathered_cpu), positions, torch.full_like(idx_cpu, sentinel)
    )
    arg_cpu = torch.full((dim_size, src.size(1)), sentinel, dtype=torch.long)
    arg_cpu.scatter_reduce_(0, idx_cpu, candidates, "amin", include_self=True)
    return out, arg_cpu.to(device)


def scatter_max_int32_5op(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for the previous *native* path: value + int32 arg in 5 ops.

    Reproduces the on-device tensor-op implementation the fused Metal kernel
    replaces (scatter_reduce, gather, eq, where, scatter_reduce), so the
    benchmark shows what fusing into a single atomic pass saved.
    """
    device = src.device
    out = torch.full(
        (dim_size, src.size(1)), float("-inf"), device=device, dtype=src.dtype
    )
    idx = index.unsqueeze(-1).expand_as(src)
    out.scatter_reduce_(0, idx, src, "amax", include_self=True)

    gathered = out.gather(0, idx)
    sentinel = src.size(0)
    positions = (
        torch.arange(src.size(0), dtype=torch.int32, device=device)
        .view(-1, 1)
        .expand_as(idx)
    )
    candidates = torch.where(
        src.eq(gathered), positions, torch.full_like(positions, sentinel)
    )
    arg = torch.full((dim_size, src.size(1)), sentinel, dtype=torch.int32, device=device)
    arg.scatter_reduce_(0, idx, candidates, "amin", include_self=True)
    out = out.masked_fill(arg == sentinel, 0)
    return out, arg.long()


def build_cases(
    device: str, dtype: torch.dtype
) -> dict[str, Callable[[torch.Tensor, torch.Tensor, int], Any]]:
    """Map an implementation label to a callable under test."""
    cases: dict[str, Callable[[torch.Tensor, torch.Tensor, int], Any]] = {
        "scatter_sum": lambda s, i, n: ops.scatter_sum(s, i, dim=0, dim_size=n),
        "scatter_mean": lambda s, i, n: ops.scatter_mean(s, i, dim=0, dim_size=n),
        # scatter_max on MPS now dispatches to the fused Metal kernel.
        "scatter_max": lambda s, i, n: ops.scatter_max(s, i, dim=0, dim_size=n),
    }
    if device == "mps":
        cases["scatter_max_int32_5op"] = (
            lambda s, i, n: scatter_max_int32_5op(s, i, n)
        )
        cases["scatter_max_cpu_arg"] = (
            lambda s, i, n: scatter_max_mps_cpu_arg(s, i, n)
        )
    return cases


def run(
    edge_counts: list[int], warmup: int, iters: int, dtype: torch.dtype
) -> dict[str, Any]:
    """Run every case on MPS and CPU across the requested edge counts."""
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    else:
        logger.warning("MPS not available; running CPU-only benchmarks.")

    results: list[dict[str, Any]] = []
    for edges in edge_counts:
        nodes = max(1, edges // FANOUT)
        for device in devices:
            src, index = make_inputs(edges, nodes, device, dtype)
            for label, fn in build_cases(device, dtype).items():
                timing = time_op(
                    lambda: fn(src, index, nodes), device, warmup, iters
                )
                row = {
                    "op": label,
                    "device": device,
                    "edges": edges,
                    "nodes": nodes,
                    **timing,
                }
                results.append(row)
                logger.info(
                    "%-22s %-4s edges=%-9d %8.3f ms",
                    label,
                    device,
                    edges,
                    timing["ms_median"],
                )

    return {
        "meta": {
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dtype": str(dtype),
            "feature_dim": FEATURE_DIM,
            "fanout": FANOUT,
            "warmup": warmup,
            "iters": iters,
            "devices": devices,
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/results.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--edges",
        type=int,
        nargs="*",
        default=DEFAULT_EDGE_COUNTS,
        help="Edge counts to sweep.",
    )
    args = parser.parse_args()

    report = run(args.edges, args.warmup, args.iters, torch.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Benchmark the fused ``sampled_op`` Metal kernel on Apple Silicon (MPS).

``sampled_op`` gathers ``left[left_index]`` / ``right[right_index]`` and applies
an elementwise op. On MPS it fuses the two gathers and the arithmetic into a
single Metal pass. Two questions drive these benchmarks:

1. How does the fused kernel compare to the CPU kernel as the sampled edge count
   grows?
2. What does fusing buy over the index_select composite -- the same math done as
   two ``index_select`` gathers plus a binary op -- on the same device?
   (``sampled_add`` vs ``sampled_add_composite_mps``.)

Timing is MPS-aware (warmup + ``torch.mps.synchronize()`` around each call).
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

# Sampled-edge counts. Node tables are edges // FANOUT rows each (both operands
# are indexed), so this mimics a bilinear edge score over a node embedding table.
DEFAULT_EDGE_COUNTS = [10_000, 50_000, 100_000, 500_000, 1_000_000]
FANOUT = 10
FEATURE_DIM = 64


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def time_op(
    fn: Callable[[], Any], device: str, warmup: int, iters: int
) -> dict[str, float]:
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


def make_inputs(edges: int, nodes: int, device: str, dtype: torch.dtype):
    """Build (left, right, left_index, right_index) for a sampled edge op."""
    g = torch.Generator().manual_seed(edges)
    left = torch.randn(nodes, FEATURE_DIM, generator=g, dtype=torch.float32)
    right = torch.randn(nodes, FEATURE_DIM, generator=g, dtype=torch.float32)
    li = torch.randint(0, nodes, (edges,), generator=g, dtype=torch.long)
    ri = torch.randint(0, nodes, (edges,), generator=g, dtype=torch.long)
    return (
        left.to(device=device, dtype=dtype),
        right.to(device=device, dtype=dtype),
        li.to(device),
        ri.to(device),
    )


def sampled_add_composite_mps(left, right, li, ri):
    """Reference: two index_selects + a binary op (what the fused kernel folds)."""
    return left.index_select(0, li) + right.index_select(0, ri)


def build_cases(
    device: str,
) -> dict[str, Callable[..., Any]]:
    cases: dict[str, Callable[..., Any]] = {
        "sampled_add": lambda l, r, li, ri: ops.sampled_add(l, r, li, ri),
    }
    if device == "mps":
        cases["sampled_add_composite_mps"] = sampled_add_composite_mps
    return cases


def run(edge_counts, warmup, iters, dtype) -> dict[str, Any]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    else:
        logger.warning("MPS not available; running CPU-only benchmarks.")

    results: list[dict[str, Any]] = []
    for edges in edge_counts:
        nodes = max(1, edges // FANOUT)
        for device in devices:
            left, right, li, ri = make_inputs(edges, nodes, device, dtype)
            for label, fn in build_cases(device).items():
                timing = time_op(
                    lambda: fn(left, right, li, ri), device, warmup, iters
                )
                results.append(
                    {
                        "op": label,
                        "device": device,
                        "edges": edges,
                        "nodes": nodes,
                        **timing,
                    }
                )
                logger.info(
                    "%-28s %-4s edges=%-9d %8.3f ms",
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
    parser.add_argument("--out", default="benchmarks/sampled_results.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--edges", type=int, nargs="*", default=DEFAULT_EDGE_COUNTS
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

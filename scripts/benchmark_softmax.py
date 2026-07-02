#!/usr/bin/env python3
"""Benchmark ``softmax_csr`` on Apple Silicon (MPS) vs CPU.

``softmax_csr`` normalizes ``src`` within each CSR row. On MPS it is a native
composite over the already-native CSR kernels (``segment_max_csr`` ->
``gather_csr`` -> exp -> ``segment_sum_csr`` -> ``gather_csr`` -> divide), so no
data leaves the GPU. Two questions drive these benchmarks:

1. How does the native MPS composite compare to the CPU kernel as graphs grow?
2. What does routing through the dedicated CSR Metal kernels buy over a naive
   pure-``torch`` MPS path that does the same math with generic ``scatter_reduce``
   / ``gather`` tensor ops? (``softmax_csr`` vs ``softmax_csr_naive_mps``.)

Timing is MPS-aware: every measured region is bracketed by
``torch.mps.synchronize()`` and each configuration is warmed up first to exclude
first-call kernel compilation. Results are written as JSON for
``scripts/plot_benchmarks.py`` to render.
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

# (edges, feature_dim). Rows (CSR segments) are derived as edges // FANOUT so
# each softmax group normalizes over ~FANOUT logits, mimicking attention.
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
    """Return median/mean/stdev milliseconds for ``fn`` with warmup and sync."""
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
    edges: int, rows: int, device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (src, indptr) pair emulating grouped-softmax over CSR rows."""
    generator = torch.Generator().manual_seed(edges)
    src = (
        torch.randn(edges, FEATURE_DIM, generator=generator, dtype=torch.float32)
        * 2.0
    )
    counts = torch.bincount(
        torch.randint(0, rows, (edges,), generator=generator), minlength=rows
    )
    indptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).long()
    return src.to(device=device, dtype=dtype), indptr.to(device)


def softmax_csr_naive_mps(src: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
    """Same max->sub->exp->sum->div math via generic tensor ops (no pyg kernels).

    Isolates what routing through the dedicated CSR Metal kernels buys over a
    plain-``torch`` implementation on the same device.
    """
    n = indptr.size(0) - 1
    counts = indptr[1:] - indptr[:-1]
    row = torch.repeat_interleave(
        torch.arange(n, device=src.device), counts
    )  # [E]
    idx = row.unsqueeze(-1).expand_as(src)

    mx = torch.full((n, src.size(1)), float("-inf"), device=src.device, dtype=src.dtype)
    mx.scatter_reduce_(0, idx, src, "amax", include_self=True)
    shifted = (src - mx.gather(0, idx)).exp()

    ssum = torch.zeros((n, src.size(1)), device=src.device, dtype=src.dtype)
    ssum.scatter_add_(0, idx, shifted)
    return shifted / ssum.gather(0, idx)


def build_cases(
    device: str,
) -> dict[str, Callable[[torch.Tensor, torch.Tensor], Any]]:
    """Map an implementation label to a callable under test."""
    cases: dict[str, Callable[[torch.Tensor, torch.Tensor], Any]] = {
        "softmax_csr": lambda s, p: ops.softmax_csr(s, p),
    }
    if device == "mps":
        cases["softmax_csr_naive_mps"] = softmax_csr_naive_mps
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
        rows = max(1, edges // FANOUT)
        for device in devices:
            src, indptr = make_inputs(edges, rows, device, dtype)
            for label, fn in build_cases(device).items():
                timing = time_op(lambda: fn(src, indptr), device, warmup, iters)
                results.append(
                    {
                        "op": label,
                        "device": device,
                        "edges": edges,
                        "rows": rows,
                        **timing,
                    }
                )
                logger.info(
                    "%-24s %-4s edges=%-9d %8.3f ms",
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
    parser.add_argument("--out", default="benchmarks/softmax_results.json")
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

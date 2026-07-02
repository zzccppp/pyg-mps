#!/usr/bin/env python3
"""Benchmark ``segment_matmul`` on Apple Silicon (MPS) vs CPU.

``segment_matmul`` multiplies row-segments of a stacked ``input`` by per-segment
weight matrices -- the core of heterogeneous-GNN linear layers. On MPS, when all
segments share a row count, it runs as a single batched ``bmm`` (Apple GEMM);
otherwise a per-segment ``mm`` loop. Two questions drive these benchmarks:

1. How does the native MPS path compare to the CPU kernel as the problem grows?
2. What does the batched ``bmm`` fast path buy over a naive per-segment ``mm``
   loop on the same device? (``segment_matmul`` vs ``segment_matmul_loop_mps``.)

Uniform segment sizes are used so the batched fast path is exercised. Timing is
MPS-aware (warmup + ``torch.mps.synchronize()`` around each call). Results are
written as JSON for ``scripts/plot_benchmarks.py`` to render.
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

# (num_segments) sweep; each segment has ROWS_PER_SEG rows of a K->N linear map.
DEFAULT_SEGMENT_COUNTS = [8, 16, 32, 64, 128, 256]
ROWS_PER_SEG = 128
K = 64
N = 64


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


def make_inputs(segments: int, device: str, dtype: torch.dtype):
    g = torch.Generator().manual_seed(segments)
    total = segments * ROWS_PER_SEG
    inp = torch.randn(total, K, generator=g, dtype=torch.float32)
    other = torch.randn(segments, K, N, generator=g, dtype=torch.float32)
    ptr = torch.arange(0, total + 1, ROWS_PER_SEG, dtype=torch.long)
    return (
        inp.to(device=device, dtype=dtype),
        ptr.to(device),
        other.to(device=device, dtype=dtype),
    )


def segment_matmul_loop_mps(inp, ptr, other):
    """Naive per-segment mm loop -- the batched fast path's on-device baseline."""
    parts = []
    ptr_cpu = ptr.cpu()
    for i in range(ptr.numel() - 1):
        s, e = int(ptr_cpu[i]), int(ptr_cpu[i + 1])
        parts.append(inp[s:e] @ other[i])
    return torch.cat(parts, dim=0)


def build_cases(device: str) -> dict[str, Callable[..., Any]]:
    cases: dict[str, Callable[..., Any]] = {
        "segment_matmul": lambda i, p, o: ops.segment_matmul(i, p, o),
    }
    if device == "mps":
        cases["segment_matmul_loop_mps"] = segment_matmul_loop_mps
    return cases


def run(segment_counts, warmup, iters, dtype) -> dict[str, Any]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")
    else:
        logger.warning("MPS not available; running CPU-only benchmarks.")

    results: list[dict[str, Any]] = []
    for segments in segment_counts:
        for device in devices:
            inp, ptr, other = make_inputs(segments, device, dtype)
            for label, fn in build_cases(device).items():
                timing = time_op(lambda: fn(inp, ptr, other), device, warmup, iters)
                results.append(
                    {
                        "op": label,
                        "device": device,
                        "segments": segments,
                        "rows_per_seg": ROWS_PER_SEG,
                        **timing,
                    }
                )
                logger.info(
                    "%-26s %-4s segments=%-5d %8.3f ms",
                    label,
                    device,
                    segments,
                    timing["ms_median"],
                )

    return {
        "meta": {
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dtype": str(dtype),
            "rows_per_seg": ROWS_PER_SEG,
            "K": K,
            "N": N,
            "warmup": warmup,
            "iters": iters,
            "devices": devices,
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/matmul_results.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--segments", type=int, nargs="*", default=DEFAULT_SEGMENT_COUNTS
    )
    args = parser.parse_args()

    report = run(args.segments, args.warmup, args.iters, torch.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

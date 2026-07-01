#!/usr/bin/env python3
"""Render charts from ``benchmarks/results.json`` for the benchmark report.

Produces two figures:

- ``scatter_max_arg.png``: wall-clock time of ``scatter_max`` on MPS with the
  on-device int32 arg path, versus the earlier MPS-value/CPU-arg path, versus
  pure CPU. Isolates what moving the arg on-device saved.
- ``native_vs_cpu.png``: MPS-over-CPU speedup for each scatter reduction as the
  graph grows, showing where native MPS starts to pay off.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _select(
    rows: list[dict[str, Any]], op: str, device: str
) -> tuple[list[int], list[float]]:
    """Return (edges, ms_median) sorted by edges for one op/device series."""
    picked = sorted(
        (r for r in rows if r["op"] == op and r["device"] == device),
        key=lambda r: r["edges"],
    )
    return [r["edges"] for r in picked], [r["ms_median"] for r in picked]


def plot_scatter_max_arg(rows: list[dict[str, Any]], out: Path) -> None:
    """Compare the int32 on-device arg path against the CPU-arg round-trip."""
    series = {
        "native MPS (int32 arg, on-device)": ("scatter_max", "mps"),
        "MPS value + CPU arg (previous path)": ("scatter_max_cpu_arg", "mps"),
        "CPU": ("scatter_max", "cpu"),
    }
    styles = {
        "native MPS (int32 arg, on-device)": {"marker": "o", "color": "#1b7837", "lw": 2},
        "MPS value + CPU arg (previous path)": {
            "marker": "s",
            "color": "#d95f02",
            "ls": "--",
            "lw": 2,
        },
        "CPU": {"marker": "^", "color": "#7570b3", "ls": ":", "lw": 2},
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (op, device) in series.items():
        edges, ms = _select(rows, op, device)
        if edges:
            ax.plot(edges, ms, label=label, **styles[label])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("edges (messages)")
    ax.set_ylabel("time per call (ms, median)")
    ax.set_title("scatter_max on MPS: on-device int32 arg vs CPU-arg round-trip")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()

    # Annotate the speedup of native over the CPU-arg path at the smallest size.
    n_edges, n_ms = _select(rows, "scatter_max", "mps")
    c_edges, c_ms = _select(rows, "scatter_max_cpu_arg", "mps")
    if n_ms and c_ms:
        speedup = c_ms[0] / n_ms[0]
        ax.annotate(
            f"{speedup:.1f}x faster",
            xy=(n_edges[0], n_ms[0]),
            xytext=(n_edges[0] * 1.5, n_ms[0] * 0.4),
            arrowprops={"arrowstyle": "->", "color": "#1b7837"},
            color="#1b7837",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out)


def plot_native_vs_cpu(rows: list[dict[str, Any]], out: Path) -> None:
    """Plot MPS-over-CPU speedup per op as a function of graph size."""
    ops = ["scatter_sum", "scatter_mean", "scatter_max"]
    colors = {"scatter_sum": "#1b7837", "scatter_mean": "#2166ac", "scatter_max": "#b2182b"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for op in ops:
        m_edges, m_ms = _select(rows, op, "mps")
        _, c_ms = _select(rows, op, "cpu")
        if not m_ms or not c_ms:
            continue
        speedup = [c / m for c, m in zip(c_ms, m_ms)]
        ax.plot(m_edges, speedup, marker="o", lw=2, color=colors[op], label=op)

    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.6)
    ax.text(
        ax.get_xlim()[0], 1.02, "MPS faster above / CPU faster below",
        fontsize=8, color="black", alpha=0.7,
    )
    ax.set_xscale("log")
    ax.set_xlabel("edges (messages)")
    ax.set_ylabel("speedup (CPU time / MPS time)")
    ax.set_title("Native MPS scatter speedup over CPU vs graph size")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="benchmarks/results.json")
    parser.add_argument("--out-dir", default="benchmarks")
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = data["results"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_scatter_max_arg(rows, out_dir / "scatter_max_arg.png")
    plot_native_vs_cpu(rows, out_dir / "native_vs_cpu.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

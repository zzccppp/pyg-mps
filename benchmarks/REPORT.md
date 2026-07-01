# Scatter-Family MPS Benchmark Report

This report measures the `pyg-lib` scatter operators on Apple Silicon and
quantifies two things:

1. what the **on-device int32 argmin/argmax** path buys over the earlier
   approach that computed arg indices through a CPU round-trip, and
2. when the **native MPS** scatter kernels actually beat CPU.

## Setup

| Item | Value |
|------|-------|
| Platform | macOS 26.5.1, arm64 (Apple Silicon) |
| PyTorch | 2.12.1 |
| `pyg-lib` | 0.8.0 (local source build, MPS patches applied) |
| Feature dim | 64 |
| Workload | `edges` messages aggregated into `edges / 10` nodes |
| Timing | warmup 20, 100 iters, `torch.mps.synchronize()` around each call, median reported |

Reproduce with:

```bash
./scripts/uv_stage.sh benchmark   # writes benchmarks/results.json
./scripts/uv_stage.sh report      # writes the PNGs below
```

## Result 1 — on-device int32 arg vs CPU-arg round-trip

`scatter_min`/`scatter_max` need an arg index, but MPS has no int64
`scatter_reduce`. The earlier shim reduced values on MPS and recovered the arg
on CPU (copy in, int64 reduce, copy back) on **every call**. The current kernel
does the whole arg reduction on-device in int32 and widens to int64 at the end.

![scatter_max: on-device int32 arg vs CPU-arg round-trip](scatter_max_arg.png)

| edges | native MPS (ms) | MPS + CPU arg (ms) | speedup |
|------:|----------------:|-------------------:|--------:|
| 10,000 | 0.78 | 1.72 | **2.20×** |
| 50,000 | 3.32 | 6.91 | **2.08×** |
| 100,000 | 13.15 | 16.13 | 1.23× |
| 500,000 | 76.04 | 88.06 | 1.16× |
| 1,000,000 | 184.26 | 214.24 | 1.16× |

**Reading:** the win is largest (≈2×) at 10k–50k edges — typical mini-batch /
neighborhood-sampling scale — because the CPU round-trip is a fixed
per-call latency (device→host→device copies plus a host-side kernel launch)
that dominates when the reduction itself is cheap. At 1M edges the copies are
amortized against a much heavier reduction, but removing them still saves ~16%.
The old path was also consistently **slower than plain CPU** at small sizes
(1.72 vs 1.47 ms at 10k) — i.e. it was worse than not using the GPU at all —
whereas the native path is faster than both.

## Result 2 — native MPS vs CPU

![Native MPS scatter speedup over CPU vs graph size](native_vs_cpu.png)

| edges | sum MPS/CPU (ms) | mean MPS/CPU (ms) | max MPS/CPU (ms) |
|------:|-----------------:|------------------:|-----------------:|
| 10,000 | 0.77 / 0.33 | 0.60 / 0.36 | 0.78 / 1.47 |
| 50,000 | 1.33 / 1.42 | 1.38 / 1.54 | 3.32 / 7.29 |
| 100,000 | 4.79 / 3.24 | 4.91 / 3.15 | 13.15 / 14.41 |
| 500,000 | 23.24 / 21.56 | 23.45 / 27.61 | 76.04 / 86.93 |
| 1,000,000 | 60.30 / 85.67 | 61.01 / 99.71 | 184.26 / 205.92 |

**Reading:** native MPS is **not a blanket win** on this hardware. For
`scatter_sum`/`scatter_mean` the GPU only pays off past ~500k edges (1.4–1.6× at
1M), and loses at small graphs where dispatch overhead outweighs the compute and
Apple's unified memory keeps CPU competitive. `scatter_max` benefits from MPS
across the whole range because its heavier arithmetic (a 5-op arg derivation)
gives the GPU more to chew on. This is the honest, data-driven takeaway: ship the
native kernels, but a production message-passing layer on small graphs should
not assume MPS is automatically faster.

### The reproducible 100k dip

The MPS `scatter_sum`/`scatter_mean` curves show a real, reproducible slowdown
near 100k edges (4.79 ms at 100k vs 1.33 ms at 50k — a 3.6× jump for 2× data,
stable across runs). This is not measurement noise; it looks like an MPS
allocation/tiling threshold. It is left in the charts rather than smoothed away
because it is a genuine characteristic of the backend at this size.

## What this implies for next steps

- **Fuse the `scatter_max` arg kernel.** At 1M edges `scatter_max` costs ~3×
  `scatter_sum` on MPS (184 vs 60 ms) because the on-device arg path is five
  tensor ops (reduce, gather, eq, where, reduce). A single hand-written Metal
  kernel that computes value and arg in one pass is the highest-value follow-up.
- **Keep point-cloud/spline ops CPU-assisted.** They are preprocessing, called
  rarely; the benchmarks confirm effort belongs on the scatter hot path.
- **Guard small-graph paths.** Below ~50k edges, CPU can beat MPS for
  sum/mean — worth a device heuristic in downstream code.

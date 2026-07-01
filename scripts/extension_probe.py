#!/usr/bin/env python3
"""Probe legacy PyG optional extensions one by one."""

from __future__ import annotations

import argparse
import importlib
import json
import traceback
from typing import Any, Callable

import torch


def result(status: str, detail: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"status": status}
    if detail:
        data["detail"] = detail
    if extra:
        data.update(extra)
    return data


def run_case(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return result("ok", extra=fn())
    except NotImplementedError as exc:
        detail = str(exc)
        if "not currently implemented for the MPS device" in detail:
            return result("unsupported", f"native MPS kernel missing: {detail}")
        return result("failed", f"{type(exc).__name__}: {exc}", {"traceback": traceback.format_exc(limit=8)})
    except RuntimeError as exc:
        detail = str(exc)
        if "must be CPU tensor" in detail or "must be CPU" in detail:
            return result(
                "unsupported",
                f"CPU-only kernel reached by non-CPU tensor: {detail}",
                {"traceback": traceback.format_exc(limit=8)},
            )
        return result("failed", f"{type(exc).__name__}: {exc}", {"traceback": traceback.format_exc(limit=8)})
    except Exception as exc:  # noqa: BLE001 - probe should report all failures.
        return result("failed", f"{type(exc).__name__}: {exc}", {"traceback": traceback.format_exc(limit=8)})


def module_info(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return result("missing", f"{type(exc).__name__}: {exc}")
    return result("ok", extra={"version": getattr(mod, "__version__", None), "path": getattr(mod, "__file__", None)})


def tensor_info(tensor: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "device": str(tensor.device), "dtype": str(tensor.dtype)}


def build_cases(device: str) -> dict[str, Callable[[], dict[str, Any]]]:
    cases: dict[str, Callable[[], dict[str, Any]]] = {}

    def torch_scatter_case() -> dict[str, Any]:
        from torch_scatter import scatter_add

        src = torch.ones(6, 3, device=device)
        index = torch.tensor([0, 1, 0, 2, 1, 3], dtype=torch.long, device=device)
        out = scatter_add(src, index, dim=0, dim_size=4)
        return {"out": tensor_info(out), "sum": float(out.sum().detach().cpu())}

    def torch_sparse_case() -> dict[str, Any]:
        from torch_sparse import SparseTensor

        row = torch.tensor([0, 1, 1, 2], dtype=torch.long, device=device)
        col = torch.tensor([1, 0, 2, 1], dtype=torch.long, device=device)
        value = torch.ones(row.numel(), device=device)
        adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(3, 3))
        x = torch.randn(3, 2, device=device)
        out = adj.matmul(x)
        return {"out": tensor_info(out), "sum": float(out.sum().detach().cpu())}

    def torch_cluster_knn_case() -> dict[str, Any]:
        from torch_cluster import knn_graph

        x = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            device=device,
        )
        out = knn_graph(x, k=2)
        return {"out": tensor_info(out)}

    def torch_cluster_grid_case() -> dict[str, Any]:
        from torch_cluster import grid_cluster

        x = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            device=device,
        )
        size = torch.tensor([0.5, 0.5], device=device)
        out = grid_cluster(x, size)
        return {"out": tensor_info(out)}

    def torch_spline_conv_case() -> dict[str, Any]:
        import torch_spline_conv

        pseudo = torch.rand(4, 1, device=device)
        kernel_size = torch.tensor([3], dtype=torch.long, device=device)
        is_open_spline = torch.tensor([1], dtype=torch.uint8, device=device)
        basis, weight_index = torch_spline_conv.spline_basis(
            pseudo, kernel_size, is_open_spline, 1
        )
        x = torch.randn(4, 3, device=device)
        weight = torch.randn(3, 3, 2, device=device)
        out = torch_spline_conv.spline_weighting(x, weight, basis, weight_index)
        return {"basis": tensor_info(basis), "weight_index": tensor_info(weight_index), "out": tensor_info(out)}

    cases["torch_scatter.scatter_add"] = torch_scatter_case
    cases["torch_sparse.SparseTensor.matmul"] = torch_sparse_case
    cases["torch_cluster.knn_graph"] = torch_cluster_knn_case
    cases["torch_cluster.grid_cluster"] = torch_cluster_grid_case
    cases["torch_spline_conv.spline"] = torch_spline_conv_case
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = parser.parse_args()

    report: dict[str, Any] = {
        "device": args.device,
        "modules": {
            name: module_info(name)
            for name in [
                "torch_scatter",
                "torch_sparse",
                "torch_cluster",
                "torch_spline_conv",
            ]
        },
        "tests": {},
    }

    if args.device == "mps" and not torch.backends.mps.is_available():
        report["tests"]["device_ready"] = result(
            "failed", "torch.backends.mps.is_available() is false"
        )
    else:
        report["tests"] = {
            name: run_case(fn) for name, fn in build_cases(args.device).items()
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    failed = any(case.get("status") == "failed" for case in report["tests"].values())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

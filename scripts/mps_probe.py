#!/usr/bin/env python3
"""Probe PyTorch/PyG MPS behavior after each dependency stage."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


OPTIONAL_MODULES = [
    "torch_geometric",
    "pyg_lib",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "torch_spline_conv",
]

SANDBOX_ENV_KEYS = [
    "CODEX_SANDBOX",
    "CODEX_ENV",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "UV_PROJECT_ENVIRONMENT",
    "UV_CACHE_DIR",
]

OPTIONAL_DEPENDENCY_MARKERS = [
    "requires 'pyg-lib",
    "requires 'torch-cluster",
    "requires 'torch-spline-conv",
    "requires 'torch-scatter",
    "requires 'torch-sparse",
]


class SkipCase(Exception):
    """Signal that a probe case is not applicable in the current stage."""


def result(status: str, detail: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"status": status}
    if detail:
        data["detail"] = detail
    if extra:
        data.update(extra)
    return data


def run_case(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        value = fn()
        if isinstance(value, dict):
            return result("ok", extra=value)
        return result("ok")
    except SkipCase as exc:
        return result("skipped", str(exc))
    except ImportError as exc:
        detail = str(exc)
        if any(marker in detail for marker in OPTIONAL_DEPENDENCY_MARKERS):
            return result("skipped", f"optional dependency missing: {detail}")
        return result(
            "failed",
            detail=f"{type(exc).__name__}: {exc}",
            extra={"traceback": traceback.format_exc(limit=8)},
        )
    except NotImplementedError as exc:
        detail = str(exc)
        if "not currently implemented for the MPS device" in detail:
            return result(
                "unsupported",
                f"native MPS kernel missing: {detail}",
                extra={"traceback": traceback.format_exc(limit=8)},
            )
        return result(
            "failed",
            detail=f"{type(exc).__name__}: {exc}",
            extra={"traceback": traceback.format_exc(limit=8)},
        )
    except Exception as exc:  # noqa: BLE001 - probe should capture all failures.
        return result(
            "failed",
            detail=f"{type(exc).__name__}: {exc}",
            extra={"traceback": traceback.format_exc(limit=8)},
        )


def module_info(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - import failures are the signal.
        return result("missing", f"{type(exc).__name__}: {exc}")

    version = getattr(mod, "__version__", None)
    path = getattr(mod, "__file__", None)
    return result("ok", extra={"version": version, "path": path})


def torch_metadata(torch: Any) -> dict[str, Any]:
    mps = getattr(torch.backends, "mps", None)
    metadata = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "mac_ver": platform.mac_ver()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torch_file": getattr(torch, "__file__", None),
        "mps_built": bool(mps and mps.is_built()),
        "mps_available": bool(mps and mps.is_available()),
        "environment": {key: os.environ.get(key) for key in SANDBOX_ENV_KEYS},
    }
    if hasattr(torch, "mps"):
        try:
            metadata["mps_device_count"] = torch.mps.device_count()
        except Exception as exc:  # noqa: BLE001 - diagnostic only.
            metadata["mps_device_count_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def torch_core_cases(torch: Any, device: str) -> dict[str, Any]:
    cases: dict[str, Callable[[], Any]] = {}

    def tensor_create() -> dict[str, Any]:
        x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
        return {"device": str(x.device), "sum": float(x.sum().detach().cpu())}

    def matmul_backward() -> dict[str, Any]:
        x = torch.randn(8, 8, device=device, requires_grad=True)
        y = (x @ x.T).sum()
        y.backward()
        return {"grad_norm": float(x.grad.norm().detach().cpu())}

    def indexing() -> dict[str, Any]:
        x = torch.arange(20, dtype=torch.float32, device=device).reshape(5, 4)
        index = torch.tensor([0, 3, 1, 1], device=device)
        y = x.index_select(0, index)
        return {"shape": list(y.shape), "sum": float(y.sum().detach().cpu())}

    def scatter_add() -> dict[str, Any]:
        src = torch.ones(10, 3, device=device)
        index = torch.tensor([0, 1, 0, 2, 1, 3, 0, 2, 3, 3], device=device)
        out = torch.zeros(4, 3, device=device)
        out.index_add_(0, index, src)
        return {"shape": list(out.shape), "sum": float(out.sum().detach().cpu())}

    def sparse_coo_mm() -> dict[str, Any]:
        indices = torch.tensor([[0, 1, 1], [2, 0, 2]], device=device)
        values = torch.tensor([3.0, 4.0, 5.0], device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (2, 3), device=device)
        dense = torch.randn(3, 4, device=device)
        out = torch.sparse.mm(sparse, dense)
        return {"shape": list(out.shape), "sum": float(out.sum().detach().cpu())}

    cases["tensor_create"] = tensor_create
    cases["matmul_backward"] = matmul_backward
    cases["indexing"] = indexing
    cases["scatter_add_index_add"] = scatter_add
    cases["sparse_coo_mm"] = sparse_coo_mm

    return {name: run_case(fn) for name, fn in cases.items()}


def pyg_cases(torch: Any, device: str) -> dict[str, Any]:
    cases: dict[str, Callable[[], Any]] = {}

    def data_to_device() -> dict[str, Any]:
        from torch_geometric.data import Data

        data = Data(
            x=torch.randn(4, 3),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        ).to(device)
        return {"x_device": str(data.x.device), "edge_index_device": str(data.edge_index.device)}

    def gcn_forward_backward() -> dict[str, Any]:
        from torch_geometric.nn import GCNConv

        x = torch.randn(5, 4, device=device, requires_grad=True)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 0, 2], [1, 2, 3, 4, 0, 2, 0]],
            dtype=torch.long,
            device=device,
        )
        conv = GCNConv(4, 2).to(device)
        out = conv(x, edge_index)
        loss = out.pow(2).mean()
        loss.backward()
        return {"shape": list(out.shape), "sum": float(out.sum().detach().cpu())}

    def knn_graph_case() -> dict[str, Any]:
        from torch_geometric.nn import knn_graph

        x = torch.randn(8, 3, device=device)
        edge_index = knn_graph(x, k=2)
        return {"shape": list(edge_index.shape), "edge_device": str(edge_index.device)}

    def spline_conv_case() -> dict[str, Any]:
        from torch_geometric.nn import SplineConv

        x = torch.randn(4, 3, device=device, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long, device=device)
        pseudo = torch.rand(edge_index.size(1), 1, device=device)
        conv = SplineConv(3, 2, dim=1, kernel_size=3).to(device)
        out = conv(x, edge_index, pseudo)
        out.sum().backward()
        return {"shape": list(out.shape), "sum": float(out.sum().detach().cpu())}

    if module_info("torch_geometric")["status"] == "ok":
        cases["pyg_data_to_device"] = data_to_device
        cases["pyg_gcn_forward_backward"] = gcn_forward_backward
        cases["pyg_knn_graph"] = knn_graph_case
        cases["pyg_spline_conv"] = spline_conv_case

    return {name: run_case(fn) for name, fn in cases.items()}


def pyg_lib_cases(torch: Any, device: str) -> dict[str, Any]:
    cases: dict[str, Callable[[], Any]] = {}

    try:
        ops = importlib.import_module("pyg_lib.ops")
    except Exception:
        return cases

    def point_cloud() -> Any:
        return torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            device=device,
        )

    def direct_knn() -> dict[str, Any]:
        x = point_cloud()
        out = ops.knn(x, x, k=2)
        return {"shape": list(out.shape), "device": str(out.device)}

    def direct_radius() -> dict[str, Any]:
        x = point_cloud()
        out = ops.radius(x, x, r=1.1, max_num_neighbors=4)
        return {"shape": list(out.shape), "device": str(out.device)}

    def direct_nearest() -> dict[str, Any]:
        x = point_cloud()
        out = ops.nearest(x, x)
        return {"shape": list(out.shape), "device": str(out.device)}

    def direct_fps() -> dict[str, Any]:
        x = point_cloud()
        ptr = torch.tensor([0, x.size(0)], dtype=torch.long, device=device)
        out = ops.fps(x, ptr, ratio=0.5, random_start=False)
        return {"shape": list(out.shape), "device": str(out.device)}

    def direct_grid_cluster() -> dict[str, Any]:
        x = point_cloud()
        size = torch.tensor([0.5, 0.5], device=device)
        out = ops.grid_cluster(x, size)
        return {"shape": list(out.shape), "device": str(out.device)}

    def direct_spline_basis() -> dict[str, Any]:
        pseudo = torch.rand(4, 1, device=device)
        kernel_size = torch.tensor([3], dtype=torch.long, device=device)
        is_open_spline = torch.tensor([1], dtype=torch.uint8, device=device)
        basis, weight_index = ops.spline_basis(pseudo, kernel_size, is_open_spline, degree=1)
        return {
            "basis_shape": list(basis.shape),
            "basis_device": str(basis.device),
            "weight_index_device": str(weight_index.device),
        }

    def direct_spline_weighting() -> dict[str, Any]:
        x = torch.randn(4, 3, device=device)
        weight = torch.randn(3, 3, 2, device=device)
        basis = torch.rand(4, 2, device=device)
        weight_index = torch.tensor([[0, 1], [1, 2], [0, 2], [1, 1]], dtype=torch.long, device=device)
        out = ops.spline_weighting(x, weight, basis, weight_index)
        return {"shape": list(out.shape), "device": str(out.device)}

    def direct_scatter_sum() -> dict[str, Any]:
        src = torch.ones(6, 3, device=device)
        index = torch.tensor([0, 1, 0, 2, 1, 3], dtype=torch.long, device=device)
        out = ops.scatter_sum(src, index, dim=0, dim_size=4)
        return {"shape": list(out.shape), "device": str(out.device), "sum": float(out.sum().detach().cpu())}

    def scatter_inputs() -> tuple[Any, Any]:
        src = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [-1.0, 2.0, 0.0],
                [2.0, 3.0, 4.0],
                [1.0, -2.0, 2.0],
                [5.0, 1.0, -3.0],
            ],
            device=device,
        )
        index = torch.tensor([0, 1, 0, 2, 1, 3], dtype=torch.long, device=device)
        return src, index

    def direct_scatter_mul() -> dict[str, Any]:
        src, index = scatter_inputs()
        out = ops.scatter_mul(src, index, dim=0, dim_size=4)
        return {"shape": list(out.shape), "device": str(out.device), "sum": float(out.sum().detach().cpu())}

    def direct_scatter_mean() -> dict[str, Any]:
        src, index = scatter_inputs()
        out = ops.scatter_mean(src, index, dim=0, dim_size=4)
        return {"shape": list(out.shape), "device": str(out.device), "sum": float(out.sum().detach().cpu())}

    def direct_scatter_min() -> dict[str, Any]:
        src, index = scatter_inputs()
        out, arg_out = ops.scatter_min(src, index, dim=0, dim_size=4)
        return {
            "shape": list(out.shape),
            "device": str(out.device),
            "arg_shape": list(arg_out.shape),
            "arg_device": str(arg_out.device),
            "sum": float(out.sum().detach().cpu()),
        }

    def direct_scatter_max() -> dict[str, Any]:
        src, index = scatter_inputs()
        out, arg_out = ops.scatter_max(src, index, dim=0, dim_size=4)
        return {
            "shape": list(out.shape),
            "device": str(out.device),
            "arg_shape": list(arg_out.shape),
            "arg_device": str(arg_out.device),
            "sum": float(out.sum().detach().cpu()),
        }

    cases["pyg_lib_knn"] = direct_knn
    cases["pyg_lib_radius"] = direct_radius
    cases["pyg_lib_nearest"] = direct_nearest
    cases["pyg_lib_fps"] = direct_fps
    cases["pyg_lib_grid_cluster"] = direct_grid_cluster
    cases["pyg_lib_spline_basis"] = direct_spline_basis
    cases["pyg_lib_spline_weighting"] = direct_spline_weighting
    cases["pyg_lib_scatter_sum"] = direct_scatter_sum
    cases["pyg_lib_scatter_mul"] = direct_scatter_mul
    cases["pyg_lib_scatter_mean"] = direct_scatter_mean
    cases["pyg_lib_scatter_min"] = direct_scatter_min
    cases["pyg_lib_scatter_max"] = direct_scatter_max

    return {name: run_case(fn) for name, fn in cases.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="manual", help="Name of the current dependency stage.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"], help="Device to test.")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "stage": args.stage,
        "device": args.device,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "modules": {},
        "tests": {},
    }

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001
        report["torch_import"] = result("failed", f"{type(exc).__name__}: {exc}")
        payload = json.dumps(report, indent=2, sort_keys=True)
        print(payload)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(payload + "\n", encoding="utf-8")
        return 1

    report["metadata"] = torch_metadata(torch)
    report["modules"]["torch"] = module_info("torch")
    for module_name in OPTIONAL_MODULES:
        report["modules"][module_name] = module_info(module_name)

    if args.device == "mps" and not report["metadata"]["mps_available"]:
        detail = "torch.backends.mps.is_available() is false"
        if report["metadata"]["environment"].get("CODEX_SANDBOX") == "seatbelt":
            detail += "; result may be a Codex sandbox false negative"
        report["tests"]["device_ready"] = result("failed", detail)
    else:
        report["tests"]["torch_core"] = torch_core_cases(torch, args.device)
        report["tests"]["pyg"] = pyg_cases(torch, args.device)
        report["tests"]["pyg_lib"] = pyg_lib_cases(torch, args.device)

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")

    failed = any(
        case.get("status") == "failed"
        for group in report.get("tests", {}).values()
        for case in (group.values() if isinstance(group, dict) else [group])
        if isinstance(case, dict)
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

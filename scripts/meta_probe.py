#!/usr/bin/env python3
"""Probe pyg-lib Meta dispatch registrations."""

from __future__ import annotations

import importlib
import json
import traceback
from typing import Any, Callable

import torch


def run_case(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"status": "ok", **fn()}
    except Exception as exc:  # noqa: BLE001 - probe reports failures.
        return {
            "status": "failed",
            "detail": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }


def main() -> int:
    ops = importlib.import_module("pyg_lib.ops")
    src = torch.empty(6, 3, device="meta")
    index = torch.empty(6, 3, dtype=torch.long, device="meta")

    cases: dict[str, Callable[[], dict[str, Any]]] = {
        "scatter_sum": lambda: tensor_summary(
            ops.scatter_sum(src, index, dim=0, dim_size=4)
        ),
        "scatter_mul": lambda: tensor_summary(
            ops.scatter_mul(src, index, dim=0, dim_size=4)
        ),
        "scatter_mean": lambda: tensor_summary(
            ops.scatter_mean(src, index, dim=0, dim_size=4)
        ),
        "scatter_min": lambda: {
            "out": tensor_summary(ops.scatter_min(src, index, dim=0, dim_size=4)[0]),
            "arg": tensor_summary(ops.scatter_min(src, index, dim=0, dim_size=4)[1]),
        },
        "scatter_max": lambda: {
            "out": tensor_summary(ops.scatter_max(src, index, dim=0, dim_size=4)[0]),
            "arg": tensor_summary(ops.scatter_max(src, index, dim=0, dim_size=4)[1]),
        },
    }

    report = {name: run_case(fn) for name, fn in cases.items()}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(item["status"] == "ok" for item in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

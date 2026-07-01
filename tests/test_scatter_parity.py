"""Numerical parity tests for the ``pyg-lib`` MPS scatter-family kernels.

The MPS dispatch shims added under ``pyg_lib/csrc/ops/mps`` are only useful if
they reproduce the reference CPU kernels. These tests treat the CPU ``pyg-lib``
kernel as ground truth and assert that the MPS path matches it across dtypes,
tie-breaking, empty groups, negative values, and multi-dimensional inputs. When
``torch_scatter`` is importable it is used as an independent cross-check.

Run with::

    ./scripts/uv_stage.sh parity
    # or
    .venv/bin/python -m pytest tests/test_scatter_parity.py -v
"""

from __future__ import annotations

from typing import Callable, Optional

import pytest
import torch

pytest.importorskip("pyg_lib", reason="pyg-lib is required for parity tests")
from pyg_lib import ops  # noqa: E402

MPS_AVAILABLE = torch.backends.mps.is_available()

pytestmark = pytest.mark.skipif(
    not MPS_AVAILABLE, reason="MPS backend is not available on this host"
)

# Per-dtype tolerances for floating-point value comparisons. Reduced-precision
# dtypes accumulate more error, especially through scatter_mean's division.
_TOLERANCES: dict[torch.dtype, dict[str, float]] = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float16: {"rtol": 1e-2, "atol": 1e-3},
    torch.bfloat16: {"rtol": 2e-2, "atol": 1e-2},
}

_FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _assert_value_close(
    got: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype
) -> None:
    """Assert floating-point parity using dtype-aware tolerances."""
    tol = _TOLERANCES[dtype]
    torch.testing.assert_close(
        got.float().cpu(), ref.float().cpu(), rtol=tol["rtol"], atol=tol["atol"]
    )


def _run_pair(
    op: Callable[..., object],
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int,
    dim_size: Optional[int],
) -> tuple[object, object]:
    """Run ``op`` on both the MPS and CPU copies of the same inputs."""
    mps_out = op(src.to("mps"), index.to("mps"), dim=dim, dim_size=dim_size)
    cpu_out = op(src.cpu(), index.cpu(), dim=dim, dim_size=dim_size)
    return mps_out, cpu_out


# ---------------------------------------------------------------------------
# Value-only reductions: scatter_sum, scatter_mul, scatter_mean
# ---------------------------------------------------------------------------

_VALUE_OPS = {
    "scatter_sum": ops.scatter_sum,
    "scatter_mul": ops.scatter_mul,
    "scatter_mean": ops.scatter_mean,
}


@pytest.mark.parametrize("op_name", list(_VALUE_OPS))
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_value_reduction_parity(op_name: str, dtype: torch.dtype) -> None:
    """MPS value reductions match the CPU kernel across dtypes."""
    torch.manual_seed(0)
    # Keep magnitudes small so scatter_mul stays representable in fp16.
    src = (torch.randn(64, 8) * 0.5).to(dtype)
    index = torch.randint(0, 10, (64,), dtype=torch.long)

    mps_out, cpu_out = _run_pair(_VALUE_OPS[op_name], src, index, 0, 12)

    assert mps_out.device.type == "mps"
    _assert_value_close(mps_out, cpu_out, dtype)


@pytest.mark.parametrize("dim", [0, 1])
def test_value_reduction_multidim(dim: int) -> None:
    """scatter_sum matches the CPU kernel when reducing along either axis."""
    torch.manual_seed(1)
    src = torch.randn(16, 20)
    idx_1d = torch.randint(0, 5, (src.size(dim),), dtype=torch.long)
    view = [1] * src.dim()
    view[dim] = src.size(dim)
    index = idx_1d.view(view).expand_as(src).contiguous()

    mps_out, cpu_out = _run_pair(ops.scatter_sum, src, index, dim, None)
    _assert_value_close(mps_out, cpu_out, torch.float32)


def test_empty_index() -> None:
    """An empty index yields the CPU-shaped zero/identity output on MPS."""
    src = torch.zeros(0, 4)
    index = torch.zeros(0, dtype=torch.long)
    mps_out, cpu_out = _run_pair(ops.scatter_sum, src, index, 0, 3)
    assert mps_out.device.type == "mps"
    _assert_value_close(mps_out, cpu_out, torch.float32)


# ---------------------------------------------------------------------------
# arg reductions: scatter_min, scatter_max (the int32 on-device arg path)
# ---------------------------------------------------------------------------

_ARG_OPS = {"scatter_min": ops.scatter_min, "scatter_max": ops.scatter_max}


@pytest.mark.parametrize("op_name", list(_ARG_OPS))
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_argreduction_parity(op_name: str, dtype: torch.dtype) -> None:
    """MPS min/max values and int32-derived arg indices match the CPU kernel."""
    torch.manual_seed(2)
    src = (torch.randn(128, 6) * 3.0).to(dtype)
    index = torch.randint(0, 16, (128,), dtype=torch.long)

    (mps_val, mps_arg) = _ARG_OPS[op_name](
        src.to("mps"), index.to("mps"), dim=0, dim_size=20
    )
    (cpu_val, cpu_arg) = _ARG_OPS[op_name](
        src.cpu(), index.cpu(), dim=0, dim_size=20
    )

    assert mps_val.device.type == "mps"
    assert mps_arg.device.type == "mps", "arg index must stay on-device (int32 path)"
    assert mps_arg.dtype == torch.int64, "arg index must be widened to int64"

    _assert_value_close(mps_val, cpu_val, dtype)
    torch.testing.assert_close(mps_arg.cpu(), cpu_arg.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("op_name", list(_ARG_OPS))
def test_argreduction_ties_first_occurrence(op_name: str) -> None:
    """Tied extrema resolve to the first source position, matching the CPU kernel."""
    # Rows 0 and 1 share group 0 with an identical extreme value in column 0.
    src = torch.tensor(
        [[7.0, 1.0], [7.0, 2.0], [3.0, 9.0], [3.0, 9.0]],
    )
    index = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    _, mps_arg = _ARG_OPS[op_name](src.to("mps"), index.to("mps"), dim=0, dim_size=2)
    _, cpu_arg = _ARG_OPS[op_name](src.cpu(), index.cpu(), dim=0, dim_size=2)
    torch.testing.assert_close(mps_arg.cpu(), cpu_arg.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("op_name", list(_ARG_OPS))
def test_argreduction_empty_groups(op_name: str) -> None:
    """Groups with no contributing element get value 0 and arg == src.size(dim)."""
    src = torch.randn(6, 3)
    # dim_size 10 leaves several groups empty (indices only cover 0..2).
    index = torch.tensor([0, 1, 0, 2, 1, 2], dtype=torch.long)
    dim_size = 10

    mps_val, mps_arg = _ARG_OPS[op_name](
        src.to("mps"), index.to("mps"), dim=0, dim_size=dim_size
    )
    cpu_val, cpu_arg = _ARG_OPS[op_name](
        src.cpu(), index.cpu(), dim=0, dim_size=dim_size
    )
    _assert_value_close(mps_val, cpu_val, torch.float32)
    torch.testing.assert_close(mps_arg.cpu(), cpu_arg.cpu(), rtol=0, atol=0)
    # Empty groups (rows 3..9) use the sentinel src.size(dim) == 6.
    assert torch.all(mps_arg.cpu()[3:] == src.size(0))


@pytest.mark.parametrize("op_name", list(_ARG_OPS))
def test_arg_points_to_reduced_value(op_name: str) -> None:
    """Semantic invariant: src at each returned arg equals the reduced value."""
    torch.manual_seed(3)
    src = torch.randn(50, 4)
    index = torch.randint(0, 8, (50,), dtype=torch.long)
    dim_size = 8

    val, arg = _ARG_OPS[op_name](
        src.to("mps"), index.to("mps"), dim=0, dim_size=dim_size
    )
    val, arg = val.cpu(), arg.cpu()
    populated = arg < src.size(0)
    gathered = src.gather(0, arg.clamp(max=src.size(0) - 1))
    torch.testing.assert_close(
        gathered[populated], val[populated], rtol=1e-5, atol=1e-6
    )


# ---------------------------------------------------------------------------
# Independent cross-check against torch_scatter (optional dependency)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", list(_ARG_OPS))
@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_metal_heavy_contention_ties(op_name: str, dtype: torch.dtype) -> None:
    """Fused Metal path: heavy contention + many exact ties, value+arg exact.

    Few nodes and many integer-valued edges force thousands of collisions per
    output cell with frequent ties -- the case where an atomic race or a bad
    tie-break in the kernel would surface. Integer values in [-4, 4] are exactly
    representable in fp16/bf16, and the kernel promotes to float32 losslessly, so
    every dtype must match the CPU kernel exactly (first-occurrence tie-break).
    """
    torch.manual_seed(7)
    E, F, N = 60_000, 8, 32  # ~1900 edges/node
    src = torch.randint(-4, 5, (E, F)).to(dtype)  # integer values => many ties
    index = torch.randint(0, N, (E,), dtype=torch.long)

    val, arg = _ARG_OPS[op_name](src.to("mps"), index.to("mps"), dim=0, dim_size=N)
    ref_val, ref_arg = _ARG_OPS[op_name](src.cpu(), index.cpu(), dim=0, dim_size=N)
    assert val.dtype == dtype
    torch.testing.assert_close(val.cpu(), ref_val, rtol=0, atol=0)
    torch.testing.assert_close(arg.cpu(), ref_arg, rtol=0, atol=0)


@pytest.mark.parametrize("op_name", list(_ARG_OPS))
def test_fallback_genuine_2d_index(op_name: str) -> None:
    """A non-broadcast 2-D index bypasses the Metal fast path and stays correct."""
    torch.manual_seed(8)
    src = torch.randn(1000, 4)
    # Distinct target per (row, col): stride(1) != 0, so the kernel falls back.
    index = torch.randint(0, 20, (1000, 4), dtype=torch.long)

    val, arg = _ARG_OPS[op_name](src.to("mps"), index.to("mps"), dim=0, dim_size=20)
    ref_val, ref_arg = _ARG_OPS[op_name](src.cpu(), index.cpu(), dim=0, dim_size=20)
    torch.testing.assert_close(val.cpu(), ref_val, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(arg.cpu(), ref_arg, rtol=0, atol=0)


@pytest.mark.parametrize("op_name", ["scatter_min", "scatter_max"])
def test_cross_check_torch_scatter(op_name: str) -> None:
    """Cross-validate the MPS arg path against torch_scatter on CPU."""
    torch_scatter = pytest.importorskip("torch_scatter")
    reference = getattr(torch_scatter, op_name)

    torch.manual_seed(4)
    src = torch.randn(96, 5)
    index = torch.randint(0, 12, (96,), dtype=torch.long)
    dim_size = 12

    val, arg = _ARG_OPS[op_name](
        src.to("mps"), index.to("mps"), dim=0, dim_size=dim_size
    )
    ref_val, ref_arg = reference(src, index, dim=0, dim_size=dim_size)

    torch.testing.assert_close(val.cpu(), ref_val, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(arg.cpu(), ref_arg, rtol=0, atol=0)

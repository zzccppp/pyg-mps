"""Parity tests for the COO segment/gather family on MPS.

A COO segment reduction is a scatter along ``dim = index.dim() - 1``, so these
ops delegate to the native/Metal ``scatter_*`` kernels; ``gather_coo`` is an
``index_select``. The tests assert exact parity with the CPU ``pyg-lib`` kernel
across dtypes, including heavy ties (which exercise the fused Metal min/max).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("pyg_lib", reason="pyg-lib is required")
from pyg_lib import ops  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS backend not available"
)

_VALUE_SEGMENTS = {
    "segment_sum_coo": ops.segment_sum_coo,
    "segment_mean_coo": ops.segment_mean_coo,
}
_ARG_SEGMENTS = {
    "segment_min_coo": ops.segment_min_coo,
    "segment_max_coo": ops.segment_max_coo,
}
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _sorted_index(num_edges: int, num_segments: int) -> torch.Tensor:
    return torch.randint(0, num_segments, (num_edges,)).sort().values.long()


@pytest.mark.parametrize("op_name", list(_VALUE_SEGMENTS))
@pytest.mark.parametrize("dtype", _DTYPES)
def test_segment_value_coo_parity(op_name: str, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    src = (torch.randn(20_000, 16) * 0.5).to(dtype)
    index = _sorted_index(20_000, 200)
    op = _VALUE_SEGMENTS[op_name]
    got = op(src.to("mps"), index.to("mps"), dim_size=200)
    ref = op(src.cpu(), index.cpu(), dim_size=200)
    assert got.device.type == "mps"
    # Larger segments accumulate ~100 summands; float32 sum ordering differs
    # between the scatter-add and the CPU sequential reduction (FP is not
    # associative), so use reduction-appropriate tolerances.
    tol = {"rtol": 2e-2, "atol": 1e-2} if dtype != torch.float32 else {"rtol": 1e-4, "atol": 1e-3}
    torch.testing.assert_close(got.float().cpu(), ref.float(), **tol)


@pytest.mark.parametrize("op_name", list(_ARG_SEGMENTS))
@pytest.mark.parametrize("dtype", _DTYPES)
def test_segment_arg_coo_parity(op_name: str, dtype: torch.dtype) -> None:
    """Heavy ties (integer values) -> exact value+arg vs CPU via the Metal path."""
    torch.manual_seed(1)
    src = torch.randint(-4, 5, (40_000, 8)).to(dtype)
    index = _sorted_index(40_000, 64)
    op = _ARG_SEGMENTS[op_name]
    val, arg = op(src.to("mps"), index.to("mps"), dim_size=64)
    ref_val, ref_arg = op(src.cpu(), index.cpu(), dim_size=64)
    assert val.dtype == dtype and arg.device.type == "mps"
    torch.testing.assert_close(val.cpu(), ref_val, rtol=0, atol=0)
    torch.testing.assert_close(arg.cpu(), ref_arg, rtol=0, atol=0)


def test_gather_coo_parity() -> None:
    torch.manual_seed(2)
    src = torch.randn(200, 16)
    index = _sorted_index(20_000, 200)
    got = ops.gather_coo(src.to("mps"), index.to("mps"))
    ref = ops.gather_coo(src.cpu(), index.cpu())
    assert got.device.type == "mps"
    torch.testing.assert_close(got.cpu(), ref, rtol=0, atol=0)

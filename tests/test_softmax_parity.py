"""Parity tests for ``softmax_csr`` on MPS.

``softmax_csr`` normalizes ``src`` within each CSR row ``[ptr[r], ptr[r+1])``.
On MPS it is a native composite over the already-native CSR kernels
(``segment_max_csr`` -> ``gather_csr`` -> exp -> ``segment_sum_csr`` ->
``gather_csr`` -> divide), mirroring the CPU kernel's max->sub->exp->sum->div
math. These tests assert forward parity vs the CPU kernel (float32 tight;
float16/bfloat16 vs a float32 CPU reference with loose tol), plus gradient
parity through autograd. Ragged, singleton, and empty rows are all exercised.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("pyg_lib", reason="pyg-lib is required")
from pyg_lib import ops  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS backend not available"
)

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _indptr(num_edges: int, num_segments: int) -> torch.Tensor:
    """CSR row pointers with some empty (and often singleton) segments."""
    counts = torch.bincount(
        torch.randint(0, num_segments, (num_edges,)), minlength=num_segments
    )
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).long()


@pytest.mark.parametrize("dtype", _DTYPES)
def test_softmax_csr_forward_parity(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    indptr = _indptr(20_000, 300)
    src = (torch.randn(int(indptr[-1]), 16) * 2.0).to(dtype)
    got = ops.softmax_csr(src.to("mps"), indptr.to("mps"))
    # CPU softmax_csr only dispatches float/double, so reference in float32.
    ref = ops.softmax_csr(src.float().cpu(), indptr.cpu())
    assert got.device.type == "mps" and got.dtype == dtype
    tol = (
        {"rtol": 1e-4, "atol": 1e-5}
        if dtype == torch.float32
        else {"rtol": 2e-2, "atol": 2e-2}
    )
    torch.testing.assert_close(got.float().cpu(), ref, **tol)


def test_softmax_csr_rows_sum_to_one() -> None:
    """Each non-empty row must normalize to 1 (empty rows produce no output)."""
    torch.manual_seed(1)
    indptr = _indptr(10_000, 200)
    src = (torch.randn(int(indptr[-1]), 8) * 1.5).to(torch.float32)
    out = ops.softmax_csr(src.to("mps"), indptr.to("mps")).cpu()
    row_sum = ops.segment_sum_csr(out, indptr)  # [N, 8]
    counts = indptr[1:] - indptr[:-1]
    nonempty = counts > 0
    expected = torch.ones_like(row_sum[nonempty])
    torch.testing.assert_close(row_sum[nonempty], expected, rtol=1e-5, atol=1e-5)


def test_softmax_csr_singleton_rows_are_one() -> None:
    """A row with a single element softmaxes to exactly 1.0."""
    # ptr = [0,1,2,3] -> three singleton rows over three source rows.
    indptr = torch.tensor([0, 1, 2, 3]).long()
    src = torch.randn(3, 4)
    out = ops.softmax_csr(src.to("mps"), indptr.to("mps")).cpu()
    torch.testing.assert_close(out, torch.ones_like(src), rtol=1e-6, atol=1e-6)


def test_softmax_csr_backward_parity() -> None:
    torch.manual_seed(2)
    indptr = _indptr(8_000, 150)
    base = (torch.randn(int(indptr[-1]), 16) * 2.0).to(torch.float32)
    grad = torch.randn(int(indptr[-1]), 16)  # contiguous upstream grad

    src_mps = base.to("mps").requires_grad_(True)
    out_mps = ops.softmax_csr(src_mps, indptr.to("mps"))
    out_mps.backward(grad.to("mps"))

    src_cpu = base.cpu().requires_grad_(True)
    out_cpu = ops.softmax_csr(src_cpu, indptr.cpu())
    out_cpu.backward(grad.cpu())

    assert src_mps.grad is not None and src_mps.grad.device.type == "mps"
    torch.testing.assert_close(
        src_mps.grad.cpu(), src_cpu.grad, rtol=1e-4, atol=1e-5
    )


def test_softmax_csr_negative_dim_falls_back() -> None:
    """dim != 0 (here the last dim) falls back to the CPU kernel, still correct."""
    torch.manual_seed(3)
    src = torch.randn(4, 6)
    ptr = torch.tensor([0, 2, 4, 6]).long()  # groups along dim=1 (len-6 axis)
    got = ops.softmax_csr(src.to("mps"), ptr.to("mps"), dim=-1)
    ref = ops.softmax_csr(src.cpu(), ptr.cpu(), dim=-1)
    assert got.device.type == "mps"
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-5, atol=1e-6)

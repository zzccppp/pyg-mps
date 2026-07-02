"""Parity tests for ``grouped_matmul`` / ``segment_matmul`` on MPS.

Both are heterogeneous batched matmuls. On MPS they dispatch to Apple's GEMM
via ``at::matmul`` / ``at::mm`` / ``at::bmm``: ``grouped_matmul`` loops per pair,
``segment_matmul`` uses a single batched ``bmm`` when all segments share a row
count and a per-segment ``mm`` loop otherwise. These tests assert forward parity
vs the CPU kernel (f32 tight, f16 loose) across uniform, ragged, and empty
segments, plus gradient parity.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("pyg_lib", reason="pyg-lib is required")
from pyg_lib import ops  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS backend not available"
)

_DTYPES = [torch.float32, torch.float16]


def _tol(dtype: torch.dtype) -> dict:
    return (
        {"rtol": 1e-4, "atol": 1e-4}
        if dtype == torch.float32
        else {"rtol": 1e-2, "atol": 1e-2}
    )


# --- grouped_matmul --------------------------------------------------------

@pytest.mark.parametrize("dtype", _DTYPES)
def test_grouped_matmul_forward_parity(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    shapes = [(5, 16, 32), (3, 32, 8), (11, 8, 24), (1, 64, 4)]
    inputs = [(torch.randn(m, k) * 0.5).to(dtype) for m, k, _ in shapes]
    others = [(torch.randn(k, n) * 0.5).to(dtype) for _, k, n in shapes]

    got = ops.grouped_matmul([x.to("mps") for x in inputs],
                             [w.to("mps") for w in others])
    ref = ops.grouped_matmul([x.float() for x in inputs],
                             [w.float() for w in others])
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g.device.type == "mps" and g.dtype == dtype
        torch.testing.assert_close(g.float().cpu(), r, **_tol(dtype))


def test_grouped_matmul_backward_parity() -> None:
    torch.manual_seed(1)
    shapes = [(5, 16, 32), (7, 8, 24)]
    inputs = [torch.randn(m, k) for m, k, _ in shapes]
    others = [torch.randn(k, n) for _, k, n in shapes]
    grads = [torch.randn(m, n) for m, _, n in shapes]

    im = [x.to("mps").requires_grad_(True) for x in inputs]
    wm = [w.to("mps").requires_grad_(True) for w in others]
    outs_m = ops.grouped_matmul(im, wm)
    torch.autograd.backward(outs_m, [g.to("mps") for g in grads])

    ic = [x.clone().requires_grad_(True) for x in inputs]
    wc = [w.clone().requires_grad_(True) for w in others]
    outs_c = ops.grouped_matmul(ic, wc)
    torch.autograd.backward(outs_c, grads)

    for a, b in zip(im, ic):
        torch.testing.assert_close(a.grad.cpu(), b.grad, rtol=1e-4, atol=1e-4)
    for a, b in zip(wm, wc):
        torch.testing.assert_close(a.grad.cpu(), b.grad, rtol=1e-4, atol=1e-4)


# --- segment_matmul --------------------------------------------------------

@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "ptr_list",
    [
        [0, 4, 8, 12],       # uniform (batched bmm fast path)
        [0, 5, 8, 15, 16],   # ragged (per-segment mm loop)
        [0, 0, 6, 6, 10],    # empty leading/interior segments
    ],
)
def test_segment_matmul_forward_parity(dtype, ptr_list) -> None:
    torch.manual_seed(2)
    ptr = torch.tensor(ptr_list).long()
    total, k, n = int(ptr[-1]), 16, 32
    g = ptr.numel() - 1
    inp = (torch.randn(total, k) * 0.5).to(dtype)
    other = (torch.randn(g, k, n) * 0.5).to(dtype)

    got = ops.segment_matmul(inp.to("mps"), ptr.to("mps"), other.to("mps"))
    ref = ops.segment_matmul(inp.float(), ptr, other.float())
    assert got.device.type == "mps" and got.dtype == dtype
    assert got.shape == (total, n)
    torch.testing.assert_close(got.float().cpu(), ref, **_tol(dtype))


def test_segment_matmul_backward_parity() -> None:
    torch.manual_seed(3)
    ptr = torch.tensor([0, 5, 8, 15]).long()
    total, k, n = int(ptr[-1]), 16, 32
    g = ptr.numel() - 1
    inp = torch.randn(total, k)
    other = torch.randn(g, k, n)
    grad = torch.randn(total, n)

    im = inp.to("mps").requires_grad_(True)
    wm = other.to("mps").requires_grad_(True)
    out_m = ops.segment_matmul(im, ptr.to("mps"), wm)
    out_m.backward(grad.to("mps"))

    ic = inp.clone().requires_grad_(True)
    wc = other.clone().requires_grad_(True)
    out_c = ops.segment_matmul(ic, ptr, wc)
    out_c.backward(grad)

    torch.testing.assert_close(im.grad.cpu(), ic.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(wm.grad.cpu(), wc.grad, rtol=1e-4, atol=1e-4)

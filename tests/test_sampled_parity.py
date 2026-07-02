"""Parity tests for the ``sampled_op`` family (sampled_add/sub/mul/div) on MPS.

``sampled_op`` gathers ``left[left_index]`` / ``right[right_index]`` (or uses the
operands directly) and applies an elementwise op. On MPS it is a fused
gather+arith Metal kernel that resolves both source rows per output cell in one
pass; non-hot layouts fall back to a native ``index_select`` composite. These
tests assert exact/near-exact forward parity vs the CPU kernel across the 4 ops,
all index combinations, and f32/f16/bf16, plus gradient parity.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("pyg_lib", reason="pyg-lib is required")
from pyg_lib import ops  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS backend not available"
)

_OPS = {
    "sampled_add": ops.sampled_add,
    "sampled_sub": ops.sampled_sub,
    "sampled_mul": ops.sampled_mul,
    "sampled_div": ops.sampled_div,
}
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# index presence: (has_left_index, has_right_index)
_INDEX_MODES = [(True, True), (True, False), (False, True), (False, False)]


def _make(nl: int, nr: int, m: int, f: int, dtype: torch.dtype):
    torch.manual_seed(nl + nr + m + f)
    left = (torch.randn(nl, f) * 1.5).to(dtype)
    # keep magnitudes away from 0 so div stays well-conditioned across dtypes
    right = (torch.randn(nr, f).abs() + 0.5).to(dtype)
    li = torch.randint(0, nl, (m,)).long()
    ri = torch.randint(0, nr, (m,)).long()
    return left, right, li, ri


@pytest.mark.parametrize("op_name", list(_OPS))
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("mode", _INDEX_MODES)
def test_sampled_forward_parity(op_name, dtype, mode) -> None:
    has_li, has_ri = mode
    # When an operand has no index it must have exactly M rows (fused-path
    # contract); pick M accordingly per mode.
    m = 2_000
    nl = m if not has_li else 512
    nr = m if not has_ri else 512
    left, right, li, ri = _make(nl, nr, m, 16, dtype)
    li_arg = li if has_li else None
    ri_arg = ri if has_ri else None
    op = _OPS[op_name]

    got = op(left.to("mps"), right.to("mps"),
             None if li_arg is None else li_arg.to("mps"),
             None if ri_arg is None else ri_arg.to("mps"))
    ref = op(left.float().cpu(), right.float().cpu(), li_arg, ri_arg)
    assert got.device.type == "mps" and got.dtype == dtype
    tol = (
        {"rtol": 1e-5, "atol": 1e-6}
        if dtype == torch.float32
        else {"rtol": 2e-2, "atol": 2e-2}
    )
    torch.testing.assert_close(got.float().cpu(), ref, **tol)


@pytest.mark.parametrize("op_name", list(_OPS))
def test_sampled_backward_parity(op_name) -> None:
    left, right, li, ri = _make(512, 512, 2_000, 16, torch.float32)
    op = _OPS[op_name]
    grad = torch.randn(2_000, 16)

    lm = left.to("mps").requires_grad_(True)
    rm = right.to("mps").requires_grad_(True)
    out_m = op(lm, rm, li.to("mps"), ri.to("mps"))
    out_m.backward(grad.to("mps"))

    lc = left.cpu().requires_grad_(True)
    rc = right.cpu().requires_grad_(True)
    out_c = op(lc, rc, li, ri)
    out_c.backward(grad)

    assert lm.grad is not None and lm.grad.device.type == "mps"
    torch.testing.assert_close(lm.grad.cpu(), lc.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(rm.grad.cpu(), rc.grad, rtol=1e-4, atol=1e-5)


def test_sampled_int32_index_composite_fallback() -> None:
    """int32 indices are not eligible for the fused path -> composite fallback."""
    left, right, li, ri = _make(512, 512, 2_000, 16, torch.float32)
    li32, ri32 = li.int(), ri.int()
    got = ops.sampled_add(left.to("mps"), right.to("mps"),
                          li32.to("mps"), ri32.to("mps"))
    ref = ops.sampled_add(left.cpu(), right.cpu(), li32, ri32)
    assert got.device.type == "mps"
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-5, atol=1e-6)

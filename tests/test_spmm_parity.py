"""Parity tests for the fused ``spmm_csr`` GNN-aggregation kernel on MPS.

``spmm_csr`` computes ``out[i] = REDUCE_e weight[e] * x[col[e]]`` over the CSR
edge range of each destination node -- the fused gather+reduce that replaces
PyG's gather + ``scatter_add`` aggregation. On MPS it is an atomic-free per-row
Metal kernel. These tests assert parity vs a gather + scatter reference for
sum/mean/max, weighted and unweighted, across dtypes, including empty rows.
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
_REDUCES = ["sum", "mean", "max"]


def _csr(num_edges: int, num_nodes: int):
    """Random CSR-by-target graph with some empty rows."""
    tgt = torch.randint(0, num_nodes, (num_edges,)).sort().values
    col = torch.randint(0, num_nodes, (num_edges,)).long()
    deg = torch.bincount(tgt, minlength=num_nodes)
    indptr = torch.cat([deg.new_zeros(1), deg.cumsum(0)]).long()
    return indptr, col, deg


def _reference(x, indptr, col, weight, reduce):
    """gather + scatter reference (float32) for one destination-CSR graph."""
    n, f = indptr.numel() - 1, x.size(1)
    deg = indptr[1:] - indptr[:-1]
    row = torch.repeat_interleave(torch.arange(n), deg)
    msg = x.float().index_select(0, col)
    if weight is not None:
        msg = msg * weight.float().unsqueeze(-1)
    idx = row.unsqueeze(-1).expand(-1, f)
    if reduce == "max":
        out = torch.full((n, f), float("-inf"))
        out.scatter_reduce_(0, idx, msg, "amax", include_self=True)
        out[deg == 0] = 0.0
    else:
        out = torch.zeros(n, f)
        out.scatter_add_(0, idx, msg)
        if reduce == "mean":
            out = out / deg.clamp(min=1).unsqueeze(-1).float()
    return out


@pytest.mark.parametrize("reduce", _REDUCES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("weighted", [False, True])
def test_spmm_csr_parity(reduce: str, dtype: torch.dtype, weighted: bool) -> None:
    torch.manual_seed(0)
    n, e, f = 800, 12_000, 32
    indptr, col, _ = _csr(e, n)
    x = (torch.randn(n, f) * 0.5).to(dtype)
    w = (torch.randn(e) * 0.5).to(dtype) if weighted else None

    got = ops.spmm_csr(x.to("mps"), indptr.to("mps"), col.to("mps"),
                       None if w is None else w.to("mps"), reduce)
    ref = _reference(x, indptr, col, w, reduce)
    assert got.device.type == "mps" and got.dtype == dtype
    tol = ({"rtol": 1e-4, "atol": 1e-4} if dtype == torch.float32
           else {"rtol": 3e-2, "atol": 3e-2})
    torch.testing.assert_close(got.float().cpu(), ref, **tol)


def test_spmm_csr_matches_cpu_kernel() -> None:
    """MPS kernel matches the pyg-lib CPU kernel exactly-ish in float32."""
    torch.manual_seed(1)
    indptr, col, _ = _csr(20_000, 1_000)
    x = torch.randn(1_000, 16)
    w = torch.randn(int(indptr[-1]))
    for reduce in _REDUCES:
        got = ops.spmm_csr(x.to("mps"), indptr.to("mps"), col.to("mps"),
                           w.to("mps"), reduce).cpu()
        ref = ops.spmm_csr(x, indptr, col, w, reduce)
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_spmm_max_csr_value_and_arg(dtype: torch.dtype) -> None:
    """Max SpMM: value matches a scatter-amax reference; arg picks the winner."""
    torch.manual_seed(2)
    n, e, f = 600, 9_000, 24
    indptr, col, deg = _csr(e, n)
    x = (torch.randn(n, f) * 0.5).to(dtype)

    out, arg = ops.spmm_max_csr(x.to("mps"), indptr.to("mps"), col.to("mps"), None)
    ref = _reference(x, indptr, col, None, "max")
    assert out.device.type == "mps" and arg.device.type == "mps"
    assert arg.dtype == torch.long and arg.shape == (n, f)
    tol = ({"rtol": 1e-4, "atol": 1e-4} if dtype == torch.float32
           else {"rtol": 3e-2, "atol": 3e-2})
    torch.testing.assert_close(out.float().cpu(), ref, **tol)

    # arg consistency: for non-empty rows, x[arg] equals the reported max value.
    arg_c = arg.cpu()
    valid = arg_c < n
    gathered = torch.gather(x.float(), 0, arg_c.clamp(max=n - 1))
    torch.testing.assert_close(gathered[valid], out.float().cpu()[valid],
                               rtol=1e-3, atol=1e-3)
    # empty rows -> sentinel arg == n and value 0
    empty = deg == 0
    if empty.any():
        assert (arg_c[empty] == n).all()


def test_spmm_max_csr_matches_cpu_kernel() -> None:
    torch.manual_seed(3)
    indptr, col, _ = _csr(15_000, 800)
    x = torch.randn(800, 16)
    o_mps, a_mps = ops.spmm_max_csr(x.to("mps"), indptr.to("mps"),
                                    col.to("mps"), None)
    o_cpu, a_cpu = ops.spmm_max_csr(x, indptr, col, None)
    torch.testing.assert_close(o_mps.cpu(), o_cpu, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(a_mps.cpu(), a_cpu, rtol=0, atol=0)


def test_spmm_csr_all_empty_rows() -> None:
    """A graph with no edges yields an all-zero output."""
    n, f = 10, 4
    indptr = torch.zeros(n + 1, dtype=torch.long)
    col = torch.empty(0, dtype=torch.long)
    x = torch.randn(n, f)
    out = ops.spmm_csr(x.to("mps"), indptr.to("mps"), col.to("mps"), None, "sum")
    assert out.device.type == "mps"
    torch.testing.assert_close(out.cpu(), torch.zeros(n, f))

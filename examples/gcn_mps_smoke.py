#!/usr/bin/env python3
"""Minimal PyG model that runs on MPS without optional native PyG kernels."""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TinyGCN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = GCNConv(4, 8)
        self.conv2 = GCNConv(8, 2)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.conv1(data.x, data.edge_index).relu()
        return self.conv2(x, data.edge_index)


def main() -> int:
    device = pick_device()
    data = Data(
        x=torch.randn(6, 4),
        edge_index=torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 0, 2, 4],
                [1, 2, 3, 4, 5, 0, 2, 4, 0],
            ],
            dtype=torch.long,
        ),
        y=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long),
    ).to(device)

    model = TinyGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(10):
        optimizer.zero_grad()
        out = model(data)
        loss = torch.nn.functional.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

    print(f"device={device}")
    print(f"loss={float(loss.detach().cpu()):.6f}")
    print(f"logits_device={out.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

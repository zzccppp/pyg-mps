#!/usr/bin/env bash
set -euo pipefail

mkdir -p src

clone_or_update() {
  local url="$1"
  local path="$2"
  if [[ -d "${path}/.git" ]]; then
    git -C "${path}" fetch --depth 1 origin
  else
    git clone --depth 1 "${url}" "${path}"
  fi
}

clone_or_update https://github.com/pyg-team/pytorch_geometric.git src/pytorch_geometric
clone_or_update https://github.com/pyg-team/pyg-lib.git src/pyg-lib
clone_or_update https://github.com/rusty1s/pytorch_scatter.git src/pytorch_scatter
clone_or_update https://github.com/rusty1s/pytorch_sparse.git src/pytorch_sparse
clone_or_update https://github.com/rusty1s/pytorch_cluster.git src/pytorch_cluster
clone_or_update https://github.com/rusty1s/pytorch_spline_conv.git src/pytorch_spline_conv

git -C src/pyg-lib submodule update --init --recursive
git -C src/pytorch_sparse submodule update --init --recursive
git -C src/pytorch_scatter submodule update --init --recursive
git -C src/pytorch_cluster submodule update --init --recursive
git -C src/pytorch_spline_conv submodule update --init --recursive

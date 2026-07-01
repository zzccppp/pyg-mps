#!/usr/bin/env bash
set -euo pipefail

stage="${1:-probe}"
arg="${2:-}"

venv_dir="${UV_PROJECT_ENVIRONMENT:-.venv}"
python_bin="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"
venv_python="${venv_dir}/bin/python"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PWD}/.uv-cache}"

warn_sandbox() {
  if [[ "${CODEX_SANDBOX:-}" == "seatbelt" ]]; then
    cat >&2 <<'EOF'
Warning: CODEX_SANDBOX=seatbelt. MPS availability can be a false negative here.
Run MPS validation from a direct outer shell for authoritative results.
EOF
  fi
}

ensure_venv() {
  if [[ ! -x "${venv_python}" ]]; then
    uv venv --python "${python_bin}" "${venv_dir}"
  fi
}

cpu_count() {
  "${venv_python}" - <<'PY'
import os
print(os.cpu_count() or 4)
PY
}

apply_local_patches() {
  local patch
  for patch in patches/pyg-lib/*.patch; do
    [[ -e "${patch}" ]] || continue
    local rel_patch="../../${patch}"
    if git -C src/pyg-lib apply --check "${rel_patch}" 2>/dev/null; then
      git -C src/pyg-lib apply "${rel_patch}"
    elif git -C src/pyg-lib apply --reverse --check "${rel_patch}" 2>/dev/null; then
      true
    else
      echo "pyg-lib patch did not apply cleanly: ${patch}" >&2
      exit 1
    fi
  done
}

probe() {
  local probe_stage="${1:-${stage}}"
  local device="${2:-mps}"
  warn_sandbox
  mkdir -p reports
  "${venv_python}" scripts/mps_probe.py \
    --stage "${probe_stage}" \
    --device "${device}" \
    --out "reports/${probe_stage}-${device}.json"
}

probe_with_mps_fallback() {
  local probe_stage="${1:-${stage}-fallback}"
  warn_sandbox
  mkdir -p reports
  PYTORCH_ENABLE_MPS_FALLBACK=1 "${venv_python}" scripts/mps_probe.py \
    --stage "${probe_stage}" \
    --device mps \
    --out "reports/${probe_stage}-mps.json"
}

install_with_probe() {
  local probe_stage="$1"
  shift
  ensure_venv
  uv pip install "$@"
  probe "${probe_stage}" mps
}

torch_version() {
  "${venv_python}" - <<'PY'
import torch
print(torch.__version__.split("+", 1)[0])
PY
}

pyg_wheel_index() {
  local version
  version="$(torch_version)"
  printf 'https://data.pyg.org/whl/torch-%s+cpu.html\n' "${version}"
}

install_pyg_native() {
  local package="$1"
  ensure_venv
  local wheel_index="${PYG_WHL_INDEX:-$(pyg_wheel_index)}"
  uv pip install "${package}" --find-links "${wheel_index}"
  probe "${package}" mps
}

install_mps_stack() {
  ensure_venv
  uv pip install torch torch-geometric
  local wheel_index="${PYG_WHL_INDEX:-$(pyg_wheel_index)}"
  uv pip install pyg-lib --find-links "${wheel_index}"
}

install_local_pyg() {
  ensure_venv
  uv pip install --no-deps -e src/pytorch_geometric
}

install_local_pyg_lib() {
  ensure_venv
  apply_local_patches
  CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(cpu_count)}" \
    uv pip install --reinstall --no-build-isolation --no-deps -e src/pyg-lib
}

install_local_optionals() {
  ensure_venv
  uv pip install scipy
  uv pip install --reinstall --no-build-isolation --no-deps \
    -e src/pytorch_scatter \
    -e src/pytorch_sparse \
    -e src/pytorch_cluster \
    -e src/pytorch_spline_conv
}

verify_mps_stack() {
  ensure_venv
  probe pyg-macos-native mps || true
  probe_with_mps_fallback pyg-macos-fallback || true
  "${venv_python}" scripts/report_summary.py
}

case "${stage}" in
  init)
    ensure_venv
    "${venv_python}" --version
    ;;
  sync-sources)
    scripts/sync_sources.sh
    ;;
  bootstrap)
    install_mps_stack
    verify_mps_stack
    ;;
  install-mps)
    install_mps_stack
    ;;
  install-local-pyg)
    install_local_pyg
    ;;
  install-local-pyg-lib)
    install_local_pyg_lib
    ;;
  install-local-optionals)
    install_local_optionals
    ;;
  install-local)
    install_local_pyg
    install_local_pyg_lib
    ;;
  verify-mps)
    verify_mps_stack
    ;;
  meta)
    ensure_venv
    "${venv_python}" scripts/meta_probe.py
    ;;
  extensions-cpu)
    ensure_venv
    "${venv_python}" scripts/extension_probe.py --device cpu
    ;;
  extensions-mps)
    ensure_venv
    warn_sandbox
    "${venv_python}" scripts/extension_probe.py --device mps
    ;;
  parity)
    ensure_venv
    warn_sandbox
    "${venv_python}" -c "import pytest" 2>/dev/null || uv pip install pytest
    "${venv_python}" -m pytest tests/test_scatter_parity.py -v
    ;;
  benchmark)
    ensure_venv
    warn_sandbox
    "${venv_python}" scripts/benchmark_scatter.py
    ;;
  report)
    ensure_venv
    "${venv_python}" -c "import matplotlib" 2>/dev/null || uv pip install matplotlib
    "${venv_python}" scripts/plot_benchmarks.py
    ;;
  example)
    ensure_venv
    "${venv_python}" examples/gcn_mps_smoke.py
    ;;
  list)
    ensure_venv
    uv pip list
    ;;
  summary)
    ensure_venv
    "${venv_python}" scripts/report_summary.py
    ;;
  probe)
    ensure_venv
    probe "${arg:-manual}" mps
    ;;
  probe-fallback)
    ensure_venv
    probe_with_mps_fallback "${arg:-manual-fallback}"
    ;;
  probe-cpu)
    ensure_venv
    probe "${arg:-manual-cpu}" cpu
    ;;
  torch)
    install_with_probe torch torch
    ;;
  pyg-core)
    install_with_probe pyg-core torch-geometric
    ;;
  pyg-lib)
    install_pyg_native pyg-lib
    ;;
  pyg-lib-fallback)
    ensure_venv
    probe_with_mps_fallback pyg-lib-fallback
    ;;
  torch-scatter)
    install_pyg_native torch-scatter
    ;;
  torch-sparse)
    install_pyg_native torch-sparse
    ;;
  torch-cluster)
    install_pyg_native torch-cluster
    ;;
  torch-spline-conv)
    install_pyg_native torch-spline-conv
    ;;
  *)
    cat >&2 <<EOF
Unknown stage: ${stage}

Known stages:
  init
  sync-sources
  bootstrap
  install-mps
  install-local
  install-local-pyg
  install-local-pyg-lib
  install-local-optionals
  verify-mps
  meta
  extensions-cpu
  extensions-mps
  parity
  benchmark
  report
  example
  list
  summary
  probe [name]
  probe-fallback [name]
  probe-cpu [name]
  torch
  pyg-core
  pyg-lib
  pyg-lib-fallback
  torch-scatter
  torch-sparse
  torch-cluster
  torch-spline-conv
EOF
    exit 2
    ;;
esac

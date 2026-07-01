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

case "${stage}" in
  init)
    ensure_venv
    "${venv_python}" --version
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

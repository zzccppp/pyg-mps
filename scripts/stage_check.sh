#!/usr/bin/env bash
set -euo pipefail

stage="${1:-manual}"
device="${2:-mps}"

mkdir -p reports
"${PYTHON:-python3}" scripts/mps_probe.py \
  --stage "${stage}" \
  --device "${device}" \
  --out "reports/${stage}-${device}.json"

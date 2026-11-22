#!/usr/bin/env bash
# Run captum_ig_tpu.py on the configured TPU VM through the TPU agent suite's
# unified compute layer (no hand-written gcloud ssh/scp). Needs: this folder's
# .venv (setup.sh), gcloud auth, and `source ../scripts/env.sh` for gcloud.
set -euo pipefail
cd "$(dirname "$0")"
PKG=tpu-agent-suite@0.7.0
RT=~/.tpu-agent-suite/packages/$PKG/framework/runtime.py
PY=.venv/bin/python
OUT=results
mkdir -p "$OUT"
compute() { "$PY" "$RT" compute --package "$PKG" "$@"; }

echo "== preflight (tools, credentials, VM state)"; compute preflight
echo "== inspect (live chips on the VM, over ssh)";  compute inspect --json | tee "$OUT/inspect.json"
echo "== run captum_ig_tpu.py on the VM"
compute run --code captum_ig_tpu.py --timeout 900 \
  --output results.json:"$OUT/results.json" \
  --output xprof.xplane.pb:"$OUT/ig.xplane.pb" \
  -- --profile 2>&1 | tee "$OUT/run.log"
echo "== done: $OUT/results.json, $OUT/ig.xplane.pb"

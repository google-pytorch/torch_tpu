#!/usr/bin/env bash
# Reproduce this workspace from scratch: clone + build the TPU agent suite,
# install it here for Claude Code, create the host venv, save the compute
# profile. Idempotent. Does not touch the TPU VM (that is run_captum_ig.sh).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SUITE=${SUITE:-$HERE/../upstream/tpu-agent-suite}
VER=0.7.0

[ -d "$SUITE" ] || git clone --depth 1 https://github.com/google-pytorch/tpu-agent-suite "$SUITE"
uv venv --python 3.12 "$HERE/.venv" -q
uv pip install -q --python "$HERE/.venv/bin/python" -r "$SUITE/package/requirements-client.txt"
PY="$HERE/.venv/bin/python"

( cd "$SUITE" && "$PY" framework/compiler.py translate package --harness claude-code \
    --output dist/claude-code/tpu-agent-suite )
"$PY" "$SUITE/framework/installer.py" install "$SUITE/dist/claude-code/tpu-agent-suite" \
  --destination "$HERE" --non-interactive --skip-host-check

RT=~/.tpu-agent-suite/packages/tpu-agent-suite@$VER/framework/runtime.py
"$PY" "$RT" compute --package tpu-agent-suite@$VER setup apply --profile default --answers "$(cat <<JSON
{"backend":"gce_tpu_vm","project":"${TPU_PROJECT:-gen-lang-client-0320701242}",
 "zone":"${TPU_ZONE:-us-east5-b}","tpu_name":"${TPU_NAME:-xai-v6e4}",
 "python_interpreter":"~/env/bin/python3","internal_ip":false}
JSON
)"
"$PY" "$RT" config --package tpu-agent-suite@$VER show

#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

USAGE="Usage: $0 --bucket <gcs_bucket> --tpu <tpu_name> [--dest <install_dir>]"

BUCKET=""
TPU_NAME=""
DEST_DIR="/opt/torch_tpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)
      BUCKET="$2"
      shift 2
      ;;
    --tpu)
      TPU_NAME="$2"
      shift 2
      ;;
    --dest)
      DEST_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "$USAGE"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "$USAGE" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BUCKET" || -z "$TPU_NAME" ]]; then
  echo "Error: --bucket and --tpu are required" >&2
  echo "$USAGE" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_FILE="$SCRIPT_DIR/torch-tpu-rbe-agent.service"

rm -rf "$DEST_DIR/ci/tools/rbe_tpu"
mkdir -p "$DEST_DIR/ci/tools/rbe_tpu"
cp -r "$SCRIPT_DIR/." "$DEST_DIR/ci/tools/rbe_tpu/"

cat > /etc/default/torch-tpu-rbe-agent <<EOF
TORCH_TPU_RELAY_BUCKET=$BUCKET
TORCH_TPU_RELAY_TPU=$TPU_NAME
PYTHONPATH=$DEST_DIR
EOF

chmod 0600 /etc/default/torch-tpu-rbe-agent

cp "$SERVICE_FILE" /etc/systemd/system/torch-tpu-rbe-agent.service
systemctl daemon-reload
systemctl enable --now torch-tpu-rbe-agent.service
systemctl restart torch-tpu-rbe-agent.service

echo "Installed and started torch-tpu-rbe-agent for $TPU_NAME against bucket $BUCKET"

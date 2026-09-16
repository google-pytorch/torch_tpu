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

PROJECT="rbe-tpu-oss"
BUCKET="torch-tpu-mailbox-eval"
CONCURRENCY=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)
      BUCKET="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

deploy_one() {
  local line="$1"
  local vm zone
  vm="$(echo "$line" | awk '{print $1}')"
  zone="$(echo "$line" | awk '{print $2}')"
  [[ -n "$vm" && -n "$zone" ]] || return 0

  echo "[deploy] Starting $vm in $zone..."
  gcloud compute tpus tpu-vm scp --recurse \
    "${REPO_ROOT}/ci/tools/rbe_tpu" \
    "${vm}:/tmp/" \
    --zone="$zone" \
    --project="$PROJECT" >/dev/null 2>&1 || {
      echo "[deploy] ERROR: scp failed for $vm ($zone)" >&2
      return 1
    }

  gcloud compute tpus tpu-vm ssh "$vm" \
    --zone="$zone" \
    --project="$PROJECT" -- \
    "sudo /tmp/rbe_tpu/install_agent.sh --bucket $BUCKET --tpu $vm" >/dev/null 2>&1 || {
      echo "[deploy] ERROR: install failed for $vm ($zone)" >&2
      return 1
    }

  echo "[deploy] Completed $vm ($zone)"
}
export -f deploy_one
export REPO_ROOT PROJECT BUCKET

echo "Fetching TPU VM inventory from project $PROJECT..."
# Parse full resource name projects/rbe-tpu-oss/locations/<zone>/nodes/<name>
VM_LIST=$(gcloud compute tpus tpu-vm list --zone=- --project="$PROJECT" \
  --format="value(name.basename(), name.segment(3))")

echo "Deploying mailbox agent to 28 VMs with concurrency $CONCURRENCY..."
printf "%s\n" "$VM_LIST" | xargs -n 2 -P "$CONCURRENCY" bash -c 'deploy_one "$0 $1"'

echo "Checking registered fleet in gs://$BUCKET/fleet/..."
gcloud storage ls "gs://$BUCKET/fleet/" | wc -l
echo "Fleet deployment complete."

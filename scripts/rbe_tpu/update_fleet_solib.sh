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

update_one() {
  local line="$1"
  local vm zone
  vm="$(echo "$line" | awk '{print $1}')"
  zone="$(echo "$line" | awk '{print $2}')"
  [[ -n "$vm" && -n "$zone" ]] || return 0

  echo "[solib-sync] Starting $vm in $zone..."
  local remote_cmd
  remote_cmd=$(cat <<'EOF'
TOKEN=$(python3 -c "import urllib.request, json; req = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token', headers={'Metadata-Flavor': 'Google'}); print(json.loads(urllib.request.urlopen(req).read().decode())['access_token'])")
curl -s -f -H "Authorization: Bearer $TOKEN" "https://storage.googleapis.com/storage/v1/b/torch-tpu-mailbox-eval/o/base%2Fsolib.tar.gz?alt=media" -o /tmp/solib.tar.gz
sudo tar -xzf /tmp/solib.tar.gz -C /tmp/torch_tpu_relay/base
rm -f /tmp/solib.tar.gz
sudo sysctl -w vm.overcommit_memory=1 >/dev/null 2>&1 || true
EOF
)

  gcloud compute tpus tpu-vm ssh "$vm" \
    --zone="$zone" \
    --project="$PROJECT" -- \
    "$remote_cmd" >/dev/null 2>&1 || {
      echo "[solib-sync] ERROR: failed for $vm ($zone)" >&2
      return 1
    }

  echo "[solib-sync] Completed $vm ($zone)"
}
export -f update_one
export PROJECT BUCKET

echo "Fetching TPU VM inventory from project $PROJECT..."
VM_LIST=$(gcloud compute tpus tpu-vm list --zone=- --project="$PROJECT" \
  --format="value(name.basename(), name.segment(3))")

echo "Syncing solib to 28 VMs with concurrency $CONCURRENCY..."
printf "%s\n" "$VM_LIST" | xargs -n 2 -P "$CONCURRENCY" bash -c 'update_one "$0 $1"'

echo "Solib fleet sync complete."

#!/usr/bin/env bash
# Usage: scripts/tpu_vm.sh {create|setup|sync|run <cmd...>|delete}
# create/delete BILL or stop billing; run them deliberately.
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/env.sh
: "${TPU_PROJECT:?set TPU_PROJECT}"; : "${TPU_ZONE:?set TPU_ZONE}"
G=(--project "$TPU_PROJECT" --zone "$TPU_ZONE")
case "${1:-}" in
  create) gcloud compute tpus tpu-vm create "$TPU_NAME" "${G[@]}" \
            --accelerator-type "$TPU_TYPE" --version "$TPU_RUNTIME" ;;
  setup)  gcloud compute tpus tpu-vm ssh "$TPU_NAME" "${G[@]}" --command \
            'sudo apt-get update -qq && sudo apt-get install -y -qq python3.12 python3.12-venv && python3.12 -m venv ~/env && ~/env/bin/pip install -q torch_tpu "transformers>=5" captum pytest' ;;
  sync)   gcloud compute tpus tpu-vm scp --recurse torchtpu_xai tests scripts/tpu_smoke.py hf_tpu_bench.py \
            "$TPU_NAME":~/ "${G[@]}" ;;
  run)    shift; gcloud compute tpus tpu-vm ssh "$TPU_NAME" "${G[@]}" --command "cd ~ && $*" ;;
  delete) gcloud compute tpus tpu-vm delete "$TPU_NAME" "${G[@]}" --quiet ;;
  *) echo "usage: $0 {create|setup|sync|run <cmd>|delete}"; exit 1 ;;
esac

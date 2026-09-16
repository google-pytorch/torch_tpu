# Operator Guide: TPU Mailbox Runner with RBE

This guide explains how to set up, operate, and monitor the TPU mailbox runner.

## Overview

The mailbox runner connects RBE CPU workers to TPU VMs through a private Google Cloud Storage (GCS) bucket.

- **RBE CPU workers** run the worker process as a test runner wrapper. Each worker binds to one healthy TPU VM, uploads test inputs (blobs and layers) to the bucket, drops work tickets in the TPU VM's mailbox, and waits for results.
- **TPU VMs** run the agent process as a systemd daemon. Each agent heartbeats its health into the fleet registry, polls its private mailbox, executes tests locally on the TPU chip, and writes outputs and exit statuses back to the bucket.

## 1. GCS Bucket Setup

Create a dedicated private bucket with Uniform Bucket-Level Access. Do not use public cache buckets.

```bash
gcloud storage buckets create gs://torch-tpu-relay-mailbox \
  --project=rbe-tpu-oss \
  --location=us-central1 \
  --uniform-bucket-level-access
```

### Lifecycle Rules

Mailbox work items, results, and ephemeral logs should expire after 24 hours to prevent bucket bloat. Shared layer blobs can persist longer.

Create `lifecycle.json`:
```json
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {
        "age": 1,
        "matchesPrefix": ["mailbox/"]
      }
    },
    {
      "action": {"type": "Delete"},
      "condition": {
        "age": 7,
        "matchesPrefix": ["blobs/"]
      }
    }
  ]
}
```

Apply the lifecycle policy:
```bash
gcloud storage buckets update gs://torch-tpu-relay-mailbox --lifecycle-file=lifecycle.json
```

## 2. IAM and Service Accounts

Two roles interact with the bucket: the RBE worker identity and the TPU VM identity.

### Worker Service Account (`rbe-worker@...`)

The RBE CPU worker requires access to write work items and blobs, and read results:
- Storage Object Creator and Storage Object Viewer on `gs://torch-tpu-relay-mailbox` (or custom role granting read/write on `bindings/*`, `mailbox/*`, `blobs/*`, `fleet/*`).

### Agent Service Account (`tpu-vm@...`)

The TPU VM requires access to read work items, write results, update fleet registration, and release bindings:
- Storage Object User on `gs://torch-tpu-relay-mailbox`.

## 3. TPU Agent Deployment

Each TPU VM runs the agent daemon under systemd.

### Install on a Single VM

```bash
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --zone="${ZONE}" --project=rbe-tpu-oss -- \
  sudo /opt/torch_tpu/ci/tools/rbe_tpu/install_agent.sh \
    --bucket torch-tpu-relay-mailbox \
    --tpu "${TPU_NAME}"
```

### Deploy Across Fleet

To push updates across all 28 VMs in the fleet:

```bash
for vm in $(gcloud compute tpus tpu-vm list --project=rbe-tpu-oss --format="value(name)"); do
  zone=$(gcloud compute tpus tpu-vm list --project=rbe-tpu-oss --filter="name=$vm" --format="value(zone)")
  echo "Deploying agent to $vm in $zone..."
  gcloud compute tpus tpu-vm scp --recurse ci/tools/rbe_tpu "${vm}:/tmp/" --zone="$zone" --project=rbe-tpu-oss
  gcloud compute tpus tpu-vm ssh "$vm" --zone="$zone" --project=rbe-tpu-oss -- \
    "sudo /tmp/rbe_tpu/install_agent.sh --bucket torch-tpu-relay-mailbox --tpu $vm"
done
```

### Agent Service Management

On any TPU VM:
```bash
sudo systemctl status torch-tpu-rbe-agent
sudo journalctl -u torch-tpu-rbe-agent -f
```

## 4. RBE Worker Pool Configuration

Configure a Foundry/RBE worker pool with:
- `worker_count: 28` (1 worker per TPU VM in the fleet)
- `max_concurrent_actions: 1` (strict: guarantees 1:1 pairing and prevents queuing multiple tests on one worker)
- `network: PRIVATE` (requires Private Google Access to reach GCS)
- `user_service_accounts: ["rbe-worker@rbe-tpu-oss.iam.gserviceaccount.com"]`

## 5. Bazel Test Execution

Invoke tests pointing to the mailbox worker:

```bash
bazel test //tests/... \
  --config=rbe \
  --test_env=TORCH_TPU_RELAY_BUCKET=torch-tpu-relay-mailbox \
  --run_under=//ci/tools/rbe_tpu:worker_main
```

## 6. Health Signal and Quarantining

The agent continuously monitors local hardware health:
- Every test run output is classified for fatal TPU hardware errors (such as missing vfio device nodes or libtpu init crashes).
- If two consecutive test runs hit fatal infrastructure errors, the agent probes `/dev/vfio` and device nodes.
- If the probe fails, the agent revokes its binding immediately, sets its state to `quarantined` in `fleet/<tpu>.json`, and stops taking work.
- The active worker notices the revocation immediately, drops the broken chip, binds to another free chip, and retries the test.
- The quarantined agent probes every 60 seconds. After 3 consecutive healthy probes and the expiration of the backoff window (120s to 3600s), the agent marks itself healthy and re-enters the pool.

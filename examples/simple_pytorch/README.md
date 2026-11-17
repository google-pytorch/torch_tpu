# Simple PyTorch on TorchTPU

This directory provides a minimal single-device PyTorch example
(`simple_pytorch.py`) demonstrating TPU initialization and tensor placement
(`device="tpu"`), along with the reference single-device Kubernetes `Job`
manifest for running on Google Kubernetes Engine (GKE).

## Overview

`simple_pytorch.py` initializes the primary TPU device (`torch.device("tpu")`),
moves a CPU tensor to the TPU, and allocates a tensor directly on the TPU.

## Running the Example

### 1. From the Command Line (TPU VM or Container Shell)

From this directory, run:

```bash
python3 ./simple_pytorch.py
```

#### Expected Output

```text
TPU Initialization complete. Primary device: tpu:0
cpu_tensor is on: cpu
tpu_tensor is on: tpu:0
direct_tpu_tensor is on: tpu:0
```

### 2. Running on GKE

`torchtpu_singledevice_slice_job.yaml` is pre-configured to run
`simple_pytorch.py` inside the container (`command:` block):

```bash
cd /workspace/examples/simple_pytorch/
python3 ./simple_pytorch.py
```

To run it:

1.  If you haven't set up your cluster yet, complete the **Before we begin**
    section in
    [Get started with TorchTPU on GKE](../../docs/get_started_on_gke.md).
2.  Submit the job by following the **Run a single-device workload (one
    TensorCore of a 4-chip host)** section in the same guide. It uses this
    manifest.

## Files in This Directory

-   `simple_pytorch.py`: Minimal single-device PyTorch script on TPU.
-   `torchtpu_singledevice_slice_job.yaml`: Reference Kubernetes `Job` manifest
    for single-device workloads on TPU v7x.

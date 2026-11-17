# Manual Tensor Parallelism on TorchTPU

This directory provides a minimal example of **manual Tensor Parallelism (TP)**
on TorchTPU, where the model explicitly shards weights and invokes
`torch.distributed` collective operations (`all_reduce`) across TPU ranks.

## Overview

-   **`model.py`**: Implements a 3-layer feed-forward network (`MyModel`)
    supporting both unsharded execution (`FeedForward`) and Megatron-LM style
    tensor parallelism (`FeedForwardTensorParallel` using `ColumnParallelLinear`
    and `RowParallelLinear`).
-   **`tp_worker.py`**: Initializes the `tpu_dist` distributed process group,
    runs the unsharded model on CPU on rank 0 as a reference baseline, runs the
    tensor-parallel model across TPU cores (`device="tpu"`), and verifies
    numerical equivalence using `torch.testing.assert_close`.

## Running the Example

### 1. From the Command Line (TPU VM or Container Shell)

#### Single-Host (8 TPU cores via `torchrun`)

From this directory, run the single-host shell launcher:

```bash
bash ./launch_singlehost.sh
```

Or run via the Python elastic launcher:

<!-- doc-test: launch_doc_test -->

```bash
python3 ./launch.py
```

#### Multi-Host (via `torchrun`)

When `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` are set across
hosts (or auto-detected via `python3 -m torch.tpu.distributed`), run:

```bash
bash ./launch_multihost.sh
```

### 2. Running on GKE

If you haven't set up your cluster yet, first complete the **Before we begin**
section in
[Get started with TorchTPU on GKE](../../../docs/get_started_on_gke.md).

#### Single-Host (4 Chips / 8 TPU Cores)

`examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` is pre-configured
to run the single-host launcher inside the container (`command:` block):

```bash
cd /workspace/examples/tensor_parallel/manual/
bash ./launch_singlehost.sh
```

To submit it, follow the **Single host, multiple chips (4 chips / 8 devices on
one host)** section in
[Get started with TorchTPU on GKE](../../../docs/get_started_on_gke.md).

#### Multi-Host (64 Chips / 128 TPU Cores)

`examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml` is
pre-configured to run the multi-host launcher across a 16-pod (64-chip)
Kubernetes `JobSet` (`command:` block):

```bash
cd /workspace/examples/tensor_parallel/manual/
bash ./launch_multihost.sh
```

To submit it, follow the **Multiple hosts (64 chips / 128 devices across 16
hosts)** section in
[Get started with TorchTPU on GKE](../../../docs/get_started_on_gke.md).

## Files in This Directory

-   `model.py`: Unsharded and manual tensor-parallel feed-forward model
    definitions.
-   `tp_worker.py`: Distributed worker script running inference and numerical
    parity verification against a CPU reference.
-   `launch_singlehost.sh`: Single-host `torchrun` launcher script (8 TPU
    cores).
-   `launch_multihost.sh`: Multi-host `torchrun` launcher script.
-   `launch.py`: Python `torch.distributed.launcher` test runner.

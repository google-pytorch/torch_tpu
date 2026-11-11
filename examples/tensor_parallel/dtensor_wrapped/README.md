# DTensor Wrapper-Based Tensor Parallelism on TorchTPU

This directory demonstrates **Tensor Parallelism (TP)** on TorchTPU using
PyTorch's native `DTensor` and
`torch.distributed.tensor.parallel.parallelize_module` APIs—sharding a standard
`nn.Linear`-based model across TPU cores without rewriting the model definition.

## Overview

-   **`model.py`**: Defines a standard, unsharded 3-layer feed-forward network
    (`MyModel`) built with `torch.nn.Linear` and `torch.nn.SiLU`.
-   **`dtp_worker.py`**: Initializes a 1D TPU `DeviceMesh`, applies a
    declarative tensor-parallel plan (`PrepareModuleInput`, `ColwiseParallel`,
    `RowwiseParallel`, and `PrepareModuleOutput`) via `parallelize_module`, and
    verifies that the sharded TPU output matches the single-CPU reference model
    using `torch.testing.assert_close`.

## Running the Example

### 1. From the Command Line (TPU VM or Container Shell)

From this directory, run the 8-core TPU shell launcher (`torchrun`):

```bash
bash ./launch.sh
```

Or run via the Python elastic launcher:

<!-- doc-test: launch_doc_test -->

```bash
python3 ./launch.py
```

### 2. Running on GKE

To run this example on a single-host 4-chip (8-core) TPU v7x slice on GKE,
update the container `command:` in
`examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` to run:

```bash
cd /workspace/examples/tensor_parallel/dtensor_wrapped/
bash ./launch.sh
```

Then:

1.  If you haven't set up your cluster yet, complete the **Before we begin**
    section in
    [Get started with TorchTPU on GKE](../../../docs/get_started_on_gke.md).
2.  Submit the job by following the **Single host, multiple chips (4 chips / 8
    devices on one host)** section in
    [Get started with TorchTPU on GKE](../../../docs/get_started_on_gke.md).

## Files in This Directory

-   `model.py`: Standard `nn.Linear` feed-forward model definition.
-   `dtp_worker.py`: Distributed worker applying `parallelize_module` on a TPU
    `DeviceMesh` and verifying numerical parity.
-   `launch.sh`: `torchrun` CLI launcher script.
-   `launch.py`: Python `torch.distributed.launcher` test runner.

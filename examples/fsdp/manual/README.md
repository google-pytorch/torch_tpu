# Manual Fully Sharded Data Parallel (DIY FSDP) on TorchTPU

This directory provides a self-contained example of implementing **Fully Sharded
Data Parallel (FSDP)** from scratch ("DIY FSDP") on Cloud TPUs using explicit
PyTorch functional collectives (`torch.distributed._functional_collectives`),
activation checkpointing (`torch.utils.checkpoint`), and
`torch.compile(backend="tpu", fullgraph=True)`.

For the full tutorial and code walkthrough comparing unsharded layers with
`ManualFSDPLinear`, see [manual_fsdp.md](manual_fsdp.md).

## Overview

-   **Explicit Parameter Sharding & Functional Collectives**: Each rank stores a
    `1 / world_size` shard of each layer's weight matrix and reconstructs full
    weights during forward via `all_gather_tensor`, whose autograd adjoint
    automatically performs `reduce_scatter` during backward.
-   **ZeRO-3 Memory Behavior via Activation Checkpointing**: Wrapping each
    layer's forward pass in `torch.utils.checkpoint.checkpoint(...,
    use_reentrant=False)` frees the unsharded weight buffer immediately after
    forward and re-all-gathers it during backward.
-   **Whole-Graph Compilation**: Compiles the fused forward and backward step
    into a single XLA program with `@torch.tpu.distributed.spmd_safe` and
    `torch.compile(..., backend="tpu", fullgraph=True)`.

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

To run the ZeRO-3 memory verification test:

<!-- doc-test: manual_fsdp_memory_doc_test -->

```bash
python3 ./manual_fsdp_memory_test.py
```

### 2. Running on GKE

To run this example on a single-host 4-chip (8-core) TPU v7x slice on GKE,
update the container `command:` in
`examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` to run:

```bash
cd /workspace/examples/fsdp/manual/
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

-   `manual_fsdp.md`: Detailed walkthrough of the manual FSDP implementation and
    ZeRO-3 memory verification.
-   `model.py`: `ManualFSDPLinear` and `ManualSimpleNN` implementations.
-   `simple_model.py`: Unsharded reference `SimpleNN` model.
-   `manual_fsdp_worker.py`: Distributed training worker verifying numerical
    parity against the unsharded CPU reference.
-   `manual_fsdp_memory_test.py`: Memory test verifying ZeRO-3 activation
    checkpointing behavior.
-   `launch.sh`: `torchrun` CLI launcher script.
-   `launch.py`: Python `torch.distributed.launcher` test runner.

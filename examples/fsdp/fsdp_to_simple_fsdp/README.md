# SimpleFSDP on TorchTPU

This directory provides a self-contained example and side-by-side migration
guide for using **SimpleFSDP (`torch_tpu.examples.fsdp.simple_fsdp`)** on Cloud
TPUs.

For the full side-by-side comparison between PyTorch FSDPv2 (`fully_shard`) and
SimpleFSDP (`data_parallel`), see
[fsdp_to_simple_fsdp.md](fsdp_to_simple_fsdp.md).

## Overview

`simple_fsdp` is a compile-first Fully Sharded Data Parallel (FSDP)
implementation designed for compiler-optimized communication patterns. It was
initially introduced in the TorchTitan repo and a specific commit was vendored
into `torch_tpu.examples.fsdp.simple_fsdp`. `simple_fsdp` is one of the
recommended approaches to implement FSDP (and HSDP) on TorchTPU:

-   **DTensor Parametrization**: Shards parameters as `DTensor`s at rest and
    expresses forward `all-gather` and backward `reduce-scatter` operations as
    `DTensor.redistribute` calls when parameters are accessed during forward and
    backward (with the autograd backward of `redistribute` performing the
    gradient `reduce-scatter` during backward).
-   **Whole-Graph XLA Compilation**: Avoids dynamic Python runtime hooks so the
    entire forward pass, loss, and backward pass compile into a single XLA
    program without graph breaks.
-   **Single-Call Wrapping & HSDP**: Wraps the entire model via
    `simple_fsdp.data_parallel(...)` without manual per-layer `fully_shard`
    calls or prefetch bucket tuning. Like FSDPv2, Hybrid Sharded Data Parallel
    (`mode="hybrid_shard"` on a 2D `DeviceMesh`) can also be expressed directly.
-   **Composes with Activation & Distributed Checkpointing**: Composes directly
    with Activation Checkpointing (`torch.utils.checkpoint.checkpoint`), which
    reduces memory usage by rematerializing activations and freeing unsharded
    weights between layers, and Distributed Checkpointing
    (`torch.distributed.checkpoint`), which saves and loads weights directly in
    their sharded `DTensor` form.

## Quick Start: How to Use SimpleFSDP

### 1. Initialize the TPU DeviceMesh

```python
import torch
from torch import distributed as dist
import torch.distributed.tensor as dt

dist.init_process_group(backend="tpu_dist")
world_size = dist.get_world_size()
mesh = dt.init_device_mesh("tpu", (world_size,), mesh_dim_names=("fsdp",))
```

### 2. Apply Activation Checkpointing and `simple_fsdp.data_parallel`

Activation checkpointing (`torch.utils.checkpoint.checkpoint`) is recommended on
each parameterized submodule so that unsharded weights are freed after each
layer's forward pass and re-all-gathered during backward recomputation:

```python
import functools
from torch.utils import checkpoint as checkpoint_utils
from examples.fsdp import simple_fsdp

# 1. Configure mixed precision (optional)
mp_policy = simple_fsdp.MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
)

# 2. Checkpoint parameterized layers
for layer in model.children():
  if list(layer.parameters()):
    layer.forward = functools.partial(
        checkpoint_utils.checkpoint, layer.forward, use_reentrant=False
    )

# 3. Shard the model with SimpleFSDP
model = simple_fsdp.data_parallel(
    model,
    device_mesh=mesh,
    mode="fully_shard",  # "replicate", "fully_shard", or "hybrid_shard"
    mp_policy=mp_policy,
)
```

### 3. Compile the Training Step

> **Note on Gradient Scaling**: SimpleFSDP uses `Partial(reduce_op="sum")` for
> gradient reduction across data-parallel ranks (unlike FSDPv2, which averages
> gradients by default). Scale your loss by `1.0 / world_size` before calling
> `.backward()` (or scale your learning rate) to match average gradient
> reduction.

```python
compiled_model = torch.compile(model, backend="tpu")

def train_step(model, x, y, world_size):
  y_pred = model(x)
  loss = loss_fn(y_pred, y)
  (loss / world_size).backward()
  return loss.detach()
```

### 4. Additional Options

-   **Custom Sharding Dimension (`shard_dim`)**: Pass an `int` or a `dict[str,
    int]` mapping parameter names to the dimension along which they should be
    sharded (default `0`). Parameters whose size along the sharding dimension is
    smaller than the mesh dimension are automatically replicated.
-   **Ignoring Parameters (`ignored_params`)**: Pass a `set[nn.Parameter]` of
    parameters (such as frozen or tied weights) to skip during sharding.
-   **Inspecting Underlying `DTensor`s (`disable_active_parametrization`)**:
    Accessing `model.weight` normally invokes the all-gather parametrization.
    Use `with simple_fsdp.disable_active_parametrization():` to inspect the
    underlying sharded `DTensor` parameter or its `.grad` directly.

## Running the Example

From this directory, run the 8-core TPU launcher script:

```bash
./run_fsdp_to_simple_fsdp.sh
```

## Files in This Directory

-   `fsdp_to_simple_fsdp.md`: Detailed walkthrough migrating from PyTorch FSDPv2
    to SimpleFSDP.
-   `fsdp_to_simple_fsdp_worker.py`: Distributed worker script comparing
    compiled FSDPv2 and compiled SimpleFSDP step times and numerical parity.
-   `model.py`: `FeedForwardBlock` toy model used by the worker.
-   `run_fsdp_to_simple_fsdp.sh`: CLI launcher script.

# Migrating from FSDPv2 to SimpleFSDP on TorchTPU

This tutorial demonstrates how to migrate a model using **PyTorch FSDPv2
(`fully_shard`)** to **SimpleFSDP
(`torchtitan.experiments.graph_trainer.simple_fsdp.data_parallel`)** on TPUs.

Under most cases, we expect SimpleFSDP to provide significant speedups over
FSDPv2 for TPU-based training.

We assume you are familiar with
[FSDPv2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

## Why SimpleFSDP?

SimpleFSDP is a **compile-first** Fully Sharded Data Parallel implementation
designed specifically to maximize performance with `torch.compile` and XLA:

1.  **Compile-First Architecture**: Instead of dynamic runtime hooks, SimpleFSDP
    expresses parameter all-gathers and gradient reduce-scatters directly as
    `DTensor` redistributions on parameter access.
2.  **Single Whole-Graph Compilation**: `torch.compile(backend="tpu")` traces
    the entire forward and backward pass into a single XLA graph, allowing the
    XLA compiler to schedule and overlap communication and computation globally.
3.  **Simpler Code compared to FSDPv2**: No manual per-layer `fully_shard`
    wrapping or manual forward/backward prefetch bucket tuning
    (`set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch`).

--------------------------------------------------------------------------------

## 1. Check TPU Topology

First, verify the available TPU cores and topology:

```python
from torch_tpu._internal.distributed import tpu_topology

try:
  topology, count = tpu_topology.get_tpu_topology()
  print(f"Detected TPU Topology: {topology}")
  print(f"Available TPU Cores (Count): {count}")
except Exception as e:
  print(f"Failed to detect topology: {e}")
```

Output on an 8-core TPU (`v6e-8`):

```text
Detected TPU Topology: 2,4,1
Available TPU Cores (Count): 8
```

--------------------------------------------------------------------------------

## 2. Code Comparison: FSDPv2 vs SimpleFSDP

Below are the key changes needed when converting FSDPv2
(`torch.distributed.fsdp.fully_shard`) code to SimpleFSDP
(`torchtitan.experiments.graph_trainer.simple_fsdp.data_parallel`).

### Before: FSDPv2 (`fully_shard`)

```python
import functools
from torch.distributed import fsdp
from torch.utils import checkpoint as checkpoint_utils

# 1. Configure MixedPrecisionPolicy from torch.distributed.fsdp
mp_policy = fsdp.MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
)

# 2. Apply activation checkpointing and FSDP sharding to each parameterized
#    submodule, then shard the root module.
layers = [layer for layer in model.children() if list(layer.parameters())]
for layer in layers:
  layer.forward = functools.partial(
      checkpoint_utils.checkpoint, layer.forward, use_reentrant=False
  )
  fsdp.fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
fsdp.fully_shard(model, mesh=mesh, mp_policy=mp_policy)

# 3. Manually configure forward & backward prefetching for overlap
for i, layer in enumerate(layers):
  if i < len(layers) - 2:
    layer.set_modules_to_forward_prefetch([layers[i + 1], layers[i + 2]])
  if i >= 2:
    layer.set_modules_to_backward_prefetch([layers[i - 1], layers[i - 2]])
```

### After: SimpleFSDP (`data_parallel`)

```python
import functools
from torch.utils import checkpoint as checkpoint_utils
from torchtitan.experiments.graph_trainer import simple_fsdp

# 1. Configure MixedPrecisionPolicy from simple_fsdp
mp_policy = simple_fsdp.MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
)

# 2. Apply activation checkpointing to each parameterized submodule so
#    unsharded weights are freed after forward and re-all-gathered in backward.
for layer in model.children():
  if list(layer.parameters()):
    layer.forward = functools.partial(
        checkpoint_utils.checkpoint, layer.forward, use_reentrant=False
    )

# 3. Wrap the entire model in a single call (no manual prefetch tuning needed!)
model = simple_fsdp.data_parallel(
    model,
    device_mesh=mesh,
    mode="fully_shard",
    mp_policy=mp_policy,
)
```

--------------------------------------------------------------------------------

## 3. Compiling the Training Step & Gradient Reduction Scaling

With SimpleFSDP, the training step is compiled using `torch.compile(...,
backend="tpu")`. The parameter redistributions (`all-gather` in forward,
`reduce-scatter` in backward) are captured directly into the XLA graph.

> **Important Migration Note on Gradient Reduction**: PyTorch FSDPv2
> (`fully_shard`) defaults to `AVG` gradient reduction across data-parallel
> ranks, whereas SimpleFSDP uses `Partial(reduce_op="sum")` (sum reduction).
> When migrating from FSDPv2 to SimpleFSDP, scale the loss by `1.0 / world_size`
> before calling `.backward()` (or scale your learning rate by `1.0 /
> world_size`) to maintain identical gradient magnitude.

```python
def simple_fsdp_train_step(model, x, y, world_size):
  y_pred = model(x)
  loss = nn.MSELoss()(y_pred.float(), y.float())
  # Scale by 1/world_size to match FSDPv2's AVG gradient reduction
  (loss / world_size).backward()
  return loss.detach(), y_pred.detach()

compiled_train_step = torch.compile(simple_fsdp_train_step, backend="tpu")
```

--------------------------------------------------------------------------------

## 4. Launch Distributed Comparison and Numerical Verification

You can run the comparison script across 8 TPU cores using the shell launcher:

```bash
./run_fsdp_to_simple_fsdp.sh
```

### Expected Output

The script builds both a compiled FSDPv2 model (`torch.compile(fsdp2_model,
backend="tpu")`) and a compiled SimpleFSDP model
(`torch.compile(simple_fsdp_train_step, backend="tpu")`) from identical initial
weights, runs both across 5 training steps, and verifies numerical equivalence
(`assert_close`) and steady-state step speedups:

```text
Step 0: FSDPv2 loss=0.952936 (6664.90 ms) | SimpleFSDP loss=0.952937 (7038.18 ms)
Step 1: FSDPv2 loss=0.952835 (281.05 ms) | SimpleFSDP loss=0.952836 (80.30 ms)
Step 2: FSDPv2 loss=0.952739 (170.04 ms) | SimpleFSDP loss=0.952740 (214.29 ms)
Step 3: FSDPv2 loss=0.952636 (168.63 ms) | SimpleFSDP loss=0.952636 (45.68 ms)
Step 4: FSDPv2 loss=0.952551 (187.02 ms) | SimpleFSDP loss=0.952553 (46.23 ms)
✅ FSDPv2 to SimpleFSDP migration tutorial verified successfully!
```

We see some variance on time as there is no warm-up. However, we can see that
SimpleFSDP has a clear speed advantage over FSDPv2.

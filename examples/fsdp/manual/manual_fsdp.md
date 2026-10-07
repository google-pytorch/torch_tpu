# Manual Fully Sharded Data Parallel (DIY FSDP) on TorchTPU

This tutorial demonstrates how to implement **Fully Sharded Data Parallel
(FSDP)** from scratch ("DIY FSDP") on TPUs using explicit PyTorch functional
collectives (`torch.distributed._functional_collectives`), activation
checkpointing (`torch.utils.checkpoint`), and `torch.compile(backend="tpu")`.

Building FSDP by adding collectives directly into the model definition will give
similar performance as `simple_fsdp`. In both cases, the XLA compiler will
optimize the timing of collectives as well as combining smaller collectives when
profitable.

Unlike `simple_fsdp`, adding collectives directly into the model definition
clearly places the computation and communications code in the same place, aiding
in performance tuning. The tradeoff is that it requires changes to the model
definition code, which may not always be feasible.

## How Manual FSDP Works on TorchTPU

1.  **Parameter Sharding**: Each rank stores only a `1 / world_size` shard of
    each layer's weight matrix in TPU memory.
2.  **Functional Collectives & Autograd Adjoints**: During the forward pass,
    each layer reconstructs its full weight tensor via
    `all_gather_tensor(self.weight, gather_dim=0, group=self.group)`. Because
    PyTorch functional collectives register autograd adjoint formulas, the
    backward pass automatically invokes the corresponding `reduce_scatter` to
    produce sharded weight gradients on each rank—without writing a custom
    `torch.autograd.Function`.
3.  **ZeRO-3 Memory Behavior via Activation Checkpointing**: Wrapping each
    layer's forward computation in `torch.utils.checkpoint.checkpoint(...,
    use_reentrant=False)` ensures that the unsharded `all_gather` weight buffer
    is freed immediately after the layer's forward pass and re-materialized
    (re-all-gathered) during the backward pass, rather than being saved across
    the forward-backward boundary.
4.  **Whole-Graph Compilation**: Compiling the fused forward and backward step
    with `torch.compile(..., backend="tpu", fullgraph=True)` traces the
    all-gathers, matmuls, and reduce-scatters into a single XLA computation.

--------------------------------------------------------------------------------

## 1. Check TPU Topology

First, verify the available TPU cores and topology:

```python
import os
import torch
from torch.tpu import distributed

try:
  distributed.set_tpu_launch_env()
  topology = os.environ.get("TORCH_TPU_TOPOLOGY")
  count = torch.accelerator.device_count()
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

## 2. Code Comparison: Unsharded Layer vs. Manual FSDP Layer

Below is the comparison between a standard unsharded model (`SimpleNN` in
`simple_model.py`) and the manually sharded implementation (`ManualFSDPLinear`
and `ManualSimpleNN` in `model.py`).

### Reference: Unsharded Model (`simple_model.py`)

```python
import torch
from torch import nn


class SimpleNN(nn.Module):
  """Two linear layers and a nonlinear activation (reference unsharded model)."""

  def __init__(self, in_features: int, out_features: int):
    super().__init__()
    self.linear1 = nn.Linear(in_features, in_features)
    self.silu = nn.SiLU()
    self.linear2 = nn.Linear(in_features, out_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.linear1(x)
    x = self.silu(x)
    x = self.linear2(x)
    return x
```

### Manual FSDP Model (`model.py`)

```python
import math
from typing import Optional

import torch
from torch import distributed as dist
from torch import nn
import torch.distributed._functional_collectives as _func_col
import torch.nn.functional as F
from torch.utils import checkpoint as checkpoint_utils


class ManualFSDPLinear(nn.Module):
  """Linear layer implementing manual Fully Sharded Data Parallelism.

  The weight matrix (out_features, in_features) is sharded along dimension 0
  across all ranks, while the bias vector (out_features,) is kept unsharded.
  Collectives have adjoint formulas registered with PyTorch autograd,
  allowing all-gather in forward to automatically reduce-scatter in backward.
  """

  def __init__(
      self,
      in_features: int,
      out_features: int,
      group: Optional[dist.ProcessGroup] = None,
  ):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.group: dist.ProcessGroup = (
        group
        if group is not None
        else dist.distributed_c10d._get_default_group()
    )

    world_size = dist.get_world_size(self.group)
    if out_features % world_size != 0:
      raise ValueError(
          f"out_features ({out_features}) must be divisible by world_size"
          f" ({world_size})"
      )
    self.shard_out_features = out_features // world_size

    # Sharded weight parameter: shape (out_features // world_size, in_features)
    self.weight = nn.Parameter(
        torch.empty(self.shard_out_features, in_features)
    )
    # Unsharded bias parameter: shape (out_features,)
    self.bias = nn.Parameter(torch.empty(out_features))

    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Initializes weights with Kaiming uniform and biases to zero."""
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    nn.init.zeros_(self.bias)

  def get_full_weight(self) -> torch.Tensor:
    """Reconstructs the full weight tensor via all-gather across all ranks."""
    return _func_col.all_gather_tensor(
        self.weight, gather_dim=0, group=self.group
    )

  def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, self.get_full_weight(), self.bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Wrap the layer in activation checkpointing so unsharded weights
    # are freed immediately after forward and re-all-gathered during backward.
    return checkpoint_utils.checkpoint(
        self._forward_impl,
        x,
        use_reentrant=False,
    )


class ManualSimpleNN(nn.Module):
  """Two-layer model using manual FSDP linear layers."""

  def __init__(
      self,
      in_features: int,
      out_features: int,
      group: Optional[dist.ProcessGroup] = None,
  ):
    super().__init__()
    self.linear1 = ManualFSDPLinear(
        in_features=in_features,
        out_features=in_features,
        group=group,
    )
    self.silu = nn.SiLU()
    self.linear2 = ManualFSDPLinear(
        in_features=in_features,
        out_features=out_features,
        group=group,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.linear1(x)
    x = self.silu(x)
    x = self.linear2(x)
    return x
```

--------------------------------------------------------------------------------

## 3. Compiling the Fused Forward & Backward Step

In `manual_fsdp_worker.py`, the forward and backward passes are fused into a
single compiled function using `torch.compile(..., backend="tpu",
fullgraph=True)`:

*   **SPMD-Safe Decoration (`@torch.tpu.distributed.spmd_safe`)**: Marks the
    function as executing identical TPU operations and collectives in the same
    order and with the same shapes across all ranks, allowing TorchTPU to fuse
    collectives with surrounding operations into the same compiled graph.
*   **Loss Scaling (`loss / world_size`)**: Because the backward adjoint of
    `all_gather_tensor` is a **sum** `reduce_scatter` across ranks, dividing the
    local MSE loss by `world_size` produces the exact average gradient across
    the global batch.
*   **Unsharded Bias Gradient Reduction**: Because `bias` is replicated rather
    than sharded across ranks, its local gradients are explicitly summed across
    ranks using `_func_col.all_reduce(m.bias.grad, "sum", group=m.group)`.

```python
@torch.tpu.distributed.spmd_safe
def fwd_and_bwd(
    model_to_run: nn.Module, data: torch.Tensor, y_true: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  y_pred = model_to_run(data)
  loss = nn.functional.mse_loss(y_pred, y_true) / world_size
  loss.backward()
  # Unsharded parameters (biases) are replicated across ranks, so their
  # gradients are summed across ranks via all_reduce.
  for m in model_to_run.modules():
    if isinstance(m, model.ManualFSDPLinear) and m.bias.grad is not None:
      m.bias.grad = _func_col.all_reduce(m.bias.grad, "sum", group=m.group)
  return loss.detach(), y_pred.detach()


# Compiles the fused forward and backward step with torch.compile.
compiled_fwd_and_bwd = torch.compile(
    fwd_and_bwd, backend="tpu", fullgraph=True
)
```

--------------------------------------------------------------------------------

## 4. Launch Distributed Training and Numerical Verification

You can run the manual FSDP worker across 8 TPU cores using the shell launcher
(`torchrun`):

```bash
./launch.sh
```

Or via the Python elastic launcher:

```bash
python3 launch.py
```

### Expected Output

`manual_fsdp_worker.py` trains the compiled `ManualSimpleNN` model across 8 TPU
cores for 10 steps (`in_features=4096`, `out_features=8192`, `bfloat16`),
reconstructs an identical unsharded `SimpleNN` reference model on CPU on rank 0,
and verifies with `torch.testing.assert_close` that the sharded gradients and
predictions match the single-device reference at every step:

```text
I0406 20:51:04.244971 1335197 manual_fsdp_worker.py:159] rank: 0, tpu_tp_output: tensor([[ 0.2590, -0.3518, -0.0562,  ...,  0.3712, -0.4614,  0.0546],
        [-0.0882,  0.2994,  0.1932,  ...,  0.3408,  0.2054, -0.5855],
        [-0.1815,  0.1157,  0.2587,  ..., -0.3204,  0.2286, -0.1753],
        ...,
        [ 0.1436, -0.2248, -0.2446,  ..., -0.0322, -0.0315,  0.3694],
        [ 0.1497, -0.0725, -0.1072,  ...,  0.1292,  0.3652, -0.0369],
        [-0.5171,  0.1021, -0.4535,  ..., -0.4090,  0.5435, -0.2597]],
       grad_fn=<AddmmBackward0>)
```

--------------------------------------------------------------------------------

## 5. Verify ZeRO-3 Memory Behavior

To verify that activation checkpointing frees the unsharded weights (`32.0 MiB`
for the `4096x4096` layer and `64.0 MiB` for the `8192x4096` layer in
`bfloat16`) during the forward pass instead of retaining them for backward, run
`manual_fsdp_memory_test.py`:

```bash
python3 manual_fsdp_memory_test.py
```

### Expected Output

Each rank holds only `12.02 MiB` of sharded weights (`4.0 MiB + 8.0 MiB` plus
biases) before forward, retains only `0.13 MiB` of activations across the
forward-backward boundary (`< 20 MiB` threshold), and frees the transient
all-gather buffers (`fwd_peak_delta=96.38 MiB` in steady state):

```text
I0406 20:54:56.826658 1403959 manual_fsdp_memory_test.py:99] Memory summary across steps:
  step 0: mem_before_fwd=12.28 MiB, mem_after_fwd=12.41 MiB, fwd_retained=0.13 MiB, fwd_peak_delta=192.38 MiB
  step 1: mem_before_fwd=12.02 MiB, mem_after_fwd=12.15 MiB, fwd_retained=0.13 MiB, fwd_peak_delta=96.38 MiB
  step 2: mem_before_fwd=12.02 MiB, mem_after_fwd=12.15 MiB, fwd_retained=0.13 MiB, fwd_peak_delta=96.38 MiB
  step 3: mem_before_fwd=12.02 MiB, mem_after_fwd=12.15 MiB, fwd_retained=0.13 MiB, fwd_peak_delta=96.38 MiB
```

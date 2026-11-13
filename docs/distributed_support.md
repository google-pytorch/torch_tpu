# TorchTPU distributed support

## Distributed support

TorchTPU implements distributed communication at the PyTorch ProcessGroup level
(`c10d::Backend`) under the `tpu_dist` backend name. By intercepting collectives
at the ProcessGroup layer, TorchTPU supports standard `torch.distributed` APIs
in both eager mode and `torch.compile`, and works directly with PyTorch
distributed frameworks including DTensor, DistributedDataParallel (DDP), FSDPv2,
and [SimpleFSDP](https://arxiv.org/abs/2411.00284).

Like PyTorch on GPU, TorchTPU enforces a one-process-per-device execution model.
Under the hood, TPU-to-TPU communication does not use a separate runtime
communication library (such as NCCL); instead, collective operations are lowered
directly into
[StableHLO collective operations](https://openxla.org/stablehlo/spec#collective_ops)
(such as `stablehlo.all_reduce`, `stablehlo.all_gather`,
`stablehlo.reduce_scatter`, and `stablehlo.all_to_all`) and compiled by XLA into
the TPU executable.

### Key architectural differences from GPU

While TorchTPU preserves standard `torch.distributed` semantics, programming
TPUs via StableHLO and PjRT introduces three key differences compared to GPU
backends:

*   **Pre-initialization hardware topology discovery:** On a multi-GPU host,
    each process launched by torchrun can inspect all attached GPUs, select its
    assigned device via `cuda:{local_rank}`, and move models to that device
    before calling `torch.distributed.init_process_group()`. On a TPU host, each
    worker process is bound to a single TPU device and cannot own the full set
    of devices on its own. Before initializing the PjRT runtime or calling
    `torch.distributed.init_process_group()`, the environment must specify the
    physical mesh geometry of the
    [TPU slice](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)
    (`TORCH_TPU_TOPOLOGY`), the local coordinator addresses used to construct
    the Inter-Chip Interconnect (ICI) mesh (`TORCH_TPU_SLICEBUILDER_ADDRESSES`),
    and the total number of participating TPU devices (`WORLD_SIZE`).
*   **Global Replica Group Resolution for Subgroups:** In PyTorch, a collective
    call such as
    [`torch.distributed.all_reduce`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)
    only receives the ProcessGroup (subgroup) in which the calling rank
    participates. In contrast,
    [`stablehlo.all_reduce`](https://openxla.org/stablehlo/spec#all_reduce)
    requires a complete description of all participating subgroups across the
    default process group (`replica_groups`), even those the current process
    does not belong to:

    ```
    %0 = "stablehlo.all_reduce"(%arg0) <{
      replica_groups = dense<[[4, 0, 1, 3], [5, 2, 6, 7]]> : tensor<2x4xi64>
    }> ({ ... })
    ```

    TorchTPU's `tpu_dist` backend automatically exchanges subgroup membership
    across all ranks in the default process group during collective setup and
    constructs the global `replica_groups` and `device_assignment` required by
    XLA.
*   **SPMD Compilation vs. MPMD Execution:** The XLA compiler and TPU
    communication stack are designed for Single Program, Multiple Data (SPMD)
    execution, where every rank compiles and executes an identical StableHLO
    module containing matching collective operations. However, PyTorch programs
    can exhibit Multiple Program, Multiple Data (MPMD) behavior across ranks—for
    example, due to rank-conditional branches (`if dist.get_rank() == 0:`) or
    uneven tensor sharding when a dimension is not evenly divisible by the
    number of devices. If different ranks execute different StableHLO modules
    containing collectives, mismatched synchronization flag IDs or buffer
    offsets across ranks can cause runtime deadlocks or memory corruption.

### Collective execution and fusion modes

TorchTPU runs each collective as a skinny collective. To let XLA overlap compute
and communication, you can opt in to either or both of two mechanisms that fuse
collectives with the surrounding computation. The two are independent and can be
combined: collectives inside an `@spmd_safe` function (eager or covering a whole
`torch.compile` graph) are always fused, and, when enabled, the Compilation
Handshake decides for any other `torch.compile` graphs.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Mode | Configuration | Graph Behavior | Best For |
| :---- | :---- | :---- | :---- |
| Skinny collectives (default behavior) | Always on for any collective that the opt-ins below don't fuse | Compiles each collective op into its own standalone ("skinny") StableHLO module by inserting graph breaks around collectives. | General MPMD safety out of the box, at the cost of preventing XLA from overlapping compute and communication. |
| `@spmd_safe` decorator (opt-in) | `@torch.tpu.distributed.spmd_safe` | Disables graph breaks around collectives inside the decorated function, fusing compute and collectives into a single StableHLO module. | Known-SPMD functions or training steps (e.g., SimpleFSDP or manual tensor parallelism) to maximize XLA compute-communication overlap. |
| Compilation Handshake (opt-in) | `TORCH_TPU_SPMD_HANDSHAKE_STAGE=COMPILE_STAGE` or `DISPATCH_STAGE` (default: OFF) | In torch.compile graphs, checks that the compiled FX graphs match across all ranks; fuses collectives if they do and falls back to skinny collectives if they don't. | Automatically enabling collective fusion in `torch.compile` while guarding against accidental MPMD divergence. |
<!-- mdformat on -->

#### Using `@torch.tpu.distributed.spmd_safe`

When you know a function or model step is strictly SPMD across all ranks (no
rank-dependent control flow or uneven tensor shapes), decorate it with
`@torch.tpu.distributed.spmd_safe`. Under `torch.compile` (and `DEFER_AND_FUSE`
eager mode), this removes graph breaks around collectives so XLA can schedule
asynchronous communication alongside matrix computation:

```python
import torch
import torch.distributed as dist

@torch.compile
@torch.tpu.distributed.spmd_safe
def spmd_train_step(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
  out = x @ weight
  dist.all_reduce(out, op=dist.ReduceOp.AVG)
  return torch.nn.functional.silu(out)
```

Only apply `@torch.tpu.distributed.spmd_safe` to code that executes the exact
same sequence of tensor and collective operations on every rank; applying it to
MPMD code can produce different StableHLO modules across ranks and cause TPU
execution deadlocks.

Under `torch.compile`, `@spmd_safe` must cover the whole compiled function, as
in the example above. Compiling a function in which only some of the operations
are `@spmd_safe` raises an error; compile the `@spmd_safe` function directly
instead. On PyTorch earlier than 2.16, a graph break inside an `@spmd_safe`
function causes TorchDynamo to run the wrapper frame eagerly and drop the
annotation on the traced subgraphs, so pass `fullgraph=True` when compiling
functions that should not graph break.

#### Configuring the compilation handshake

Instead of manually annotating functions with `@spmd_safe`, you can enable
TorchTPU's Compilation Handshake via the `TORCH_TPU_SPMD_HANDSHAKE_STAGE`
environment variable. When `torch.compile` traces a graph containing global
collectives, TorchTPU runs a consensus check across all ranks to verify that
their compiled FX graphs match before fusing collectives into the graph:

*   `OFF` (Default):** Disables the handshake; collectives use default Skinny
    Collectives unless marked `@torch.tpu.distributed.spmd_safe`.
*   `COMPILE_STAGE`: Verifies graph consensus across ranks prior to PjRT
    compilation. If all ranks produce identical graphs, graph breaks around
    collectives are elided. Recommended for standard `torch.compile` workloads
    where all ranks compile together.
*   `DISPATCH_STAGE`: Compiles collectives without graph breaks and verifies
    consensus across ranks at dispatch time, recompiling with split collectives
    if ranks diverge. This mode is necessary for workloads that can trigger
    rank-specific recompilations; it has higher overhead than `COMPILE_STAGE`.

Below is an example of rank-specific recompilation that is supported by
`DISPATCH_STAGE`, whereas `COMPILE_STAGE` would time out because only
`divergent_rank` recompiles on iteration 1:

```python
iteration = 0
divergent_rank = ...

def divergence_after_iteration(x, y):
  z = torch.add(x, y)
  if rank == divergent_rank and iteration == 1:
    z = torch.mul(z, 5.0)
  dist.all_reduce(z)
  return z

compiled_fn = torch.compile(divergence_after_iteration, backend=backend)
res0 = compiled_fn(x, y)
iteration += 1
# IMPORTANT: If Handshake is in COMPILE_STAGE a recompilation happens only on `divergent_rank`
# this would initiate a Handshake ONLY ON THAT RANK which would timeout which is why we provide
# a DISPATCH_STAGE mode.
res1 = compiled_fn(x, y)
```

## Distributed initialization

Initializing a distributed workload in TorchTPU uses standard
`torch.distributed` APIs, paired with TorchTPU's launch environment helper to
populate the TPU topology and slice-builder environment variables:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Environment Variable | Description | Example |
| :---- | :---- | :---- |
| `TORCH_TPU_TOPOLOGY` | Physical mesh geometry of the TPU slice. | `"2,2,1"`, `"2,2,1,2"`, `"4,4,4"` |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | Comma-separated `host:port` addresses (one per device) used to construct the ICI mesh. | `"localhost:8471,localhost:8472,..."` |
<!-- mdformat on -->

You need to set these variables before starting the workers. In common setups (a
single host, or a GKE TPU slice), TorchTPU can determine the correct values for
you: run

```bash
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
```

In your launch script before torchrun, or call
`torch.tpu.distributed.set_tpu_launch_env()` in your parent Python launcher
process before spawning workers. Both leave any variable you have already set
unchanged, so you can override individual values.

Below is the initialization flow for a distributed TorchTPU workload, followed
by a complete worker and launcher example:

![Distributed Initialization Flow](./images/distributed.png)

### 1. Worker script (`train.py`)

Inside each worker process, initialize the `tpu_dist` backend and acquire the
local TPU device via `torch.device("tpu")`:

```python
import torch
import torch.distributed as dist

def worker_main():
  # 1. Initialize the distributed process group using the tpu_dist backend`
  dist.init_process_group(backend="tpu_dist")

  # 2. Query rank, world size, and acquire the local TPU device
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  device = torch.device("tpu")

  print(f"[Rank {rank}/{world_size}] Initialized on {device}")

  # 3. Perform a distributed collective operation across TPU workers
  x = torch.ones(4, 4, device=device) * (rank + 1)
  dist.all_reduce(x, op=dist.ReduceOp.SUM)

  print(f"[Rank {rank}] Reduced sum: {x[0, 0].item()}")

  # 4. Clean up process group resources
  dist.destroy_process_group()

if __name__ == "__main__":
  worker_main()
```

### 2. Launching with torchrun

Use the `torch.tpu.distributed` CLI module to detect the TPU hardware topology
and export the required launch variables before invoking `torchrun`:

```bash
# Detect and export TORCH_TPU_TOPOLOGY, TORCH_TPU_SLICEBUILDER_ADDRESSES,
# WORLD_SIZE, NODE_RANK, and multi-host coordination variables:
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
# Launch 1 worker process per TPU device on the host:
torchrun --nproc_per_node=$WORLD_SIZE train.py
```

Alternatively, if you launch workers from Python (for example, with
`torch.distributed.launcher.elastic_launch`), call
`torch.tpu.distributed.set_tpu_launch_env()` in the parent process prior to
spawning workers. Note that the default process group encompasses all devices in
the
[TPU slice](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices).

## Optimal device mesh mapping

To maximize collective bandwidth and minimize latency, align your model's
parallelism dimensions with the physical TPU interconnect hierarchy:

*   **Inter-Chip Interconnect (ICI)**: High-speed direct links connecting TPU
    chips in a 2D/3D torus or twisted-torus mesh within a slice (tens to
    hundreds of GB/s per link).
*   **Data Center Network (DCN)**: Host-level Ethernet connecting separate TPU
    slices in multi-slice workloads (lower bandwidth and higher latency than
    ICI).

Use `torch.tpu.distributed.topology_aware_mesh()` to compute a rank layout
aligned with physical ICI rings. Order your logical mesh dimensions from
**lowest communication intensity** (outermost) to **highest communication
intensity** (innermost):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Parallelism Type | Typical Collectives | Communication Frequency | Recommended Physical Placement |
| :---- | :---- | :---- | :---- |
| **Tensor Parallel (TP)** | `all_reduce`, `reduce_scatter`, `all_gather` | Every transformer layer (Forward + Backward) | Fastest local ICI ring (same tray/host, contiguous 1-hop devices) |
| **Context / Sequence Parallel (CP/SP)** | `all_to_all`, Ring Attention | Per attention layer | Local ICI axes (within slice) |
| **Expert Parallel (EP)** | `all_to_all`, `ragged_all_to_all` | Per MoE router layer | Local ICI axes (within slice) |
| **Fully Sharded Data Parallel (FSDP)** | `all_gather`, `reduce_scatter` | Per module / transformer block | ICI axes (within slice) |
| **Replicated Data Parallel (DP)** | `all_reduce` (gradients only) | Once per training step | Outer ICI axis or DCN (across slices) |
<!-- mdformat on -->

### Example: Constructing a topology-aware 2D DeviceMesh

```python
import torch
from torch.distributed.device_mesh import DeviceMesh

# Logical mesh shape: (dp=4, tp=2) on an 8-device slice,
# ordered from least communication-intensive (dp) to most intensive (tp):
mesh_shape = (4, 2)
dim_names = ("dp", "tp")

# Compute an optimal rank tensor aligned with physical ICI rings
optimal_rank_tensor = torch.tpu.distributed.topology_aware_mesh(
    mesh_shape=mesh_shape,
)

# Instantiate the PyTorch DeviceMesh
mesh_2d = DeviceMesh(
    "tpu",
    mesh=optimal_rank_tensor,
    mesh_dim_names=dim_names,
    preserve_rank_order=True,
)
dp_mesh = mesh_2d["dp"]
tp_mesh = mesh_2d["tp"]
```

For multi-slice (hybrid DCN + ICI) workloads, passing `dcn_mesh_shape` alongside
`mesh_shape` is optional and only needed to choose the split explicitly: by
default, `torch.tpu.distributed.topology_aware_mesh()` detects the slices
automatically and maps the outermost non-trivial mesh dimension (typically data
parallel) across slices over DCN and the inner dimensions within each slice over
ICI.

## Distributed Pallas Kernels

TorchTPU supports custom
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) kernels that perform
direct inter-device Remote DMA (RDMA) over ICI without host CPU involvement.
This enables you to author custom point-to-point or ring communication
kernels—such as Ring Attention, custom All-to-All shuffles, pipeline-parallel
shifts, or halo exchanges—and invoke them as native PyTorch custom ops in both
eager mode and `torch.compile`.

### Writing and Registering a Communicating Pallas Kernel

Integrating a distributed Pallas kernel involves three steps:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Stage | Key Steps & Operations |
| :---- | :---- |
| **1. Pallas Kernel Body** | Query logical coordinates with `jax.lax.axis_index` and `jax.lax.axis_size`, construct an asynchronous transfer descriptor via `pltpu.make_async_remote_copy(..., device_id=(target,))`, and manage completion with `.start()` and `.wait()`. |
| **2. `pl.pallas_call` & `jax.shard_map`** | Allocate DMA semaphores (`pltpu.SemaphoreType.DMA`) in `scratch_shapes` and wrap the kernel with `jax.shard_map` over a rank-aligned, world-sized `jax.sharding.Mesh` and `PartitionSpec`. |
| **3. TorchTPU Registration** | Register a type-annotated JAX wrapper function using `torch.tpu.pallas.jax_op`, passing `mesh`, `input_partition_specs`, and `output_partition_specs`. |
<!-- mdformat on -->

```python
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import torch

P = jax.sharding.PartitionSpec

# 1. Pallas Kernel Body: Asynchronously sends local buffer to neighbor rank (rank + 1)
def pallas_send_to_next_kernel(x_ref, y_ref, send_sem, recv_sem):
  rank = jax.lax.axis_index("x")
  world_size = jax.lax.axis_size("x")
  dest_id = (rank + 1) % world_size
  # Trigger hardware Remote DMA over TPU interconnect
  async_copy = pltpu.make_async_remote_copy(
      src_ref=x_ref,
      dst_ref=y_ref,
      send_sem=send_sem,
      recv_sem=recv_sem,
      device_id=(dest_id,),
  )
  async_copy.start()
  async_copy.wait()

# 2. JAX Wrapper with ShardMap and DMA Semaphores
def build_jax_fn(mesh):
  def local_pallas_call(x: jax.Array) -> jax.Array:
    return pl.pallas_call(
        pallas_send_to_next_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(x.shape)],
            out_specs=pl.BlockSpec(x.shape),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),  # send & recv semaphores
        ),
    )(x)

  return jax.shard_map(
      local_pallas_call,
      mesh=mesh,
      in_specs=P("x"),
      out_specs=P("x"),
      check_vma=False,
  )

# 3. Create the TorchTPU callable wrapper
def get_send_to_next_op(world_size: int):
  mesh = torch.tpu.pallas.get_pallas_mesh(
      axis_names=("x",), mesh_shape=(world_size,)
  )
  jax_fn = build_jax_fn(mesh)
  return torch.tpu.pallas.jax_op(
      "pallas::send_to_next",
      jax_fn,
      mesh=mesh,
      input_partition_specs=[P("x")],
      output_partition_specs=[P("x")],
  )
```

### Rank-aligned meshes and device assignment

Device IDs and ranks assigned are not 1 to 1 in many TPU architectures: ranks
are assigned such that the neighbors are the adjacent devices (rank 1 is next to
rank 0 and rank 2) but the device IDs are 0, 1, and 4 for v7x in this scenario.
To smooth the translation, torchTPU provides `torch.tpu.pallas.get_pallas_mesh`,
which provides a Pallas/JAX mesh which matches exactly the rank ID that is used
throughout the rest of the Torch application.

*   **Rank-aligned mesh construction**:
    `torch.tpu.pallas.get_pallas_mesh(axis_names=..., mesh_shape=...,
    process_group=...)` queries the active PyTorch `ProcessGroup`'s
    rank-to-device mapping and constructs a `jax.sharding.Mesh` of abstract
    compile-only devices without initializing a JAX PjRT backend.
*   **World-sized mesh requirement**: Distributed Pallas kernels require a mesh
    whose total device count equals `dist.get_world_size()`. For
    multi-dimensional parallel topologies (such as data + tensor parallelism),
    pass a `mesh_shape` whose product equals `world_size` - for example,
    `torch.tpu.pallas.get_pallas_mesh(axis_names=("dp", "tp"), mesh_shape=(2,
    4))`.
*   **Unified collective and kernel device assignment**: When `mesh` is provided
    to `torch.tpu.pallas.jax_op`, TorchTPU extracts the mesh's rank-to-device
    mapping as the XLA `DeviceAssignment` for the kernel and reconciles it with
    any standard `torch.distributed` collectives fused into the same compiled
    module. This guarantees that `jax.lax.axis_index`, mesh-addressed DMAs
    (`device_id={axis: target}`), and PyTorch collectives all route consistently
    according to PyTorch rank order.

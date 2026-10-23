# TorchTPU 分布式支持 { #torchtpu-distributed-support }

## 分布式支持 { #distributed-support }

TorchTPU 在 PyTorch ProcessGroup 层（`c10d::Backend`）以 `tpu_dist` 后端名称实现分布式通信。通过在 ProcessGroup 层拦截集合通信操作，TorchTPU 在 eager 模式和 `torch.compile` 下均支持标准 `torch.distributed` API，并可直接与 PyTorch 分布式框架配合使用，包括 DTensor、DistributedDataParallel（DDP）、FSDPv2 和 [SimpleFSDP](https://arxiv.org/abs/2411.00284)。

与 GPU 上的 PyTorch 一样，TorchTPU 强制采用每个设备一个进程的执行模型。在底层，TPU 之间的通信不使用单独的运行时通信库（例如 NCCL），而是将集合通信操作直接降级为 [StableHLO 集合通信操作](https://openxla.org/stablehlo/spec#collective_ops)（例如 `stablehlo.all_reduce`、`stablehlo.all_gather`、`stablehlo.reduce_scatter` 和 `stablehlo.all_to_all`），再由 XLA 编译到 TPU 可执行文件中。

### 与 GPU 的关键架构差异 { #key-architectural-differences-from-gpu }

虽然 TorchTPU 保留了标准的 `torch.distributed` 语义，但与 GPU 后端相比，通过 StableHLO 和 PjRT 对 TPU 编程会带来三个关键差异：

*   **初始化前的硬件拓扑发现**：在多 GPU 主机上，torchrun 启动的每个进程都可以检查所有已连接的 GPU，通过 `cuda:{local_rank}` 选择分配给它的设备，并在调用 `torch.distributed.init_process_group()` 之前将模型移到该设备上。在 TPU 主机上，每个 worker 进程都绑定到单个 TPU 设备，无法独自拥有全部设备。在初始化 PjRT 运行时或调用 `torch.distributed.init_process_group()` 之前，环境必须指定 [TPU 切片](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)的物理网格几何结构（`TORCH_TPU_TOPOLOGY`）、用于构建芯片间互连（ICI）网格的本地协调器地址（`TORCH_TPU_SLICEBUILDER_ADDRESSES`），以及参与的 TPU 设备总数（`WORLD_SIZE`）。
*   **子组的全局副本组解析**：在 PyTorch 中，[`torch.distributed.all_reduce`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) 等集合通信调用只接收调用方 rank 所参与的 ProcessGroup（子组）。相比之下，[`stablehlo.all_reduce`](https://openxla.org/stablehlo/spec#all_reduce) 需要完整描述默认进程组中所有参与的子组（`replica_groups`），即使是当前进程不属于的子组也需要描述：

    ```
    %0 = "stablehlo.all_reduce"(%arg0) <{
      replica_groups = dense<[[4, 0, 1, 3], [5, 2, 6, 7]]> : tensor<2x4xi64>
    }> ({ ... })
    ```

    TorchTPU 的 `tpu_dist` 后端会在集合通信设置期间，在默认进程组的所有 rank 之间自动交换子组成员信息，并构建 XLA 所需的全局 `replica_groups` 和 `device_assignment`。
*   **SPMD 编译与 MPMD 执行**：XLA 编译器和 TPU 通信栈专为单程序多数据（SPMD）执行而设计，在这种模式下，每个 rank 都会编译并执行一个相同的 StableHLO 模块，其中包含相互匹配的集合通信操作。然而，PyTorch 程序在不同 rank 之间可能表现出多程序多数据（MPMD）行为，例如由于依赖 rank 的条件分支（`if dist.get_rank() == 0:`），或者当某个维度不能被设备数整除时出现的不均匀的张量分片。如果不同的 rank 执行包含集合通信的不同 StableHLO 模块，各 rank 之间不匹配的同步标志 ID 或缓冲区偏移量可能会导致运行时死锁或内存损坏。

### 集合通信执行和融合模式 { #collective-execution-and-fusion-modes }

TorchTPU 将每个集合通信作为瘦集合通信（skinny collective）运行。为了让 XLA 能够将计算与通信重叠，您可以选择启用两种机制中的一种或两种，将集合通信与周围的计算融合。这两种机制相互独立，可以组合使用：`@spmd_safe` 函数内的集合通信始终会被融合；启用编译握手后，由它决定每个 `torch.compile` 计算图中其余集合通信的处理方式。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 模式 | 配置 | 计算图行为 | 最适合的场景 |
| :---- | :---- | :---- | :---- |
| 瘦集合通信（默认行为） | 对于下方选择启用的机制未融合的任何集合通信，始终开启 | 通过在集合通信周围插入图中断，将每个集合通信操作编译为独立的（“瘦”）StableHLO 模块。 | 开箱即用地提供通用的 MPMD 安全性，代价是 XLA 无法将计算与通信重叠。 |
| `@spmd_safe` 装饰器（选择启用） | `@torch.tpu.distributed.spmd_safe` | 停用被装饰函数内集合通信周围的图中断，将计算和集合通信融合到单个 StableHLO 模块中。 | 已知为 SPMD 的函数或训练步骤（例如 SimpleFSDP 或手动张量并行），以最大限度地实现 XLA 计算与通信的重叠。 |
| 编译握手（选择启用） | `TORCH_TPU_SPMD_HANDSHAKE_STAGE=COMPILE_STAGE` 或 `DISPATCH_STAGE`（默认：OFF） | 在 torch.compile 计算图中，检查所有 rank 上已编译的 FX 计算图是否一致；如果一致，则融合集合通信，否则回退到瘦集合通信。 | 在 `torch.compile` 中自动启用集合通信融合，同时防止意外的 MPMD 分歧。 |
<!-- mdformat on -->

#### 使用 `@torch.tpu.distributed.spmd_safe` { #using-torchtpudistributedspmd_safe }

如果您确定某个函数或模型步骤在所有 rank 上严格遵循 SPMD（没有依赖 rank 的控制流，也没有不均匀的张量形状），请使用 `@torch.tpu.distributed.spmd_safe` 对其进行装饰。在 `torch.compile`（以及 `DEFER_AND_FUSE` eager 模式）下，这会移除集合通信周围的图中断，让 XLA 可以将异步通信与矩阵计算一起调度：

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

请仅对在每个 rank 上执行完全相同的张量操作和集合通信操作序列的代码应用 `@torch.tpu.distributed.spmd_safe`；将其应用于 MPMD 代码可能会在各 rank 之间生成不同的 StableHLO 模块，并导致 TPU 执行死锁。

#### 配置编译握手 { #configuring-the-compilation-handshake }

除了使用 `@spmd_safe` 手动注解函数之外，您还可以通过 `TORCH_TPU_SPMD_HANDSHAKE_STAGE` 环境变量启用 TorchTPU 的编译握手。当 `torch.compile` 跟踪包含全局集合通信的计算图时，TorchTPU 会在所有 rank 之间运行一致性检查，在将集合通信融合到计算图中之前，验证各 rank 已编译的 FX 计算图是否一致：

*   `OFF`（默认）：停用握手；除非标记了 `@torch.tpu.distributed.spmd_safe`，否则集合通信使用默认的瘦集合通信。
*   `COMPILE_STAGE`：在 PjRT 编译之前验证各 rank 之间的计算图一致性。如果所有 rank 生成的计算图完全相同，则省略集合通信周围的图中断。建议用于所有 rank 一起编译的标准 `torch.compile` 工作负载。
*   `DISPATCH_STAGE`：在没有图中断的情况下编译集合通信，并在分派时验证各 rank 之间的一致性；如果各 rank 出现分歧，则使用拆分的集合通信重新编译。对于可能触发特定 rank 重新编译的工作负载，必须使用此模式；它的开销比 `COMPILE_STAGE` 更高。

以下是 `DISPATCH_STAGE` 支持的特定 rank 重新编译的示例；而在 `COMPILE_STAGE` 下，由于只有 `divergent_rank` 会在第 1 次迭代时重新编译，因此会超时：

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

## 分布式初始化 { #distributed-initialization }

在 TorchTPU 中初始化分布式工作负载时，使用标准 `torch.distributed` API，并配合 TorchTPU 的启动环境辅助工具来填充 TPU 拓扑和切片构建器环境变量：

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 环境变量 | 说明 | 示例 |
| :---- | :---- | :---- |
| `TORCH_TPU_TOPOLOGY` | TPU 切片的物理网格几何结构。 | `"2,2,1"`、`"2,2,1,2"`、`"4,4,4"` |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | 用于构建 ICI 网格的 `host:port` 地址（每个设备一个），以英文逗号分隔。 | `"localhost:8471,localhost:8472,..."` |
<!-- mdformat on -->

您需要在启动 worker 之前设置这些变量。在常见设置（单台主机或 GKE TPU 切片）中，TorchTPU 可以为您确定正确的值：在启动脚本中，于 torchrun 之前运行

```bash
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
```

或者在启动 worker 之前，在父 Python 启动器进程中调用 `torch.tpu.distributed.set_tpu_launch_env()`。这两种方法都不会更改您已设置的任何变量，因此您可以覆盖个别值。

下图展示了分布式 TorchTPU 工作负载的初始化流程，随后是完整的 worker 和启动器示例：

![分布式初始化流程](./images/distributed.png)

### 1. Worker 脚本（`train.py`） { #1-worker-script-trainpy }

在每个 worker 进程中，初始化 `tpu_dist` 后端，并通过 `torch.device("tpu")` 获取本地 TPU 设备：

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

### 2. 使用 torchrun 启动 { #2-launching-with-torchrun }

在调用 `torchrun` 之前，使用 `torch.tpu.distributed` CLI 模块检测 TPU 硬件拓扑并导出所需的启动变量：

```bash
# Detect and export TORCH_TPU_TOPOLOGY, TORCH_TPU_SLICEBUILDER_ADDRESSES,
# WORLD_SIZE, NODE_RANK, and multi-host coordination variables:
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
# Launch 1 worker process per TPU device on the host:
torchrun --nproc_per_node=$WORLD_SIZE train.py
```

或者，如果您从 Python 启动 worker（例如使用 `torch.distributed.launcher.elastic_launch`），请在启动 worker 之前，在父进程中调用 `torch.tpu.distributed.set_tpu_launch_env()`。请注意，默认进程组涵盖 [TPU 切片](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)中的所有设备。

## 最佳设备网格映射 { #optimal-device-mesh-mapping }

为了最大限度地提高集合通信带宽并降低延迟，请将模型的并行维度与 TPU 的物理互连层次结构对齐：

*   **芯片间互连（ICI）**：高速直连链路，在切片内以 2D/3D 环面或扭曲环面网格连接 TPU 芯片（每条链路的带宽为数十到数百 GB/s）。
*   **数据中心网络（DCN）**：主机级以太网，在多切片工作负载中连接不同的 TPU 切片（带宽比 ICI 低，延迟比 ICI 高）。

使用 `torch.tpu.distributed.topology_aware_mesh()` 计算与物理 ICI 环对齐的 rank 布局。将逻辑网格维度按照从**通信强度最低**（最外层）到**通信强度最高**（最内层）的顺序排列：

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 并行类型 | 典型集合通信 | 通信频率 | 推荐的物理放置位置 |
| :---- | :---- | :---- | :---- |
| **张量并行（TP）** | `all_reduce`、`reduce_scatter`、`all_gather` | 每个 Transformer 层（前向 + 反向） | 最快的本地 ICI 环（同一托盘/主机上、1 跳相邻的连续设备） |
| **上下文/序列并行（CP/SP）** | `all_to_all`、Ring Attention | 每个注意力层 | 本地 ICI 轴（切片内） |
| **专家并行（EP）** | `all_to_all`、`ragged_all_to_all` | 每个 MoE 路由层 | 本地 ICI 轴（切片内） |
| **完全分片数据并行（FSDP）** | `all_gather`、`reduce_scatter` | 每个模块/Transformer 块 | ICI 轴（切片内） |
| **复制数据并行（DP）** | `all_reduce`（仅梯度） | 每个训练步骤一次 | 外层 ICI 轴或 DCN（跨切片） |
<!-- mdformat on -->

### 示例：构建拓扑感知的 2D DeviceMesh { #example-constructing-a-topology-aware-2d-devicemesh }

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

对于多切片（DCN + ICI 混合）工作负载，可以选择在传递 `mesh_shape` 的同时传递 `dcn_mesh_shape`，但只有在需要明确选择拆分方式时才需要这样做：默认情况下，`torch.tpu.distributed.topology_aware_mesh()` 会自动检测切片，将最外层的非平凡网格维度（通常是数据并行）通过 DCN 映射到各个切片，并将内层维度通过 ICI 映射到每个切片内部。

## 分布式 Pallas kernel { #distributed-pallas-kernels }

TorchTPU 支持自定义 [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) kernel，这类 kernel 可以通过 ICI 执行设备间的直接远程 DMA（RDMA），无需主机 CPU 参与。借助这项功能，您可以编写自定义的点对点或环形通信 kernel（例如 Ring Attention、自定义 All-to-All 混洗、流水线并行移位或 halo 交换），并在 eager 模式和 `torch.compile` 下将其作为原生 PyTorch 自定义算子调用。

### 编写和注册通信 Pallas kernel { #writing-and-registering-a-communicating-pallas-kernel }

集成分布式 Pallas kernel 包括三个步骤：

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 阶段 | 关键步骤和操作 |
| :---- | :---- |
| **1. Pallas kernel 主体** | 使用 `jax.lax.axis_index` 和 `jax.lax.axis_size` 查询逻辑坐标，通过 `pltpu.make_async_remote_copy(..., device_id=(target,))` 构建异步传输描述符，并使用 `.start()` 和 `.wait()` 管理完成状态。 |
| **2. `pl.pallas_call` 和 `jax.shard_map`** | 在 `scratch_shapes` 中分配 DMA 信号量（`pltpu.SemaphoreType.DMA`），并使用 `jax.shard_map` 在与 rank 对齐、大小等于 world size 的 `jax.sharding.Mesh` 和 `PartitionSpec` 上封装 kernel。 |
| **3. TorchTPU 注册** | 使用 `torch.tpu.pallas.jax_op` 注册带类型注解的 JAX 封装函数，并传递 `mesh`、`input_partition_specs` 和 `output_partition_specs`。 |
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

### 与 rank 对齐的网格和设备分配 { #rank-aligned-meshes-and-device-assignment }

在许多 TPU 架构中，设备 ID 与所分配的 rank 并非一一对应：rank 的分配方式让相邻的 rank 位于相邻的设备上（rank 1 与 rank 0 和 rank 2 相邻），但在这种情况下，v7x 上对应的设备 ID 是 0、1 和 4。为了简化这种转换，TorchTPU 提供了 `torch.tpu.pallas.get_pallas_mesh`，它会提供一个 Pallas/JAX 网格，与 Torch 应用其余部分中使用的 rank ID 完全一致。

*   **构建与 rank 对齐的网格**：`torch.tpu.pallas.get_pallas_mesh(axis_names=..., mesh_shape=...,
    process_group=...)`
    查询当前活跃的 PyTorch `ProcessGroup` 的 rank 到设备映射，并构建由仅用于编译的抽象设备组成的 `jax.sharding.Mesh`，而无需初始化 JAX PjRT 后端。
*   **网格大小须等于 world size**：分布式 Pallas kernel 要求网格的设备总数等于 `dist.get_world_size()`。对于多维并行拓扑（例如数据并行 + 张量并行），请传递乘积等于 `world_size` 的 `mesh_shape`，例如 `torch.tpu.pallas.get_pallas_mesh(axis_names=("dp", "tp"), mesh_shape=(2,
    4))`。
*   **统一集合通信和 kernel 的设备分配**：当向 `torch.tpu.pallas.jax_op` 提供 `mesh` 时，TorchTPU 会提取网格的 rank 到设备映射，作为该 kernel 的 XLA `DeviceAssignment`，并使其与融合到同一已编译模块中的任何标准 `torch.distributed` 集合通信保持一致。这可确保 `jax.lax.axis_index`、按网格寻址的 DMA（`device_id={axis: target}`）和 PyTorch 集合通信都按照 PyTorch rank 顺序进行一致的路由。

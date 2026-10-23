# TorchTPU API 参考 { #torchtpu-api-reference }

本文档是 TorchTPU 公共 Python API 和环境变量的参考指南，旨在帮助您：

*   **从 CUDA 迁移到 TPU**：将常见的 `torch.cuda` 调用映射到 PyTorch 与设备无关的 [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html) API。
*   **使用 TPU 专属功能**：通过 **`torch.tpu`** 模块配置执行模式、MXU 数值精度、分布式拓扑、自定义 Pallas/TPU 算子、可跟踪的优化器以及 XProf 性能分析。
*   **配置运行时**：通过环境变量调整编译缓存、调试检查和多主机启动设置。

> **Note:** 如需查看自动生成的、涵盖所有公共 Python API 和环境变量的完整权威列表，请参阅 [TorchTPU 公共 API（`public_apis.md`）](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/public_apis.md)。

## **核心设备查询** { #core-device-queries }

在已安装 TorchTPU 的环境中导入 `torch` 时，后端会自动注册 `"tpu"` 设备和 `torch.tpu` 模块。如果 PyTorch 提供了与设备无关的 [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html) API，请使用它而不是后端专属 API，以便您的代码可在不同加速器之间移植。对于没有通用等效 API 的 TPU 专属硬件属性，请使用 `torch.tpu`。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA API | TPU / 加速器 API | 说明 |
| :--- | :--- | :--- |
| `torch.device("cuda")` | `torch.accelerator.current_accelerator()` | 当 TPU 是当前活跃的加速器时，返回 `torch.device("tpu")`。 |
| `torch.cuda.is_available()` | `torch.accelerator.is_available()` | 检查 TPU 后端是否可用。 |
| `torch.cuda.is_initialized()` | `torch.tpu.is_initialized()` | 检查 TorchTPU 的运行时状态是否已初始化。 |
| `torch.cuda.device_count()` | `torch.accelerator.device_count()` | 当前进程可见的设备数量。 |
| `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | 返回当前活跃的 TPU 设备索引。 |
| `torch.cuda.set_device(rank)` | `torch.accelerator.set_device_index(rank)` | 为当前进程设置活跃的 TPU 设备索引。 |
| `torch.cuda.device(rank)` | `torch.accelerator.device_index(rank)` | 临时设置活跃设备索引的上下文管理器。 |
| `torch.cuda.get_device_name(device=None)` | `torch.tpu.get_device_properties(device=None).name` | 返回所连接 TPU 的可读型号名称（例如 `"TPU v6e"`）。 |
| `torch.cuda.get_device_properties(device=None)` | `torch.tpu.get_device_properties(device=None)` | 返回 TPU 设备的静态属性（包括 `name`、`devices_per_chip`、`hbm_bytes_per_chip`、`hbm_bytes_per_device`、`bf16_flops_per_chip`、`bf16_flops_per_device`、`sparse_cores_per_device` 和 `sparse_core_simd_width`）。 |
<!-- mdformat on -->

此表中的 `torch.accelerator` API 需要 PyTorch 2.11 或更高版本。在极少数情况下，如果您使用的是较旧的 PyTorch 版本，请改用对应的 `torch.tpu` API（例如 `torch.tpu.device_count()` 或 `torch.tpu.current_device()`）。在 PyTorch 2.11 及更高版本中，这些冗余的 `torch.tpu` API 属于 TorchTPU 内部 API，从用户代码直接调用会引发错误。

```python
import torch

# Initialize hardware
device = torch.device("tpu")

print(f"torch.accelerator.is_available(): {torch.accelerator.is_available()}")
print(f"torch.accelerator.device_count(): {torch.accelerator.device_count()}")
print(
    "torch.accelerator.current_device_index():"
    f" {torch.accelerator.current_device_index()}"
)
print(
    "torch.tpu.get_device_properties().name:"
    f" {torch.tpu.get_device_properties().name}"
)
```

## **随机数生成（RNG）** { #random-number-generation-rng }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA API | TPU / 加速器 API | 说明 |
| :--- | :--- | :--- |
| `torch.cuda.manual_seed(seed)` | `torch.tpu.manual_seed(seed)` | 为当前活跃 TPU 设备的随机数生成器设置种子。 |
| `torch.cuda.manual_seed_all(seed)` | `torch.tpu.manual_seed_all(seed)` | 为所有本地 TPU 设备的 RNG 设置种子。 |
| `torch.cuda.initial_seed()` | `torch.accelerator.random.initial_seed()` | 返回当前 TPU 设备的初始随机种子。 |
| `torch.cuda.get_rng_state()` | `torch.accelerator.random.get_rng_state()` | 返回当前设备的 RNG 状态张量。 |
| `torch.cuda.set_rng_state(new_state, device=...)` | `torch.tpu.set_rng_state(new_state, device=...)` | 为当前进程的活跃 TPU 设备设置 RNG 状态张量。 |
| `torch.cuda.default_generators` | `torch.tpu.default_generators` | 延迟初始化的默认 `torch.Generator` 实例元组，按本地设备序号索引。 |
<!-- mdformat on -->

此表中的 `torch.accelerator.random` API 需要 PyTorch 2.13 或更高版本。在极少数情况下，如果您使用的是较旧的 PyTorch 版本，请改用 `torch.tpu.initial_seed()` 或 `torch.tpu.get_rng_state()`。在 PyTorch 2.13 及更高版本中，这些 `torch.tpu` API 属于 TorchTPU 内部 API，从您的代码调用会引发错误。

```python
# Set the seed across all cores
torch.tpu.manual_seed_all(42)
print("Seeded all TPU cores with value 42.")
```

## **自动混合精度（AMP）** { #automatic-mixed-precision-amp }

TorchTPU 使用 `"tpu"` 设备类型字符串与 PyTorch 的标准 [`torch.amp.autocast`](https://docs.pytorch.org/docs/stable/amp.html) 框架集成。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA API | TPU API | 说明 |
| :--- | :--- | :--- |
| `torch.amp.autocast("cuda", ...)` | `torch.amp.autocast("tpu", ...)` | 启用自动混合精度的上下文管理器/装饰器（在 TPU 上默认为 `bfloat16`）。 |
| `torch.cuda.get_amp_supported_dtype()` | `torch.tpu.get_amp_supported_dtype()` | 列出 TPU 上支持的 dtype（`bfloat16`、`float16`、`float32`、`float64`、`float8_*`、整数、布尔和复数类型）。 |
<!-- mdformat on -->

```python
supported_dtypes = torch.tpu.get_amp_supported_dtype()
print(f"AMP supported dtypes: {supported_dtypes}")
```

## **Stream 和事件** { #streams-and-events }

对于 stream 管理和设备同步，请使用标准 [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html) API，而不是后端专属 API。仅在直接构造 TPU `Stream` 或 `Event` 对象或者配置 TPU 核心绑定时才使用 `torch.tpu`。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA API | TPU / 加速器 API | 说明 |
| :--- | :--- | :--- |
| `torch.cuda.current_stream()` | `torch.accelerator.current_stream()` | 返回设备的当前活跃 stream。 |
| `torch.cuda.set_stream(stream)` | `torch.accelerator.set_stream(stream)` | 设置活跃 stream。 |
| `torch.cuda.default_stream(device=None)` | `torch.tpu.default_stream(device=None)` | 返回设备的默认 `TpuStream`。 |
| `torch.cuda.stream(stream)` | `torch.tpu.stream(stream)` | 选择指定 `TpuStream` 的上下文管理器。 |
| `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` | 等待 TPU 设备上所有待处理的工作完成。 |
| `torch.cuda.Stream` | `torch.tpu.Stream(device=None, priority=0, *, pinned=None)` | `torch.Stream` 的子类。支持 `synchronize()`、`record_event()`、`wait_event()`、`wait_stream()` 以及核心绑定（`pinned="dense"` 或 `"sparse"`，通过 `is_dense` / `is_sparse` 查询）。 |
| `torch.cuda.Event` | `torch.tpu.Event(enable_timing=False)` | `torch.Event` 的子类。支持 `record()`、`synchronize()`、`query()` 和 `elapsed_time(end_event)`。 |
<!-- mdformat on -->

此表中的 `torch.accelerator` API 需要 PyTorch 2.11 或更高版本。在极少数情况下，如果您使用的是较旧的 PyTorch 版本，请改用 `torch.tpu.current_stream()`、`torch.tpu.set_stream()` 或 `torch.tpu.synchronize()`。

> **Note:** stream 是 CUDA 的概念，在 TPU 中没有直接对应的概念。在 TorchTPU 的延迟执行模型中，XLA 编译器和 PjRt 运行时会根据数据依赖关系自动调度操作并使其重叠执行，因此无需手动管理多个 stream。

## **TPU 专属 API** { #tpu-specific-apis }

除了标准的 CUDA 等效 API 之外，`torch.tpu` 还提供专门针对 TPU 执行模式、MXU 精度、XLA 编译、分布式拓扑、自定义 Pallas kernel、可跟踪优化器和性能分析而设计的 API。

### **执行模式和编译器选项** { #execution-modes-and-compiler-options }

控制 XLA 如何将 eager 操作排队、融合和编译。如需了解架构详解，请参阅 [TorchTPU 执行模式](./torchtpu_execution_model.md)。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 说明 |
| :--- | :--- |
| `torch.tpu.EagerMode` | 控制 eager 执行的枚举：`DEFER_AND_FUSE`、`DEFER_NEVER` 和 `DEFER_NEVER_AND_LAUNCH_BLOCKING`。 |
| `torch.tpu.eager_mode` | 全局获取或设置当前活跃的 `EagerMode`。 |
| `torch.tpu.set_eager_mode(mode)` | 临时设置活跃 `EagerMode` 的上下文管理器。 |
| `torch.tpu.compiler.custom_compiler_options(options)` | 上下文管理器，将自定义 XLA 编译器标志传递给在该代码块内具体化的 eager 操作。 |
<!-- mdformat on -->

```python
print(f"Default eager_mode: {torch.tpu.eager_mode}")
with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_AND_FUSE):
  x = torch.ones((4, 4), device="tpu") + 1
  print(
      f"Inside context manager: {torch.tpu.eager_mode}, sum={x.sum().item()}"
  )
```

### **数值精度** { #numerical-precision }

控制矩阵乘法单元（MXU）精度以及中间结果的 `float32` 累加。如需详细指导，请参阅[数值精度](./numerical_precision.md)。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 说明 |
| :--- | :--- |
| `torch.tpu.Precision` | 控制 `float32` 矩阵乘法和卷积的 MXU 精度的枚举：`Precision.DEFAULT`、`Precision.HIGH` 和 `Precision.HIGHEST`。 |
| `torch.tpu.precision(level)` | 上下文管理器，为代码块内的矩阵乘法和卷积操作设置 `Precision` 级别。 |
| `torch.tpu.allow_excess_precision` | 全局获取或设置 XLA 的 `--xla_allow_excess_precision` 标志，使融合运算的中间值保持为 `float32`。 |
<!-- mdformat on -->

```python
a = torch.randn(64, 64, device="tpu", dtype=torch.float32)
b = torch.randn(64, 64, device="tpu", dtype=torch.float32)
with torch.tpu.precision(torch.tpu.Precision.HIGH):
  c = torch.matmul(a, b)
print(f"Computed HIGH-precision matmul with output shape: {tuple(c.shape)}")
```

### **分布式训练和拓扑** { #distributed-training-and-topology }

TorchTPU 将 `"tpu_dist"`（别名也为 `"tpu"`）注册为 `"tpu"` 设备的默认 `torch.distributed` 后端。请参阅[在 GKE 上开始使用 TorchTPU](./get_started_on_gke.md) 和 [FSDP 最佳实践](./case_studies_fsdp.md)。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 说明 |
| :--- | :--- |
| `torch.tpu.topology_aware_mesh(mesh_shape, ...)` | 为 `DeviceMesh("tpu", mesh=...)` 计算形状为 `mesh_shape` 的拓扑感知 rank 布局张量。 |
| `torch.tpu.distributed.set_tpu_launch_env(...)` | 在启动 worker 之前填充 TPU 拓扑和多主机启动环境变量。 |
| `torch.tpu.distributed.spmd_safe(func)` | 装饰器，声明所有 rank 在 `func` 中执行相同的 TPU 操作，从而启用 eager 集合通信融合。 |
| `torch.tpu.window_num_stripes` | 获取或设置点对点传输的最大并发 DMA 条带数。 |
| `torch.tpu.window_stripe_chunk_mb` | 获取或设置点对点传输中每个 DMA 条带的最小载荷大小（以 MB 为单位）。 |
<!-- mdformat on -->

### **TPU 自定义算子（`torch.tpu.ops`）和 Pallas（`torch.tpu.pallas`）** { #tpu-custom-operators-torchtpuops-and-pallas-torchtpupallas }

TorchTPU 在 `torch.tpu.ops` 下提供针对硬件优化的算子，并在 `torch.tpu.pallas` 下提供自定义 JAX/Pallas kernel 封装器。请参阅[使用 Pallas 添加自定义 kernel](./custom_kernels.md)。

#### **通用、注意力、MoE 和集合通信算子** { #general-attention-moe-and-collective-operators }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 说明 |
| :--- | :--- |
| `torch.tpu.ops.max_pool{1,2,3}d(...)` | 在 TPU 上执行 1D、2D 或 3D 最大池化，无需具体化索引张量。 |
| `torch.tpu.ops.max_pool{1,2,3}d_backward(...)` | 在不使用索引张量的情况下计算 `max_pool{1,2,3}d` 的反向传播。 |
| `torch.tpu.ops.optimization_barrier(inputs)` | 阻止 XLA 跨屏障移动、融合计算或对计算去重。 |
| `torch.tpu.ops.splash_sdpa(q, k, v, ...)` | 基于 TPU Pallas Splash Attention kernel 的缩放点积注意力。 |
| `torch.tpu.ops.ragged_dot(lhs, rhs, group_sizes, ...)` | 用于混合专家（MoE）工作负载的分组矩阵乘法。 |
| `torch.tpu.ops.ragged_dot_weight_grad(lhs, grad_output, group_sizes)` | 计算 `ragged_dot` 的权重梯度。 |
| `torch.tpu.ops.align_jagged_to_multiple(nt, ...)` | 将 `torch.jagged` `NestedTensor` 缓冲区填充到指定倍数，以满足 TPU 对齐要求。 |
| `torch.tpu.ops.strip_jagged_padding(nt, ...)` | 移除由 `align_jagged_to_multiple` 添加的尾部填充。 |
| `torch.tpu.ops.all_gather_dim(input, dim, group=None)` | 在 `group` 中对 `input` 执行 all-gather，并沿维度 `dim` 拼接。 |
| `torch.tpu.ops.reduce_scatter_dim(input, dim, group=None, ...)` | 在 `group` 中对 `input` 执行归约，并沿维度 `dim` 分散各个分块。 |
| `torch.tpu.ops.ragged_all_to_all(...)` | 在进程组内的各个 rank 之间交换可变长度的张量切片。 |
| `torch.tpu.ops.send(tensors, dst, tag)` / `torch.tpu.ops.recv(tensors, src, tag)` | 在默认 TPU 进程组上执行点对点张量发送和接收。 |
| `torch.tpu.pallas.jax_op(name, fn=None, ...)` | 将 JAX/Pallas 函数注册为可跟踪的 PyTorch 自定义算子。 |
<!-- mdformat on -->

#### **SparseCore 嵌入算子** { #sparsecore-embedding-operators }

这些算子需要配备 SparseCore 的 TPU。`torch.tpu.get_device_properties()` 的 `sparse_cores_per_device` 属性表示每个设备的 SparseCore 数量（在没有 SparseCore 的 TPU 上为 0），`sparse_core_simd_width` 表示其 SIMD 宽度。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 说明 |
| :--- | :--- |
| `torch.tpu.ops.sparse_dense_matmul(...)` | 在 TPU SparseCore 上执行 CSR 嵌入查找和聚合。 |
| `torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(...)` / `_with_adagrad(...)` / `_with_adam(...)` | 计算 `sparse_dense_matmul` 的反向梯度，并在 SparseCore 上执行融合的原地 SGD、Adagrad 或 Adam 嵌入表更新。 |
| `torch.tpu.ops.sparse_dense_matmul_activation_unstack(...)` | 将堆叠的 SparseCore 嵌入激活值拆分为各特征的张量。 |
| `torch.tpu.ops.sparse_dense_matmul_gradient_stack(...)` | 将各特征的激活梯度堆叠为单个张量，供 SparseCore 使用。 |
| `torch.tpu.ops.sparse_gather(...)` | 在 TPU SparseCore 上使用 CSR 索引从 2D 张量中收集行。 |
| `torch.tpu.ops.sparse_gather_backward(...)` | 在 TPU SparseCore 上对 `sparse_gather` 的梯度执行 scatter-add。 |
| `torch.tpu.ops.sparse_iota(...)` | 在 SparseCore 上为 CSR 矩阵中的非零元素生成行内位置索引。 |
| `torch.tpu.ops.sparse_map_row(...)` | 在 SparseCore 上将自定义的 MLIR 逐行计算映射到稀疏 CSR 矩阵。 |
<!-- mdformat on -->

### **融合训练步骤和优化器（`torch.tpu.optimizers`）** { #fused-training-step-and-optimizers-torchtpuoptimizers }

`torch.tpu.optimizers` 会将前向传播、损失计算、反向传播和函数式优化器步骤跟踪为单个已编译的 TPU 计算图。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 说明 |
| :--- | :--- |
| `torch.tpu.optimizers.SingleTraceTrainer(model, optimizer=None, ...)` | 管理函数式参数（`params`）和优化器状态（`param_group`），并通过 `make_compiled_train_step(...)` 编译完整的训练步骤。 |
| `torch.tpu.optimizers.Optimizer(lr=..., weight_decay=...)` | 无状态函数式优化器的基类（提供 `defaults`、`init_param_group` 和 `step`）。 |
| `torch.tpu.optimizers.AdamW(...)` | 可跟踪的函数式 AdamW 优化器（`AdamW.ParamGroup`）。 |
| `torch.tpu.optimizers.Muon(...)` | 可跟踪的函数式 Muon 优化器（`Muon.ParamGroup`）。 |
| `torch.tpu.optimizers.Sgd(...)` | 可跟踪的函数式 SGD 优化器（`Sgd.ParamGroup`）。 |
<!-- mdformat on -->

### **性能分析（`torch.tpu.profiler`）** { #profiling-torchtpuprofiler }

为 XProf 和 TensorBoard 配置 TPU 硬件跟踪和 Python 跟踪。请参阅[适用于 TorchTPU 的 XProf](./performance_profiling_with_xprof.md)。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 说明 |
| :--- | :--- |
| `torch.tpu.profiler.TpuProfilerConfig(...)` | 传递给 `torch.profiler.profile(..., experimental_config=...)` 的实验性配置，用于为 XProf 配置主机、TPU 设备和 Python 跟踪。 |
<!-- mdformat on -->

## **环境变量** { #environment-variables }

TorchTPU 的行为由多个环境变量控制，这些变量用于配置 XLA 编译器、TPU 驱动程序（`libtpu`）以及 PyTorch 自身的调试子系统。

> **关键：进程全局初始化** TorchTPU 环境变量会在后端初始化时（通常是在 `import torch` 或 `torch.device("tpu")` 期间）立即**锁定**。

### **设备可见性、运行时和分布式启动** { #device-visibility-runtime-and-distributed-launch }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 环境变量 | 说明 |
| :--- | :--- |
| `TPU_VISIBLE_DEVICES` | 以英文逗号分隔的、当前进程可见的 TPU 芯片索引（类似于 `CUDA_VISIBLE_DEVICES`）。 |
| `RANK` | 当前进程在分布式作业中的全局 rank。 |
| `LOCAL_RANK` | 当前进程在主机内的本地 rank。 |
| `WORLD_SIZE` | 分布式作业中的进程总数。 |
| `NNODES` | 多主机作业中的主机节点总数。 |
| `NODE_RANK` | 当前主机节点在多主机作业中的 rank。 |
| `MASTER_ADDR` | `torch.distributed` 的 rank 0 协调节点的地址。 |
| `MASTER_PORT` | `MASTER_ADDR` 上用于 `torch.distributed` 会合（rendezvous）的端口。 |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | 设置为 `"0"` 可停用 `torch_tpu` 后端的自动注册。 |
| `TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS` | 启用后，允许从 PyTorch 核心之外调用内部 `torch.tpu` 设备 API（即存在 `torch.accelerator` 等效 API 的那些 API），此时只发出一次警告，而不会引发 `RuntimeError`。请仅将其用作临时解决方案；建议迁移到 `torch.accelerator.*`，此变量可能会在未来版本中移除。 |
| `LIBTPU_INIT_ARGS` | 以空格分隔的初始化标志，传递给 `libtpu` 和 PJRT 运行时。 |
| `TPU_PREMAPPED_BUFFER_SIZE` | PJRT 客户端的预映射主机到设备内存缓冲区大小（以字节为单位）。 |
| `TMPDIR` | 运行时和编译器使用的临时目录。 |
<!-- mdformat on -->

### **硬件拓扑和多主机协调** { #hardware-topology-and-multi-host-coordination }

这些变量用于指定 TPU 硬件拓扑和进程网络。在大多数情况下，它们由 `torch.tpu.distributed.set_tpu_launch_env()`、TorchTPU 启动器或 Google Cloud 环境自动设置。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 环境变量 | 说明 |
| :--- | :--- |
| `TORCH_TPU_TOPOLOGY` | TPU 切片的物理拓扑（例如 `2x2x1`、`2x2x4`）。 |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | TPU 切片构建器服务的 worker 地址，格式为 `host:port`，以英文逗号分隔。 |
| `TORCH_TPU_DEVICES_PER_HOST` | 推导切片构建器地址时每台主机的 TPU 芯片数。 |
| `TORCH_TPU_BASE_PORT` | 推导切片构建器地址时的起始端口号。 |
| `TORCH_TPU_SPMD_HANDSHAKE_STAGE` | 各 rank 验证集合通信计算图在各 rank 之间是否一致的阶段。 |
| `TORCHCOMM_RANK` | `torchcomm` 的进程 rank（优先级高于 `RANK`）。 |
| `TORCHCOMM_SIZE` | `torchcomm` 的 world size（优先级高于 `WORLD_SIZE`）。 |
<!-- mdformat on -->

### **执行模式、编译和缓存** { #execution-mode-compilation-and-caching }

这些变量控制 PyTorch 操作和计算图如何分派、降级、由 XLA 优化，以及如何在多次运行和多个 rank 之间缓存。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 环境变量 | 说明 |
| :--- | :--- |
| `TORCH_TPU_DEFER_AND_FUSE` | 启用后（`"1"`/`"true"`），将默认 eager 模式设置为 `EagerMode.DEFER_AND_FUSE`。 |
| `TORCH_TPU_LAUNCH_BLOCKING` | 启用后（`"1"`/`"true"`），将默认 eager 模式设置为 `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING`。 |
| `TORCH_TPU_TIER2_COMPILATION_CACHE` | `/dev/shm/torch_tpu_cache/` 下用于主机本地编译缓存的目录名称（或 `"disabled"`）。 |
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | 在多台主机和多次运行之间共享的持久性第 3 层编译缓存的路径（例如 `gs://...` 或本地路径）。 |
| `XLA_FLAGS` | 以空格分隔的 XLA 编译器标志（例如 `--xla_dump_hlo_as_text --xla_dump_to=...`）。 |
<!-- mdformat on -->

### **调试、安全、性能分析和跟踪** { #debugging-safety-profiling-and-tracing }

这些变量控制 TorchTPU 的调试上下文、运行时安全检查和性能分析器输出。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 环境变量 | 说明 |
| :--- | :--- |
| `TORCH_SHOW_CPP_STACKTRACES` | 设置为 `"1"` 时，在 Python `RuntimeError` 异常中附加 C++ 回溯信息。 |
| `TORCH_TPU_ENABLE_DEBUG_CHECKS` | 设置为 `"1"` 时，在 TPU 上启用运行时调试检查和边界检查。 |
| `TORCH_TPU_PROFILER_OUTPUT_DIR` | 未设置 `TpuProfilerConfig.run_dir` 时，XProf 跟踪文件的备用输出目录。 |
| `TORCH_TPU_XPROF_SESSION_ID` | 在多主机 worker 之间共享的 XProf 性能分析会话 ID。 |
| `TORCH_TRACE` | 结构化编译器和调度器跟踪日志（`tlparse`）的输出目录。 |
<!-- mdformat on -->

### **环境变量配置代码段** { #environment-variable-configuration-snippets }

#### **1. C++ 堆栈轨迹（`TORCH_SHOW_CPP_STACKTRACES`）** { #1-c-stacktraces-torch_show_cpp_stacktraces }

在导入 `torch` 之前设置 `TORCH_SHOW_CPP_STACKTRACES=1`，可以更轻松地诊断发生在 C++ 后端或 XLA 深处的错误。您收到的将不再是简单的 Python `RuntimeError`，而是详细的 `C++
CapturedTraceback`，其中显示确切的执行帧。

```python
import os
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
import torch

device = torch.device("tpu")
a = torch.randn(10, device=device)
b = torch.randn(20, device=device)
# Raises RuntimeError enriched with C++ Error Trace and Captured Traceback:
c = a + b
```

#### **2. 内部调试检查和边界验证（`TORCH_TPU_ENABLE_DEBUG_CHECKS`）** { #2-internal-debug-checks-boundary-validation-torch_tpu_enable_debug_checks }

默认情况下，TPU 为了提高速度会跳过边界检查。启用 `TORCH_TPU_ENABLE_DEBUG_CHECKS=1` 可以捕获与越界索引相关的隐蔽 bug（例如在 `torch.take` 中），但由于需要额外的 CPU/TPU 同步，执行速度会变慢。原本会静默失败或导致数据损坏的操作会引发明确的异常，例如 `take(): expected indices to be
in range`：

```python
import os
os.environ["TORCH_TPU_ENABLE_DEBUG_CHECKS"] = "1"
import torch

device = torch.device("tpu")
x = torch.tensor([1.0, 2.0, 3.0], device=device)
indices = torch.tensor([5], device=device)  # Out of bounds
# Raises an explicit out-of-bounds exception when materialized:
y = torch.take(x, indices).cpu()
```

#### **3. 转储 XLA IR / HLO（`XLA_FLAGS`）** { #3-dumping-xla-ir-hlo-xla_flags }

通过设置 `XLA_FLAGS`，您可以强制编译器将高级优化器（HLO）文本转储到某个目录。转储目录（例如 `/tmp/hlo_dump`）将包含：

*   **`*.after_optimizations.txt`**：最终的 HLO 计算图，显示硬件级操作。
*   **`*.flagfile`**：当前生效的 XLA 编译器标志的完整列表（可用于验证 `xla_optimization_level`）。
*   **`*tpu_comp_env.txt`**：编译期间使用的内部 TPU 环境设置。

```python
import os

dump_dir = "/tmp/hlo_dump"
os.makedirs(dump_dir, exist_ok=True)
os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump_dir}"

import torch
device = torch.device("tpu")
x = torch.randn(10, device=device)
y = (x + 1).to("cpu")
```

#### **4. 多层编译缓存（`TORCH_TPU_TIER2_COMPILATION_CACHE` 和 `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`）** { #4-multi-tiered-compilation-caching-torch_tpu_tier2_compilation_cache-torch_tpu_tier3_compilation_cache_root }

TorchTPU 使用多层缓存，避免在多次运行和多个 rank 之间重复编译：

*   **第 2 层（`TORCH_TPU_TIER2_COMPILATION_CACHE`）** 存储在主机内的共享内存中。它需要一个目录**名称**（例如 `"tpu_tier2_cache"`），后端会自动将其放在 `/dev/shm/torch_tpu_cache/<name>` 中。
*   **第 3 层（`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`）** 是在多台主机和多个会话之间共享的持久性编译缓存（通常是 `gs://` 存储桶 URI 或共享文件系统路径）。

已编译的二进制工件（`*.bin` 可执行文件）会异步保存到缓存目录中，并在后续运行中重复使用，从而大幅降低 JIT 编译延迟：

```python
import os

tier2_name = "tpu_tier2_cache"
tier2_path = os.path.join("/dev/shm/torch_tpu_cache", tier2_name)
tier3_path = "/tmp/tpu_tier3_cache"  # Or "gs://<bucket_name>/<path_to_cache>"
os.makedirs(tier2_path, exist_ok=True)
os.makedirs(tier3_path, exist_ok=True)

# TORCH_TPU_TIER2_COMPILATION_CACHE expects just the directory NAME inside /dev/shm/torch_tpu_cache/
os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = tier2_name
os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = tier3_path

import torch
device = torch.device("tpu")
a = torch.randn(1024, 1024, device=device)
_ = (a @ a).to("cpu")
```

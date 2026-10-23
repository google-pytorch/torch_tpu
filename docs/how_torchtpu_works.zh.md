# TorchTPU 的工作原理 { #how-torchtpu-works }

## TPU 基础知识 { #tpu-primer }

[Google TPU](https://cloud.google.com/tpu) 是专门为执行大规模张量运算而设计的专用 ASIC，而张量运算正是深度学习的基础。不同代际的 TPU [在架构上有所不同](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)，但概括来说，一个 TPU 芯片包含：

*   一个或多个 TensorCore，每个 TensorCore 都是单线程处理器，由矩阵乘法单元（MXU）、向量单元（VPU）和标量单元组成。
*   一个或多个 SparseCore，专门用于稀疏运算，TensorCore 可以将运算卸载到 SparseCore 上。
*   大容量高带宽内存（HBM）：[TPU 8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) 中为 288 GB，带宽 8.6 TB/s，可用于存储机器学习数据并与 CPU 共享。
*   双向芯片间互连（ICI）：[TPU 8t 和 8i](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/) 中为 19.2 TBps，可在用户配置的 [TPU 切片](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)内实现 TPU 芯片之间的快速通信。TPU 切片是 [TPU Pod](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu-pod) 的逻辑子组，由预先配置数量的 TPU 芯片组成，这些芯片以 3D 环面（TPU 7、TPU 8t）或 Boardfly（TPU 8i）拓扑相连。
*   双向数据中心网络（DCN）：[TPU 8t](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/) 中为 400 GBps，可在不同切片的 TPU 芯片之间进行通信。

TPU 通过 [XLA 编译器](https://github.com/openxla/xla)栈进行编程，有两个不同的入口点：[StableHLO](https://openxla.org/stablehlo/spec) 和 [Mosaic](https://docs.jax.dev/en/latest/pallas/design/design.html#lowering-pallas-to-mosaic-for-tpu)。实际的 TPU ISA 是专有的，可能会发生变化，并且不公开。实际上，可以将 StableHLO（Mosaic 更是如此）视为相当于 PTX 的一层。下图为整体示意图：

![TorchTPU 的工作原理](./images/torch_tpu_inputs_to_tpu.svg)

C++ 程序可以通过实例化 [PjRT Client](https://openxla.org/xla/pjrt/cpp_api_overview) 来访问本地 TPU。PjRT Client 的 API 提供了以下方法：将 MLIR（StableHLO 或 Mosaic）编译为 TPU 可执行文件、启动该可执行文件，以及在 TPU 与主机之间复制数据。请注意，TPU 执行是异步的，CPU 可以通过等待关联的 PjRTBuffer 来等待 TPU 结果就绪。StableHLO 遵循[静态单赋值（SSA）范式](https://en.wikipedia.org/wiki/Static_single-assignment_form)，因此不允许原地修改，也就是说，写操作总是会产生一个新值（至少在概念上如此）。这一点对 TorchTPU 尤为重要，因为 TorchTPU 通过视图分解和反演，提供了大家熟悉的 PyTorch“视图”语义（包括修改）。XLA 编译器假定张量形状在编译时已知。TorchTPU 通过[有界动态性](./dynamic_shapes.md)放宽了这一限制。我们认识到这对 PyTorch 工作负载是一项限制，因此一直在添加对有界动态性的支持，即形状各维度的最大上限在编译时已知，而实际维度要到运行时才能确定。

## TorchTPU：支持自动加载的树外后端 { #torchtpu-an-out-of-tree-backend-with-autoload-support }

按照[当前的最佳实践](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html)，额外的后端在树外构建并在运行时注册。我们使用一个注册为 `PrivateUse1` 的新 C++ 后端扩展了 PyTorch 调度器。此外，我们还使用标准的 `at::Tensor` 机制定义了特定于设备的张量存储实现（通过 `c10::StorageImpl` 中的 `c10::UniqueVoidPtr`）。

我们在设计 TorchTPU 时注重易用性。当用户在已安装 TorchTPU wheel 的 TPU 机器上导入 PyTorch 时，Python 会自动加载 TorchTPU wheel。自动加载该 wheel 会设置后端，并将 `PrivateUse1` 名称替换为 `tpu`。这样，在简单示例中，TorchTPU 程序与树内后端并无区别，PyTorch 程序只需很少修改甚至无需修改即可在 TPU 上运行：

```python
import torch
import torch.distributed as dist

def main():
    device = torch.device("tpu")
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    word_size = dist.get_world_size()
    ...
    model = model.to(device)
    ...
```

由于 TPU 会成为当前加速器，而 `tpu_dist` 是 TorchTPU 的默认后端，因此无需显式指定它们：

```python
import torch
import torch.distributed as dist

def main():
    device = torch.accelerator.current_accelerator()
    dist.init_process_group()
    rank = dist.get_rank()
    word_size = dist.get_world_size()
    ...
    model = model.to(device)
    ...
```

## 捕获计算图 { #capturing-the-graph }

TorchTPU 在 ATen 层级跟踪 eager 算子，依赖的是作为后端基础设施一部分的 PyTorch 算子分派钩子。TorchTPU 算子被构造为 StableHLO 程序，并以单个或成组的形式传递给 XLA，使编译器能够对操作进行大量融合，从而更有效地利用硬件。当 TorchTPU 收到来自 PyTorch 调度器的回调调用时，运行时会：

*   创建一个 Deferred Op（延迟算子）实例，其中包含：
    *   一个 StableHLO 构建器对象，调用时会返回一个实现 `aten::mul` 语义的 StableHLO 函数，如下所示（请注意，任何隐式广播和 dtype 提升都在其他地方处理）

```
func.func @main(%arg0: tensor<4x16xbfloat16>, %arg1: tensor<4x16xbfloat16>) -> tensor<4x16xbfloat16> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x16xbfloat16>
  return %0 : tensor<4x16xbfloat16>
}
```

*   一个可选的静态算子参数映射（例如 `aten::native_batch_norm` 的 training 和 momentum），用于计算缓存键；
*   返回一个新的 `at::Tensor`，其 `c10::StorageImpl`（通过间接方式）指向该 Deferred Op 实例。

到达适当的同步点（取决于执行模式）后，TorchTPU 将：

*   分析算子区域；
*   确定哪些算子要具体化为可执行代码；
*   为该图创建编译缓存键；
*   如果缓存中已存在，则获取该可执行文件并分派到 TPU 上执行；
*   如果缓存中不存在，则异步编译该对象，将其添加到缓存中，然后分派执行；
*   如果所到达的同步点需要，则可能会等待值就绪并将其复制回来。

我们会在 [TorchTPU 执行模型](./torchtpu_execution_model.md)中详细讨论各种执行模式。

## TorchTPU 编译缓存 { #the-torchtpu-compilation-cache }

为了避免不必要的编译，TorchTPU 采用了多层级编译缓存：

*   **第 1 层**：进程本地的内存缓存，用于在同一进程内共享编译结果。这是三层中最快的一层，并始终启用。
*   **第 2 层**：机器本地的共享内存，支持在同一主机上跨进程共享。这对多设备训练/推理很有用。对于分布式程序，第 2 层默认启用；但对于单设备程序，它没有任何好处，因此处于停用状态。
*   **第 3 层**：由远程文件系统（例如 Google Cloud Storage）支持的持久性缓存，可在不同主机、程序和作业重启之间重复使用。在 Colab 和数据中心等执行环境中运行 TPU 程序时，这很有用。此层默认停用，但可以通过指定存储位置来启用。

每个条目都是一个编译好的 TPU 可执行文件，代表由一个 ATen 算子或一系列 ATen 算子生成的 StableHLO 计算图。StableHLO 图由指纹表示，指纹综合了图的结构、输入/输出的张量形状以及编译器选项（包括 TPU 硬件、libtpu 和 PjRt 运行时的版本）。

## `torch.compile` 支持 { #torchcompile-support }

为了与其他支持 `torch.compile()` 的硬件保持一致的用户体验，我们注册了一个新的 TorchDynamo [后端](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html)，用法如下：`compiled_model = torch.compile(model,
backend="tpu")`。

Dynamo 通过传递 `torch.fx.GraphModule` 和一个示例输入张量列表来调用 TorchTPU 后端。此时，运行时会像在 eager 模式下一样跟踪所提供的图，但所有算子都会被延迟，不会在 TPU 上执行。运行时会为整个 FX 图生成并编译一个 StableHLO 程序，并返回一个封装该调用的 Python 可调用对象。请注意，TorchTPU 不使用 `TorchInductor`，而是复用随 TPU 架构一同发展起来的成熟 XLA 编译器栈。

## 更多详情 { #more-detail }

在以下文档中，我们将详细介绍：

*   TPU 支持的[内存模型](./torchtpu_memory_model.md)及其对 PyTorch 的影响；
*   TorchTPU 支持的[执行模式](./torchtpu_execution_model.md)；
*   对[自定义 kernel](./custom_kernels.md) 的支持；
*   在 TPU 集群上进行[分布式执行](./distributed_support.md)。

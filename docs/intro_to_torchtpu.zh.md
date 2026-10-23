# TorchTPU 简介 { #introduction-to-torchtpu }

<!-- Link to colab and to generated notebook -->

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)

TorchTPU 是一个面向 [Google Cloud TPU](https://cloud.google.com/tpu) 的开源原生 PyTorch 后端，由 Google 与 Meta 的 PyTorch 团队合作开发。借助 TorchTPU，开发者只需对代码做少量修改，就能在 TPU 硬件上运行现有的 PyTorch 工作负载，并将训练和推理工作负载从单个芯片扩展到多切片 TPU Pod。

## 为什么要使用 TorchTPU？ { #why-use-torchtpu }

Google TPU 是专为驱动深度学习的矩阵和向量运算而打造的专用集成电路（ASIC），可在大规模场景下提供高吞吐量、高能效和高性价比。虽然 TPU 过去一直与 JAX 和 TensorFlow 生态系统搭配使用，但 TorchTPU 将原生 TPU 支持直接带入了 [PyTorch 生态系统](https://github.com/google-pytorch/torch_tpu)，其设计刻意遵循以下三项指导原则：

*   **原生融入 PyTorch**：TorchTPU 直接与 PyTorch 集成，不依赖 JAX 运行时。它保留了标准的 PyTorch 张量视图和原地修改语义，采用 PyTorch 的“每个设备一个进程”执行模型，并通过 Dynamo 支持 `torch.compile`。
*   **开箱即用**：将工作负载从 GPU 移植到 TPU，首先只需将 `device="cuda"` 改为 `device="tpu"`，并将 `backend="nccl"` 改为 `backend="tpu_dist"`。未经修改的 PyTorch 模型可以立即在 eager 模式下运行，在诊断运行时或编译器问题时会提供清晰的错误消息。
*   **生产规模的性能**：除了逐个算子的 eager 执行之外，TorchTPU 还引入了延迟融合 eager 模式、从 Dynamo 到 XLA 的编译器流水线、可避免因形状变化而重新编译的有界动态性，以及自定义 kernel API，帮助您达到硬件的峰值利用率。

## 快速入门：在 TPU 上运行 { #quickstart-running-on-tpu }

在以下代码单元中，您可以看到一个在 TPU 上运行的简单 PyTorch 程序示例。启用 TPU 运行时后，您可以[在 Colab 上运行此示例](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)。

```python
import torch

device = torch.device("tpu")
x = torch.randn((4, 4), device=device)
y = torch.ones((4, 4), device=device)

result = torch.matmul(x, y).cpu()
print(f"Executed on device: {x.device}")
print(f"Result shape on CPU: {result.shape}, sum: {result.sum():.4f}")
```

## 主要功能 { #key-features }

*   **全面的算子和模型覆盖**：支持 450 个核心 ATen 算子（包括前向和反向传播），并在 5,000 个热门 HuggingFace 模型上持续验证，涵盖 LLM（Llama、Qwen、Gemma、DeepSeek、Mistral）、视觉模型（ViT、ResNet、ConvNeXt、DINOv2）、音频模型（Whisper、Wav2Vec2）以及扩散流水线（FLUX、Stable Diffusion）。
*   **三种 eager 执行模式**：
    *   **Strict（默认）**：一次将一个 ATen 算子异步分派到 TPU，与 GPU 上标准的 PyTorch eager 行为一致。
    *   **Debug（`TORCH_TPU_LAUNCH_BLOCKING=1`）**：每个算子都与 CPU 同步运行，并启用运行时边界检查，以简化调试。
    *   **DeferAndFuse（`TORCH_TPU_DEFER_AND_FUSE=1`）**：将 ATen 算子序列延迟并融合为经过优化的计算块，直到遇到具体化点（例如 `.cpu()` 或 `.item()`），从而显著减少高带宽内存（HBM）流量并加速 eager 执行。
*   **使用 `torch.compile` 进行图编译**：注册 `"tpu"` Dynamo 后端（`torch.compile(model)`），将整个 PyTorch FX 图降级（lower）为 [StableHLO](https://openxla.org/stablehlo/spec)，并使用 [OpenXLA 编译器](https://openxla.org/xla)进行编译。对动态形状（`torch._dynamo.mark_dynamic`）的积极支持使编译后的图能够处理不同的批次大小或序列长度（不超过指定上限），而无需重新编译。
*   **有界动态性**：通过针对动态上限进行编译，原生缓解了因批次大小和序列长度变化而导致的重新编译。
*   **多层级编译缓存**：通过三个缓存层级消除冗余的 XLA 编译：
    *   **第 1 层**：进程本地的内存缓存（始终启用）。
    *   **第 2 层**：主机本地的共享内存，用于在多设备作业中跨进程复用。
    *   **第 3 层**：持久性远程缓存（例如 Cloud Storage），用于跨主机和作业重启复用。
*   **原生分布式原语**：通过 `backend="tpu_dist"` 在 `torch.distributed` 进程组（`c10d::Backend`）层级实现 TPU 集合通信，支持 `DistributedDataParallel`（DDP）、`FullyShardedDataParallel`（FSDP2 和 SimpleFSDP）以及 `DTensor`。您还可以使用 `torch.tpu.distributed.topology_aware_mesh` 将并行维度与物理 TPU 互连对齐。
*   **自定义 TPU kernel**：使用 [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) 编写自定义 kernel（可通过 `torch.tpu.pallas.jax_op` 调用，并直接支持芯片间 RDMA 集合通信），使用实验性的 [Helion](https://pytorch.org/blog/helion/) DSL 编写基于 tile 的 kernel，或从 Tokamax 导入经过优化的预构建 TPU kernel。
*   **生态系统集成**：可与标准的 PyTorch 推理服务和训练框架以及性能分析套件开箱即用，包括 `torch.profiler`（TensorBoard/XProf）、vLLM、SGLang、TorchTitan、Ray 和 TorchRec。

## 限制和未来工作 { #limitations-and-future-work }

与任何产品一样，TorchTPU 也存在已知的限制，这些限制会随着时间推移逐步解决。虽然我们的首个版本已为主要工作负载提供了稳健的开箱即用覆盖，但 TorchTPU 是一项协作的、由社区驱动的工作。请查看我们的[功能集和路线图](./feature_set.md)，了解即将推出的内容，或者直接在 GitHub 上[提交 PR](https://github.com/google-pytorch/torch_tpu)，帮助我们解决这些限制。这是一个社区项目，我们很期待听到您的声音，并邀请您通过 [GitHub](https://github.com/google-pytorch/torch_tpu)、[https://discuss.pytorch.org/c/tpu](https://discuss.pytorch.org/c/tpu)、[https://dev-discuss.pytorch.org/c/tpu/](https://dev-discuss.pytorch.org/c/tpu/) 和 [torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com) 与我们合作。

*   动态形状会导致重新编译
    *   在 TorchTPU 上，`torch.compile` 默认设置 `dynamic=False`。`dynamic=None` 或 `dynamic=True` 尚不支持用于生产环境。[请参阅关于动态形状的讨论](./dynamic_shapes.md)
    *   例如，在 Hugging Face LLM 上运行 decode/generate 步骤时，除非启用有界动态性，否则每一步都会涉及重新编译。
*   视图算子在 TPU 上的开销可能很大（请参阅 [TorchTPU 内存模型](./torchtpu_memory_model.md)）。
*   带集合通信的 MPMD 在功能上受支持，但目前还无法达到峰值性能。集合通信与计算的重叠存在[一些限制](./distributed_support.md)：
    *   完整的 stream 支持（包括 SparseCore 卸载）即将推出。
    *   请将 SimpleFSDP 与 `torch.compile()` 搭配使用，而不是将 FSDPv2 与 eager 或 `torch.compile` 搭配使用，请参阅 [FSDP 最佳实践](./case_studies_fsdp.md)
    *   在通信模块上从 XLA 优化中获得峰值性能（即启用计算/通信重叠）的推荐方式是：在**用户保证**完全相同的区域上使用 `spmd_safe` 装饰器
*   TorchTPU 目前不原生支持 FlexAttention
*   当前一代 TPU 不支持以低于 fp32 的精度（例如 tf32）进行矩阵乘法累加。
*   一个 TPU 设备只能由一个进程访问，这限制了对共置 RL 的支持（请注意，一个 TPU 芯片可以关联一个或多个设备，这些设备可以由不同的操作系统进程访问）。

**其他建议：**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 限制 | 建议 |
| :--- | :--- |
| eager 模式的预热时间不为零 | 启用持久性编译缓存 |
| Torch.compile 耗时较长 | 启用持久性编译缓存 |
| PyTorch 中的 CNN 默认使用通道优先（channel-first）布局，在 TPU 上效率很低 | 使用此 [PyTorch 选项](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)设置通道在后（channel last）布局，以获得更好的性能。 |
| 由于 TPU 与其他架构的拓扑不同，默认的设备网格映射可能不是最优的 | 使用 `topology_aware_mesh` 获得最优的设备网格映射，并可自定义设备网格映射 |
<!-- mdformat on -->

## 开始使用 { #getting-started }

本文档集分为四个结构化部分，帮助您从初始设置一路推进到分布式生产部署：

*   **TorchTPU 入门**：分步演示如何配置环境，并在[虚拟机](./get_started_on_a_vm.md)上或[通过 GKE](./get_started_on_gke.md) 部署工作负载。
*   **深入了解**：深入探讨 [TPU 内存模型](./torchtpu_memory_model.md)、[分布式执行](./distributed_support.md)和不同的[执行模式](./torchtpu_execution_model.md)。
*   **案例研究**：端到端的实践方案，涵盖 [Flux 模型](./case_studies_flux.md)的优化、[Muse Glimmer](./case_studies_muse_glimmer_30b_training.md) 训练、[端到端 LLM 推理](./case_studies_muse_glimmer_30b_inference.md)以及 [FSDP](./case_studies_fsdp.md) 的有效应用。
*   **分析工作负载**：[性能分析](./performance_profiling_with_xprof.md)、管理[数值精度](./numerical_precision.md)

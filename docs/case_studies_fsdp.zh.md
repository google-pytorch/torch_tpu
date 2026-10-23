# FSDP 最佳实践 { #fsdp-best-practices }

完全分片数据并行（Fully Sharded Data Parallel，FSDP）会将模型参数、梯度和优化器状态分片到各个数据并行 rank 上，使每个 TPU 核心只存储模型的 `1 / world_size`。在某一层运行之前，系统会通过 **all-gather** 重建该层的完整权重；在反向传播期间，梯度会通过 **reduce-scatter** 进行合并并重新分片。

本指南介绍在 TorchTPU 上使用 FSDP 的推荐方式，并指向 [`examples/fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp) 中可运行的示例。

在本指南中，您将了解：

1.  在 TorchTPU 上**应选择哪种 FSDP 方式**。
2.  **SimpleFSDP（推荐）**：以编译为先的 FSDP，只需一次封装调用。
3.  **手动 FSDP**：直接在模型定义中编写集合通信操作。
4.  适用于所有方式的**最佳实践**。
5.  **运行示例**：启动命令以及每个示例检查的内容。

## 1. 选择 FSDP 方式 { #1-choosing-an-fsdp-approach }

```
On TPU, performance comes from compiling the whole training step into a single
XLA program with `torch.compile(backend="tpu")`. When the XLA compiler sees all
forward, backward, and collective operations at once, it can schedule
collectives to overlap with computation and combine small collectives when
profitable.

Choose an FSDP approach based on how well it composes with whole-graph
compilation:
```

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
    | 方式 | 适用场景 | 示例 |
    | :------- | :------------- | :------ |
    | **SimpleFSDP** (`simple_fsdp.data_parallel`) | **推荐的默认方式。** 您希望在不修改模型定义的情况下使用 FSDP 或 HSDP。| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) |
    | **手动 FSDP**（手动集合通信） | 您拥有模型代码，并希望将通信代码放在计算代码旁边，以便进行性能调优。 | [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) |
    | **PyTorch FSDPv2** (`fully_shard`) | 尚未迁移的现有代码。它可以正常运行，但其运行时钩子会导致图中断，而且要获得有竞争力的性能，还需要即将推出的、将 stream 卸载到 TPU SparseCore 的支持。
.| [迁移指南](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md) |
<!-- mdformat on -->

```
SimpleFSDP and manual FSDP both express the collectives as part of the traced
graph, so XLA can optimize them together with the rest of the step. PyTorch
FSDPv2 (`fully_shard`) relies on Python pre-forward and post-backward hooks.
These hooks split the step into many smaller graphs, even when the model is
compiled.

For a side-by-side comparison of compiled FSDPv2 and compiled SimpleFSDP, see
the
[FSDPv2 to SimpleFSDP tutorial](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md).
```

## 2. 推荐：SimpleFSDP { #2-recommended-simplefsdp }

[`simple_fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/simple_fsdp.py) 是一种专为与 `torch.compile` 配合使用而设计的 FSDP 实现。它最初在 TorchTitan 中引入，`examples/fsdp/simple_fsdp.py` 中提供了一个仅依赖 PyTorch 的独立副本。

SimpleFSDP 将每个参数存储为分片的 `DTensor`。访问参数时，参数化（parametrization）会调用 `DTensor.redistribute` 对其执行 all-gather。`redistribute` 的 autograd 反向传播会执行梯度的 reduce-scatter。由于没有运行时钩子，整个步骤可以在没有图中断（graph break）的情况下完成编译。只需调用一次 `simple_fsdp.data_parallel(...)` 即可对整个模型进行分片，无需逐层封装或调整预取设置；在 2D `DeviceMesh` 上使用 `mode="hybrid_shard"` 即可支持 HSDP。

如需了解如何使用 SimpleFSDP，请参阅：

*   [SimpleFSDP README](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/README.md)：快速入门和可用选项。
*   [FSDPv2 到 SimpleFSDP 教程](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md)：如何迁移现有的 `fully_shard` 代码。

## 3. 替代方案：手动 FSDP { #3-alternative-manual-fsdp }

如果您能控制模型定义，可以直接使用手动集合通信操作（`torch.distributed._functional_collectives`）和 `torch.compile` 编写 FSDP。每一层只存储自己的权重分片，并在前向传播中对完整权重执行 all-gather。函数式集合通信操作带有 autograd 公式，因此 all-gather 的反向传播就是 reduce-scatter，您无需编写自定义的 `torch.autograd.Function`。

这种方式让通信代码与计算代码放在一起，可以更方便地进行性能调优。代价是您必须修改模型定义。

如需了解如何实现手动 FSDP，请参阅[手动 FSDP 指南](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual/manual_fsdp.md)。

## 4. 最佳实践 { #4-best-practices }

以下建议适用于 SimpleFSDP 和手动 FSDP。

1.  **使用 `torch.compile(backend="tpu")` 编译整个训练步骤**。如果无法编译整个步骤，至少应将前向传播、损失计算和反向传播一起编译。只编译模型模块无法生成单个计算图，这会限制 XLA 将集合通信与计算重叠的效果。
2.  **对每个分片层使用激活检查点**。将每一层的前向传播封装在 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` 中。这样，未分片的权重会在每层前向传播结束后释放，而不是一直保留到反向传播，从而实现 FSDP 预期的 ZeRO-3 内存行为。
3.  **确保分片维度能被网格大小整除**。手动 FSDP 要求满足这一点。SimpleFSDP 会复制太小而无法分片的参数。使用 `shard_dim` 选择一个可以均匀划分的维度。
4.  **初始化分片权重时，为每个 rank 设置不同的种子**。如果每个 rank 都用相同的种子初始化自己的分片，完整权重就会是同一个随机块在各 rank 间的重复，这可能会影响收敛。如需获得可复现且互不相同的分片，请为每个 rank 设置一个与 rank 相关的种子值，例如 `torch.manual_seed(base_seed + dist.get_rank())`。
5.  **让所有 rank 运行相同的程序**。不要将强制具体化的操作（例如 `.item()`）放在 `if
    rank == 0` 之类的特定于 rank 的分支中。
6.  **当在整个网格上完全分片无法扩展时，请使用 HSDP**。如果您的 ICI 切片太大，无法有效地对模型进行分片，或者在其上分片会产生过多的网络流量，请在 2D 网格上使用 `mode="hybrid_shard"`。在多个切片上运行时，最好的做法几乎总是在较慢的 DCN 上进行复制（数据并行），并在每个切片内部进行分片。
7.  **在扩大规模之前检查内存和数值**。使用 `assert_close` 与未分片或 FSDPv2 参考实现进行比较，并测量在前向-反向边界上保留的内存。示例中包含一个一致性检查（`fsdp_to_simple_fsdp_worker.py`）和一个内存测试（`manual_fsdp_memory_test.py`）。

## 5. 运行示例 { #5-running-the-examples }

每个示例都附带一个启动脚本，可在 8 个 TPU 核心（例如 `v6e-8`）上运行：

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 示例 | 启动命令 | 检查内容 |
:-------- | :------------- | :------------- |
| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) | `./run_fsdp_to_simple_fsdp.sh` | 编译后的 FSDPv2 与编译后的 <br> SimpleFSDP 的损失和预测结果一致， <br> 并报告二者的 <br> 步骤时间。 |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `./launch.sh` 或 `python3 launch.py` | 手动 FSDP 的梯度和预测结果 <br> 与单设备 CPU 参考结果一致。 |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `python3 manual_fsdp_memory_test.py` | 未分片的权重在 <br> 前向传播后被释放。 |
<!-- mdformat on -->

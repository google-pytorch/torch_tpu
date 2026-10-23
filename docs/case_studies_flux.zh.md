# FLUX 优化 { #flux-optimization }

<!-- Link to colab and to generated notebook -->

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_flux.ipynb)

本案例研究介绍了我们如何基于 [FLUX](https://github.com/black-forest-labs/flux) 扩散 Transformer `DoubleStreamBlock` 的 PyTorch 参考实现，在序列长度为 32,768 时，借助 TorchTPU 让训练步骤在 Cloud TPU 上的速度**提升至 1.48 倍**。

本页是完整教程的精简配套内容，通过简短、独立的代码段重点介绍每项优化背后的关键概念。如需查看完整的模型代码、基准测试、XProf 跟踪记录和测试，请参阅：

*   [FLUX 优化教程](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md)：完整讲解每个里程碑，包括 XProf 跟踪记录以及复现每项基准测试的说明。
*   [为 PyTorch 编写自定义 Pallas kernel](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)：分步讲解里程碑 4 中使用的 Pallas SplashAttention kernel。

该实现基于 [FLUX 的 TorchTitan 实现](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/)和 [black-forest-labs/flux](https://github.com/black-forest-labs/flux/blob/main/src/flux/) 代码库。

如果您是第一次接触这些示例，可以先[通过我们的其他文档](./get_started_on_gke.md)熟悉如何在 GKE 上运行示例。

## 结果概览 { #results-at-a-glance }

单个 `DoubleStreamBlock` 的平均训练步骤时间（前向传播 + 损失 + 反向传播 + 优化器步骤）：

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 配置 | 平均步骤时间 | 相对基准的加速比 |
| :--- | ---: | ---: |
| 基准参考实现（eager） | 178.61 ms | 1.00x |
| 仅 Pallas SplashAttention（eager） | 149.81 ms | 1.19x |
| 仅零拷贝视图（eager） | 178.30 ms | 1.00x |
| 仅 `torch.compile` | 159.33 ms | 1.12x |
| 零拷贝视图 + 融合 RoPE + Pallas SplashAttention（eager） | 156.73 ms | 1.14x |
| 零拷贝视图 + `torch.compile` | 159.08 ms | 1.12x |
| 零拷贝视图 + 融合 RoPE + `torch.compile` | 153.60 ms | 1.16x |
| **零拷贝视图 + 融合 RoPE + Pallas SplashAttention + `torch.compile`** | **122.51 ms** | **1.46x** |
<!-- mdformat on -->

**测量设置：**

-   **硬件**：单个 TPU v6e 芯片（32 GiB HBM）。
-   **软件**：Python 3.14、PyTorch 2.15（预发布版）、JAX 0.12、einops 0.8.2，以及从源代码构建的 TorchTPU。
-   **模型形状**：`batch_size=2`、`hidden_size=1024`、`num_heads=8`（`head_dim=128`）、`img_seq_len=32256`、`txt_seq_len=512`（总序列长度为 32,768）。
-   **精度**：bfloat16 格式的块参数和输入。
-   **SplashAttention 块大小**：`block_q`、`block_kv` 以及 dK/dV 反向传播 tile 均为 2048。
-   **计时**：3 个预热步骤，之后是 10 个计时步骤。计时步骤会连续分派，并在最后一个步骤之后运行一次 `torch.accelerator.synchronize()`，因此平均步骤时间（总时间除以步骤数）包含设备执行时间。在计时步骤期间，性能分析器处于停用状态。

在此序列长度下，注意力占据了步骤的大部分时间，因此 SplashAttention 和编译带来的收益最大。在较短的序列长度下，编译的作用更大（序列长度为 640 时约为 3 倍），而注意力 kernel 的作用较小。

这些经验适用于 TPU 上的大多数 Transformer 模型：

1.  **避免隐藏的布局复制**：优先使用零拷贝视图（`.view()`、`torch.unbind()`），而不是会具体化新 HBM 缓冲区的重排操作。
2.  **少存储，多计算**：将位置嵌入表示为紧凑的逐元素数学运算，而不是具体化的矩阵。
3.  **编译整个块**：`torch.compile()` 让 XLA 可以融合逐元素运算链，并消除中间结果在 HBM 之间的往返。
4.  **必要时使用自定义 kernel**：对于长序列长度下的注意力等受内存带宽限制的操作，Pallas kernel 可让您明确控制片上分块。

### 如何阅读本页 { #how-to-read-this-page }

每个里程碑都以一对小节的形式呈现：

-   **优化前（参考实现）**：原始实现及其瓶颈。
-   **优化后（优化实现）**：优化后的实现，随后检查其结果是否与参考实现相同。

代码单元的第一行带有对应的 `# BEFORE (reference)` 或 `# AFTER
(optimized)` 注释作为标记。

**前提条件**：里程碑 1 的参考实现代码单元使用 [einops](https://github.com/arogozhnikov/einops)（`pip install einops`）。

```python
import torch

device = torch.device("tpu")
print(f"Running on {device}.")
```

## 里程碑 1：零拷贝张量视图 { #milestone-1-zero-copy-tensor-views }

在 `DoubleStreamBlock` 中，图像流和文本流会将形状为 `(B, L, 3 * H * D)` 的隐藏状态投影为 query、key 和 value。如何将该投影拆分为 Q、K 和 V，决定了 TPU 在训练期间是否需要在 HBM 中复制数据。

### 优化前（参考实现）：`einops.rearrange` { #before-reference-einopsrearrange }

参考实现将 `3`（Q/K/V）轴移到最前面：

```python
img_q, img_k, img_v = einops.rearrange(
    img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
)
```

在前向传播中，`einops.rearrange` 返回投影的跨步视图，因此此时还没有移动数据。开销出现在**反向传播**中：为了让梯度沿着轴重排的路径传回，autograd 会在新的缓冲区中具体化梯度，这在 TPU 上表现为每个注意力块中额外的一次 HBM 到 HBM 复制（在 XProf 中显示为 `copy` 操作）。

```python
# BEFORE (reference): einops.rearrange moves the Q/K/V axis to the front.
import einops

num_heads, head_dim = 4, 32
qkv = torch.randn(2, 16, 3 * num_heads * head_dim, device=device)

q_before, _k, _v = einops.rearrange(
    qkv, "B L (K H D) -> K B L H D", K=3, H=num_heads
)
print(f"q shape: {tuple(q_before.shape)}")
```

### 优化后（优化实现）：`.view()` + `torch.unbind()` { #after-optimized-view-torchunbind }

优化后的实现让 `3` 轴紧邻 `(H, D)`，并使用 `torch.unbind` 进行拆分。这样，反向传播会将 Q、K 和 V 的梯度直接堆叠到原始布局中，无需额外复制。在结果表中，这项优化对步骤时间的影响很小（eager 模式下从 **178.61 ms 降至 178.30 ms**，编译模式下从 **159.33 ms 降至 159.08 ms**），因为复制操作只占步骤时间的一小部分，但这项优化可以零成本地消除该复制。

> **Note:** 此里程碑中的代码单元仅展示两种方法会生成相同的 Q、K 和 V。在前向传播中，两者都返回相同的跨步视图，因此输出看起来完全一致。`einops.rearrange` 带来的额外 HBM 复制只出现在反向传播中，需要在完整训练步骤的 XProf 跟踪记录中查看，而不会体现在代码单元的输出中。如需查看跟踪记录，请参阅 [FLUX 优化教程的里程碑 1](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-1)。

```python
# AFTER (optimized): keep Q/K/V adjacent to (H, D) and unbind.
_b, _l, _ = qkv.shape
q_after, _k, _v = torch.unbind(
    qkv.view(_b, _l, 3, num_heads, head_dim), dim=2
)
print(f"q shape: {tuple(q_after.shape)}")
print(f"Matches reference: {torch.equal(q_after.cpu(), q_before.cpu())}")
```

## 里程碑 2：融合的代数 RoPE { #milestone-2-fused-algebraic-rope }

FLUX 会对 query 和 key 应用多轴旋转位置嵌入（RoPE）。以下代码单元针对单个 RoPE 轴实现了这两个版本。

### 优化前（参考实现）：2x2 旋转矩阵 { #before-reference-2x2-rotation-matrices }

参考实现会为每个频率对具体化一个完整的 `2x2` 旋转矩阵 `[[cos,
-sin], [sin, cos]]`，并将其作为矩阵-向量积应用到重塑后的 6D query/key 张量上。
这会将 `cos` 和 `sin` 各存储两次，额外的堆叠和重塑操作会在 XProf 中显示为大型复制块。

```python
# BEFORE (reference): materialize 2x2 rotation matrices, then contract.
def rope_ref(pos, dim, theta=10000):
  """Returns rotation matrices of shape (B, S, D/2, 2, 2)."""
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  out = torch.stack(
      [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)],
      dim=-1,
  )
  return out.reshape(*out.shape[:-1], 2, 2).float()

def apply_rope_ref(xq, xk, freqs_cis):
  """6D matrix-vector contraction against the rotation matrices."""
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return (
      xq_out.reshape(*xq.shape).type_as(xq),
      xk_out.reshape(*xk.shape).type_as(xk),
  )

_b, _s, _h, rope_head_dim = 2, 64, 4, 32
positions = (
    torch.arange(_s, dtype=torch.float32, device=device)
    .unsqueeze(0)
    .expand(_b, _s)
)
rope_xq = torch.randn(_b, _s, _h, rope_head_dim, device=device)
rope_xk = torch.randn(_b, _s, _h, rope_head_dim, device=device)

# Add a broadcast head dimension: (B, S, 1, D/2, 2, 2).
_freqs_ref = rope_ref(positions, rope_head_dim).unsqueeze(2)
q_rope_ref, k_rope_ref = apply_rope_ref(rope_xq, rope_xk, _freqs_ref)

print(f"Rotation matrix freqs shape: {tuple(_freqs_ref.shape)}")
```

### 优化后（优化实现）：紧凑的 `[cos, sin]` 对 + 逐元素旋转 { #after-optimized-compact-cos-sin-pairs-elementwise-rotation }

优化后的实现只存储紧凑的 `[cos, sin]` 对，并直接通过逐元素数学运算执行旋转：

```python
# (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
xq_out_0 = xq_0 * cos - xq_1 * sin
xq_out_1 = xq_0 * sin + xq_1 * cos
```

这样可以将位置嵌入占用的空间减半，映射为融合乘加指令，并在编译时与周围的投影和注意力操作融合。在零拷贝视图和 `torch.compile` 的基础上，它将步骤时间从 **159.08 ms 降至 153.60 ms（3.4%）**。

由于两个版本对相同的操作数执行相同的乘法和加法，因此它们会生成完全相同的结果。以下代码单元在相同的输入上运行优化后的版本，并将其与参考实现进行比较：

> **Note:** 这些代码单元展示了更小的频率张量，以及两个版本会生成完全相同的结果。只有当 RoPE 在已编译的 `DoubleStreamBlock` 中运行时才会体现出加速效果，此时逐元素旋转会与周围的操作融合。如需查看 XProf 跟踪记录，请参阅 [FLUX 优化教程的里程碑 2](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-2)。

```python
# AFTER (optimized): store compact [cos, sin] pairs, rotate elementwise.
def rope(pos, dim, theta=10000):
  """Returns compact [cos, sin] pairs of shape (B, S, D/2, 2)."""
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  return torch.stack([torch.cos(out).float(), torch.sin(out).float()], dim=-1)

def apply_rope(xq, xk, freqs_cis):
  """Direct 2D algebraic rotation with elementwise math."""
  cos = freqs_cis[..., 0]
  sin = freqs_cis[..., 1]
  b, s, h, d = xq.shape
  xq_ = xq.float().view(b, s, h, d // 2, 2)
  xk_ = xk.float().view(b, s, h, d // 2, 2)
  xq_0, xq_1 = xq_[..., 0], xq_[..., 1]
  xk_0, xk_1 = xk_[..., 0], xk_[..., 1]
  xq_out = torch.stack(
      [xq_0 * cos - xq_1 * sin, xq_0 * sin + xq_1 * cos], dim=-1
  ).view(b, s, h, d)
  xk_out = torch.stack(
      [xk_0 * cos - xk_1 * sin, xk_0 * sin + xk_1 * cos], dim=-1
  ).view(b, s, h, d)
  return xq_out.type_as(xq), xk_out.type_as(xk)

# Add a broadcast head dimension: (B, S, 1, D/2, 2).
freqs = rope(positions, rope_head_dim).unsqueeze(2)
_q_new, _k_new = apply_rope(rope_xq, rope_xk, freqs)

print(f"Compact freqs shape:         {tuple(freqs.shape)}")
print(
    "Max |q_ref - q_new|:        "
    f" {(q_rope_ref.cpu() - _q_new.cpu()).abs().max().item():.3e}"
)
print(
    "Max |k_ref - k_new|:        "
    f" {(k_rope_ref.cpu() - _k_new.cpu()).abs().max().item():.3e}"
)
torch.testing.assert_close(
    _q_new.cpu(), q_rope_ref.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
torch.testing.assert_close(
    _k_new.cpu(), k_rope_ref.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
```

## 里程碑 3：整块 XLA 编译 { #milestone-3-whole-block-xla-compilation }

### 优化前（参考实现）：eager 执行 { #before-reference-eager-execution }

在 eager 模式下，每个 PyTorch 操作都会启动一个独立的 TPU 程序，因此中间激活值需要在 HBM 中往返，主机也经常因分派而停顿。在结果表中，基准 `DoubleStreamBlock` 在 eager 模式下每个步骤耗时 **178.61 ms**。

以下代码单元以 eager 方式逐个操作地运行里程碑 2 中的融合 RoPE：

```python
# BEFORE (reference): eager execution, one TPU program per op.
q_eager, k_eager = apply_rope(rope_xq, rope_xk, freqs)
print("Ran fused RoPE eagerly.")
```

### 优化后（优化实现）：`torch.compile()` { #after-optimized-torchcompile }

使用 `torch.compile()` 编译整个 `DoubleStreamBlock`，会将其降级为单个 XLA 计算图，让编译器可以融合逐元素操作、归一化和残差加法：

```python
block = DoubleStreamBlock(
    hidden_size=1024,
    num_heads=8,
    mlp_ratio=4.0,
    config=OptimizationConfig(
        use_zero_copy_view=True,
        use_fused_rope=True,
        compile=True,
    ),
)
compiled_block = block.maybe_compile()  # Wraps the block in torch.compile().
```

编译将步骤时间从 **178.61 ms 降至 159.33 ms（1.12 倍）**。由于在序列长度为 32,768 时注意力占据了步骤的大部分时间，因此收益有限。在序列长度为 640 时，编译后的速度约为原来的 3 倍。

同样的模式适用于任何函数。以下代码单元会编译融合 RoPE，并检查其结果是否与 eager 执行一致：

> **Note:** 这些代码单元仅检查编译不会改变结果。融合效果和加速效果会体现在完整 `DoubleStreamBlock` 的 XProf 跟踪记录和步骤时间测量结果中，而不会体现在代码单元的输出中。如需查看 eager 与编译模式的跟踪记录对比，请参阅 [FLUX 优化教程的里程碑 3](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-3)。

```python
# AFTER (optimized): compile into a single fused XLA graph.
compiled_apply_rope = torch.compile(apply_rope, backend="tpu")
_q_compiled, _k_compiled = compiled_apply_rope(rope_xq, rope_xk, freqs)

torch.testing.assert_close(
    _q_compiled.cpu(), q_eager.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
torch.testing.assert_close(
    _k_compiled.cpu(), k_eager.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
print("Compiled fused RoPE matches eager execution.")
```

## 里程碑 4：自定义 Pallas SplashAttention kernel { #milestone-4-custom-pallas-splashattention-kernel }

在长序列长度（例如 `S = 2048` 或 `S = 4096`）下，完整的联合注意力会受内存带宽限制。此里程碑仅以代码段形式展示；如需查看完整的可运行实现，请参阅 [Pallas kernel 详解](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)。

### 优化前（参考实现）：标准 PyTorch SDPA { #before-reference-standard-pytorch-sdpa }

```python
attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

未分块的注意力会在 HBM 中具体化一个 `O(S^2)` 的分数矩阵。

### 优化后（优化实现）：Pallas SplashAttention { #after-optimized-pallas-splashattention }

```python
attn = splash_sdpa(q, k, v, is_causal=False)  # Drop-in replacement.
```

`splash_sdpa` 是一个自定义 [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) SplashAttention kernel，它以片上 tile 为单位计算注意力，基于 [SplashAttention 的 TorchTitan 实现](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/kernels/splash_attention.py)。

该 kernel 通过 `torch.tpu.pallas.jax_op` 提供给 PyTorch。其前向和反向函数使用 `jax.Array` 类型进行注解，`jax_op` 会据此推断 PyTorch 自定义算子的 schema。前向传播会保存 softmax 的 `logsumexp` 残差，这样反向 kernel 就无需重新运行前向计算：

```python
torch_fwd_fn = torch.tpu.pallas.jax_op(f"pallas::{op_fwd_name}", splash_fn)
torch_bwd_fn = torch.tpu.pallas.jax_op(f"pallas::{op_bwd_name}", splash_bwd_fn)


def setup_context(ctx, inputs, output):
  q, k, v = inputs
  out, lse = output
  ctx.save_for_backward(q, k, v, out, lse)


def backward(ctx, grad_out, grad_lse):
  del grad_lse
  q, k, v, out, lse = ctx.saved_tensors
  return torch_bwd_fn(q, k, v, out, lse, grad_out)


torch_fwd_fn.register_autograd(backward, setup_context=setup_context)
```

> **Important:** **运行时不会执行 JAX。** Pallas 随 `jax` 软件包一起分发，因此 kernel 使用 `jax` 语法编写。不过，`torch.tpu.pallas.jax_op` 只在跟踪时使用一次 JAX，借助抽象的 shape/dtype 占位符将 kernel 导出为 MLIR 模块。在训练和推理期间，无论是在 eager 模式下还是在 `torch.compile` 下，kernel 都直接在 `torch.Tensor` TPU 缓冲区上运行。

**结果**：在结果表中，仅使用 SplashAttention 时，速度是 eager 基准的 **1.19 倍**（从 178.61 ms 降至 149.81 ms）。在启用零拷贝视图和融合 RoPE 的情况下，无论是在 eager 模式（**从 182.61 ms 降至 156.73 ms，1.17 倍**）还是编译模式（**从 153.60 ms 降至 122.51 ms，1.25 倍**）下，它都比标准 SDPA 更快。

这些结果取决于针对工作负载配置 kernel 的方式。使用默认的 512 个 token 的块大小时，在相同序列长度下，编译模式中的 SplashAttention 比标准 SDPA 慢约 10%，而使用 float32 输入时会更慢。在动手编写自定义 kernel 之前，请针对您的序列长度对几种块大小进行基准测试，以 bfloat16 运行 kernel，并进行性能分析。

**了解详情：**

*   **Pallas SplashAttention 示例（本案例研究中使用）**：[`splash_attention.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention.py) 包含完整的 kernel 集成，[`splash_attention_test.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention_test.py) 则验证前向结果和梯度与 PyTorch SDPA 是否一致。[kernel 详解](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)分步说明了块大小、掩码、autograd 注册和测试。
*   **Pallas 基础知识**：[使用 Pallas 编写自定义 kernel](./custom_kernels.md) 介绍了如何使用 TorchTPU 编写 Pallas kernel。

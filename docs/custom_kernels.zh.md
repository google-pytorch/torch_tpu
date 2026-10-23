# TorchTPU 自定义 kernel { #torchtpu-custom-kernels }

虽然 TorchTPU 能原生加速标准的 PyTorch 框架算子（例如 [`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)），但您可能仍需要集成自定义 kernel（内核），以加速模型特有的操作、融合操作以降低内存带宽开销，或实现标准 PyTorch 算子尚未覆盖的新型架构。

## 选择集成方式 { #choosing-an-integration-approach }

根据您的用例，TorchTPU 提供三种在 TPU 上使用自定义 kernel 的方式：

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 方式 | 状态 | 最适合 |
| :---- | :---- | :---- |
| **Tokamax** | 可用于生产环境 | 无需编写 kernel 代码，即可直接加速常见的 LLM/MoE 构建块。 |
| **Pallas (`torch.tpu.pallas.jax_op`)** | 可用于生产环境 | 用 JAX/Pallas 编写定制的、性能最优的 TPU kernel，并从 PyTorch 调用。 |
| **Helion** | 实验性 | 使用 PyTorch 原生 DSL 编写可在多种加速器（例如 GPU、TPU）之间移植的 kernel。 |
<!-- mdformat on -->

## 1. 使用 Tokamax 中的预构建 kernel { #1-using-pre-built-kernels-from-tokamax }

在从头编写自定义 kernel 之前，请先检查 [Tokamax](http://openxla.org/tokamax) 中是否已提供该算子。Tokamax 是一个高性能 TPU kernel 库，原生支持与 PyTorch 互操作。

### 何时使用 { #when-to-use }

当您的模型需要标准的高性能构建块时，请使用 Tokamax，包括：

*   **混合专家（MoE）**：分组矩阵乘法、token gather 和 scatter。
*   **注意力变体**：Paged Attention、Gated DeltaNet、Kimi Delta Attention 和 Multi-head Latent Attention（MLA）。
*   **选择**：经过优化的 `TopK` 计算。

### 工作原理 { #how-it-works }

Tokamax kernel 使用 JAX/Pallas 编写，并配有集成的自动调优（例如自动搜索 tile 大小）、低开销的 XProf 基准测试，以及跨 TPU 代际的回归测试。在底层，Tokamax 使用 `torch.tpu.pallas.jax_op` 透明地处理框架互操作，并直接向您的模型公开原生 PyTorch API。

## 2. 编写自定义 Pallas kernel { #2-writing-custom-pallas-kernels }

如果您需要 PyTorch 或 Tokamax 未覆盖的定制操作，可以使用 [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) 编写 TPU kernel，并使用 `torch.tpu.pallas.jax_op` 将其注册为 PyTorch 算子。

### 如何集成 Pallas kernel { #how-to-integrate-a-pallas-kernel }

1.  使用输入和输出内存引用（`Ref`）**定义 Pallas kernel**。
2.  使用 `pl.pallas_call` **将 kernel 封装到 JAX 函数中**，以指定输出的形状和 dtype。
3.  通过 `torch.tpu.pallas.jax_op` **向 TorchTPU 注册 JAX 函数**。

```python
import jax
from jax.experimental import pallas as pl
import torch

# 1. Define the Pallas kernel
def add_vectors_pallas_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y

# 2. Wrap the Pallas kernel into a JAX function
def add_vectors_jax(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      add_vectors_pallas_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )(x, y)

# 3. Wrap it for PyTorch using the functional interface
add_vectors_fn = torch.tpu.pallas.jax_op(
    "pallas::add_vectors",
    add_vectors_jax,
)

# 4. Call it like a standard PyTorch function
x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
y = torch.tensor([4.0, 5.0, 6.0], device="tpu")
z = add_vectors_fn(x, y)
```

## 3. 使用 Helion 编写可移植 kernel（实验性） { #3-writing-portable-kernels-with-helion-experimental }

如果您希望 kernel 能在多种加速器（例如 TPU、GPU）之间移植，可以使用 [Helion](https://helionlang.com/)，这是一种用于编写 kernel 的 PyTorch 原生 DSL。

> **Note:** TorchTPU 对 Helion 的支持目前处于**实验性**阶段。对于生产工作负载，请使用 Tokamax 或 `torch.tpu.pallas.jax_op`。

### 如何编写 Helion kernel { #how-to-write-a-helion-kernel }

使用 `@helion.kernel` 装饰您的函数。使用标准 PyTorch 操作完成主机端设置，并使用 `hl.tile` 定义在分区 tile 上运行的设备执行循环：

```python
import helion
import helion.language as hl
import torch

# 1. Define the kernel in Helion
@helion.kernel
def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  # Standard PyTorch host-side setup
  x, y = torch.broadcast_tensors(x, y)
  out = torch.empty_like(x)

  # Device execution loop over automatically partitioned tiles
  for tile in hl.tile(out.size()):
    out[tile] = x[tile] + y[tile]

  return out

# 2. Call it like a standard PyTorch function
x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
y = torch.tensor([4.0, 5.0, 6.0], device="tpu")
z = add_vectors(x, y)
```

如需详细了解 Helion，请参阅 [Helion 文档](https://helionlang.com/)，在 [PyTorch 开发者讨论论坛](https://dev-discuss.pytorch.org/c/helion/31)中提问，或[在 GitHub 上报告问题](https://github.com/pytorch/helion/issues)。

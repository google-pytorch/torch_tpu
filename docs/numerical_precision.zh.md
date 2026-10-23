# 在 TorchTPU 中管理数值精度 { #manage-numerical-precision-in-torchtpu }

Google TPU 将浮点运算分配给两个硬件单元，以平衡吞吐量和数值准确性：

*   **矩阵乘法单元（MXU）**：使用原生 `bfloat16` 输入（8 位有效数）和 32 位 `float32` 累加来执行收缩运算（例如 `torch.matmul`、`torch.mm`、`torch.bmm` 和卷积）。
*   **向量执行单元（VPU）**：以完整的 IEEE-754 `float32`（24 位有效数）执行非收缩运算（例如逐元素算术运算、激活函数、层归一化和 softmax）。

本指南介绍如何配置 `float32` 矩阵乘法精度模式、验证 CPU 与 TPU 之间的数值一致性、控制输出累加器类型和超额精度，以及在代码中验证 MXU 的逐位行为。

### 比较 CUDA 和 TPU 上的 PyTorch 精度 { #compare-pytorch-precision-on-cuda-and-tpu }

TorchTPU 支持标准的 PyTorch 精度 API（`torch.set_float32_matmul_precision` 和 `torch.get_float32_matmul_precision`），以及 TPU 专用的控制项（`torch.tpu.Precision` 和 `torch.tpu.precision`）：

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 维度          | CUDA 上的 PyTorch                      | TPU 上的 PyTorch（TorchTPU）              |
| :----------------- | :----------------------------------- | :------------------------------------- |
| **默认精度模式** | `"highest"`（纯 `float32`） | `"medium"`（`Precision.DEFAULT`）       |
| **快速数学运算的加速器原语** | Tensor Core（`TF32`） | 脉动 MXU（`bfloat16`）             |
| **快速数学运算的位数** | 1 个符号位 + 8 个指数位 + 10 个尾数位（19 位 `TF32`） | 1 个符号位 + 8 个指数位 + 7 个尾数位（16 位 `bfloat16`） |
| **累加精度** | `float32`（`"highest"`、`"high"`），`TF32`（`"medium"`） | 所有模式均为 `float32` |
| **更高精度的实现机制** | 专用的 `float32`/`TF32` 硬件单元 | 多遍分解（3 遍或 6 遍 MXU 计算） |
| **`high`/`highest` 对性能的影响** | 基准速度或略有减慢（约 1–2 倍） | `"high"` 的 MXU 计算开销约为 3 倍；`"highest"` 约为 6 倍 |
| **作用域 API** | 全局 `torch.set_float32_matmul_precision` | 全局 <br> `torch.set_float32_matmul_precision` + 限定作用域的 `torch.tpu.precision` |
<!-- mdformat on -->

## 指定精度模式 { #specify-precision-modes }

在 MXU 上对 `float32` 张量进行乘法运算时，TorchTPU 会将 PyTorch 的精度级别映射到 `torch.tpu.Precision` 枚举：

*   `"medium"` / `Precision.DEFAULT`（1 遍）：**默认。**将 `float32` 输入舍入为 `bfloat16`（对有效数的最高 8 位进行乘法运算），并以 `float32` 累加。推荐用于几乎所有训练和推理工作负载。
*   `"high"` / `Precision.HIGH`（3 遍）：将 `float32` 输入拆分为高位和低位两个 `bfloat16` 分量，并计算三个交叉项（对有效数的前 16 位进行乘法运算），MXU 计算开销约为 3 倍。
*   `"highest"` / `Precision.HIGHEST`（6 遍）：将 `float32` 输入拆分为三个 `bfloat16` 分量，并计算六个交叉项（对有效数的全部 24 位进行乘法运算），以接近完整的 IEEE-754 `float32` 精度，MXU 计算开销约为 6 倍。

您可以通过两种方式配置矩阵乘法精度：

*   **全局配置**：调用 `torch.set_float32_matmul_precision("medium"
    | "high" | "highest")` 为进程设置默认模式。
*   **限定作用域的上下文管理器**：使用 `with torch.tpu.precision(mode):`，仅对数值敏感的层或子模块应用更高的精度。
*   **性能警告**：选择 `"high"` 或 `"highest"` 会发出一次性警告（`TORCH_WARN_ONCE`）。除非您观察到与精度相关的收敛问题，否则请保留默认的 `"medium"` 模式。

> **Note:** 进入或退出 `torch.tpu.precision` 会导致 `torch.compile` 图中断。使用 `torch.compile(..., fullgraph=True)` 时，请在编译函数外部使用 `torch.set_float32_matmul_precision(...)` 设置精度，或者将外层调用封装在 `with
> torch.tpu.precision(...):` 中。

使用 `torch.tpu.precision` 比较三种模式下的稳定状态执行时间：

```python
import time
import torch

device = torch.device("tpu")
precision = torch.tpu.precision  # Context manager
Precision = torch.tpu.Precision  # Enum class

print(f"Default matmul precision: {torch.get_float32_matmul_precision()}")

size = (8, 2048, 2048)
a = torch.randn(size, dtype=torch.float32, device=device)
b = torch.randn(size, dtype=torch.float32, device=device)

def run_matmul():
  res = torch.matmul(a, b)
  torch.accelerator.synchronize()
  return res

for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
  with precision(mode):
    # Warm up to compile the graph before timing steady-state execution
    run_matmul()
    start = time.perf_counter()
    run_matmul()
    print(f"{mode} time: {time.perf_counter() - start:.4f}s")
```

### 比较各精度模式的准确性 { #compare-accuracy-across-precision-modes }

在相同输入上将 TPU 输出与 CPU `float32` 参考结果进行比较，查看每种精度模式如何减少数值误差：

```python
# CPU float32 reference result
c_ref = torch.matmul(a.cpu(), b.cpu())

for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
  with precision(mode):
    res = torch.matmul(a, b)
    diff = res.cpu() - c_ref
    print(f"Mode: {mode}")
    print(f"  Max Absolute Error: {diff.abs().max():.6e}")
    print(
        "  Mean Relative Error: "
        f" {(diff.abs() / (c_ref.abs() + 1e-8)).mean():.6e}\n"
    )
```

## 验证数值一致性 { #verify-numerical-parity }

将模型移植到 TPU 或调试收敛差异时，请在 CPU 和 TPU 上使用同步的权重和相同的输入运行同一个层或子模块，然后使用 `torch.testing.assert_close` 比较输出，以验证数值一致性：

```python
import torch.nn as nn

# 1. Set up identical models in bfloat16 for a fair comparison
model_cpu = nn.Linear(10, 10).cpu().to(torch.bfloat16)
model_tpu = nn.Linear(10, 10).to(device).to(torch.bfloat16)

# 2. Synchronize weights (mandatory for parity checks)
model_tpu.load_state_dict(model_cpu.state_dict())

# 3. Create identical input data for both devices
input_data = torch.randn(5, 10, dtype=torch.bfloat16)
input_tpu = input_data.to(device)

# 4. Run the forward pass
out_cpu = model_cpu(input_data)
out_tpu = model_tpu(input_tpu)

# 5. Compare results
# Note: TorchTPU's bfloat16 accumulation and fusion differences can exceed
# default tolerances, generating the `assert_close` diagnostic report below.
try:
  torch.testing.assert_close(out_tpu.cpu(), out_cpu)
  print("Parity check passed!")
except AssertionError as e:
  print("Parity check failed (expected for hardware precision drift):\n")
  print(e)
```

### 解读详细的不匹配信息 { #interpret-detailed-mismatches }

当张量超出容差时，`torch.testing.assert_close` 会输出一份摘要，帮助您区分正常的浮点漂移和逻辑错误：

```text
Tensor-likes are not close!

Mismatched elements: 1 / 50 (2.0%)
Greatest absolute difference: 0.006596 at index (3, 9) (up to 1e-05 allowed)
Greatest relative difference: 0.165375 at index (3, 9) (up to 0.016 allowed)
```

*   **Mismatched elements**：如果有很高比例的元素存在较小的绝对差异，这通常表明是标准的舍入差异，或编译器融合导致的累加顺序差异，而不是错误。
*   **Greatest absolute / relative difference**：报告最大误差及其张量索引，以便您检查离群值或选择合适的 `atol` 和 `rtol` 阈值。

### 一致性最佳实践 { #best-practices-for-parity }

*   **在两种设备上使用相同的数据类型**：将 TPU 上的 `bfloat16` 与 CPU 上的 `bfloat16` 进行比较，或将 TPU 上的 `float32`（使用 `Precision.HIGHEST`）与 CPU 上的 `float32` 进行比较。将 `bfloat16` 与 `float32` 进行比较，衡量的是量化误差，而不是硬件漂移。
*   **同时检查梯度和激活值**：调用 `loss.backward()` 后，使用相同的模式将 `model_tpu.weight.grad.cpu()` 与 `model_cpu.weight.grad` 进行比较。
*   **从标准容差开始**：以 [PyTorch 测试文档](https://docs.pytorch.org/docs/stable/testing.html)中的默认容差为基准，仅在降低精度累加的预期范围内放宽 `atol` 或 `rtol`。

## 指定输出精度 { #specify-output-precision }

使用 `torch.mm` 或 `torch.bmm` 对 `bfloat16` 张量进行乘法运算时，MXU 会先以 `float32` 累加乘积，然后再将最终输出舍入回 `bfloat16`。传入 `out_dtype=torch.float32` 可直接返回 `float32` 累加器的结果，而不进行向下转换：

```python
a_bmm = torch.randn((2, 256, 256), dtype=torch.bfloat16, device=device)
# Using out_dtype to force float32 output accumulator precision
res_bmm = torch.bmm(a_bmm, a_bmm, out_dtype=torch.float32)
print(f"Input dtype: {a_bmm.dtype}")
print(f"Output dtype: {res_bmm.dtype}")
```

## 控制超额精度 { #control-excess-precision }

由于 TPU MXU 以 `float32` 累加 `bfloat16` 矩阵乘法，XLA 编译器可以通过将中间值保留为 `float32`（而不是在各步骤之间将其向下舍入为 `bfloat16`）来优化融合操作。保留这些额外的中间位称为*超额精度*（excess precision）。

您可以使用 `torch.tpu.allow_excess_precision` 全局配置此行为（它会设置底层的 `--xla_allow_excess_precision` 编译器标志，默认值为 `False`）：

*   **停用超额精度（`False`，默认）**：设置 `torch.tpu.allow_excess_precision = False`，强制编译器保留到 `bfloat16` 的中间向下转换，使融合操作与 eager 模式下的 `bfloat16` 舍入行为一致。
*   **允许超额精度（`True`）**：如需允许编译器省略中间向下转换，并将融合的中间结果保留为 `float32`，请设置 `torch.tpu.allow_excess_precision =
    True`。

> **Important:** 请在脚本的最开始、运行或编译任何 TPU 操作之前设置 `torch.tpu.allow_excess_precision`。XLA 运行时会在第一次编译开始时锁定此标志；之后尝试将其更改为其他值会引发 `RuntimeError`。

## 了解 TPU MXU 精度 { #understand-tpu-mxu-precision }

了解 TPU MXU 如何处理浮点数，就能理解超额精度和多遍 `float32` 模式为什么会以这样的方式工作：

*   **范围与精度（`bfloat16`）**：[Google Cloud 关于 `bfloat16` 的研究](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)表明，只要保留 `float32` 的宽动态范围（这对范数和梯度等运算至关重要），神经网络就可以在降低尾数精度的情况下进行训练，而不会损失模型准确性。`float32` 有 23 个尾数位（包括隐含的前导位在内，有效数精度为 24 位），而 `bfloat16` 保留了相同的 8 位指数范围，但只有 7 个尾数位（有效数精度为 8 位，机器 epsilon 为 `2**-7`，约 0.78%）。
*   **交换格式与算术格式**：浮点系统区分**交换格式**（数字在内存中的存储方式）和**算术格式**（硬件如何计算中间运算并进行舍入，例如标准 ALU 中额外的 3 个保护位、舍入位和粘滞位）。`bfloat16` 可以指其中任意一种。
*   **沿收缩维度累加**：将形状为 `[I, J] @ [J, K]` 的两个矩阵相乘，会计算一个由形状为 `[1,
    J] @ [J, 1]` 的内积组成的网格，每个内积都沿**收缩维度**
    `J` 对 `J` 个逐元素乘积求和。由于 `bfloat16` 的机器 epsilon 为 `2**-7`（约 `0.008`），在 `bfloat16` 中累加 `J` 个乘积会导致很大的舍入误差。因此，MXU 接受 `bfloat16` 输入（在 TPU v7x 等较新的代际上还接受 `fp8`），并以 `float32` 累加中间和。这与 ALU 中的保护位、舍入位和粘滞位在舍入为输出格式之前所起的作用相同。
*   **输入/输出格式不匹配**：`bfloat16` 输入与 `float32` 累加器的组合，既是*超额精度*（在融合操作之间保留 `float32` 中间值）的根本原因，也是*低于预期的精度*（1 遍 `float32` 矩阵乘法将输入截断为 8 位 `bfloat16` 有效数）的根本原因。相比之下，TPU 向量执行单元（VPU）会对 `float32` 的全部 23 个尾数位进行运算（次正规数等少数 IEEE-754 边缘情况除外）。

### `float32` 的多遍多项式展开 { #multi-pass-polynomial-expansion-for-float32 }

由于 MXU 只接受 `bfloat16` 输入（8 位有效数），因此无法在一遍计算中完成完整的 24 位 `float32` 矩阵乘法。在实践中，大多数现代模型都以 `bfloat16` 存储权重，很少执行 `float32` 矩阵乘法。当需要完整的 `float32` 矩阵乘法精度时，XLA 会使用 [Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations](https://arxiv.org/pdf/1904.06376)（Henry、Tang 和 Heinecke）中的算法，将每个 24 位 `float32` 有效数表示为三个 8 位 `bfloat16` 值之和（`High + Med + Low`），并展开它们的乘积：

```text
(a + b + c)(d + e + f) = ad + ae + af + bd + be + bf + cd + ce + cf
```

将两个三项之和相乘会得到九个 `bfloat16` 乘积，其中三个最低阶项在 `float32` 中总是会下溢：

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|          | High        | Med                      | Low                      |
| :------- | :---------- | :----------------------- | :----------------------- |
| **High** | High x High *（1 遍）* | High x Med *（3 遍）*    | High x Low *（6 遍）*    |
| **Med**  | Med x High *（3 遍）* | Med x Med *（6 遍）*     | Med x Low *（下溢）* |
| **Low**  | Low x High *（6 遍）* | Low x Med *（下溢）* | Low x Low *（下溢）* |
<!-- mdformat on -->

### 评估 3 遍和 6 遍的性能 { #evaluate-3-pass-and-6-pass-performance }

虽然 `Precision.HIGH`（3 遍）和 `Precision.HIGHEST`（6 遍）分别需要 3 倍和 6 倍的 MXU 计算遍数，但对端到端的影响取决于您的工作负载：

*   **较大的收缩维度**：在使用 `float32` 累加训练且具有较大收缩维度的现代模型（例如 DeepSeek-V3）中，数千个元素上的累加噪声和不确定性远远超过低阶尾数交叉项的贡献，因此 1 遍（`Precision.DEFAULT`）就已足够。
*   **受内存带宽限制的模型**：较旧的 `float32` 模型对从高带宽内存（HBM）传输的每个字节执行的算术运算量通常不足以让 MXU（或 GPU Tensor Core）饱和。在这些受内存限制的工作负载中，3 遍和 6 遍执行所需的额外 MXU 计算会使用原本空闲的 MXU 周期，因此端到端的减慢程度会低于 3 倍或 6 倍。

## 验证逐位精度行为 { #validate-bitwise-precision-behavior }

您可以利用以下两条二进制浮点规则构造输入，以验证 MXU 如何处理尾数位和超额精度：

*   `1 + 2**-n` 会创建一个二进制有效数为 `1.000...1000...` 的数，在第 `n` 个尾数位上放置一个小数位 `1`。
*   `(1 + m) * (1 + n) = 1 + m + n + mn`：将尾数第 6 位和第 7 位分别置位的两个数相乘，得到的结果会在第 6、7 位以及第 `6 + 7 = 13` 位上置位。由于 `bfloat16` 只有 7 个尾数位，第 13 位会超出 `bfloat16` 的表示范围，只有保留在 `float32` 累加器中时才能保留下来：

```py
value = (1 + 2**-6) * (1 + 2**-7)  # 1.0000_01 * 1.0000_001 = 1.0000_0110_0000_1
```

将这些值封装到 `256 x 256` 矩阵中，可以触发在 MXU 上的执行，并让您检查第 13 个尾数位是被舍入掉还是被保留：

```python
# Python uses float64 by default.
bit_6 = 1 + 2**-6
bit_7 = 1 + 2**-7
bits_6_7 = 1.0 + 2**-6 + 2**-7
bits_6_7_13 = 1.0 + 2**-6 + 2**-7 + 2**-13

print("=== Python CPU values ===")
print(f"{bit_6=:.20f}")
print(f"{bit_7=:.20f}")
print(f"{bits_6_7=:.20f}")
print(f"{bits_6_7_13=:.20f}")
print(f"{(bit_6 * bit_7)=:.20f}")
assert bit_6 * bit_7 == bits_6_7_13

print("=== PyTorch TPU values ===")

eye = torch.eye(256, dtype=torch.bfloat16, device=device)
bit_6_tensor = eye.clone() * bit_6
bit_7_tensor = eye.clone() * bit_7
bits_6_7_tensor = eye.float().clone() * bits_6_7
bits_6_7_13_tensor = eye.float().clone() * bits_6_7_13

# Use a 1-element vector rather than a 0D scalar because PyTorch's type
# promotion rules ignore 0D scalar tensor dtypes when paired with 2D tensors.
one_float32 = torch.tensor([1.0], dtype=torch.float32, device=device)

print(f"{bit_6_tensor[0,0]=:.20f}")
print(f"{bit_7_tensor[0,0]=:.20f}")
print(f"{bits_6_7_tensor[0,0]=:.20f}")
print(f"{bits_6_7_13_tensor[0,0]=:.20f}")

def mul_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
  return a @ b * c

# In eager execution, `a @ b` rounds to bfloat16 before `* c`, dropping bit 13.
print("=== TPU values: Eager ===")
eager_res = mul_add(bit_6_tensor, bit_7_tensor, one_float32)[0, 0]
print(f"{eager_res=:.20f}")
assert eager_res == bits_6_7_tensor[0, 0]

# With allow_excess_precision = False, the compiled graph also rounds `a @ b`
# to bfloat16 before multiplying by `c`, dropping bit 13.
print("=== TPU values: compiled with allow_excess_precision = False ===")
torch.compiler.reset()
torch.tpu.allow_excess_precision = False
mul_add_compiled_false = torch.compile(mul_add, backend="tpu")
print(f"{torch.tpu.allow_excess_precision=}")
res_false = mul_add_compiled_false(bit_6_tensor, bit_7_tensor, one_float32)[
    0, 0
]
print(f"{res_false=:.20f}")

try:
  assert res_false == bits_6_7_tensor[0, 0]
  print("  [PASSED] Successfully disabled excess precision.")
except AssertionError:
  print(
      "  [WARNING] Failed to disable excess precision (XLA compiled another"
      " graph with default 'True' earlier in this session)."
  )

# Changing allow_excess_precision after XLA compilation has already started in
# this process raises a RuntimeError because compiler options are locked on
# the first compilation.
#
# When `torch.tpu.allow_excess_precision = True` is set at the start of a
# fresh Python session before any TPU operations or compilations run, XLA
# keeps `a @ b` in float32 when fusing with `* c`, preserving the 13th
# mantissa bit (res_true == bits_6_7_13_tensor[0, 0] == 1.0235595703125).
print("=== TPU values: compiled with allow_excess_precision = True ===")
torch.compiler.reset()
try:
  torch.tpu.allow_excess_precision = True
  mul_add_compiled_true = torch.compile(mul_add, backend="tpu")
  print(f"{torch.tpu.allow_excess_precision=}")
  res_true = mul_add_compiled_true(bit_6_tensor, bit_7_tensor, one_float32)[
      0, 0
  ]
  print(f"{res_true=:.20f}")
  assert res_true == bits_6_7_13_tensor[0, 0]
  print("  [PASSED] Successfully enabled excess precision.")
except (RuntimeError, AssertionError) as e:
  print(
      "  [EXPECTED] Cannot change excess precision after first compilation:"
      f" {e}"
  )
```

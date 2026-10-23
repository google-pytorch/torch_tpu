# 动态形状 { #dynamic-shapes }

<!-- Link to colab and to generated notebook -->

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/dynamic_shapes.ipynb)

当您使用 `torch.compile(model)` 编译模型时，TorchTPU 会将计算降级为 StableHLO，并使用 XLA 编译器进行编译（请参阅 [TorchTPU 的工作原理](./how_torchtpu_works.md)）。XLA 编译器传统上是围绕**静态已知的张量形状**设计的，这样就可以提前布局高带宽内存（HBM）缓冲区、融合算子流水线，并针对 TPU 的脉动阵列（MXU）对矩阵乘法进行分块。如果在 `torch.compile` 工作负载中，输入张量的维度在多次调用之间发生变化，那么针对每个确切形状进行编译会使 Dynamo 的形状守卫（guard）失效，并在每遇到一个新形状时触发代价高昂的完整 XLA 重新编译。

为了在不频繁重新编译的情况下支持**动态形状**，TorchTPU 使用了**有界动态性（Bounded Dynamism）**：动态维度在编译时的上限是固定的，而实际的运行时维度大小由 XLA 动态跟踪。

## TorchTPU 如何处理动态形状 { #how-dynamic-shapes-are-handled-by-torchtpu }

当您使用默认动态性（即 `torch.compile(model, dynamic=None)`）编译模型或函数（输入形状在多次调用之间发生变化），或者使用 `torch._dynamo.mark_dynamic` 显式标记动态维度时，Dynamo 会跟踪计算并生成一个 FX 图，其中变化的维度表示为符号整数变量（`torch.SymInt`），而不是固定常量。TorchTPU 按如下方式处理这些符号维度：

### 编译 { #compilation }

*   TorchTPU 拦截跟踪得到的 FX 图，为每个符号维度确定一个总体的 `[lower_bound, upper_bound]` 范围（来自 `mark_dynamic(..., min=..., max=...)`，或由 Dynamo 根据初始运行时大小 `S` 推断），并将该范围划分为若干个连续的桶（bucket），每个桶与 `bucket_size` 的倍数对齐（默认为 `128`，与 TPU 的 128 通道 tile 维度一致），桶的数量最多为 `num_dynamic_buckets`（默认为 `1`，最大为 `8`）：
    *   第一个桶覆盖从 `lower_bound` 到 `S` 处或之后的下一个 `128` 的倍数（`U_1`），第二个桶覆盖从 `U_1 + 1` 到下一个 `128` 的倍数（`U_2`），依此类推（例如 `[lower_bound, 128]`、`[129, 256]`……）。
    *   TorchTPU 会向 Dynamo 注册一个覆盖所有桶完整跨度的范围守卫，因此只有当维度增长到超过总体 `upper_bound` 时，Dynamo 才会触发重新编译。
*   对于每个桶，TorchTPU 会将符号维度替换为该桶的静态上限 `U`，将实际的运行时维度大小作为输入传入，并将图降级为带有有界张量类型的 StableHLO，这些类型在所有操作中同时跟踪静态上限 `U` 和当前的运行时大小。
*   XLA 会针对每个已配置的桶上限 `U` 编译生成的模块。

### 执行 { #execution }

*   在运行时，TorchTPU 会根据动态维度的实际大小选择相应的已编译桶可执行文件。

*   执行已编译的模块时，XLA 会将动态输入张量填充到桶的静态上限 `U`，并将超出当前运行时大小的填充区域中的值重置为代数单位元值，以确保填充元素不会影响输出结果。

*   对于动态输出张量，填充保留在缓冲区的末尾，这样 TorchTPU 只需轻量地更新形状元数据即可将缓冲区返回给 PyTorch，而无需对张量进行物理切片。

## 典型用例：自回归解码 { #motivating-use-case-autoregressive-decoding }

促使 TorchTPU 引入有界动态性的一个主要工作负载是使用动态增长的 KV 缓存进行**自回归 LLM 解码**。在标准的解码循环中，每一步都会通过 `torch.cat` 沿序列维度将新 token 的 key 和 value 状态追加到缓存的历史记录中：

*   **预填充（prefill）**：输入提示的形状为 `[B, S]`（批次大小 `B`，提示序列长度 `S`），生成形状为 `[B, H_kv, S, D]` 的 KV 缓存（`H_kv` 个 key/value 头，头维度 `D`）。
*   **解码第 1 步**：输入 token 的形状为 `[B, 1]`，与缓存拼接后得到形状 `[B, H_kv, S + 1, D]`。
*   **解码第 `t` 步**：KV 缓存的序列长度增长到 `S + t`。

使用静态形状编译（`dynamic=False`）时，Dynamo 会对 KV 缓存的确切序列长度设置守卫。每个解码步骤都会改变张量形状，使守卫失效，并**在生成每个 token 时触发完整的 XLA 重新编译**。除非您重写模型以预分配静态缓冲区并执行原地修改，否则原本毫秒级的解码步骤会变成数秒的编译。

**有界动态性**无需重写模型即可解决这个问题：随着 KV 缓存在各个解码步骤中增长（`S + 1`、`S + 2`……），已编译桶（`<= 128`、`<= 256`……）内的所有形状都会复用同一个 XLA 可执行文件。

如需了解将有界动态性应用于完整 LLM 解码循环（并与静态 KV 缓存预分配进行比较）的实践演示，请参阅[端到端 LLM 推理](./case_studies_muse_glimmer_30b_inference.md)。

## 示例：张量不断增长的自回归循环 { #example-autoregressive-loop-with-a-growing-tensor }

为了解静态编译（`dynamic=False`）和使用默认动态性（`dynamic=None`）的有界动态性在实践中的表现，我们来看一个最小的自回归循环，其中每一步都使用 `torch.cat` 沿序列维度（`dim=2`）追加一个新 token：

```python
import time
import torch
from torch._functorch._aot_autograd import autograd_cache

torch._logging.set_logs()  # pylint: disable=protected-access
device = torch.device("tpu")

def decode_step(
    kv_cache: torch.Tensor, new_token: torch.Tensor
) -> torch.Tensor:
  return torch.cat([kv_cache, new_token], dim=2)

def run_decode_loop(
    compiled_step, num_steps: int = 10, prefix: str = ""
) -> None:
  kv_cache = torch.ones((1, 8, 16, 128), dtype=torch.bfloat16, device=device)
  new_token = torch.ones((1, 8, 1, 128), dtype=torch.bfloat16, device=device)
  torch.accelerator.synchronize()

  for step in range(num_steps):
    start = time.perf_counter()
    kv_cache = compiled_step(kv_cache, new_token)
    torch.accelerator.synchronize()
    step_ms = (time.perf_counter() - start) * 1000
    print(
        f"{prefix}Step {step + 1}/{num_steps} (seq_len={kv_cache.shape[2]}):"
        f" {step_ms:.2f} ms",
        flush=True,
    )
```

### 1. 使用 `dynamic=False` 运行 { #1-running-with-dynamicfalse }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()
torch._dynamo.config.recompile_limit = 64

step_static = torch.compile(decode_step, backend="tpu", dynamic=False)
run_decode_loop(step_static, prefix="[dynamic=False] ")
```

示例输出：

```text
[dynamic=False] Step 1/10 (seq_len=17): 49.54 ms
[dynamic=False] Step 2/10 (seq_len=18): 43.51 ms
[dynamic=False] Step 3/10 (seq_len=19): 39.52 ms
[dynamic=False] Step 4/10 (seq_len=20): 38.63 ms
[dynamic=False] Step 5/10 (seq_len=21): 38.40 ms
[dynamic=False] Step 6/10 (seq_len=22): 38.23 ms
[dynamic=False] Step 7/10 (seq_len=23): 38.11 ms
[dynamic=False] Step 8/10 (seq_len=24): 38.55 ms
[dynamic=False] Step 9/10 (seq_len=25): 37.89 ms
[dynamic=False] Step 10/10 (seq_len=26): 37.75 ms
```

**发生了什么**：使用 `dynamic=False` 时，Dynamo 会对 `kv_cache` 的确切序列长度（`16`、`17`、`18`……）设置守卫。由于 `torch.cat` 在每次迭代中都会将 `dim=2` 增加 `1`，形状守卫在每次调用时都会失败，并在每一步触发完整的 XLA 重新编译。

### 2. 使用 `dynamic=None` 运行（有界动态性） { #2-running-with-dynamicnone-bounded-dynamism }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()

step_dynamic = torch.compile(decode_step, backend="tpu", dynamic=None)
run_decode_loop(step_dynamic, prefix="[dynamic=None] ")
```

示例输出：

```text
[dynamic=None] Step 1/10 (seq_len=17): 49.59 ms
[dynamic=None] Step 2/10 (seq_len=18): 206.68 ms
[dynamic=None] Step 3/10 (seq_len=19): 64.29 ms
[dynamic=None] Step 4/10 (seq_len=20): 0.51 ms
[dynamic=None] Step 5/10 (seq_len=21): 0.43 ms
[dynamic=None] Step 6/10 (seq_len=22): 0.38 ms
[dynamic=None] Step 7/10 (seq_len=23): 0.40 ms
[dynamic=None] Step 8/10 (seq_len=24): 0.44 ms
[dynamic=None] Step 9/10 (seq_len=25): 0.45 ms
[dynamic=None] Step 10/10 (seq_len=26): 0.37 ms
```

**发生了什么**：使用 `dynamic=None` 时，只有前 3 次迭代会产生编译开销，从第 4 步开始，每一步的执行时间降至约 0.4 ms（快约 100 倍），且没有任何重新编译：

*   **第 1 步（`seq_len=16 -> 17`）**：Dynamo 首次看到输入，并为 `seq_len=16` 编译一个静态可执行文件。
*   **第 2 步（`seq_len=17 -> 18`）**：由于 `dim=2` 从 `16` 变为 `17`，Dynamo 会自动将 `dim=2` 标记为动态（`SymInt`），TorchTPU 则使用初始输入张量布局，为 `[17, 128]` 桶编译一个有界动态可执行文件。
*   **第 3 步（`seq_len=18 -> 19`）**：第 2 步生成的有界动态输出缓冲区采用 XLA 的填充输出布局，因此 TorchTPU 会针对这种稳定状态下的 XLA 布局再次编译 `[17,
    128]` 桶。
*   **第 4–10 步（`seq_len=19 -> 26`）**：序列长度（在 `<= 128` 桶内）和 XLA 缓冲区布局都与缓存的可执行文件匹配，因此后续每一步都会直接分派预编译的 XLA 可执行文件，无需重新编译。

> **Tip:** 如果您事先知道哪个维度会发生变化，可以在第一步之前调用 `torch._dynamo.mark_dynamic(kv_cache, 2, min=16, max=128)`。这会让 Dynamo 在第 1 步就立即将 `dim=2` 作为动态维度进行跟踪，从而跳过第 1 步的初始静态编译，并允许您指定明确的 `[min, max]` 边界。

## 当前限制 { #current-limitations }

TorchTPU 中的有界动态性正在积极演进，目前主要针对**自回归 KV 缓存**用例进行了优化。在将其应用于您的模型时，请注意以下限制：

*   **桶数量限制和无界动态性**：每次跟踪最多编译 **8 个桶**（`num_dynamic_buckets <= 8`）。目前尚不支持无界动态性：如果动态维度增长超过已编译的最高桶的上限，就会触发重新编译。
*   **Dynamo 形状约束**：桶上限必须满足当前生效的 Dynamo 形状约束（例如 `reshape` 或 `view` 带来的整除要求）。否则，TorchTPU 会回退为使用单个桶，其上限为 `2 * S`（初始运行时大小 `S` 的两倍）。
*   **计算和 HBM 开销取决于桶上限**：由于执行是在填充到桶静态上限 `U` 的缓冲区上运行的，因此 HBM 占用和 FLOPs 会随 `U` 而不是实际运行时大小增长。如果实际大小远小于 `U`，而您设置了非常大的 `bucket_size` 或 `mark_dynamic(..., max=U)`，就需要承担 `U` 对应的计算和内存开销。
*   **XLA 布局变化**：即使张量的形状落在已编译的桶内，只要输入缓冲区底层的 XLA 物理布局发生变化，也会触发该桶的重新编译。
*   **没有持久性编译缓存（`AOTAutogradCache`）**：有界动态可执行文件目前还不会序列化到 `AOTAutogradCache`，因此每次启动新进程时都会重新编译。
*   **算子覆盖范围**：并非所有 PyTorch 算子都已支持有界动态形状。
*   **将视图作为模型输入**：对将张量视图直接作为输入传递给已编译模型的支持有限。
*   **数据相关的动态性**：不支持输出形状取决于运行时张量值的操作（例如 `torch.nonzero`）。
*   **自定义 kernel**：有界动态形状不支持自定义 TPU kernel（例如通过 `torch.tpu.pallas.jax_op` 注册的 Pallas kernel）。
*   **dynamic=True**：不支持 torch.compile() 的 *dynamic=True* 模式。

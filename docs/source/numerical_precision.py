# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: skip-file
"""Precision Management in TorchTPU."""

import marimo


__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # Manage numerical precision in TorchTPU

    Google TPUs split floating-point operations between two hardware units to
    balance throughput and numerical accuracy:

    *   **Matrix Multiply Unit (MXU):** Executes contracting operations (such as
        `torch.matmul`, `torch.mm`, `torch.bmm`, and convolutions) using native
        `bfloat16` inputs (8-bit significand) with 32-bit `float32` accumulation.
    *   **Vector Execution Unit (VPU):** Executes non-contracting operations (such
        as element-wise arithmetic, activations, layer normalization, and softmax)
        in full IEEE-754 `float32` (24-bit significand).

    This guide covers how to configure `float32` matrix multiplication precision
    modes, verify numerical parity between CPU and TPU, control output accumulator
    types and excess precision, and validate bitwise MXU behavior in code.

    ### Compare PyTorch precision on CUDA and TPU

    TorchTPU supports standard PyTorch precision APIs
    (`torch.set_float32_matmul_precision` and `torch.get_float32_matmul_precision`)
    alongside TPU-specific controls (`torch.tpu.Precision` and
    `torch.tpu.precision`):

    | Dimension          | PyTorch on CUDA                      | PyTorch on TPU (TorchTPU)              |
    | :----------------- | :----------------------------------- | :------------------------------------- |
    | **Default precision mode** | `"highest"` (pure `float32`) | `"medium"` (`Precision.DEFAULT`)       |
    | **Fast math accelerator primitive** | Tensor Cores (`TF32`) | Systolic MXUs (`bfloat16`)             |
    | **Fast math bits** | 1 sign + 8 exponent + 10 mantissa bits (19-bit `TF32`) | 1 sign + 8 exponent + 7 mantissa bits (16-bit `bfloat16`) |
    | **Accumulation precision** | `float32` (`"highest"`, `"high"`), `TF32` (`"medium"`) | `float32` across all modes |
    | **Higher precision mechanism** | Dedicated hardware `float32`/`TF32` units | Multi-pass decomposition (3x or 6x MXU passes) |
    | **Performance impact of `high`/`highest`** | Baseline speed or modest slowdown (~1–2x) | ~3x MXU compute cost for `"high"`; ~6x for `"highest"` |
    | **Scoping API** | Global `torch.set_float32_matmul_precision` | Global <br> `torch.set_float32_matmul_precision` + scoped `torch.tpu.precision` |

    ## Specify precision modes

    When multiplying `float32` tensors on the MXU, TorchTPU maps PyTorch's precision
    levels to the `torch.tpu.Precision` enum:

    *   `"medium"` / `Precision.DEFAULT` (1-pass): **Default.** Rounds `float32`
        inputs to `bfloat16` (multiplying the top 8 bits of significand) and
        accumulates in `float32`. Recommended for nearly all training and inference
        workloads.
    *   `"high"` / `Precision.HIGH` (3-pass): Splits `float32` inputs into high and
        low `bfloat16` components and computes three cross-terms (multiplying the
        first 16 bits of significand) at roughly 3x the MXU compute cost.
    *   `"highest"` / `Precision.HIGHEST` (6-pass): Splits `float32` inputs into
        three `bfloat16` components and computes six cross-terms (multiplying all 24
        bits of significand) to approximate full IEEE-754 `float32` accuracy at
        roughly 6x the MXU compute cost.

    You can configure matrix multiplication precision in two ways:

    *   **Global configuration:** Call `torch.set_float32_matmul_precision("medium"
        | "high" | "highest")` to set the default mode for the process.
    *   **Scoped context manager:** Use `with torch.tpu.precision(mode):` to apply
        higher precision only to numerically sensitive layers or submodules.
    *   **Performance warning:** Selecting `"high"` or `"highest"` emits a one-time
        warning (`TORCH_WARN_ONCE`). Keep the default `"medium"` mode unless you
        observe precision-related convergence issues.

    > **Note:** Entering or exiting `torch.tpu.precision` introduces a
    > `torch.compile` graph break. When using `torch.compile(...,
    > fullgraph=True)`, set precision outside the compiled function with
    > `torch.set_float32_matmul_precision(...)` or wrap the outer call in `with
    > torch.tpu.precision(...):`.

    Use `torch.tpu.precision` to compare steady-state execution time across the
    three modes:
    """)
  return


@app.cell
def init():
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

  for _mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
    with precision(_mode):
      # Warm up to compile the graph before timing steady-state execution
      run_matmul()
      start = time.perf_counter()
      run_matmul()
      print(f"{_mode} time: {time.perf_counter() - start:.4f}s")
  return Precision, a, b, device, precision, torch


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### Compare accuracy across precision modes

    Compare the TPU outputs against a CPU `float32` reference on the same inputs to
    see how each precision mode reduces numerical error:
    """)
  return


@app.cell
def _(Precision, a, b, precision, torch):
  # CPU float32 reference result
  c_ref = torch.matmul(a.cpu(), b.cpu())

  for _mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
    with precision(_mode):
      res = torch.matmul(a, b)
      diff = res.cpu() - c_ref
      print(f"Mode: {_mode}")
      print(f"  Max Absolute Error: {diff.abs().max():.6e}")
      print(
          "  Mean Relative Error: "
          f" {(diff.abs() / (c_ref.abs() + 1e-8)).mean():.6e}\n"
      )
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## Verify numerical parity

    When porting a model to TPU or debugging convergence differences, verify
    numerical parity by running the same layer or submodule on CPU and TPU with
    synchronized weights and identical inputs, then comparing the outputs with
    `torch.testing.assert_close`:
    """)
  return


@app.cell
def numerical_parity_demo(device, torch):
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
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### Interpret detailed mismatches

    When tensors exceed tolerances, `torch.testing.assert_close` prints a summary
    that helps distinguish normal floating-point drift from a logic bug:

    ```text
    Tensor-likes are not close!

    Mismatched elements: 1 / 50 (2.0%)
    Greatest absolute difference: 0.006596 at index (3, 9) (up to 1e-05 allowed)
    Greatest relative difference: 0.165375 at index (3, 9) (up to 0.016 allowed)
    ```

    *   **Mismatched elements:** A high percentage of elements with small absolute
        differences typically indicates standard rounding or accumulation-order
        differences from compiler fusions rather than a bug.
    *   **Greatest absolute / relative difference:** Reports the largest error and
        its tensor index so you can inspect outliers or choose appropriate `atol`
        and `rtol` thresholds.

    ### Best practices for parity

    *   **Match data types on both devices:** Compare `bfloat16` on TPU against
        `bfloat16` on CPU, or `float32` on TPU (with `Precision.HIGHEST`) against
        `float32` on CPU. Comparing `bfloat16` against `float32` measures
        quantization error rather than hardware drift.
    *   **Check gradients as well as activations:** After calling `loss.backward()`,
        compare `model_tpu.weight.grad.cpu()` against `model_cpu.weight.grad` using
        the same pattern.
    *   **Start from standard tolerances:** Use the default tolerances in the
        [PyTorch testing documentation](https://docs.pytorch.org/docs/stable/testing.html)
        as a baseline, and relax `atol` or `rtol` only where expected for
        reduced-precision accumulation.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## Specify output precision

    When multiplying `bfloat16` tensors with `torch.mm` or `torch.bmm`, the MXU
    accumulates products in `float32` before rounding the final output back to
    `bfloat16`. Pass `out_dtype=torch.float32` to return the `float32` accumulator
    directly without downcasting:
    """)
  return


@app.cell
def output_precision_demo(device, torch):
  a_bmm = torch.randn((2, 256, 256), dtype=torch.bfloat16, device=device)
  # Using out_dtype to force float32 output accumulator precision
  res_bmm = torch.bmm(a_bmm, a_bmm, out_dtype=torch.float32)
  print(f"Input dtype: {a_bmm.dtype}")
  print(f"Output dtype: {res_bmm.dtype}")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## Control excess precision

    Because the TPU MXU accumulates `bfloat16` matrix multiplications in `float32`,
    the XLA compiler can optimize fused operations by keeping intermediate values in
    `float32` instead of rounding them down to `bfloat16` between steps. Retaining
    these extra intermediate bits is called *excess precision*.

    You can configure this behavior globally with `torch.tpu.allow_excess_precision`
    (which sets the underlying `--xla_allow_excess_precision` compiler flag and
    defaults to `False`):

    *   **Disable excess precision (`False`, default):** Set
        `torch.tpu.allow_excess_precision = False` to force the compiler to retain
        intermediate downcasts to `bfloat16` so fused operations match eager
        `bfloat16` rounding behavior.
    *   **Allow excess precision (`True`):** Set `torch.tpu.allow_excess_precision =
        True` to allow the compiler to elide intermediate downcasts and keep fused
        intermediate results in `float32`.

    > **Important:** Set `torch.tpu.allow_excess_precision` at the very
    > beginning of your script before running or compiling any TPU operations.
    > The XLA runtime locks this flag when the first compilation starts;
    > attempting to change it to a different value afterward raises a
    > `RuntimeError`.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## Understand TPU MXU precision

    Understanding how the TPU MXU handles floating-point numbers explains why excess
    precision and multi-pass `float32` modes work the way they do:

    *   **Range vs. precision (`bfloat16`):**
        [Google Cloud research on `bfloat16`](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
        showed that neural networks can train with reduced mantissa precision
        without losing model accuracy, provided they keep the wide dynamic range of
        `float32` (critical for operations like norms and gradients). While
        `float32` has 23 mantissa bits (24 bits of significand precision including
        the implicit leading bit), `bfloat16` keeps the same 8-bit exponent range
        with only 7 mantissa bits (8 bits of significand precision, for a machine
        epsilon of `2**-7` or ~0.78%).
    *   **Interchange formats vs. arithmetic formats:** Floating-point systems
        distinguish between **interchange formats** (how numbers are stored in
        memory) and **arithmetic formats** (how hardware computes and rounds
        intermediate operations, such as the 3 extra guard, round, and sticky bits
        in a standard ALU). `bfloat16` can refer to either.
    *   **Accumulation along the contracting dimension:** Multiplying two matrices
        of shape `[I, J] @ [J, K]` computes a grid of inner products of shape `[1,
        J] @ [J, 1]`, each summing `J` element-wise products along the **contracting
        dimension** `J`. Because `bfloat16` has a machine epsilon of `2**-7`
        (~`0.008`), accumulating `J` products in `bfloat16` would cause large
        rounding errors. Instead, the MXU accepts `bfloat16` inputs (and `fp8` on
        newer generations such as TPU v7x) and accumulates intermediate sums in
        `float32`—playing the same role as guard, round, and sticky bits in an ALU
        before rounding to the output format.
    *   **Input/output format mismatch:** Having `bfloat16` inputs and a `float32`
        accumulator is the root cause of both *excess precision* (retaining
        `float32` intermediates across fused ops) and *less-than-expected precision*
        (1-pass `float32` matmul truncating inputs to 8-bit `bfloat16`
        significands). By contrast, the TPU Vector Execution Unit (VPU) operates on
        all 23 mantissa bits for `float32` (except for minor IEEE-754 edge cases
        such as subnormals).

    ### Multi-pass polynomial expansion for `float32`

    Because an MXU only accepts `bfloat16` inputs (8-bit significand), it cannot
    compute a full 24-bit `float32` matrix multiplication in a single pass. In
    practice, most modern models store weights in `bfloat16` and rarely execute
    `float32` matmuls. When full `float32` matmul precision is needed, XLA uses the
    algorithm from
    [Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations](https://arxiv.org/pdf/1904.06376)
    (Henry, Tang, and Heinecke) to represent each 24-bit `float32` significand as a
    sum of three 8-bit `bfloat16` values (`High + Med + Low`) and expand their
    product:

    ```text
    (a + b + c)(d + e + f) = ad + ae + af + bd + be + bf + cd + ce + cf
    ```

    Multiplying two 3-term sums yields nine `bfloat16` products, of which the three
    lowest-order terms always underflow in `float32`:

    |          | High        | Med                      | Low                      |
    | :------- | :---------- | :----------------------- | :----------------------- |
    | **High** | High x High *(1-pass)* | High x Med *(3-pass)*    | High x Low *(6-pass)*    |
    | **Med**  | Med x High *(3-pass)* | Med x Med *(6-pass)*     | Med x Low *(underflows)* |
    | **Low**  | Low x High *(6-pass)* | Low x Med *(underflows)* | Low x Low *(underflows)* |

    ### Evaluate 3-pass and 6-pass performance

    Although `Precision.HIGH` (3-pass) and `Precision.HIGHEST` (6-pass) require 3x
    and 6x as many MXU passes, the end-to-end impact depends on your workload:

    *   **Large contracting dimensions:** In modern models trained with `float32`
        accumulation (such as DeepSeek-V3) that have large contracting dimensions,
        accumulation noise and non-determinism across thousands of elements dominate
        the contribution from lower-order mantissa cross-terms, making 1-pass
        (`Precision.DEFAULT`) sufficient.
    *   **Memory-bandwidth-bound models:** Older `float32` models often do not
        perform enough arithmetic per byte transferred from High Bandwidth Memory
        (HBM) to saturate the MXU (or GPU Tensor Cores). In these memory-bound
        workloads, the extra MXU passes for 3-pass and 6-pass execution use
        otherwise idle MXU cycles, resulting in less than a 3x or 6x end-to-end
        slowdown.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## Validate bitwise precision behavior

    You can verify how the MXU handles mantissa bits and excess precision by
    constructing inputs using two binary floating-point rules:

    *   `1 + 2**-n` creates a number with a binary significand of `1.000...1000...`,
        placing a fractional `1` at the `n`th mantissa bit.
    *   `(1 + m) * (1 + n) = 1 + m + n + mn`: Multiplying two numbers with bits set
        at mantissa positions 6 and 7 produces a result with bits set at positions
        6, 7, and `6 + 7 = 13`. Because `bfloat16` only holds 7 mantissa bits,
        position 13 overflows `bfloat16` and is only preserved when kept in the
        `float32` accumulator:

    ```py
    value = (1 + 2**-6) * (1 + 2**-7)  # 1.0000_01 * 1.0000_001 = 1.0000_0110_0000_1
    ```

    Wrapping these values in `256 x 256` matrices triggers execution on the MXU and
    lets you inspect whether the 13th mantissa bit is rounded away or retained:
    """)
  return


@app.cell
def excess_precision_verification(device, torch):
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
  return


if __name__ == "__main__":
  app.run()

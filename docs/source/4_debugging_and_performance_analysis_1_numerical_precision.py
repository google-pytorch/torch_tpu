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


   Google TPUs accelerate matrix multiplication using a dedicated Matrix Multiply Unit (MXU) that natively multiplies `bfloat16` inputs and accumulates results in `float32`. If you are training or running inference on PyTorch models with TorchTPU, understanding this hardware behavior helps you balance execution speed against numerical accuracy. In this guide, you learn how to configure `float32` matrix multiplication precision modes, control excess precision and output accumulator data types, verify bitwise precision on TPUs, and choose precision for reinforcement learning (RL) post-training at low learning rates.


   ## Specify precision modes


   Specify the precision modes using the `torch.tpu.Precision` enum:


   * `Precision.DEFAULT` (1-pass): The default precision. This is the fastest available precision that maintains reasonable accuracy (often `bfloat16` for the compute, but `float32` for accumulation). Multiplies only the first 8 bits.
   * `Precision.HIGH` (3-pass): Uses multiple `bfloat16` passes or hardware features to approximate `float32` precision, providing higher accuracy than `Precision.DEFAULT` at the cost of slower execution. Multiplies the first 16 bits.
   * `Precision.HIGHEST` (6-pass): Uses multiple `bfloat16` passes or hardware features to approximate or compute full `float32` precision. This is the slowest, most accurate mode. Multiplies all 24 bits, achieving full `float32` precision.


   Use `torch.tpu.precision` as a context manager to compare compilation and execution time across the three modes:
   """)
  return


@app.cell
def init():
  import time
  import torch

  # Standard hardware initialization
  torch.device("tpu")

  precision = torch.tpu.precision  # Context manager
  Precision = torch.tpu.Precision  # Enum class

  def run_matmul():
    size = (8, 2048, 2048)
    a = torch.zeros(size, device="tpu")
    b = torch.zeros(size, device="tpu")
    return torch.matmul(a, b).cpu()

  for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
    with precision(mode):
      start = time.time()
      run_matmul()
      print(f"{mode} time: {time.time() - start:.4f}s")
  return Precision, precision, torch


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
   ### Compare accuracy across precision modes


   Compare the TPU results against a CPU reference to see how each precision mode affects accuracy:
   """)
  return


@app.cell
def _(Precision, precision, torch):
  size = (8, 2048, 2048)
  rand_a = torch.randn(size, device="tpu")
  rand_b = torch.randn(size, device="tpu")

  # CPU reference result
  c_ref = torch.matmul(rand_a.cpu(), rand_b.cpu())

  for precision_mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
    with precision(precision_mode):
      res = torch.matmul(rand_a, rand_b)
      diff = res.cpu() - c_ref
      print(f"Mode: {precision_mode}")
      print(f"  Max Absolute Error: {diff.abs().max():.6e}")
      print(
          "  Mean Relative Error: "
          f" {(diff.abs() / (c_ref.abs() + 1e-8)).mean():.6e}\n"
      )
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
   ## Control excess precision


   Because the TPU MXU accumulates `bfloat16` matrix multiplications in `float32`, the XLA compiler can optimize fused operations by keeping intermediate values in `float32` instead of rounding them down to `bfloat16` between steps. Retaining these extra intermediate bits is called *excess precision*.


   You can control this behavior globally using `torch.tpu.allow_excess_precision` (which sets the underlying `--xla_allow_excess_precision` compiler flag and defaults to `False`):


   * **Disable excess precision (`False`):** Set `torch.tpu.allow_excess_precision = False` to force the compiler to retain intermediate downcasts to `bfloat16`.
   * **Allow excess precision (`True`):** Set `torch.tpu.allow_excess_precision = True` to allow the compiler to elide intermediate downcasts and keep intermediate results in `float32`.


   Set `torch.tpu.allow_excess_precision` at the beginning of your script before running or compiling operations. Changing this flag mid-session forces recompilation, and the TPU XLA runtime can latch the compiler option on the first compilation in a session, ignoring subsequent changes.
   """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
   ## Specify output precision


   While `torch.tpu.allow_excess_precision` only permits the compiler to retain higher intermediate precision, the `out_dtype` argument on PyTorch ops forces the compiler to return a value with more bits of precision:
   """)
  return


@app.cell
def output_precision_demo(torch):
  a_bmm = torch.randn((2, 256, 256), dtype=torch.bfloat16, device="tpu")
  # Using out_dtype to force float32 output accumulator precision
  res_bmm = torch.bmm(a_bmm, a_bmm, out_dtype=torch.float32)
  print(f"Input dtype: {a_bmm.dtype}")
  print(f"Output dtype: {res_bmm.dtype}")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
   ## Understand TPU MXU precision


   Google TPUs accelerate matrix multiplication in hardware using a physical module called a Matrix Multiply Unit (MXU).


   To achieve maximum speed, researchers identified an inexpensive tradeoff. [Google Cloud research on bfloat16](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) showed that neural networks can train with less precision than `float32` "without having any noticeable impact on model accuracy." The same was not true for range. Due to operations like norms, keeping the range of `float32` was important. The solution was brain float 16 (`bfloat16`): the same range as `float32`, with less precision: only 7 bits of mantissa for a machine epsilon of approximately 0.78%.


   Two related but distinct concepts in floating-point math are **interchange formats** (how memory stores floating-point numbers) and **arithmetic formats** (the formats in which hardware performs operations). An example of an arithmetic format is the use of 3 extra bits beyond the significand to handle rounding accurately during multiplication, addition, and subtraction. These three extra bits are the guard, round, and sticky bits.


   The term `bfloat16` can refer to both an interchange format and an arithmetic format.


   `float32` has 23 bits of mantissa (plus the implicit leading bit, creating a significand with 24 bits of precision). `bfloat16` only has 7 bits of mantissa, leading to 8 bits of precision.


   In its most native state, an MXU accepts matrices of `dtype` `bfloat16` (newer generations like the v7x also support `fp8`). However, it does not output `bfloat16`: because the machine epsilon from 7 bits is `2**-7` or roughly `0.008`, the rounding errors per matrix multiplication along the contracting dimension would be unacceptably high if accumulation were in `bfloat16`. For that reason, the intermediate accumulation values on an MXU are `float32`. These extra bits play the same role as the guard, round, and sticky bits in an ALU: they attempt to meet the IEEE 754 standard's goal (when accumulating many values together) that arithmetic should round as if the intermediate ops were performed in infinite precision and then rounded to the output format.


   When multiplying two matrices of shape `[I, J] @ [J, K]`, a mathematically equivalent view is to treat the matrix multiplication as a two-dimensional grid of inner products of shape `[1, J] @ [J, 1]`. Each inner product is equivalent to a dot product of two vectors of length `J`. For each of these inner or dot products, the systolic array performs `J` multiplications and accumulations. Hence, `J` is the contracting dimension because the dot product sums up and contracts away that dimension.


   **However, notice that the input and output formats are different.** This mismatch is the root of two confusing elements of modern ML frameworks: excess precision and less-than-expected precision.


   MXU precision behavior does not apply to vector math units on a TPU: those operate on all 23 bits of mantissa for `float32` (though not always strictly adhering to IEEE 754 standards, for example in the case of subnormals).


   Because MXUs only accept `bfloat16` inputs (8 bits of precision) and accumulate in `float32`, an MXU cannot calculate the full 24 bits of arithmetic precision for `float32` matrix multiplications in a single pass. In practice, this is not an issue for two reasons.


   First, recent models generally hold weights in `bfloat16` format. Deep learning is robust to rounding, and 24 bits of precision is more than enough. When running inference on recent models, `float32` matrix multiplications rarely occur.


   Second, you can compute math on all 23 bits of mantissa through an algorithm described in the paper [Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations](https://arxiv.org/pdf/1904.06376) by Greg Henry, Ping Tak Peter Tang, and Alexander Heinecke. You can express a higher-precision number as a sum of numbers in a lower-precision format, and then expand the product of sums into a sum of products via polynomial expansion:


   ```text
   (a + b + c)(d + e + f) = ad + ae + af + bd + be + bf + cd + ce + cf
   ```


   This is how MXUs that natively handle `bfloat16` multiplication implement `float32` math. Because the 8 bits of a `bfloat16` significand are one-third of a `float32` significand, representing one `float32` value requires three `bfloat16` values. Multiplying two sums of three terms expands into a sum of nine products. Upon closer analysis, the three lowest-order products always underflow in `float32`, so they rarely affect the output:


   | High | Med | Low |
   | :---- | :---- | :---- |
   | High x High | High x Med | High x Low |
   | Med x High | Med x Med | Med x Low *(underflows)* |
   | Low x High | Low x Med *(underflows)* | Low x Low *(underflows)* |


   ### Evaluate 3-pass and 6-pass performance


   In theory, the 6-pass technique requires 6 times more passes on the MXU, so a program could slow down by a factor of 6. In other words, an op that used to take one second would take six seconds. Similar math applies to the 3-pass technique. However, this slowdown rarely happens in practice for two reasons.


   First, modern models trained with `float32` accumulation like DeepSeek-V3 have large contracting dimensions where the noise and non-determinism of accumulation dominates the contribution from the product of rightward mantissa bits.


   Second, older models trained with `float32` are generally not optimized for modern TPUs (or GPUs). A simplified model of a TPU is a large arithmetic engine with limited bandwidth to High Bandwidth Memory (HBM). Older models do not perform enough arithmetic with their data to saturate their MXUs (or in the case of GPUs, TensorCores). As a result, the additional MXU passes for the 3-pass and 6-pass techniques use otherwise idle MXU cycles.
   """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
   ## Validate bitwise precision behavior


   To verify the behavior, you can manipulate mantissa bits using two rules:


   * `1 + 2**-n` creates a number that looks like `1.000...1000...`, where the fractional `1` is in the `n`th position of the mantissa.
   * `(1 + m) * (1 + n) = 1 + m + n + mn`. Multiplying two numbers with bits flipped at mantissa positions 6 and 7 creates a number with mantissa positions 6, 7, and `6 + 7 = 13` flipped. This is useful to force overflow bits.


   ```py
   value = (1 + 2**-6) * (1 + 2**-7)  # 1.0000_01 * 1.0000_001 = 1.0000_0110_0000_1
   ```


   Wrapping these in matrices triggers execution on the MXU. To ensure reproducible results, you can use the following verification pattern to validate the current behavior of your environment:
   """)
  return


@app.cell
def excess_precision_verification(torch):
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

  eye = torch.eye(256, dtype=torch.bfloat16, device="tpu")
  bit_6_tensor = eye.clone() * bit_6
  bit_7_tensor = eye.clone() * bit_7
  bits_6_7_tensor = eye.float().clone() * bits_6_7
  bits_6_7_13_tensor = eye.float().clone() * bits_6_7_13

  # This must be a vector of length one rather than a
  # scalar due to PyTorch's unique dtype promotion rules.
  # PyTorch often ignores scalar dtypes during type promotion.
  one_float32 = torch.tensor([1.0], dtype=torch.float32, device="tpu")

  print(f"{bit_6_tensor[0,0]=:.20f}")
  print(f"{bit_7_tensor[0,0]=:.20f}")
  print(f"{bits_6_7_tensor[0,0]=:.20f}")
  print(f"{bits_6_7_13_tensor[0,0]=:.20f}")

  def mul_add(
      a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
  ) -> torch.Tensor:
    return a @ b * c

  # The 13th bit will not be allowed because of eager execution.
  print("=== TPU values: Eager ===")
  print(f"{mul_add(bit_6_tensor, bit_7_tensor, one_float32)[0,0]=:.20f}")
  assert (
      mul_add(bit_6_tensor, bit_7_tensor, one_float32)[0, 0]
      == bits_6_7_tensor[0, 0]
  )

  # The 13th bit will not be allowed because of the flag.
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
  # To run with allow_excess_precision = True:
  # This must be run standalone in a fresh Python session where
  # `torch.tpu.allow_excess_precision = True` is set at the very beginning
  # before any TPU operations or compilations occur. When run standalone, the
  # compiler grants the MXU permission to retain intermediate downcasts,
  # preserving the 13th mantissa bit (res_true == 1.02355957031250000000).
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


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
   ## Choose precision for RL post-training at low learning rates


   Reinforcement learning (RL) post-training, such as GRPO fine-tuning of an LLM, uses small learning rates (the results in this section use a peak of `2.5e-6`). At these learning rates, two `bfloat16` choices can stop the model from learning without an error or a NaN: storing the master weights that the optimizer updates in `bfloat16`, and drawing the sampler's random noise in `bfloat16`. Keep both in `float32`. The forward and backward passes can still run in `bfloat16`. Both effects come from `bfloat16` rounding, not from the TPU, so they apply on any hardware.


   The examples in this section are small, so they also run on a CPU. The following cell uses a TPU when TorchTPU can initialize one, and the CPU otherwise:
   """)
  return


@app.cell
def rl_demo_device(torch):
  try:
    demo_device = torch.device("tpu")
    torch.zeros(1, device=demo_device)
  except RuntimeError:
    demo_device = torch.device("cpu")
  print(f"Running the RL precision examples on: {demo_device}")
  return (demo_device,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
   ### Keep `float32` master weights


   `bfloat16` has 8 bits of precision, so the spacing between neighboring values (one unit in the last place, or ULP) near a weight `w` is between `|w|/256` and `|w|/128`. When an optimizer adds an update to a `bfloat16` weight with round-to-nearest, any update smaller than half an ULP rounds back to the old value. An AdamW update is at most about the size of the learning rate. At a learning rate of `2.5e-6`, that is less than half an ULP for every weight with `|w| > 2**-10` (about `0.001`), so most updates are lost instead of accumulating. The run doesn't diverge; it stops improving. In the Qwen3-0.6B runs summarized at the end of this section, by step 348 only about 2% of the weight elements on one device had changed their `bfloat16` value with `bfloat16` master weights, compared with 36% with `float32` master weights.


   To avoid this:


   * Keep the parameters that the optimizer updates in `float32`, and run the forward and backward passes in `bfloat16` through mixed precision, for example with FSDP2 and `torch.distributed.fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)`. Don't cast the model with `model.to(torch.bfloat16)` before you create the optimizer.
   * If you need to save optimizer memory, store the AdamW moments in `bfloat16` instead of the weights. This did not reduce accuracy in the measured runs.
   * If you must store the weights in `bfloat16`, apply each update in `float32` and round the result to `bfloat16` stochastically instead of to nearest. Stochastic rounding rounds up or down at random, with probabilities that make the result exact on average, so small updates survive in expectation.


   The following example starts from weights at a typical LLM scale (standard deviation `0.02`) and applies 300 updates of size `2.5e-6`, each with a fixed sign per weight. This is roughly what AdamW does when a weight's gradient keeps its sign. The example compares `float32` master weights with `bfloat16` weights updated with round-to-nearest and with stochastic rounding:
   """)
  return


@app.cell
def master_weight_demo(demo_device, torch):
  def stochastic_round_to_bf16(x):
    # Round float32 `x` to one of its two bfloat16 neighbors at random. The
    # chance of rounding up equals the distance from the lower neighbor in
    # ULPs, so the result equals `x` on average.
    magnitude = x.abs()
    # bfloat16 keeps 8 significant bits, so its ULP is 2**(exponent - 7).
    # Clamp the exponent at -119 so that the ULP is never smaller than 2**-126,
    # the smallest normal float32 value. Hardware that flushes subnormal values
    # to zero, as TPUs can, would turn a smaller ULP into 0, and the division
    # below would give NaN. The clamp also maps log2(0) = -inf to -119, so
    # zeros stay zero. Inputs below 2**-119 round to a multiple of 2**-126,
    # still exact on average.
    exponent = torch.floor(torch.log2(magnitude)).clamp(min=-119)
    ulp = torch.exp2(exponent - 7)
    # Fix the exponent where log2 rounds down just above a power of two.
    ulp = torch.where(magnitude >= 256 * ulp, 2 * ulp, ulp)
    # Work in units of the ULP so that no intermediate value is subnormal.
    scaled = magnitude / ulp
    lower = torch.floor(scaled)
    round_up = torch.rand_like(x) < scaled - lower
    rounded = torch.where(round_up, lower + 1, lower) * ulp
    return (torch.sign(x) * rounded).to(torch.bfloat16)

  torch.manual_seed(0)
  lr, num_steps = 2.5e-6, 300
  initial_weights = 0.02 * torch.randn(100_000, device=demo_device)
  update = lr * torch.randn_like(initial_weights).sign()

  master_fp32 = initial_weights.clone()
  master_bf16_rne = initial_weights.to(torch.bfloat16)
  master_bf16_sr = initial_weights.to(torch.bfloat16)
  for _ in range(num_steps):
    master_fp32 += update
    master_bf16_rne = (master_bf16_rne.float() + update).to(torch.bfloat16)
    master_bf16_sr = stochastic_round_to_bf16(master_bf16_sr.float() + update)

  # Compare the bfloat16 weights that a bfloat16 forward pass would see.
  start_bf16 = initial_weights.to(torch.bfloat16).float()
  intended = num_steps * update
  for label, final_bf16 in [
      ("float32 master weights", master_fp32.to(torch.bfloat16)),
      ("bfloat16, round-to-nearest", master_bf16_rne),
      ("bfloat16, stochastic rounding", master_bf16_sr),
  ]:
    moved = final_bf16.float() - start_bf16
    changed = (moved != 0).float().mean()
    applied = (moved * intended).sum() / (intended * intended).sum()
    print(
        f"{label:>29}: {changed:6.1%} of weights changed,"
        f" {applied:.2f} of the intended update applied"
    )
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
   With round-to-nearest, only the few weights smaller than about `0.001` move, and the update as a whole is almost entirely lost. `float32` master weights and stochastic rounding both apply the full update.


   ### Draw sampling noise in `float32`


   Samplers that use the Gumbel-max trick draw uniform noise `u`, turn it into Gumbel noise `-log(-log(u))`, and pick the token with the largest `logits + noise`. The exponential-race form, `argmax(probs / -log(u))`, is equivalent. To keep the logarithm finite, samplers typically clamp `u` to at most `1 - eps` for its dtype. For `bfloat16`, `eps` is `2**-7`, which caps the Gumbel noise at about 4.85, compared with about 15.9 for `float32`. Low-probability tokens are then drawn too rarely, and tokens far below the top logit are never drawn, so the sampled tokens no longer follow the model's distribution. In RL, this biases the behavior policy that generates the training data.


   The noise is `bfloat16` whenever you draw it with `torch.rand_like(logits)` and the logits are `bfloat16`. To fix this, draw the noise in `float32` regardless of the logits dtype:


   ```py
   u = torch.rand_like(logits, dtype=torch.float32)
   ```


   The following example compares the noise cap for each dtype, and counts how often a token with a probability of about `4.5e-5` is drawn in one million samples:
   """)
  return


@app.cell
def gumbel_noise_demo(demo_device, torch):
  def gumbel_noise(u):
    # Clamp u as samplers do, then transform it in float32 so that only the
    # dtype of u differs between the two runs.
    finfo = torch.finfo(u.dtype)
    u = u.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
    return -torch.log(-torch.log(u.float()))

  torch.manual_seed(0)
  # Token 1 is 10 logits below token 0.
  two_token_logits = torch.tensor([0.0, -10.0], device=demo_device)
  num_draws = 1_000_000
  expected = num_draws * torch.softmax(two_token_logits, dim=0)[1]
  print(f"Expected draws of token 1: {expected:.1f}")
  for noise_dtype in [torch.bfloat16, torch.float32]:
    ones = torch.ones(1, device=demo_device, dtype=noise_dtype)
    noise_cap = gumbel_noise(ones).item()
    u = torch.rand(num_draws, 2, device=demo_device, dtype=noise_dtype)
    tokens = torch.argmax(two_token_logits + gumbel_noise(u), dim=-1)
    print(
        f"u in {noise_dtype}: noise cap {noise_cap:.2f},"
        f" token 1 drawn {int(tokens.sum())} times"
    )
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
   With `bfloat16` noise, the noise of the two tokens can differ by at most about 9.3 (the cap of 4.85 minus the lowest possible noise, -4.47), so token 1 can't close the gap of 10 logits and is never drawn. With `float32` noise, the count matches the expected value up to sampling error.


   ### Review measured results


   The following results come from GRPO fine-tuning of Qwen3-0.6B on GSM8K on one TPU v7x slice, with AdamW at a peak learning rate of `2.5e-6`. Accuracy is held-out GSM8K pass@1 on a fixed 512-problem test subset, averaged over the last evaluations of each run. The results cover one model and one task, so treat them as an indication of effect size, not as a guarantee for other workloads. At pretraining learning rates, the rounding argument is weaker.


   | Setting | Runs | Held-out accuracy (%) | Notes |
   | :---- | :---- | :---- | :---- |
   | `float32` master weights and sampling noise, 348 steps | 5 | 84.24 | Reference |
   | `bfloat16` master weights, round-to-nearest | 10 | 70.43 | 10/10 runs stalled |
   | `float32` master weights, `bfloat16` AdamW moments | 3 | 84.37 | No loss |
   | `bfloat16` master weights, stochastic rounding | 3 | 83.54 | Substitutes for `float32` master weights |
   | `bfloat16` master weights, 6x or 8x learning rate | 4 each | 83.38 (6x), 83.85 (8x) | Partial recovery |
   | `float32` master weights and sampling noise, 696 steps | 16 | 85.25 | |


   Rows without a step count ran 348 steps. None of the stalled runs diverged. Within one software build, `bfloat16` master weights cost 13.9 points (95% confidence interval 13.3 to 14.6 points, 7 runs against 5). With the sampling noise drawn in `bfloat16`, 35 of 35 runs collapsed, across all the learning rates and PPO ratio settings tried. With only the noise moved to `float32`, 0 of 21 runs collapsed.
   """)
  return


if __name__ == "__main__":
  app.run()

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
    # Precision Management

    # **APIs and Usage**

    ## **Excess Precision**

    When running PyTorch on TPUs with the TorchTPU backend, to force the compiler to retain intermediate down casts, which prevents excess precision, use:

    ```py
    torch.backends.tpu.allow_excess_precision = False
    ```

    To give the compiler the option, but not the obligation, to remove intermediate downcasts, possibly resulting in excess precision, use:

    ```py
    torch.backends.tpu.allow_excess_precision = True
    ```
    """)
  return


@app.cell
def init():
  import torch

  # Standard hardware initialization
  device = torch.device("tpu")
  return (torch,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Float32 Matmul Precision**

    You can control the algorithm used by the tpu backend for float32 matrix multiplications via the `torch.tpu.precision` context manager. Rather than a global flag, this context manager allows fine-grained control of matmul precisions.

    ```py
    with torch.tpu.precision(torch.tpu.Precision.HIGHEST):
      c = a @ b # float32 mat mul operation on all 23 bits of mantissa
    ```

    In TPU backends, these correspond to three modes of operation:

    * **DEFAULT (1-pass):** Multiplies only the first 8 bits
    * **HIGH (3-pass):** Multiplies the first 16 bits
    * **HIGHEST (6-pass):** Multiplies all 24 bits, achieving full float32 precision.
    """)
  return


@app.cell
def matmul_precision_demo(torch):
  # Example float32 tensors
  a_matmul = torch.randn((128, 128), dtype=torch.float32, device="tpu")
  b_matmul = torch.randn((128, 128), dtype=torch.float32, device="tpu")

  with torch.tpu.precision(torch.tpu.Precision.HIGHEST):
    c_matmul = (
        a_matmul @ b_matmul
    )  # float32 mat mul operation on all 23 bits of mantissa
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Output Precision**

    Unlike the excess precision flag, which grants the compiler the right but not the obligation to allow more precision in the outputs than the inputs, the out\_dtype arg on torch ops forces the compiler to return a value with more bits of precision.

    ```py
    a = torch.randn((256,256), dtype=torch.bfloat16)
    torch.bmm(a, a, out_dtype=torch.float32)
    ```
    """)
  return


@app.cell
def output_precision_demo(torch):
  a_bmm = torch.randn((2, 256, 256), dtype=torch.bfloat16, device="tpu")
  # Using out_dtype to force float32 output accumulator precision
  res_bmm = torch.bmm(a_bmm, a_bmm, out_dtype=torch.float32)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # **In-Depth Explanation of the TPU MXU**

    Google TPUs are built with matrix multiplication optimized on silicon in a physical module called a Matrix Multiply Unit or MXU.

    To achieve maximum speed, researchers identified an inexpensive tradeoff. [The research](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) showed that neural networks were able to train with less precision than float32  “without having any noticeable impact on model accuracy”. The same was not true for range. Due to operations like norms, float32’s range was important to keep. The solution was brain float 16 (bfloat16): the same range as float32, with less precision: only seven bits of mantissa for a machine epsilon of approximately 0.78%.

    Two related but distinct concepts in floating point math are **interchange formats** (how floating point numbers are stored) and **arithmetic formats** (the formats in which operations are performed). An example of an arithmetic format is the use of 3 extra bits beyond the significand to handle rounding accurately during multiplication, addition, and subtraction. These three extra bits are known as the guard, round, and sticky bits.

    Confusingly, bfloat16 can refer to both interchange format and arithmetic format.

    Float32 has 23 bits of mantissa (plus the implicit leading bit, creating a significand with 24 bits of precision). Bfloat16 only has 7 bits of mantissa, leading to 8 bits of precision.

    In its most native state, an MXU accepts matrices of dtype bfloat16 (newer generations like the v7x also support fp8). However, they do not output bfloat16: because the machine epsilon from 7 bits is literally 2\*\*-7 or roughly 0.008, the rounding errors per matrix multiplication along the contracting dimension would be unacceptably high if accumulation were in bfloat16. For that reason, the intermediate accumulation values on an MXU are actually float32. These extra bits play the same role as the guard, round, and sticky bits in an ALU: they are there to attempt to live up to the IEEE 754 standard’s goal (in the case of accumulating many values together): arithmetic should round as if the intermediate ops were performed in infinite precision and then rounded to the output format.

    Note on contracting dimensions: When multiplying two matrices of shape [I,J] @ [J,K], a mathematically equivalent and in this case useful equivalence is to view the matrix multiplication as two dimensional grid of inner products of shape [1,J] @ [J,1]. And of course, the inner product is equivalent to a dot product of two vectors of length J. For each of these inner / dot products, the systolic array will perform J multiplications and accumulation. Hence, J is the contracting dimension since the dimension gets summed up and contracted away by the inner / dot product.

    **However, notice that the input and output formats are different.** This mismatch is the root of two confusing elements of modern ML frameworks: excess precision and less-than-expected precision.

    Note that MXU precision issues do not apply to vector math units on a TPU: those do operate on full 23 bits of mantissa for float32 (though not always strictly adhering to IEEE 754 standards for example in the case of subnormals).

    Reiterating the on-silicon mixed precision behavior of the TPU MXU, MXUs only accept bfloat16 inputs (8 bits of precision) but accumulate in float32. When attempting to do matrix multiplications of float32 numbers, it is not possible for an MXU to calculate the full 24 bits of arithmetic precision (at least in one pass). In practice, this is not an issue for a couple of reasons.

    First, models in the past few years generally hold weights in bfloat16 format. This goes back to the original research: deep learning is very robust to rounding, and 24 bits of precision is more than enough. So for inferencing recent models, float32 matrix multiplications simply do not happen.

    Finally, it is actually possible to do math on the full 23 bits of mantissa through an algorithm described in the precisely-named paper [Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations](https://arxiv.org/pdf/1904.06376) by Greg Henry, Ping Tak Peter Tang, and  Alexander Heinecke. The intuition is that the higher precision number can be expressed as a sum of numbers in the lower precision format. Then, the product of sums can be converted to a sum of products via polynomial expansion.

    (a \+ b \+ c)  (d \+ e \+ f) \= ad \+ ae \+ af \+ bd \+ be \+ bf \+ cd \+ ce \+ cf

    This is exactly how MXUs that can only handle bfloat16 multiplication can implement float32 math. Because the 8 bits of a bfloat16’s significand are a third of a fp32’s, 3 bfloat16s are needed. A multiplication of a sum of triples is converted to a sum of nine elements. Upon closer analysis, the cells in red will always underflow in fp32, so will rarely affect the output.

    | High x High | High x Med | High x Low |
    | :---- | :---- | :---- |
    | Med x High | Med x Med | Med x low |
    | Low x High | Low x Med | Low x Low |

    ## **Performance considerations of 3-pass and 6-pass**

    In theory, the 6-pass technique will require 6 times more passes on the MXU, so a program could slow down by a factor of 6\. In other words, an op that used to take one second will now take six seconds. Similar math applies to the 3 pass technique. But this does not happen in practice for a few reasons.

    First, modern models trained with float32 accumulation like DeepSeek v3 have large contracting dimensions where the noise and non-determinism of accumulation dominates the contribution from the product of rightward mantissa bits.

    Second, older models trained with float32 are generally not optimized for modern TPUs (or GPUs). A simplified model of a TPU is a big arithmetic engine and limited bandwidth to memory (HBM). Older models simply don’t do enough arithmetic with their data to saturate their MXUs (or in the case of GPUs, TensorCores). So the additional MXU passes for the 3-pass and 6-pass technique are simply making use of otherwise idle MXU time.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # **Advanced Techniques: validating bitwise behavior easily**

    To verify the behavior, recall that mantissa bits are easily manipulated by two rules:

    1. `1 + 2**-n` creates a number that looks like 1.000...1000... where the fractional 1 is in the nth position of the mantissa
    2. `(1+m) * (1+n) = 1 + m + n + mn`. In other words, multiplying two numbers with bits flipped at mantissa positions 6 and 7 creates a number with mantissa positions  6, 7 and 5+6==13 flipped. This is useful to force overflow bits.

    ```py
    value = (1  + 2**-6) @ (1 + 2**-7) # 1.0000_01 * 1.0000_001 = 1.0000_0110_0000_1
    ```

    Wrapping these in matrices will trigger execution on the MXU. To ensure reproducible results, you can use the following verification pattern to validate the current behavior of your environment:
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
  # TLDR: scalar dtypes are often ignored.
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
  torch.tpu._clear_cache()
  torch.compiler.reset()
  torch.backends.tpu.allow_excess_precision = False
  mul_add_compiled_false = torch.compile(mul_add, backend="tpu")
  print(f"{torch.backends.tpu.allow_excess_precision=}")
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

  # The 13th bit might be flipped because of the flag.
  print("=== TPU values: compiled with allow_excess_precision = True ===")
  torch.tpu._clear_cache()
  torch.compiler.reset()
  torch.backends.tpu.allow_excess_precision = True
  mul_add_compiled_true = torch.compile(mul_add, backend="tpu")
  print(f"{torch.backends.tpu.allow_excess_precision=}")
  res_true = mul_add_compiled_true(bit_6_tensor, bit_7_tensor, one_float32)[
      0, 0
  ]
  print(f"{res_true=:.20f}")

  try:
    assert res_true == bits_6_7_13_tensor[0, 0]
    print("  [PASSED] Successfully enabled excess precision.")
  except AssertionError:
    print(
        "  [WARNING] Failed to enable excess precision (Expected: XLA option"
        " latching locked the flag to 'False' earlier in this session)."
    )
  return


if __name__ == "__main__":
  app.run()

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

import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # **Numerical Parity: Comparing TPU vs CPU**

    When porting models to TPU, you will often find that outputs differ slightly from your CPU reference. This is expected behavior due to fundamental hardware-level differences between a General Purpose CPU, a GPU, and the specialized TPU architecture.

    ### **1. Hardware-Level Differences**

    Numerical drift on TPU is primarily a result of hardware design, rather than just software optimizations:

    *   **Native Data Types**: TPUs are optimized for **BFloat16**. Unlike CPUs or GPUs, TPUs do **not** natively support Float32 at the hardware level. To execute Float32 math, the hardware must emulate it using three BFloat16 operations, which is significantly slower and can still lead to precision drift compared to standard IEEE-754 Float32.
    *   **Accumulator Precision**: The TPU's Matrix Units (MXUs) use specialized hardware accumulators. Depending on the TPU generation, these accumulators may operate in BFloat16 while a GPU might use Float32 or TF32. These differing accumulation paths lead to variations in the final result.
    *   **Systolic Array & Rounding**: TPUs use a systolic array for matrix multiplication, which optimizes for data reuse and throughput. This architectural choice means the order of operations and rounding behavior differs from the sequential logic used by CPUs.
    *   **Compiler Fusions**: The XLA compiler may fuse multiple operations (e.g., Multiply-Add) into a single hardware instruction to reduce rounding errors and improve speed, further diverging from the step-by-step execution of a CPU.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### **2. Comparing Results (Weight Synchronization)**

    To accurately compare TPU and CPU outputs, you must ensure both models start with the exact same weights. This process is used for **numerical verification only** and is not part of a standard training workflow.

    **The Parity Workflow:**
    1.  Initialize your model on the CPU.
    2.  Move an identical model to the TPU.
    3.  **Sync the weights** by loading the CPU `state_dict` into the TPU model.
    4.  **Use the same Dtype**: For a fair "apples-to-apples" comparison, cast both models to the same data type (e.g., `torch.bfloat16`) before comparing.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### **3. Using `torch.testing.assert_close`**

    PyTorch's standard assertion utility `torch.testing.assert_close` compares TPU results moved to CPU (`out_tpu.cpu()`) against the CPU reference (`out_cpu`).
    """)
  return


@app.cell
def _():
  import torch
  import torch.nn as nn

  # Standard hardware initialization
  device = torch.device("tpu")
  return nn, torch


@app.cell
def _(nn, torch):
  # 1. Setup identical models in BFloat16 for fair comparison
  model_cpu = nn.Linear(10, 10).cpu().to(torch.bfloat16)
  model_tpu = nn.Linear(10, 10).to("tpu").to(torch.bfloat16)

  # 2. Sync weights (Mandatory for Parity)
  model_tpu.load_state_dict(model_cpu.state_dict())

  # 3. Create dummy input data (same for both)
  input_data = torch.randn(5, 10).to(torch.bfloat16)
  input_tpu = input_data.to("tpu")

  # 4. Run Forward Pass
  out_cpu = model_cpu(input_data)
  out_tpu = model_tpu(input_tpu)

  # 5. Compare results
  # Note: TorchTPU's internal bfloat16 precision drift may exceed default
  # strict tolerances, generating the `assert_close` diagnostic report below.
  try:
    # Note: we move the TPU result back to CPU for comparison
    torch.testing.assert_close(out_tpu.cpu(), out_cpu)
    print("✅ Parity check passed!")
  except AssertionError as e:
    print("❌ Parity check failed (expected for hardware precision drift)\n")
    print(e)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### **4. Interpreting Detailed Mismatches**

    If the tensors exceed tolerances, `utils.assert_close` provides a detailed diagnostic report. This identifies if the drift is systematic or an outlier.

    **Example Failure Output:**
    ```
    Tensor-likes are not close!

    Mismatched elements: 1 / 50 (2.0%)
    Greatest absolute difference: 0.006596 at index (3, 9) (strict relative check failure)
    Greatest relative difference: 0.165375 at index (3, 9) (up to 0.1 allowed)

    Tolerance Suggestions:
    Strict check failed.
    To pass STRICT mode, you need BOTH:
    - rtol >= 0.165375 (1.7e-01)
    - atol >= 0.006596 (6.6e-03)
    ```

    *   **Mismatched elements**: High percentages (e.g., >70%) typically suggest standard precision loss from compiler fusions rather than a logic bug.
    *   **Tolerance Suggestions**: This analyzes the failure and calculates the exact `atol` and `rtol` needed to accommodate the hardware variance. In `STRICT` mode, you need to satisfy both thresholds.

    ### **5. Best Practices for Parity**

    *   **Dtype Consistency**: Always compare `BF16` on TPU to `BF16` on CPU. Comparing `BF16` to `FP32` will always yield significant mismatches.
    *   **Verify Gradients**: Use the same synchronization pattern to check `model.weight.grad` parity after a `loss.backward()` call.
    *   **Baseline Tolerances**: The recommended baseline tolerances are the default ones linked in the [PyTorch testing documentation](https://docs.pytorch.org/docs/stable/testing.html). When doing comparisons, you should use the `STRICT` mode in `utils.assert_close` (which is the default).
    """)
  return


if __name__ == "__main__":
  app.run()

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

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # **Numerical Parity Tools (`utils.assert_close`)**

    When porting models to the TPU, you will often find that TPU outputs differ slightly from your CPU reference. This is usually not a bug, but a result of hardware-level optimizations and the way the **XLA Compiler** re-arranges your math for the hardware.

    ### **1. Understanding Numerical Drift**

    While TorchTPU defaults to the **Highest Precision** path for **Float32**, several factors cause TPU results to deviate from the CPU:

    *   **Operator Fusion**: The compiler may fuse a sequence like ***A . B + C*** into a single instruction. This results in fewer rounding steps than the CPU, which executes each op sequentially.
    *   **Algebraic Simplification**: XLA reorders mathematical operations to optimize for the TPU's architecture — for example, to increase hardware parallelism or utilize the Matrix Units (MXUs) more efficiently. Because of this, in floating-point math, $(A + B) + C$ is not always calculated the same as ***A + (B + C)***.
    *   **Systolic Array Execution**: The TPU Matrix Units (MXUs) use specialized hardware paths for "dot-product" accumulations. To prioritize computation speed, these specialized units may not strictly adhere to the standard IEEE-754 logic used by CPUs for all operations.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### **2. The Weight Synchronization Trap**

    A common mistake in parity testing is initializing a model on the CPU and another on the TPU and comparing them immediately. Because layers use random initialization, the weights will differ, causing the test to fail.

    **The Golden Workflow:**
    1.  Initialize your model on the CPU.
    2.  Initialize an identical model on the TPU.
    3.  **Sync the weights** by loading the CPU `state_dict` into the TPU model.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### **3. Using `utils.assert_close`**

    This utility is a wrapper around PyTorch’s standard assertion engine, optimized for TPU-to-CPU comparisons. It treats the CPU as the high-precision "ground truth."
    """)
  return


@app.cell
def _():
  import torch
  import torch.nn as nn
  from tpu_utils import safe_init
  from torch_tpu._internal.utils import utils  # used only in notebooks

  # Self-healing hardware initialization for notebooks
  device = safe_init()
  return device, nn, torch, utils


@app.cell
def _(device, nn, torch, utils):
  # 1. Setup identical models
  model_cpu = nn.Linear(10, 10).cpu()
  model_tpu = nn.Linear(10, 10).to(device)

  # 2. Sync weights (Mandatory)
  model_tpu.load_state_dict(model_cpu.state_dict())

  # 3. Create identical input data
  input_data = torch.randn(5, 10)
  input_tpu = input_data.to(device)

  # 4. Run Forward Pass
  out_cpu = model_cpu(input_data)
  out_tpu = model_tpu(input_tpu)

  # 5. Compare results
  # We use PyTorch's recommended bfloat16 tolerances in STRICT mode.
  # Note: TorchTPU's internal bfloat16 precision drift will likely exceed these
  # strict tolerances, generating the `assert_close` diagnostic report below.
  try:
    utils.assert_close(
        out_tpu.cpu(),
        out_cpu,
        rtol=1.6e-2,
        atol=1e-5,
        check_value=utils.CheckValueMode.STRICT,
    )
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
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### **5. Best Practices for Parity**

    *   **Don't Cast CPU to BFloat16**: Keep your CPU reference in `float32`. BFloat16 has a smaller mantissa than Float32, making it more susceptible to precision differences. Keeping the CPU baseline in Float32 ensures a fair comparison and measures the TPU's deviation from the highest possible precision.
    *   **Baseline Tolerances**: The recommended baseline tolerances are the default ones linked in the [PyTorch testing documentation](https://docs.pytorch.org/docs/stable/testing.html). When doing comparisons, you should use the `STRICT` mode in `utils.assert_close` (which is the default).
    *   **Verify Gradients**: Check `model.weight.grad` parity after a `loss.backward()` call using the same synchronization pattern.
    """)
  return


if __name__ == "__main__":
  app.run()

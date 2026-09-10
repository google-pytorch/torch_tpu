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
"""This notebook explains the execution modes of TorchTPU."""

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
    # **Eager Mode and Execution Control**

    This guide provides an overview of TorchTPU's execution modes. TorchTPU supports multiple ways to dispatch operations to the hardware, ranging from standard eager execution to highly optimized fused execution.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Execution Modes**

    TorchTPU provides three execution modes. By default, it operates in **Strict Eager** mode, which behaves similarly to standard PyTorch on CUDA.

    | Mode | Constant | Description |
    | :--- | :--- | :--- |
    | **Strict Eager** | `EagerMode.DEFER_NEVER` | **The default mode.** Operations are dispatched one at a time and the execution is asynchronous. This aligns with standard PyTorch Eager mode on GPU. |
    | **Debug Eager** | `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING` | **Synchronous mode.** Operations are dispatched one at a time but execution is synchronous. This mode can also be enabled with the TPU_LAUNCH_BLOCKING="1" environment variable. This execution mode is similar to the CUDA_LAUNCH_BLOCKING environment variable. |
    | **Fused Eager** | `EagerMode.DEFER_AND_FUSE` | **Optimized mode.** Groups multiple operations together, allowing the XLA compiler to fuse across operation boundaries for high performance. This mode can also be enabled with the TORCH_TPU_DEFER_AND_FUSE="1" environment variable.|


    ### **Switching Modes**
    You can change the execution mode in two ways:

    #### **1. Globally (for the entire session)**
    Assign the desired mode to the `eager_mode` property:

    ```python
    from torch_tpu._internal import execution_mode as em

    # Switch to Strict Eager globally
    em.eager_mode = em.EagerMode.DEFER_NEVER
    ```

    #### **2. Locally (using a context manager)**
    Use `set_eager_mode` to temporarily override the mode for a block of code:

    ```python
    from torch_tpu._internal import execution_mode as em

    # Force every operation to execute immediately for this block
    with em.set_eager_mode(em.EagerMode.DEFER_NEVER):
        # Now you can pinpoint the exact line where an error occurs
        output = model(input_data)
    ```
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Demonstrating the Difference**

    In **Strict Eager** (default), operations are executed one at a time. In **Fused Eager**, operations are deferred and only executed as larger chunks as necessary.
    """)
  return


@app.cell
def _(torch):
  import torch

  # Standard hardware initialization
  device = torch.device("tpu")
  import time
  from torch_tpu._internal import execution_mode as em

  # Use a large tensor to make memory bandwidth the bottleneck
  size = 5000
  iterations = 50
  x = torch.randn(size, size, device="tpu", dtype=torch.bfloat16)

  def run_benchmark():
    y = x
    for _ in range(iterations):
      # These operations will be fused in Fused Eager mode
      y = y + 1.0
      y = y * 2.0
      y = y - 1.0
    # Transferring to CPU triggers the execution of all deferred ops
    return y.cpu()

  # --- Warmup ---
  # Trigger compilation for Fused Eager before measuring
  with em.set_eager_mode(em.EagerMode.DEFER_AND_FUSE):
    _ = run_benchmark()

  # --- Measure Fused Eager (DEFER_AND_FUSE) ---
  with em.set_eager_mode(em.EagerMode.DEFER_AND_FUSE):
    t0 = time.time()
    _ = run_benchmark()
    time_fused = time.time() - t0

  # --- Measure Strict Eager (DEFER_NEVER) ---
  with em.set_eager_mode(em.EagerMode.DEFER_NEVER):
    # Warmup for strict mode (ensures execution path is ready)
    _ = run_benchmark()
    t0 = time.time()
    _ = run_benchmark()
    time_strict = time.time() - t0

  print(f"Fused Eager (DEFER_AND_FUSE) Time: {time_fused:.4f} seconds")
  print(f"Strict Eager (DEFER_NEVER) Time: {time_strict:.4f} seconds")
  print(f"Speedup: {time_strict / time_fused:.1f}x")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Why Use Fused Eager?**

    While **Strict Eager** is the most intuitive for development, **Fused Eager** (`DEFER_AND_FUSE`) provides significant performance advantages for production training:

    1. **Fusion & Optimization:** Grouping operations allows XLA to reduce memory traffic by fusing multiple logical steps (like a `Linear` layer followed by a `ReLU`) into a single hardware kernel.
    2. **Memory Efficiency:** Intermediate results in a fused block often don't need to be stored in physical TPU memory (HBM), reducing the memory footprint.
    3. **Reduced Dispatch Overhead:** It minimizes the "chatter" between the CPU and TPU by sending larger, more meaningful chunks of work at once.

    > [!TIP]
    > For the absolute best performance, consider using `torch.compile(model, backend="tpu")`, which uses an even more aggressive Ahead-of-Time (AOT) compilation strategy.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Factory Methods and Materialization**

    Even in **Fused Eager** mode, certain operations trigger immediate execution (materialization):

    *   **`.item()`, `.cpu()`, `.tolist()`:** These force the data to be synchronized with the host CPU.
    *   **`print(tensor)`:** To display values, the data must be materialized.
    *   **Data-dependent control flow:** `if (tensor > 0).all():` implicitly materializes the tensor.
    """)
  return


if __name__ == "__main__":
  app.run()

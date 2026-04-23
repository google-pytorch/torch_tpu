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

    TorchTPU provides three primary execution modes. By default, it operates in **Strict Eager** mode, which behaves similarly to standard PyTorch on CUDA.

    | Mode | Constant | Description |
    | :--- | :--- | :--- |
    | **Strict Eager** | `EagerMode.DEFER_NEVER` | **The default mode.** Operations are dispatched to the hardware immediately. Simplest for debugging and aligns with CUDA behavior. |
    | **Fused Eager** | `EagerMode.DEFER_AND_FUSE` | **Optimized mode.** Groups multiple operations together, allowing the XLA compiler to optimize across operation boundaries for high performance. |
    | **Debug Eager** | `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING` | **Synchronous mode.** Executes operations immediately and waits for completion. Best for pinpointing exactly which line caused a hardware error. |

    ### **Switching Modes**
    You can change the execution mode globally or locally using the `eager_mode` context manager:

    ```python
    from torch_tpu._internal import execution_mode as em

    # Enable Fused Eager for a specific block of code
    with em.eager_mode(em.EagerMode.DEFER_AND_FUSE):
        output = model(input_data)
    ```
    """)
  return


@app.cell
def init():
  import torch

  from torch_tpu._internal import execution_mode as em

  # Standard hardware initialization
  return em, torch


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Demonstrating the Difference**

    In **Strict Eager** (default), operations happen one by one. In **Fused Eager**, operations are recorded into a graph and only executed when necessary (materialization) or when a large enough block is formed.
    """)
  return


@app.cell
def modes_demo(em, torch):
  from torch_tpu._internal.sync import sync

  def check_fusion(mode_name, mode):
    print(f"--- Mode: {mode_name} ---")
    with em.eager_mode(mode):
      # Perform a simple calculation
      x = torch.ones((2, 2), device="tpu")
      y = x + 1
      z = y * 2

      # computation_mlir returns the StableHLO graph for the given tensor.
      # If fused, it will contain 'add' and 'multiply' operations.
      mlir = sync.computation_mlir(z)
      has_add = "stablehlo.add" in mlir
      has_mul = "stablehlo.multiply" in mlir

      print(f"  Graph contains 'add' op: {has_add}")
      print(f"  Graph contains 'multiply' op: {has_mul}")

      if mode == em.EagerMode.DEFER_AND_FUSE:
        print(
            "  Status: Operations are FUSED into a single graph for"
            " optimization."
        )
      else:
        print(
            "  Status: Operations were DISPATCHED immediately (standard eager)."
        )
    print()

  # Default Behavior
  check_fusion("Strict Eager (DEFER_NEVER)", em.EagerMode.DEFER_NEVER)

  # Performance Behavior
  check_fusion("Fused Eager (DEFER_AND_FUSE)", em.EagerMode.DEFER_AND_FUSE)

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

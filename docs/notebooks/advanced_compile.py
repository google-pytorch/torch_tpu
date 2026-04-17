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
"""Advanced compilation with torch.compile notebook."""

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
    # **Torch.compile on TorchTPU: User Guide**

    This guide provides best practices for using `torch.compile` to accelerate PyTorch models on Google TPUs. By following these patterns, you can ensure that your models are optimized for the TPU's asynchronous, graph-based architecture.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **1. Setup and Environment**
    Ensure you are using **PyTorch 2.10** or later. While `backend="tpu"` is currently required, simply importing the `torch_tpu` module will soon enable auto-registration as the default backend for TPU devices.
    """)
  return


@app.cell
def _():
  import torch
  import torch_tpu

  # Initialize the TPU device
  # Idempotent and thread-safe entry point
  device = torch.device("tpu")
  print(f"TPU Device initialized: {device}")
  return device, torch


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **2. Efficient Model Loading and Compilation Warming**
    To prevent memory bloat and reduce the risk of **Out-of-Memory (OOM)** errors, avoid loading large models on the CPU and moving them to TPU via `.to(device)`. Instead, use the `torch.device` context manager to allocate parameters directly on TPU High Bandwidth Memory (HBM).
    """)
  return


@app.cell
def efficient_loading(device, mo, torch):
  class MyLargeModel(torch.nn.Module):

    def __init__(self):
      super().__init__()
      self.net = torch.nn.Sequential(
          torch.nn.Linear(10, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
      )

    def forward(self, x):
      return self.net(x)

  # Best Practice: Model and data are allocated directly on the hardware
  with torch.device(device):
    model = MyLargeModel()
    input_data = torch.randn(20, 10)

  # Initialize the compiled model with mandatory static shapes
  compiled_model = torch.compile(model, backend="tpu", dynamic=False)

  # Compilation Warming: The first forward pass triggers the actual TPU compilation
  warmup_output = compiled_model(input_data)

  mo.output.append(
      mo.md(
          "**Compilation warmup complete!** Output shape:"
          f" `{warmup_output.shape}`"
      )
  )
  return (model,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **3. Understanding the Compute Model**
    TorchTPU supports two distinct execution paths:

    * **Eager (Deferred) Mode:** Operations are recorded into a Directed Acyclic Graph (DAG). They are executed only when a value is needed (e.g., calling `.item()` or `.cpu()`).
    * **Compiled Mode:** `torch.compile` captures the entire graph Ahead-of-Time (AOT). This allows the XLA compiler to perform cross-operator fusions and layout optimizations not possible in Eager mode.
    * **Materialization:** Results from a compiled model are returned as **Materialized** tensors (physical buffers on the TPU), ensuring hardware completion before returning to Python.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **4. Architectural Constraints**

    ### **Static Shapes Requirement**
    The XLA compiler, in order to achieve maximum performance, compiles all computations using their specific (i.e. static) shapes. This means that changes to shapes (i.e. dynamic shapes) require recompilations, which can be time consuming.
    * **Strict Enforcement:** You **must** set `dynamic=False` in `torch.compile`.
    * **Behavior:** The backend will fail or trigger frequent re-compilations if shapes vary at runtime.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **5. Advanced Optimization: Compiling the Training Step**
    For maximum throughput, compile the **entire training step function** rather than just the model. This allows XLA to generate a **joint forward-backward graph**, optimizing memory reuse and buffer donation across the entire training iteration.
    > [!IMPORTANT]
    > **Stateful Optimizers:** When using stateful optimizers (like `Adam`, `AdamW`, `Adamax`, `ASGD`, `NAdam`, `RAdam`, `RMSProp`, or `Rprop`) with `torch.compile` on TPU, you **must** pass `capturable=True` during initialization. This ensures the optimizer's lazy state tensors (e.g., momentum) are safely captured into the XLA graph instead of crashing as CPU tensors.
    """)
  return


@app.cell
def training_step(device, mo, model, torch):
  import torch.optim as optim

  # Optimizer and data must be on the SAME device as the model
  # NOTE: capturable=True is required for stateful optimizers on TorchTPU!
  optimizer = optim.Adam(model.parameters(), lr=1e-3, capturable=True)
  criterion = torch.nn.MSELoss()

  @torch.compile(dynamic=False)
  def train_step(data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.detach()

  # Ensure inputs are on TPU before calling the compiled function
  target_data = torch.randn(20, 10, device=device)
  sample_input = torch.randn(20, 10, device=device)

  # Execute as a single optimized TPU kernel
  final_loss = train_step(sample_input, target_data)

  # Synchronize and move the result to the cpu for printing
  mo.output.append(
      mo.md(f"**Training step complete!** Loss: `{final_loss.item():.4f}`")
  )
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **6. Debugging and Interpreting Output**
    Use `torch._dynamo.explain()` to diagnose performance issues. This tool identifies "graph breaks" - points where the compiler must return control to the CPU, causing significant latency.

    ### **Real-World Example: Fixing a Graph Break**
    """)
  return


@app.cell
def explain_graph(device, torch):
  from torch._dynamo import explain

  class DiagnosticModel(torch.nn.Module):

    def forward(self, x):
      x = x + 1
      # BUG: Standard Python 'print' or .item() causes a graph break!
      print(f"Mean: {x.mean()}")
      return x * 2

  diag_model = DiagnosticModel().to(device)
  diag_input = torch.randn(10, 10, device=device)

  # Generate the explanation
  explanation = explain(diag_model, diag_input)
  print(explanation)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### **Interpreting the Explanation**
    * **Graphs Generated:** Ideally **1**. If you see 2 or more, XLA cannot optimize across the segments.
    * **Break Points:** Look for reasons like `print`, `item()`, or unsupported Python control flow.
    """)
  return


if __name__ == "__main__":
  app.run()

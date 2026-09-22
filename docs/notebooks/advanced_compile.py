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

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # **Torch.compile on TorchTPU**

    This guide provides best practices for using `torch.compile` to accelerate PyTorch models on Google TPUs.
    """)
  return


@app.cell
def _():
  import sys
  import torch

  # One time initialization of the TPU
  device = torch.device("tpu")
  return (torch,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **1. The `dynamic=False` Requirement**

    As of today, `torch_tpu` requires static shapes when using `torch.compile`. We do not currently support the `dynamic=True` or `dynamic=None` flags.

    **What this means for you:**
    You must always set `dynamic=False`. As a consequence, if you pass inputs with variable dimensions (such as a changing batch size at the end of an epoch, or variable sequence lengths), this will trigger a full recompilation every time the size changes. Recompilations are expensive and will reduce your training performance.

    **Best Practice:**
    Do not pass variable sizes to your compiled models. Ensure your tensor dimensions remain constant across all iterations. If your dataset naturally has varying sizes, use standard PyTorch techniques to **pad** or **slice** the data *outside* of the compiled region to maintain fixed-size input tensors.
    """)
  return


@app.cell
def efficient_loading(mo, torch):
  class MyLargeModel(torch.nn.Module):

    def __init__(self):
      super().__init__()
      self.net = torch.nn.Sequential(
          torch.nn.Linear(10, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
      )

    def forward(self, x):
      return self.net(x)

  # Best Practice: Model and data are allocated directly on the hardware
  with torch.device("tpu"):
    model = MyLargeModel()
    input_data = torch.randn(20, 10)

  # Initialize the compiled model with mandatory static shapes
  compiled_model = torch.compile(model, backend="tpu", dynamic=False)

  print(
      "Starting TPU XLA compilation (this may take 1-2 minutes)...", flush=True
  )
  # Compilation Warming: The first forward pass triggers the actual TPU compilation
  warmup_output = compiled_model(input_data)
  print("Compilation complete!", flush=True)

  mo.output.append(
      mo.md(
          "**Compilation warmup complete!** Output shape:"
          f" `{warmup_output.shape}`"
      )
  )
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **2. Graph Breaks: Why They Matter and How to Detect Them**

    A **graph break** occurs when `torch.compile` (TorchDynamo) encounters Python code, control flow, or third-party library calls that it cannot trace into an FX graph. When this happens, execution falls back to the Python interpreter.

    **Why Graph Breaks are Bad on TPU:**
    - **Lack of Fusion Across Boundaries:** The XLA compiler cannot optimize or fuse operations across a graph break boundary. Intermediate tensors must be written to and re-read from TPU HBM instead of being kept in fast on-chip memory across fused kernel pipelines.
    - **Dispatch and Launch Overhead:** Each graph break fragments your computation into separate XLA program executables. The host runtime must schedule and dispatch each sub-graph independently, introducing significant dispatch latency between offloads.

    **How to Detect Graph Breaks:**
    1. `torch._dynamo.explain(fn)(*args)`: Programmatically inspects graph break counts, break reasons, and stack traces.
    2. `torch.compile(fn, fullgraph=True)`: Fails fast on the first graph break by raising `torch._dynamo.exc.Unsupported`.
    3. `TORCH_LOGS="graph_breaks"`: Streams graph break notifications to stderr at runtime.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **3. A Common Issue: Training Steps with Graph Breaks**

    In standard PyTorch eager mode, a typical training step executes `loss = criterion(model(x), target)`, calls `loss.backward()`, and then invokes `optimizer.step()`.

    When wrapping this standard training loop with `torch.compile`, calling `loss.backward()` and `optimizer.step()` directly inside the compiled function causes graph breaks because Dynamo traces the forward pass and cannot inline the eager autograd C++ engine or mutable optimizer operations into a single static graph.
    """)
  return


@app.cell
def basic_training_breaks(mo, torch):
  class ToyLinearModule(torch.nn.Module):

    def __init__(self, in_features=16, out_features=16):
      super().__init__()
      self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
      return self.linear(x)

  with torch.device("tpu"):
    toy_model = ToyLinearModule()
    # Note: capturable=True is recommended for stateful optimizers on TPU
    optimizer = torch.optim.Adam(
        toy_model.parameters(), lr=1e-2, capturable=True
    )
    criterion = torch.nn.MSELoss()
    x = torch.randn(4, 16)
    target = torch.randn(4, 16)

  def basic_train_step(data, target_data):
    optimizer.zero_grad()
    output = toy_model(data)
    loss = criterion(output, target_data)
    loss.backward()
    optimizer.step()
    return loss

  # 1. Compile and run with standard torch.compile
  compiled_basic_step = torch.compile(
      basic_train_step, backend="tpu", dynamic=False
  )
  print("Running basic compiled training step...", flush=True)
  basic_loss = compiled_basic_step(x, target)
  print(f"Basic train step loss: {basic_loss.item():.4f}", flush=True)

  # 2. Programmatically detect graph breaks using torch._dynamo.explain
  explanation = torch._dynamo.explain(basic_train_step)(x, target)
  print(
      f"Detected {explanation.graph_break_count} graph break(s) in basic"
      " training step:",
      flush=True,
  )
  for idx, reason in enumerate(explanation.break_reasons):
    print(f"  Break {idx + 1}: {reason.reason}", flush=True)

  # 3. Verify that fullgraph=True raises an exception on graph breaks
  fullgraph_step = torch.compile(
      basic_train_step, backend="tpu", dynamic=False, fullgraph=True
  )
  try:
    fullgraph_step(x, target)
  except torch._dynamo.exc.Unsupported as e:
    print(
        f"fullgraph=True caught expected graph break: {type(e).__name__}",
        flush=True,
    )

  mo.output.append(
      mo.md(
          "**Basic Training Step Results:**<br>"
          f"- Graph breaks detected: `{explanation.graph_break_count}`<br>"
          f"- Subgraphs generated: `{len(explanation.graphs)}`<br>"
          f"- Loss: `{basic_loss.item():.4f}`"
      )
  )
  return (ToyLinearModule,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **4. Fusing the Full Training Step: `SingleTraceTrainer`**

    To prevent graph breaks during training and achieve maximum TPU utilization, the entire training step (forward pass, loss, backward gradient calculation, and optimizer update) should be fused into a **single, unified static graph**.

    The `SingleTraceTrainer` utility (`torch_tpu.extras.optimizers.single_trace_trainer`) achieves this by:
    1. Using `torch.func.grad_and_value` to compute reverse-mode automatic differentiation symbolically.
    2. Using functional optimizer updates (`adamw.TorchAdamw` or `sgd.TorchSgd`).
    3. Capturing the entire forward, loss, backward, and optimizer pipeline into a single FX graph via `make_fx`.

    This unified graph compiles with `torch.compile(..., fullgraph=True)` with **zero graph breaks**, running as a single, highly fused offload on TPU hardware.
    """)
  return


@app.cell
def single_trace_training(ToyLinearModule, mo, torch):
  from torch_tpu.extras.optimizers import adamw
  from torch_tpu.extras.optimizers import single_trace_trainer

  with torch.device("tpu"):
    no_break_model = ToyLinearModule()
    no_break_x = torch.randn(4, 16)
    no_break_target = torch.randn(4, 16)

  # Initialize SingleTraceTrainer with TorchAdamw
  trainer = single_trace_trainer.SingleTraceTrainer(
      model=no_break_model,
      optimizer=adamw.TorchAdamw(use_bfloat16_moments=False),
  )

  # Verify zero graph breaks and compile with fullgraph=True
  def compile_and_verify_breaks(graph_module, flat_inputs):
    explanation = torch._dynamo.explain(graph_module)(*flat_inputs)
    print(
        f"SingleTraceTrainer graph breaks: {explanation.graph_break_count}",
        flush=True,
    )
    return torch.compile(
        graph_module, backend="tpu", dynamic=False, fullgraph=True
    )

  train_step = trainer.make_compiled_train_step(
      no_break_x, no_break_target, compile_fn=compile_and_verify_breaks
  )

  # Execute compiled single-trace training step on TPU
  fused_loss = train_step(no_break_x, no_break_target)
  print(
      f"SingleTrace training step complete! Loss: {fused_loss.item():.4f}",
      flush=True,
  )

  mo.output.append(
      mo.md(
          "**SingleTraceTrainer Verification:**<br>"
          "- Graph breaks: `0`<br>"
          "- Compilation mode: `fullgraph=True`<br>"
          f"- Final loss: `{fused_loss.item():.4f}`"
      )
  )
  return


if __name__ == "__main__":
  app.run()

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
# pylint: skip-file  # ruff: noqa
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


if __name__ == "__main__":
  app.run()

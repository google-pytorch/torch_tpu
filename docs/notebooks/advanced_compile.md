---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

+++

# **Torch.compile on TorchTPU**

This guide provides best practices for using `torch.compile` to accelerate PyTorch models on Google TPUs.

```{code-cell} ipython3
import sys
import torch

# One time initialization of the TPU
device = torch.device("tpu")
```

## **1. The `dynamic=False` Requirement**

As of today, `torch_tpu` requires static shapes when using `torch.compile`. We do not currently support the `dynamic=True` or `dynamic=None` flags.

**What this means for you:**
You must always set `dynamic=False`. As a consequence, if you pass inputs with variable dimensions (such as a changing batch size at the end of an epoch, or variable sequence lengths), this will trigger a full recompilation every time the size changes. Recompilations are expensive and will reduce your training performance.

**Best Practice:**
Do not pass variable sizes to your compiled models. Ensure your tensor dimensions remain constant across all iterations. If your dataset naturally has varying sizes, use standard PyTorch techniques to **pad** or **slice** the data *outside* of the compiled region to maintain fixed-size input tensors.

```{code-cell} ipython3
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

print(f"Compilation warmup complete! Output shape: {warmup_output.shape}")
```

---
torch_tpu_version: 0.1.1.dev20260922
---

# **Torch.compile on TorchTPU**

This guide provides best practices for using `torch.compile` to accelerate
PyTorch models on Google TPUs.

```python
import sys
import torch

# One time initialization of the TPU
device = torch.device("tpu")
```

## **1. The `dynamic=False` Requirement**

As of today, `torch_tpu` requires static shapes when using `torch.compile`. We
do not currently support the `dynamic=True` or `dynamic=None` flags.

**What this means for you:** You must always set `dynamic=False`. As a
consequence, if you pass inputs with variable dimensions (such as a changing
batch size at the end of an epoch, or variable sequence lengths), this will
trigger a full recompilation every time the size changes. Recompilations are
expensive and will reduce your training performance.

**Best Practice:** Do not pass variable sizes to your compiled models. Ensure
your tensor dimensions remain constant across all iterations. If your dataset
naturally has varying sizes, use standard PyTorch techniques to **pad** or
**slice** the data *outside* of the compiled region to maintain fixed-size input
tensors.

```python
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
```

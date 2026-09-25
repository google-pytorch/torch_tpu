---
torch_tpu_version: 0.1.1.dev20260922
---

# Background on ATen ops

There is a third class of callables in PyTorch beyond functions and modules:
ops.

Behind the scenes of a call to a function like `torch.nn.functional.linear(a,
b)` or even one with a custom autograd function like `SigmoidNoVanishing`,
PyTorch converts these calls into ATen and c10d ops. ATen stands for "A Tensor
Library", and c10d is an abbreviation for caffe distributed. c10d ops are for
collectives to enable communication between devices. Everything else is an ATen
op. ATen contains thousands of ops, and these ops form an Intermediate
Representation (IR).

As you saw in the implementation of `torch.nn.Linear`, the forward method simply
calls `torch.nn.functional.linear`. Ultimately, that call will get translated by
this block of C++ into the aten ops of `addmm` and `.t` (transpose).

```cpp
return at::addmm(*bias, input, weight.t());
```

[PyTorch source](https://github.com/pytorch/pytorch/blob/5b4d57d8bb7dbe3ddb5dced469df4f5dd1e82a11/aten/src/ATen/native/Linear.cpp#L85)

You will now explore how to view the IR that the PyTorch IR engine creates
internally. Look at the sample implementation below. Notice that logs for aot
are enabled:

```python
torch._logging.set_logs(aot_graphs=True)
```

AOT stands for Ahead-Of-Time compilation. In this context, it basically means
the series of aten ops that your Python code was lowered into.

Inspecting the logs, you'll see the key lines: a transpose, then an addmm.

```log
t: "f32[64, 64][1, 64]cpu" = torch.ops.aten.t.default(primals_1);  primals_1 = None
addmm: "f32[32, 64][64, 1]cpu" = torch.ops.aten.addmm.default(primals_2, primals_3, t);  primals_2 = t = None
```

You now know how to view the ATen ops that PyTorch creates internally.

**Exercise for the reader**: Apply the same logging as in the sample below to
**[extending_modules_and_functions_via_composition.py](?file=extending_modules_and_functions_via_composition.py)**
and figure out which ATen ops correspond to the bit operations.

```python
  # Shift each of the 32 bits by zero to 31 positions.
  shifted = packed >> torch.arange(32, dtype=torch.uint32)
  # AKA shifted = packed.bitwise_right_shift(torch.arange(32, dtype=torch.uint32))

  # Mask out the upper bits.
  values = shifted & 1
  # AKA values = shifted.bitwise_and(1)
```

**Exercise for the reader**: Apply the same logging technique to
**[customizing_autograd_via_torch_autograd_function.py](?file=customizing_autograd_via_torch_autograd_function.py)**
and figure out which ATen ops correspond to `torch.where` and the `>` and `<=`
operator.

```python
local_grad = torch.where(
    torch.logical_and(result > 0.1, result <= 0.9),
    torch.tensor(0.25, device=result.device),
    torch.tensor(0.01, device=result.device),
)
```

*Answer*:

```
gt: "b8[7, 1][1, 1]cpu" = torch.ops.aten.gt.Scalar(sigmoid, 0.1)
le: "b8[7, 1][1, 1]cpu" = torch.ops.aten.le.Scalar(sigmoid, 0.9)
```

```python
from absl import app as absl_app
import torch

torch._logging.set_logs(aot_graphs=True)  # pylint: disable=protected-access
```

```python
@torch.compile(backend="aot_eager")
def fwd_bwd(layer_one, layer_two, data):
  output = layer_two(layer_one(data))
  output.sum().backward()
```

```python
def main():
  fwd_bwd(
      layer_one=torch.nn.Linear(64, 64),
      layer_two=torch.nn.functional.relu,
      data=torch.randn(32, 64),
  )
  print("Success.")

main()
```

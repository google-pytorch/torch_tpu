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

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
        <style>
          .nav-link {
            color: #1a73e8;
            text-decoration: none;
            font-weight: 500;
            margin: 0 2px;
          }
          .nav-link:hover {
            text-decoration: underline;
          }
          .nav-strong {
            color: #202124;
            font-weight: 600;
            margin: 0 2px;
          }
        </style>
        <div style='margin-bottom: 24px; font-size: 13px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; text-align: center; border-bottom: 1px solid #e0e0e0; padding-bottom: 12px; line-height: 1.8;'>
          <a class='nav-link' data-page='index' href='index.html'>Home</a> |
          <a class='nav-link' data-page='extending_modules_and_functions_via_composition' href='extending_modules_and_functions_via_composition.html'>Composition</a> |
          <a class='nav-link' data-page='customizing_autograd_via_torch_autograd_function' href='customizing_autograd_via_torch_autograd_function.html'>Custom Autograd</a> |
          <span class='nav-strong'>ATen Ops</span> |
          <a class='nav-link' data-page='background_on_torch_compile' href='background_on_torch_compile.html'>torch.compile</a> |
          <a class='nav-link' data-page='background_on_torch_compile_on_tpus_via_xla' href='background_on_torch_compile_on_tpus_via_xla.html'>Compile on TPU</a> |
          <a class='nav-link' data-page='quantized_sum' href='quantized_sum.html'>JAX Custom Ops</a> |
          <a class='nav-link' data-page='quant' href='quant.html'>Python Baseline</a> |
          <a class='nav-link' data-page='custom_ops_via_pallas' href='custom_ops_via_pallas.html'>Pallas TPU Kernel</a>
        </div>
        <script>
          if (window.location.search && window.location.search.indexOf('file=') !== -1) {
            document.querySelectorAll('.nav-link').forEach(link => {
              const page = link.getAttribute('data-page');
              link.href = '?file=' + page + '.py';
            });
          }
        </script>
        """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
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
    internally. Look at the sample implementation below. Notice that logs for aot are
    enabled:

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

    **Exercise for the reader**: Apply the same logging as in the sample below to **[extending_modules_and_functions_via_composition.py](?file=extending_modules_and_functions_via_composition.py)** and figure
    out which ATen ops correspond to the bit operations.

    ```python
      # Shift each of the 32 bits by zero to 31 positions.
      shifted = packed >> torch.arange(32, dtype=torch.uint32)
      # AKA shifted = packed.bitwise_right_shift(torch.arange(32, dtype=torch.uint32))

      # Mask out the upper bits.
      values = shifted & 1
      # AKA values = shifted.bitwise_and(1)
    ```

    **Exercise for the reader**: Apply the same logging technique to **[customizing_autograd_via_torch_autograd_function.py](?file=customizing_autograd_via_torch_autograd_function.py)** and figure out
    which ATen ops correspond to `torch.where` and the `>` and `<=` operator.

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
    """)
  return


@app.cell
def _():
  import torch

  torch._logging.set_logs(aot_graphs=True)  # pylint: disable=protected-access
  return (torch,)


@app.cell
def _(torch):
  @torch.compile(backend="aot_eager")
  def fwd_bwd(layer_one, layer_two, data):
    output = layer_two(layer_one(data))
    output.sum().backward()

  return (fwd_bwd,)


@app.cell
def _(fwd_bwd, torch):
  def main():
    fwd_bwd(
        layer_one=torch.nn.Linear(64, 64),
        layer_two=torch.nn.functional.relu,
        data=torch.randn(32, 64),
    )
    print("Success.")

  main()
  return


if __name__ == "__main__":
  app.run()

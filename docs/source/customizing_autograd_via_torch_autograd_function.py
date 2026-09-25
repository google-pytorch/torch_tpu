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
    # torch.autograd.Function

    `torch.autograd.Function` is the next approach to extending the functionality of
    PyTorch. It enables developers to define a custom backward pass.

    As a toy problem, suppose you decide that you want to modify the vanishing
    gradients problem of sigmoid and instead clamp the gradient to a minimum of
    0.01.

    Look at the sample implementation below. Notice that:

    *   `torch.autograd.Function` is subclassed
    *   `forward` and `backward` are overridden
    *   The `backward` implements the key clamping logic using the PyTorch functions
        `torch.where` and the Python operators `<=` and `>`.
    *   The output of the forward pass is saved for the backward pass.

    As a manual test, the main function prints a table of the outputs and grads for
    both the custom sigmoid and the regular `torch.sigmoid`.

    Input  | Custom Output | Regular Output | Custom Grad | Regular Grad
    -----: | ------------: | -------------: | ----------: | -----------:
    -100.0 | 0.0000        | 0.0000         | 0.0100      | 0.0000
    -10.0  | 0.0000        | 0.0000         | 0.0100      | 0.0000
    -1.0   | 0.2689        | 0.2689         | 0.2500      | 0.1966
    0.0    | 0.5000        | 0.5000         | 0.2500      | 0.2500
    1.0    | 0.7311        | 0.7311         | 0.2500      | 0.1966
    10.0   | 1.0000        | 1.0000         | 0.0100      | 0.0000
    100.0  | 1.0000        | 1.0000         | 0.0100      | 0.0000

    > **Warning**: In PyTorch 1.0, custom ops (a topic discussed below) were implemented
    > using this same mechanism. In PyTorch 2.0, custom ops have their own first
    > class support and should not use `torch.autograd.Function`.
    """)
  return


@app.cell
def _():
  from absl import app as absl_app
  import torch

  # Uncomment this line in a future exercise.
  torch._logging.set_logs(aot_graphs=True)  # pylint: disable=protected-access
  return (torch,)


@app.cell
def _(torch):
  class SigmoidNoVanishing(torch.autograd.Function):
    """Custom Sigmoid function to avoid vanishing gradients.

    The gradient is forced to be either 0.01 or 1.0.
    """

    @staticmethod
    def forward(ctx, x):
      result = torch.sigmoid(x)
      ctx.save_for_backward(result)
      return result

    @staticmethod
    def backward(ctx, grad_output):
      (result,) = ctx.saved_tensors
      local_grad = torch.where(
          torch.logical_and(result > 0.1, result <= 0.9),
          torch.tensor(0.25, device=result.device),
          torch.tensor(0.01, device=result.device),
      )
      return grad_output * local_grad

  return (SigmoidNoVanishing,)


@app.cell
def _(SigmoidNoVanishing, torch):
  def main():
    # Create variant of sigmoid:
    sigmoid_no_vanishing = SigmoidNoVanishing.apply
    sigmoid_no_vanishing = torch.compile(
        sigmoid_no_vanishing, backend="aot_eager"
    )

    # Input data.
    x = torch.tensor(
        [[-100.0], [-10.0], [-1.0], [0.0], [1.0], [10.0], [100.0]],
        requires_grad=True,
    )
    y = sigmoid_no_vanishing(x)

    # Reference sigmoid for comparison
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = torch.sigmoid(x_ref)

    # Verify outputs are identical
    _ = torch.allclose(y, y_ref)

    y.backward(torch.ones_like(y))
    y_ref.backward(torch.ones_like(y_ref))

    print(
        f"{'Input':>8} | {'Custom Output':>15} | {'Regular Output':>15} |"
        f" {'Custom Grad':>12} | {'Regular Grad':>12}"
    )
    print("-" * 74)
    for i in range(len(x)):
      print(
          f"{x[i].item():>8.1f} | {y[i].item():>15.4f} |"
          f" {y_ref[i].item():>15.4f} | {x.grad[i].item():>12.4f} |"
          f" {x_ref.grad[i].item():>12.4f}"
      )

  main()
  return


if __name__ == "__main__":
  app.run()

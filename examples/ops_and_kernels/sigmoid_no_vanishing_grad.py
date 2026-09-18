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

from absl import app
import torch

# Uncomment this line in a future exercise.
torch._logging.set_logs(aot_graphs=True)  # pylint: disable=protected-access


class _SigmoidNoVanishing(torch.autograd.Function):
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


# Example usage and verification
def main(argv):
  del argv
  # Create variant of sigmoid:
  sigmoid_no_vanishing = _SigmoidNoVanishing.apply
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
        f"{x[i].item():>8.1f} | {y[i].item():>15.4f} | {y_ref[i].item():>15.4f}"
        f" | {x.grad[i].item():>12.4f} | {x_ref.grad[i].item():>12.4f}"
    )


if __name__ == "__main__":
  app.run(main)

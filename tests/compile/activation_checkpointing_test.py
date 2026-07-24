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

from absl.testing import absltest
import torch
import torch._functorch.config as functorch_config
from torch_tpu._internal import compile as compile_lib


class ActivationCheckpointingTest(absltest.TestCase):

  def test_activation_checkpointing(self):
    class CheckpointedModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Linear(5, 10), torch.nn.GELU(), torch.nn.Linear(10, 2)
        )

      def forward(self, x):
        return torch.utils.checkpoint.checkpoint(
            self.block1, x, use_reentrant=False
        )

    device = torch.device("tpu")
    model = CheckpointedModel().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, capturable=True)
    criterion = torch.nn.MSELoss()
    tpu_backend = compile_lib.TpuBackend(debug=True)
    compiled = torch.compile(model, backend=tpu_backend)

    input_tensor = torch.arange(50, dtype=torch.float32, device=device).reshape(
        10, 5
    )
    target = torch.randn(10, 2, device=device)

    optimizer.zero_grad()
    outputs = compiled(input_tensor)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    self.assertLen(tpu_backend._compiled_executables, 2)
    fwd_mlir = tpu_backend._compiled_executables[0].mlir_text
    bwd_mlir = tpu_backend._compiled_executables[1].mlir_text

    self.assertNotIn(
        "stablehlo.optimization_barrier",
        fwd_mlir,
        "Found unexpected stablehlo.optimization_barrier in forward pass MLIR",
    )
    self.assertIn(
        "stablehlo.optimization_barrier",
        bwd_mlir,
        "Expected to find stablehlo.optimization_barrier in backward pass MLIR",
    )

  def test_functionalize_rng_ops(self):
    device = torch.device("tpu")
    q = torch.randn(2, 4, 128, 32, device=device, requires_grad=True)
    k = torch.randn(2, 4, 128, 32, device=device, requires_grad=True)
    v = torch.randn(2, 4, 128, 32, device=device, requires_grad=True)

    def fn(q, k, v):
      return torch.nn.functional.scaled_dot_product_attention(
          q, k, v, is_causal=True
      )

    tpu_backend = compile_lib.TpuBackend()
    compiled = torch.compile(fn, backend=tpu_backend)
    with functorch_config.patch(functionalize_rng_ops=True):
      compiled(q, k, v).sum().backward()


if __name__ == "__main__":
  absltest.main()

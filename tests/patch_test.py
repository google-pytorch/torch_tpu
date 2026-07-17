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

"""Tests for optimizer patching utility."""

import copy
from absl.testing import absltest
import torch
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal import optim
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.utils import utils as tpu_utils


class PatchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def tearDown(self):
    optim.unpatch_optimizer_graph_breaks()
    super().tearDown()

  def test_patch_optimizer_graph_breaks(self):
    # Verify that without the patch, compiling standard AdamW's step fails
    # under fullgraph=True.
    model = torch.nn.Linear(10, 5).to('tpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, capturable=True)
    tpu_backend = torch_tpu_compile.TpuBackend()

    # Populate gradients eagerly
    x = torch.randn(2, 10, device='tpu')
    model(x).sum().backward()

    # Compile optimizer step directly
    compiled_step = torch.compile(
        optimizer.step, backend=tpu_backend, fullgraph=True
    )

    # Without patching, it should fail to compile with fullgraph=True
    # due to the graph break in _use_grad wrapper
    with self.assertRaises(Exception):
      compiled_step()

    # Now apply the patch
    optim.patch_optimizer_graph_breaks()

    # Re-compile
    torch._dynamo.reset()
    compiled_step_patched = torch.compile(
        optimizer.step, backend=tpu_backend, fullgraph=True
    )

    # This should now compile and run successfully!
    compiled_step_patched()

  def test_numerical_parity(self):
    device = torch.device('tpu')
    tpu_backend = torch_tpu_compile.TpuBackend()

    # 1. Models and inputs
    model_eager = torch.nn.Linear(10, 5).to(device)
    model_patched = copy.deepcopy(model_eager)
    initial_weights = [p.clone() for p in model_eager.parameters()]

    inputs = [torch.randn(2, 10, device=device) for _ in range(5)]

    # 2. Setup optimizers
    opt_eager = torch.optim.AdamW(
        model_eager.parameters(), lr=0.1, capturable=True
    )

    # Patched compiled AdamW
    optim.patch_optimizer_graph_breaks()
    opt_patched = torch.optim.AdamW(
        model_patched.parameters(), lr=0.1, capturable=True
    )

    # 3. Define model steps (eager or compiled)
    def model_step_eager(x, model):
      loss = model(x).sum()
      loss.backward()
      return loss.detach()

    # We can compile the model step with fullgraph=True
    compiled_model_step_patched = torch.compile(
        model_step_eager, backend=tpu_backend, fullgraph=True
    )

    # 4. Compile optimizer steps directly with fullgraph=True
    compiled_opt_patched_step = torch.compile(
        opt_patched.step, backend=tpu_backend, fullgraph=True
    )

    # 5. Run steps
    for x in inputs:
      # Eager
      opt_eager.zero_grad()
      model_step_eager(x, model_eager)
      opt_eager.step()

      # Patched compiled
      opt_patched.zero_grad()
      compiled_model_step_patched(x, model_patched)
      compiled_opt_patched_step()

    # 6. Check weights parity
    for p_eager, p_patched in zip(
        model_eager.parameters(),
        model_patched.parameters(),
    ):
      tpu_utils.assert_close(p_patched, p_eager, atol=1e-5, rtol=1e-4)

    # 7. Check that weights actually updated (avoiding false positive match
    # on unmodified weights)
    for p_eager, p_init in zip(model_eager.parameters(), initial_weights):
      self.assertFalse(torch.allclose(p_eager, p_init, atol=1e-4))


if __name__ == '__main__':
  absltest.main()

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

"""Tests for TorchAX optimizer mapping."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch import nn
from perflab.torchax import optimizer
from tests import seed_test_utils

_NUM_STEPS = 3


def _run_torch_and_optax(
    torch_optimizer_cls: type[torch.optim.Optimizer],
    grads: list[np.ndarray],
    init_param: np.ndarray,
    **optimizer_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
  """Applies the same gradients with torch and the converted optax optimizer.

  Args:
    torch_optimizer_cls: The torch optimizer class to convert.
    grads: The gradient to apply at each step.
    init_param: The initial parameter value.
    **optimizer_kwargs: Hyperparameters for the torch optimizer.

  Returns:
    The final (torch, optax) parameter values.
  """
  param = nn.Parameter(torch.tensor(init_param))
  torch_opt = torch_optimizer_cls([param], **optimizer_kwargs)
  optax_opt = optimizer.get_optax_optimizer(torch_opt)

  jax_param = jnp.asarray(init_param)
  opt_state = optax_opt.init(jax_param)
  for grad in grads:
    param.grad = torch.tensor(grad)
    torch_opt.step()

    updates, opt_state = optax_opt.update(
        jnp.asarray(grad), opt_state, jax_param
    )
    jax_param = optax.apply_updates(jax_param, updates)

  return param.detach().numpy(), np.asarray(jax_param)


class OptimizerTest(seed_test_utils.RepeatableTest):

  @parameterized.named_parameters(
      dict(
          testcase_name="adam",
          torch_optimizer_cls=torch.optim.Adam,
          optimizer_kwargs=dict(lr=0.05, betas=(0.8, 0.95), eps=1e-3),
      ),
      dict(
          testcase_name="adamw",
          torch_optimizer_cls=torch.optim.AdamW,
          optimizer_kwargs=dict(
              lr=0.05, betas=(0.85, 0.98), eps=1e-3, weight_decay=0.2
          ),
      ),
      dict(
          testcase_name="adamw_default_weight_decay",
          torch_optimizer_cls=torch.optim.AdamW,
          optimizer_kwargs=dict(lr=0.05),
      ),
  )
  def test_optax_steps_match_torch(self, torch_optimizer_cls, optimizer_kwargs):
    rng = np.random.default_rng(0)
    init_param = rng.standard_normal((4, 3)).astype(np.float32)
    grads = [
        rng.standard_normal((4, 3)).astype(np.float32)
        for _ in range(_NUM_STEPS)
    ]

    torch_param, optax_param = _run_torch_and_optax(
        torch_optimizer_cls, grads, init_param, **optimizer_kwargs
    )

    # The step must actually move the parameters for the comparison to be
    # meaningful.
    self.assertGreater(np.abs(torch_param - init_param).max(), 1e-2)
    np.testing.assert_allclose(optax_param, torch_param, rtol=1e-5, atol=1e-6)

  def test_get_optax_optimizer_unsupported_raises(self):
    model = nn.Linear(2, 2)
    torch_sgd = torch.optim.SGD(model.parameters(), lr=0.01)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=benchmark
        ValueError, "Unsupported torch optimizer class: SGD"
    ):
      optimizer.get_optax_optimizer(torch_sgd)


if __name__ == "__main__":
  absltest.main()

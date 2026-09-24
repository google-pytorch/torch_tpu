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

"""Unit tests for stateless SGD optimizers on TPU comparing against official PyTorch optimizers."""

from absl.testing import absltest
import torch
from torch_tpu._internal.utils import test_utils as assert_utils
from torch_tpu.extras.optimizers import sgd
from tests import seed_test_utils
from tests.extras.optimizers import test_utils


class SGDOptimizersTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def _validate_opt_step(
      self, custom_opt, lr=1e-2, momentum=0.9, weight_decay=0.0
  ):
    params_custom, grads_custom = test_utils.make_dummy_params(self.device)
    params_ref, grads_ref = test_utils.make_dummy_params(self.device)

    pg = custom_opt.init_param_group(params_custom)
    new_pg = custom_opt.step(pg, grads_custom)

    ref_params = {
        k: torch.nn.Parameter(v.clone().detach()) for k, v in params_ref.items()
    }
    ref_opt = torch.optim.SGD(
        list(ref_params.values()),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        dampening=custom_opt.dampening,
        nesterov=custom_opt.nesterov,
    )
    for k, p in ref_params.items():
      p.grad = grads_ref[k].clone().detach()
    ref_opt.step()

    for k in params_ref:
      assert_utils.assert_close(
          new_pg.params[k],
          ref_params[k].detach(),
          rtol=1e-4,
          atol=1e-4,
          preamble=f"Mismatch in param {k}",
      )
      if momentum != 0.0:
        ref_m = ref_opt.state[ref_params[k]]["momentum_buffer"]
        assert_utils.assert_close(
            new_pg.opt_state_m[k],
            ref_m,
            rtol=1e-4,
            atol=1e-4,
            preamble=f"Mismatch in opt_state_m for {k}",
        )

  def test_reference_sgd_step(self):
    opt = sgd.ReferenceSgd(lr=1e-2, momentum=0.9)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.9)

  def test_reference_sgd_no_momentum_step(self):
    opt = sgd.ReferenceSgd(lr=1e-2, momentum=0.0)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.0)

  def test_reference_sgd_weight_decay_step(self):
    opt = sgd.ReferenceSgd(lr=1e-2, momentum=0.9, weight_decay=1e-2)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.9, weight_decay=1e-2)

  def test_reference_sgd_nesterov_step(self):
    opt = sgd.ReferenceSgd(lr=1e-2, momentum=0.9, nesterov=True)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.9)

  def test_reference_sgd_dampening_step(self):
    opt = sgd.ReferenceSgd(lr=1e-2, momentum=0.9, dampening=0.1)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.9)

  def test_reference_sgd_compiled_linear_train_step(self):
    opt = sgd.ReferenceSgd(lr=1e-2, momentum=0.9)
    test_utils.validate_compiled_linear_train_step(self, opt, self.device)

  def test_torch_sgd_step(self):
    opt = sgd.TorchSgd(lr=1e-2, momentum=0.9)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.9)

  def test_torch_sgd_no_momentum_step(self):
    opt = sgd.TorchSgd(lr=1e-2, momentum=0.0)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.0)

  def test_torch_sgd_custom_kwargs(self):
    opt = sgd.TorchSgd(
        lr=1e-2,
        momentum=0.9,
        fused=False,
        foreach=False,
        maximize=False,
    )
    self._validate_opt_step(opt, lr=1e-2, momentum=0.9)

  def test_torch_sgd_compiled_linear_train_step(self):
    opt = sgd.TorchSgd(lr=1e-2, momentum=0.9)
    test_utils.validate_compiled_linear_train_step(self, opt, self.device)


if __name__ == "__main__":
  absltest.main()

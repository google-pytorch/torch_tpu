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

"""Unit tests for stateless Muon optimizers on TPU comparing against official PyTorch optimizers."""

from absl.testing import absltest
import torch
from torch.tpu import optimizers
from torch_tpu._internal.optimizers import muon
from torch_tpu._internal.utils import test_utils as assert_utils
from tests import seed_test_utils
from tests.optimizers import test_utils


class MuonOptimizersTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def _validate_opt_step(
      self,
      custom_opt,
      lr=1e-3,
      weight_decay=0.1,
      momentum=0.95,
  ):
    params_custom, grads_custom = test_utils.make_dummy_muon_params(self.device)
    params_ref, grads_ref = test_utils.make_dummy_muon_params(self.device)

    pg = custom_opt.init_param_group(params_custom)
    new_pg = custom_opt.step(pg, grads_custom)

    ref_params = {
        k: torch.nn.Parameter(v.clone().detach()) for k, v in params_ref.items()
    }
    ref_opt = torch.optim.Muon(
        list(ref_params.values()),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=custom_opt.defaults["nesterov"],
        ns_coefficients=custom_opt.defaults["ns_coefficients"],
        eps=custom_opt.defaults["eps"],
        ns_steps=custom_opt.defaults["ns_steps"],
        adjust_lr_fn=custom_opt.defaults["adjust_lr_fn"],
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
      ref_m = ref_opt.state[ref_params[k]]["momentum_buffer"]
      assert_utils.assert_close(
          new_pg.opt_state_m[k],
          ref_m,
          rtol=1e-4,
          atol=1e-4,
          preamble=f"Mismatch in opt_state_m for {k}",
      )

  def test_reference_muon_step(self):
    opt = muon.ReferenceMuon(lr=1e-3, momentum=0.95, weight_decay=0.1)
    self._validate_opt_step(opt, lr=1e-3, momentum=0.95, weight_decay=0.1)

  def test_reference_muon_compiled_linear_train_step(self):
    opt = muon.ReferenceMuon(lr=1e-3, momentum=0.95, weight_decay=0.1)
    test_utils.validate_compiled_linear_train_step(
        self, opt, self.device, bias=False
    )

  def test_torch_muon_step(self):
    opt = optimizers.TorchMuon(lr=1e-3, momentum=0.95, weight_decay=0.1)
    self._validate_opt_step(opt, lr=1e-3, momentum=0.95, weight_decay=0.1)

  def test_torch_muon_custom_kwargs(self):
    opt = optimizers.TorchMuon(
        lr=1e-3,
        momentum=0.95,
        weight_decay=0.1,
        nesterov=False,
        adjust_lr_fn="match_rms_adamw",
    )
    self._validate_opt_step(opt, lr=1e-3, momentum=0.95, weight_decay=0.1)

  def test_torch_muon_compiled_linear_train_step(self):
    opt = optimizers.TorchMuon(lr=1e-3, momentum=0.95, weight_decay=0.1)
    test_utils.validate_compiled_linear_train_step(
        self, opt, self.device, bias=False
    )

  def test_defaults_are_read_only(self):
    for opt in (
        optimizers.TorchMuon(lr=1e-2, momentum=0.5),
        muon.ReferenceMuon(lr=1e-2, momentum=0.5),
    ):
      with self.subTest(type(opt).__name__):
        self.assertEqual(opt.defaults["lr"], 1e-2)
        self.assertEqual(opt.defaults["momentum"], 0.5)
        self.assertFalse(hasattr(opt, "lr"))
        with self.assertRaises(  # ASSERT_RAISES_OK=Python attribute check
            TypeError
        ):
          opt.defaults["lr"] = 1e-3  # pytype: disable=unsupported-operands


if __name__ == "__main__":
  absltest.main()

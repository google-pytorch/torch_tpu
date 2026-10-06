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

"""Unit tests for stateless AdamW optimizers on TPU comparing against official PyTorch optimizers."""

from absl.testing import absltest
import torch
from torch.tpu import optimizers
from torch_tpu._internal.optimizers import adamw
from torch_tpu._internal.utils import test_utils as assert_utils
from tests import seed_test_utils
from tests.optimizers import test_utils


class AdamWOptimizersTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def _validate_opt_step(self, custom_opt, lr=1e-3, weight_decay=1e-2):
    params_custom, grads_custom = test_utils.make_dummy_params(self.device)
    params_ref, grads_ref = test_utils.make_dummy_params(self.device)

    pg = custom_opt.init_param_group(params_custom)
    new_pg = custom_opt.step(pg, grads_custom)

    ref_params = {
        k: torch.nn.Parameter(v.clone().detach()) for k, v in params_ref.items()
    }
    ref_opt = torch.optim.AdamW(
        list(ref_params.values()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(custom_opt.defaults["beta1"], custom_opt.defaults["beta2"]),
        eps=custom_opt.defaults["eps"],
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
      ref_m = ref_opt.state[ref_params[k]]["exp_avg"]
      assert_utils.assert_close(
          new_pg.opt_state_m[k],
          ref_m,
          rtol=1e-4,
          atol=1e-4,
          preamble=f"Mismatch in opt_state_m for {k}",
      )
      ref_v = ref_opt.state[ref_params[k]]["exp_avg_sq"]
      assert_utils.assert_close(
          new_pg.opt_state_v[k],
          ref_v,
          rtol=1e-4,
          atol=1e-4,
          preamble=f"Mismatch in opt_state_v for {k}",
      )
      ref_step = ref_opt.state[ref_params[k]]["step"]
      ref_step_t = (
          ref_step.cpu()
          if isinstance(ref_step, torch.Tensor)
          else torch.tensor(float(ref_step))
      )
      assert_utils.assert_close(
          new_pg.opt_steps[k].cpu(),
          ref_step_t,
          rtol=1e-4,
          atol=1e-4,
          preamble=f"Mismatch in opt_steps for {k}",
      )

  def test_reference_adamw_step(self):
    opt = adamw.ReferenceAdamW(
        lr=1e-3, weight_decay=1e-2, use_bfloat16_moments=False
    )
    self._validate_opt_step(opt, lr=1e-3, weight_decay=1e-2)

  def test_torch_adamw_step(self):
    opt = optimizers.TorchAdamW(
        lr=1e-3, weight_decay=1e-2, use_bfloat16_moments=False
    )
    self._validate_opt_step(opt, lr=1e-3, weight_decay=1e-2)

  def test_torch_adamw_custom_kwargs(self):
    opt = optimizers.TorchAdamW(
        lr=1e-3,
        weight_decay=1e-2,
        use_bfloat16_moments=False,
        fused=False,
        foreach=False,
        maximize=False,
        differentiable=False,
    )
    self._validate_opt_step(opt, lr=1e-3, weight_decay=1e-2)

  def test_torch_adamw_compiled_linear_train_step(self):
    opt = optimizers.TorchAdamW(
        lr=1e-3, weight_decay=1e-2, use_bfloat16_moments=False
    )
    test_utils.validate_compiled_linear_train_step(self, opt, self.device)

  def test_defaults_are_read_only(self):
    for opt in (
        optimizers.TorchAdamW(lr=1e-2, beta1=0.5),
        adamw.ReferenceAdamW(lr=1e-2, beta1=0.5),
    ):
      with self.subTest(type(opt).__name__):
        self.assertEqual(opt.defaults["lr"], 1e-2)
        self.assertEqual(opt.defaults["beta1"], 0.5)
        self.assertFalse(hasattr(opt, "lr"))
        with self.assertRaises(  # ASSERT_RAISES_OK=Python attribute check
            TypeError
        ):
          opt.defaults["lr"] = 1e-3  # pytype: disable=unsupported-operands


if __name__ == "__main__":
  absltest.main()

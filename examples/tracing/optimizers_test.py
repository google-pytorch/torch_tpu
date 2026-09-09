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

"""Unit tests for stateless functional optimizers on TPU comparing against official PyTorch optimizers."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree
from torch_tpu._internal.compile.compiler import StaticCompiler
from torch_tpu._internal.utils import test_utils
from examples.tracing import optimizers
from torch_tpu._internal.benchmarks import xprof_adapter
from tests import seed_test_utils


def _make_dummy_params(device: torch.device):
  p1 = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32, device=device)
  p2 = torch.tensor(
      [[0.5, -0.5], [1.5, 0.0]], dtype=torch.float32, device=device
  )
  g1 = torch.tensor([0.1, -0.2, 0.5], dtype=torch.float32, device=device)
  g2 = torch.tensor(
      [[-0.1, 0.2], [0.3, -0.4]], dtype=torch.float32, device=device
  )
  params = {"weight": p1, "bias": p2}
  grads = {"weight": g1, "bias": g2}
  return params, grads


def _validate_compiled_linear_train_step(test_case, custom_opt, device):
  model = torch.nn.Linear(128, 64).to(device=device, dtype=torch.float32)
  model.train()

  pg = custom_opt.init_param_group(model)
  x = torch.randn((32, 128), dtype=torch.float32, device=device)
  target = torch.randn((32, 64), dtype=torch.float32, device=device)

  orig_params = {k: v.detach().cpu().clone() for k, v in pg.params.items()}

  def stateless_train_step(param_group, x, target):
    def loss_fn(p):
      p_orig = {k.replace("_", "."): v for k, v in p.items()}
      out = torch.func.functional_call(model, p_orig, (x,))
      return torch.nn.functional.mse_loss(out, target)

    grads, loss = torch.func.grad_and_value(loss_fn)(param_group.params)
    new_param_group = custom_opt.step(param_group, grads)
    return loss, new_param_group

  flat_inputs, in_spec = _pytree.tree_flatten((pg, x, target))

  dummy_loss = torch.tensor(0.0, device=device)
  _, out_spec = _pytree.tree_flatten((dummy_loss, pg))

  def flattened_stateless_train_step(*flat_args):
    structured_args = _pytree.tree_unflatten(flat_args, in_spec)
    p_group, inps, tgts = structured_args
    loss, new_p_group = stateless_train_step(p_group, inps, tgts)
    flat_outputs, _ = _pytree.tree_flatten((loss, new_p_group))
    return tuple(flat_outputs)

  unified_graph = make_fx(flattened_stateless_train_step)(*flat_inputs)
  compiled_executable = StaticCompiler()(unified_graph, flat_inputs)

  session = xprof_adapter.XprofSession()
  session.start_session(host_trace_level=3, enable_python_tracer=True)
  try:
    with xprof_adapter.TraceMe("ExecuteCompiledStep"):
      # Step 1
      flat_inputs, _ = _pytree.tree_flatten((pg, x, target))
      result = compiled_executable(*flat_inputs)
      loss, new_pg = _pytree.tree_unflatten(result, out_spec)

      # Step 2
      flat_inputs, _ = _pytree.tree_flatten((new_pg, x, target))
      result = compiled_executable(*flat_inputs)
      loss, new_pg = _pytree.tree_unflatten(result, out_spec)

      torch.accelerator.synchronize()
  finally:
    xprof_url = session.end_session_and_get_url()
    print(f"Xprof URL: {xprof_url}")

  test_case.assertIsNotNone(loss)
  test_case.assertFalse(torch.isnan(loss))
  test_case.assertFalse(torch.isnan(new_pg.params["weight"]).any())
  for k, orig_p in orig_params.items():
    test_case.assertFalse(
        torch.allclose(new_pg.params[k].cpu(), orig_p),
        msg=f"Param {k} was not updated post train step",
    )
  for k, step_t in new_pg.opt_steps.items():
    test_case.assertGreater(
        step_t.item(), 1.0, msg=f"Opt step {k} was not incremented"
    )


class AdamWOptimizersTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def _validate_opt_step(self, custom_opt, lr=1e-3, weight_decay=1e-2):
    params_custom, grads_custom = _make_dummy_params(self.device)
    params_ref, grads_ref = _make_dummy_params(self.device)

    pg = custom_opt.init_param_group(params_custom)
    new_pg = custom_opt.step(pg, grads_custom)

    ref_params = {
        k: torch.nn.Parameter(v.clone().detach()) for k, v in params_ref.items()
    }
    ref_opt = torch.optim.AdamW(
        list(ref_params.values()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(custom_opt.beta1, custom_opt.beta2),
        eps=custom_opt.eps,
    )
    for k, p in ref_params.items():
      p.grad = grads_ref[k].clone().detach()
    ref_opt.step()

    for k in params_ref:
      test_utils.assert_close(
          new_pg.params[k],
          ref_params[k].detach(),
          rtol=1e-4,
          atol=1e-4,
          preamble=f"Mismatch in param {k}",
      )
      ref_m = ref_opt.state[ref_params[k]]["exp_avg"]
      test_utils.assert_close(
          new_pg.opt_state_m[k],
          ref_m,
          rtol=1e-4,
          atol=1e-4,
          preamble=f"Mismatch in opt_state_m for {k}",
      )
      ref_v = ref_opt.state[ref_params[k]]["exp_avg_sq"]
      test_utils.assert_close(
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
      test_utils.assert_close(
          new_pg.opt_steps[k].cpu(),
          ref_step_t,
          rtol=1e-4,
          atol=1e-4,
          preamble=f"Mismatch in opt_steps for {k}",
      )

  def test_reference_adamw_step(self):
    opt = optimizers.ReferenceAdamw(
        lr=1e-3, weight_decay=1e-2, use_bfloat16_moments=False
    )
    self._validate_opt_step(opt, lr=1e-3, weight_decay=1e-2)

  def test_torch_adamw_step(self):
    opt = optimizers.TorchAdamw(
        lr=1e-3, weight_decay=1e-2, use_bfloat16_moments=False
    )
    self._validate_opt_step(opt, lr=1e-3, weight_decay=1e-2)

  def test_torch_adamw_custom_kwargs(self):
    opt = optimizers.TorchAdamw(
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
    opt = optimizers.TorchAdamw(
        lr=1e-3, weight_decay=1e-2, use_bfloat16_moments=False
    )
    _validate_compiled_linear_train_step(self, opt, self.device)


class SGDOptimizersTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def _validate_opt_step(
      self, custom_opt, lr=1e-2, momentum=0.9, weight_decay=0.0
  ):
    params_custom, grads_custom = _make_dummy_params(self.device)
    params_ref, grads_ref = _make_dummy_params(self.device)

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
      test_utils.assert_close(
          new_pg.params[k],
          ref_params[k].detach(),
          rtol=1e-4,
          atol=1e-4,
          preamble=f"Mismatch in param {k}",
      )
      if momentum != 0.0:
        ref_m = ref_opt.state[ref_params[k]]["momentum_buffer"]
        test_utils.assert_close(
            new_pg.opt_state_m[k],
            ref_m,
            rtol=1e-4,
            atol=1e-4,
            preamble=f"Mismatch in opt_state_m for {k}",
        )

  def test_reference_sgd_step(self):
    opt = optimizers.ReferenceSgd(lr=1e-2, momentum=0.9)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.9)

  def test_reference_sgd_no_momentum_step(self):
    opt = optimizers.ReferenceSgd(lr=1e-2, momentum=0.0)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.0)

  def test_reference_sgd_weight_decay_step(self):
    opt = optimizers.ReferenceSgd(lr=1e-2, momentum=0.9, weight_decay=1e-2)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.9, weight_decay=1e-2)

  def test_reference_sgd_nesterov_step(self):
    opt = optimizers.ReferenceSgd(lr=1e-2, momentum=0.9, nesterov=True)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.9)

  def test_reference_sgd_dampening_step(self):
    opt = optimizers.ReferenceSgd(lr=1e-2, momentum=0.9, dampening=0.1)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.9)

  def test_reference_sgd_compiled_linear_train_step(self):
    opt = optimizers.ReferenceSgd(lr=1e-2, momentum=0.9)
    _validate_compiled_linear_train_step(self, opt, self.device)

  def test_torch_sgd_step(self):
    opt = optimizers.TorchSgd(lr=1e-2, momentum=0.9)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.9)

  def test_torch_sgd_no_momentum_step(self):
    opt = optimizers.TorchSgd(lr=1e-2, momentum=0.0)
    self._validate_opt_step(opt, lr=1e-2, momentum=0.0)

  def test_torch_sgd_custom_kwargs(self):
    opt = optimizers.TorchSgd(
        lr=1e-2,
        momentum=0.9,
        fused=False,
        foreach=False,
        maximize=False,
    )
    self._validate_opt_step(opt, lr=1e-2, momentum=0.9)

  def test_torch_sgd_compiled_linear_train_step(self):
    opt = optimizers.TorchSgd(lr=1e-2, momentum=0.9)
    _validate_compiled_linear_train_step(self, opt, self.device)


if __name__ == "__main__":
  absltest.main()

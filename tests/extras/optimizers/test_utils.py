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

"""Shared test utilities for TPU tracing optimizers."""

from typing import Any

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.benchmarks import xprof_adapter


def make_dummy_params(
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
  """Creates dummy model parameters and gradients for 1D/2D tests."""
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


_make_dummy_params = make_dummy_params


def make_dummy_muon_params(
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
  """Creates dummy 2D model parameters and gradients for Muon tests."""
  p1 = torch.tensor(
      [[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]], dtype=torch.float32, device=device
  )
  p2 = torch.tensor(
      [[0.5, -0.5], [1.5, 0.0]], dtype=torch.float32, device=device
  )
  g1 = torch.tensor(
      [[0.1, -0.2, 0.5], [-0.1, 0.2, -0.3]], dtype=torch.float32, device=device
  )
  g2 = torch.tensor(
      [[-0.1, 0.2], [0.3, -0.4]], dtype=torch.float32, device=device
  )
  params = {"w1": p1, "w2": p2}
  grads = {"w1": g1, "w2": g2}
  return params, grads


_make_dummy_muon_params = make_dummy_muon_params


def validate_compiled_linear_train_step(
    test_case: Any, custom_opt: Any, device: torch.device, bias: bool = True
) -> None:
  """Validates compiled linear training step execution on TPU with custom optimizer."""
  model = torch.nn.Linear(128, 64, bias=bias).to(
      device=device, dtype=torch.float32
  )
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
  compiled_executable = compiler.StaticCompiler()(unified_graph, flat_inputs)

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


_validate_compiled_linear_train_step = validate_compiled_linear_train_step

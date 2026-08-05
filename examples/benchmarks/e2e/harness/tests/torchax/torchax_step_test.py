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

"""Tests for torchax_step."""

import sys

if sys.version_info >= (3, 14):
  # Torchax is not supported on Python 3.14+.
  sys.exit(0)

from unittest import mock

from absl.testing import absltest
from tests import seed_test_utils
import torch
import torch.nn as nn
from examples.benchmarks.e2e.harness import compile as compile_lib
from examples.benchmarks.e2e.harness import context as context_lib
from examples.benchmarks.e2e.harness import registry as registry_lib
from examples.benchmarks.e2e.harness import step_lib
from examples.benchmarks.e2e.harness import target as target_lib
from examples.benchmarks.e2e.harness.torchax import torchax_step
import torchax


class DummyModelOutput:

  def __init__(self, logits=None, loss=None, to_tuple_data=None):
    self.logits = logits
    self.loss = loss
    self._to_tuple_data = to_tuple_data

  def to_tuple(self):
    return self._to_tuple_data


class DummyModule(nn.Module):

  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(4, 2)

  def forward(self, x):
    return self.linear(x)


class TorchaxStepTest(seed_test_utils.RepeatableTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    torchax.enable_globally()
    torchax.enable_performance_mode()

  def test_prepare_inputs_for_torchax(self):
    with torchax.default_env():
      t = torch.randn(2, 3)
      inputs = {"a": t, "b": [t, 42, "hello"]}
      res = torchax_step._prepare_inputs_for_torchax(inputs)

      self.assertIsInstance(res["a"], torchax.tensor.Tensor)
      self.assertIsInstance(res["b"][0], torchax.tensor.Tensor)
      self.assertEqual(res["b"][1], 42)
      self.assertEqual(res["b"][2], "hello")

  def test_extract_jax_compatible_output(self):
    t1 = torch.randn(2, 2)
    t2 = torch.randn(2, 2)

    self.assertIsNone(torchax_step._extract_jax_compatible_output(None))
    self.assertIs(torchax_step._extract_jax_compatible_output(t1), t1)

    # Output with .logits attribute
    out_logits = DummyModelOutput(logits=t1)
    self.assertIs(torchax_step._extract_jax_compatible_output(out_logits), t1)

    # Output with .loss attribute
    out_loss = DummyModelOutput(loss=t2)
    self.assertIs(torchax_step._extract_jax_compatible_output(out_loss), t2)

    # Output with .to_tuple() returning single element vs multiple
    out_tuple1 = DummyModelOutput(to_tuple_data=(t1, None))
    self.assertIs(torchax_step._extract_jax_compatible_output(out_tuple1), t1)

    out_tuple2 = DummyModelOutput(to_tuple_data=(t1, t2))
    res_tuple2 = torchax_step._extract_jax_compatible_output(out_tuple2)
    self.assertEqual(res_tuple2, (t1, t2))

    # Output as dict or list/tuple
    res_dict = torchax_step._extract_jax_compatible_output({"a": t1, "b": None})
    self.assertEqual(res_dict, {"a": t1})

    res_list = torchax_step._extract_jax_compatible_output([t1, None])
    self.assertIs(res_list, t1)

    # Unsupported output
    self.assertIsNone(torchax_step._extract_jax_compatible_output("string"))

  def test_call_functional_model(self):
    mock_model = mock.MagicMock()
    t = torch.randn(2, 2)
    mock_model.functional_call.return_value = t

    params = {"w": torch.randn(2, 2)}
    buffers = {}

    # Dict args
    out1 = torchax_step._call_functional_model(
        mock_model, params, buffers, {"x": t}, None
    )
    self.assertIs(out1, t)
    mock_model.functional_call.assert_called_with(
        "forward", params, buffers, x=t
    )

    # Tuple args with kwargs
    mock_model.reset_mock()
    out2 = torchax_step._call_functional_model(
        mock_model, params, buffers, (t,), {"y": 1}
    )
    self.assertIs(out2, t)
    mock_model.functional_call.assert_called_with(
        "forward", params, buffers, t, y=1
    )

    # Single non-sequence arg
    mock_model.reset_mock()
    out3 = torchax_step._call_functional_model(
        mock_model, params, buffers, t, None
    )
    self.assertIs(out3, t)
    mock_model.functional_call.assert_called_with("forward", params, buffers, t)

  def test_default_loss_fn(self):
    t_scalar = torch.tensor(2.5)
    t_2d = torch.tensor([[1.0, 3.0], [2.0, 4.0]])

    # From attribute .loss
    out_obj = DummyModelOutput(loss=t_scalar)
    self.assertTrue(
        torch.equal(torchax_step._default_loss_fn(out_obj), t_scalar)
    )

    # From dict key 'loss'
    out_dict = {"loss": t_2d}
    self.assertEqual(torchax_step._default_loss_fn(out_dict).item(), 2.5)

    # From tuple/list first item
    out_list = [t_2d, "extra"]
    self.assertEqual(torchax_step._default_loss_fn(out_list).item(), 2.5)

    # Direct tensor
    self.assertEqual(torchax_step._default_loss_fn(t_2d).item(), 2.5)

    # Non-extractable object raises ValueError
    with self.assertRaises(ValueError):
      torchax_step._default_loss_fn("not_a_loss")

    # Extracted value not a Tensor raises TypeError
    with self.assertRaises(TypeError):
      torchax_step._default_loss_fn({"loss": 123})

  def test_device_kind_proxy_and_factory_context(self):
    target = target_lib.make_target(platform=target_lib.Platform.V5E_1X1)
    orig_ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    factory_ctx = torchax_step._TorchaxFactoryContext(orig_ctx)
    self.assertEqual(factory_ctx.device_kind.value, "cpu")
    self.assertEqual(factory_ctx.device_kind, orig_ctx.device_kind)
    self.assertEqual(factory_ctx.target, orig_ctx.target)

  def test_torchax_forward_stepper(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    def forward_factory(factory_ctx):
      model = DummyModule()
      x = torch.randn(2, 4)
      return model, (x,), {}

    spec = registry_lib.BenchmarkSpec(
        name="test_forward",
        factory=forward_factory,
        stepper=step_lib.StepperType.FORWARD,
    )

    wrapper = torchax_step.TorchaxForwardStepper()
    wrapper.init_with_benchmark_args(spec, ctx)

    # Test step execution before compile
    step_fn = wrapper.get_step_fn()
    out = step_fn()
    self.assertIsNotNone(out)

    # Test compile with STEP scope
    compile_config = compile_lib.CompileConfig(scope=compile_lib.Scope.STEP)
    wrapper.compile(compile_config)
    step_fn_compiled = wrapper.get_step_fn()
    out_compiled = step_fn_compiled()
    self.assertIsNotNone(out_compiled)

    # Test compile with invalid scope raises ValueError
    with self.assertRaises(ValueError):
      wrapper.compile(compile_lib.CompileConfig(scope=compile_lib.Scope.MODEL))

  def test_torchax_backward_stepper(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    class DummyTrainModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

      def forward(self, x):
        return self.linear(x)

    def train_factory(factory_ctx):
      model = DummyTrainModule()
      x = torch.randn(2, 4)
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      return model, (x,), {}, optimizer

    spec = registry_lib.BenchmarkSpec(
        name="test_train",
        factory=train_factory,
        stepper=step_lib.StepperType.TRAINING,
    )

    wrapper = torchax_step.TorchaxBackwardStepper()
    wrapper.init_with_benchmark_args(spec, ctx)

    # Step before compile
    step_fn = wrapper.get_step_fn()
    loss, new_weights, new_opt_state = step_fn()
    self.assertIsNotNone(loss)
    self.assertIsNotNone(new_weights)
    self.assertIsNotNone(new_opt_state)

    # Compile with STEP scope
    compile_config = compile_lib.CompileConfig(scope=compile_lib.Scope.STEP)
    wrapper.compile(compile_config)
    step_fn_compiled = wrapper.get_step_fn()
    loss_c, new_weights_c, new_opt_state_c = step_fn_compiled()
    self.assertIsNotNone(loss_c)
    self.assertIsNotNone(new_weights_c)
    self.assertIsNotNone(new_opt_state_c)

    # Test compile with invalid scope raises ValueError
    with self.assertRaises(ValueError):
      wrapper.compile(compile_lib.CompileConfig(scope=compile_lib.Scope.MODEL))

  def test_resolve_torchax_stepper(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    forward_spec = registry_lib.BenchmarkSpec(
        name="test_forward",
        factory=lambda f_ctx: (DummyModule(), (torch.randn(2, 4),), {}),
        stepper=step_lib.StepperType.FORWARD,
    )
    train_spec = registry_lib.BenchmarkSpec(
        name="test_train",
        factory=lambda f_ctx: (
            DummyModule(),
            (torch.randn(2, 4),),
            {},
            torch.optim.Adam(DummyModule().parameters()),
        ),
        stepper=step_lib.StepperType.TRAINING,
    )

    # Resolution by StepperType
    forward_stepper = torchax_step.resolve_torchax_stepper(
        step_lib.StepperType.FORWARD
    )
    self.assertIsInstance(forward_stepper, step_lib.Stepper)
    self.assertIsInstance(forward_stepper, torchax_step.TorchaxForwardStepper)
    forward_stepper.init_with_benchmark_args(forward_spec, ctx)

    backward_stepper = torchax_step.resolve_torchax_stepper(
        step_lib.StepperType.TRAINING
    )
    self.assertIsInstance(backward_stepper, step_lib.Stepper)
    self.assertIsInstance(backward_stepper, torchax_step.TorchaxBackwardStepper)
    backward_stepper.init_with_benchmark_args(train_spec, ctx)

    # Resolution with keyword args
    backward_stepper3 = torchax_step.resolve_torchax_stepper(
        step_lib.StepperType.TRAINING, accum_steps=1
    )
    self.assertIsInstance(
        backward_stepper3, torchax_step.TorchaxBackwardStepper
    )

    # Resolution with custom loss function raises ValueError
    with self.assertRaises(ValueError):
      torchax_step.resolve_torchax_stepper(
          step_lib.StepperType.TRAINING,
          accum_steps=1,
          compute_loss=lambda *args, **kwargs: torch.tensor(0.0),
      )

    # Invalid stepper raises KeyError
    with self.assertRaises(KeyError):
      torchax_step.resolve_torchax_stepper("unknown")


if __name__ == "__main__":
  absltest.main()

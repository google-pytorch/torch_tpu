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
from tests import oss_utils

if oss_utils.is_oss() and sys.version_info >= (3, 14):
  # `torchax` imports `flax`, and the `flax` version pinned for OSS fails to
  # import on Python 3.14+ with `KeyError: '__annotations__'`.
  sys.exit(0)

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tests import seed_test_utils
import torch
import torch.nn as nn
from perflab import compile as compile_lib
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import target as target_lib
from perflab.torchax import torchax_step
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

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # This test runs TorchAX on the JAX CPU backend on machines without a TPU.
    # `torch.accelerator.is_available()` still returns true there, so the
    # `torch.accelerator.synchronize()` call in `tearDown()` would try to
    # initialize the TPU and abort the test process.
    self.synchronize_tensors_in_tear_down = False

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

    # Output with both .loss and .logits attributes (e.g., HF CausalLMOutputWithPast
    # during training) must prioritize .loss over .logits.
    out_both = DummyModelOutput(logits=t1, loss=t2)
    self.assertIs(torchax_step._extract_jax_compatible_output(out_both), t2)

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
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
      torchax_step._default_loss_fn("not_a_loss")

    # Extracted value not a Tensor raises TypeError
    with self.assertRaises(TypeError):  # ASSERT_RAISES_OK=benchmark
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
      del factory_ctx
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
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
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
      del factory_ctx
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
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
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

    # Invalid stepper raises KeyError
    with self.assertRaises(KeyError):  # ASSERT_RAISES_OK=benchmark
      torchax_step.resolve_torchax_stepper("unknown")

  @parameterized.named_parameters(
      ("forward", step_lib.StepperType.FORWARD),
      ("training", step_lib.StepperType.TRAINING),
  )
  def test_factory_legacy_tensor_constructor(self, stepper_type):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    class LegacyTensorConstructorModule(nn.Module):

      def __init__(self):
        super().__init__()
        # `torch.Tensor(2)` fails while the TorchAX function mode is active.
        self.scale = nn.Parameter(torch.Tensor(2).uniform_())
        self.linear = nn.Linear(4, 2)

      def forward(self, x):
        return self.linear(x) * self.scale

    def factory(factory_ctx):
      del factory_ctx  # unused
      model = LegacyTensorConstructorModule()
      if stepper_type == step_lib.StepperType.TRAINING:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return model, (torch.randn(2, 4),), {}, optimizer
      return model, (torch.randn(2, 4),), {}

    spec = registry_lib.BenchmarkSpec(
        name="test_legacy_tensor_constructor",
        factory=factory,
        stepper=stepper_type,
    )

    # Exiting a nested `with env:` resets `env.enabled` to False while the
    # global modes remain active (where `torchax.disable_temporarily()` is a
    # no-op).
    with torchax.default_env():
      pass

    wrapper = torchax_step.resolve_torchax_stepper(stepper_type)
    wrapper.init_with_benchmark_args(spec, ctx)
    wrapper.compile(compile_lib.CompileConfig(scope=compile_lib.Scope.STEP))
    out = wrapper.get_step_fn()()

    if stepper_type == step_lib.StepperType.TRAINING:
      loss, new_weights, _ = out
      self.assertEqual(loss.shape, ())
      self.assertIsInstance(new_weights["scale"], torchax.tensor.Tensor)
    else:
      self.assertEqual(out.shape, (2, 2))
      self.assertIsInstance(wrapper._weights["scale"], torchax.tensor.Tensor)

  def test_factory_restores_torchax_on_exception(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    def failing_factory(factory_ctx):
      del factory_ctx  # unused
      raise ValueError("factory failed")

    spec = registry_lib.BenchmarkSpec(
        name="test_failing_factory",
        factory=failing_factory,
        stepper=step_lib.StepperType.FORWARD,
    )
    wrapper = torchax_step.TorchaxForwardStepper()
    with self.assertRaisesRegex(ValueError, "factory failed"):
      wrapper.init_with_benchmark_args(spec, ctx)
    self.assertIsInstance(torch.randn(2, device="jax"), torchax.tensor.Tensor)

  def test_parametrized_module_forward(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    class ParametrizedModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = nn.utils.parametrizations.weight_norm(nn.Linear(4, 2))

      def forward(self, x):
        return self.linear(x)

    expected = []

    def factory(factory_ctx):
      del factory_ctx  # unused
      model = ParametrizedModule()
      x = torch.randn(2, 4)
      with torch.no_grad():
        expected.append(model(x).tolist())
      return model, (x,), {}

    spec = registry_lib.BenchmarkSpec(
        name="test_parametrized",
        factory=factory,
        stepper=step_lib.StepperType.FORWARD,
    )

    wrapper = torchax_step.TorchaxForwardStepper()
    wrapper.init_with_benchmark_args(spec, ctx)
    wrapper.compile(compile_lib.CompileConfig(scope=compile_lib.Scope.STEP))
    out = wrapper.get_step_fn()()

    self.assertNotIn("linear.weight", wrapper._buffers)
    self.assertIn("linear.parametrizations.weight.original0", wrapper._weights)
    self.assertIn("linear.parametrizations.weight.original1", wrapper._weights)
    np.testing.assert_allclose(out.numpy(), expected[0], rtol=1e-5, atol=1e-6)

  def test_parametrized_module_training(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    class ParametrizedModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = nn.utils.parametrizations.weight_norm(nn.Linear(4, 2))

      def forward(self, x):
        return self.linear(x)

    def factory(factory_ctx):
      del factory_ctx  # unused
      model = ParametrizedModule()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
      return model, (torch.randn(2, 4),), {}, optimizer

    spec = registry_lib.BenchmarkSpec(
        name="test_parametrized_training",
        factory=factory,
        stepper=step_lib.StepperType.TRAINING,
    )

    wrapper = torchax_step.TorchaxBackwardStepper()
    wrapper.init_with_benchmark_args(spec, ctx)
    wrapper.compile(compile_lib.CompileConfig(scope=compile_lib.Scope.STEP))
    parametrization_inputs = (
        "linear.parametrizations.weight.original0",
        "linear.parametrizations.weight.original1",
    )
    # Copy to host memory because the step donates the weight buffers.
    weights_before = {
        k: wrapper._weights[k].numpy() for k in parametrization_inputs
    }
    _, new_weights, _ = wrapper.get_step_fn()()

    self.assertNotIn("linear.weight", wrapper._buffers)
    for k in parametrization_inputs:
      self.assertFalse(
          np.allclose(new_weights[k].numpy(), weights_before[k]),
          msg=f"{k} was not updated",
      )

  def test_unregistered_tensor_attribute_forward(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    class UnregisteredTensorModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        # A plain tensor attribute rather than a registered buffer, so
        # `model.to("jax")` does not move it.
        self.offset = torch.ones(2)

      def forward(self, x):
        return self.linear(x) + self.offset

    expected = []

    def factory(factory_ctx):
      del factory_ctx  # unused
      model = UnregisteredTensorModule()
      x = torch.randn(2, 4)
      with torch.no_grad():
        expected.append(model(x).tolist())
      return model, (x,), {}

    spec = registry_lib.BenchmarkSpec(
        name="test_unregistered_tensor_attribute",
        factory=factory,
        stepper=step_lib.StepperType.FORWARD,
    )

    wrapper = torchax_step.TorchaxForwardStepper()
    wrapper.init_with_benchmark_args(spec, ctx)
    wrapper.compile(compile_lib.CompileConfig(scope=compile_lib.Scope.STEP))
    out = wrapper.get_step_fn()()

    self.assertIsInstance(wrapper._buffers["offset"], torchax.tensor.Tensor)
    np.testing.assert_allclose(out.numpy(), expected[0], rtol=1e-5, atol=1e-6)

  def test_backward_stepper_custom_compute_loss(self):
    target = target_lib.make_target(platform=target_lib.Platform.CPU)
    ctx = context_lib.Context(
        target=target, run_scope=context_lib.RUN_SCOPE.value
    )

    def factory(factory_ctx):
      del factory_ctx  # unused
      # Reseed so that both steppers start from the same weights and inputs.
      torch.manual_seed(0)
      model = DummyModule()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      return model, (torch.randn(2, 4),), {}, optimizer

    spec = registry_lib.BenchmarkSpec(
        name="test_custom_compute_loss",
        factory=factory,
        stepper=step_lib.StepperType.TRAINING,
    )
    seen_models = []

    def negated_sum_loss(model, input_args, input_kwargs):
      seen_models.append(model)
      return -model(*input_args, **input_kwargs).sum()

    default_stepper = torchax_step.TorchaxBackwardStepper()
    default_stepper.init_with_benchmark_args(spec, ctx)
    custom_stepper = torchax_step.resolve_torchax_stepper(
        step_lib.StepperType.TRAINING, compute_loss=negated_sum_loss
    )
    custom_stepper.init_with_benchmark_args(spec, ctx)
    custom_stepper.compile(
        compile_lib.CompileConfig(scope=compile_lib.Scope.STEP)
    )
    # Copy to host memory because the compiled step donates the weight buffers.
    weights_before = {k: v.numpy() for k, v in custom_stepper._weights.items()}

    default_loss, default_weights, _ = default_stepper.get_step_fn()()
    custom_loss, custom_weights, _ = custom_stepper.get_step_fn()()

    # Check `type()` because `JittableModule` overrides `__class__` to pass
    # `isinstance()` checks for the module that it wraps.
    self.assertEqual({type(m) for m in seen_models}, {DummyModule})
    # The model has 4 outputs, so their negated sum is -4 times their mean.
    np.testing.assert_allclose(
        custom_loss.numpy(), -4 * default_loss.numpy(), rtol=1e-5, atol=1e-6
    )
    # Adam's first update is about `-lr * sign(grad)`, so the nonzero default
    # update flips sign only if the gradient is computed through
    # `compute_loss`.
    for name, before in weights_before.items():
      default_update = default_weights[name].numpy() - before
      custom_update = custom_weights[name].numpy() - before
      self.assertFalse(np.allclose(default_update, 0), msg=name)
      np.testing.assert_allclose(
          custom_update, -default_update, rtol=1e-4, atol=1e-6, err_msg=name
      )


if __name__ == "__main__":
  absltest.main()

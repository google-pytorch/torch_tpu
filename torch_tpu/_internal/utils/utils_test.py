# Copyright 2025 Google LLC
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

from collections.abc import Sequence
from typing import Any

from absl.testing import absltest
import torch
from torch_tpu._internal import dynamism
from torch_tpu._internal.utils import test_fixtures
from torch_tpu._internal.utils import utils


def fwd(
    idx: int,
    depth: int,
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...] | Any | None = None,
    /,
) -> utils.Event:
  """Helper to compactly create an ActivationTracer forward log event.

  Args:
    idx: The index of the event.
    depth: The depth of the event.
    module: The module of the event.
    args: The arguments of the event.
    output: The output of the event.

  Returns:
    An ActivationTracer forward log event.
  """
  if output is None:
    return dict(idx=idx, depth=depth, module=module, args=args)
  return dict(idx=idx, depth=depth, module=module, args=args, output=output)


def bwd(
    idx: int,
    depth: int,
    module: torch.nn.Module,
    grad_output: tuple[torch.Tensor, ...],
    grad_input: tuple[torch.Tensor, ...] | None = None,
    /,
) -> utils.Event:
  """Helper to compactly create an ActivationTracer backward log event.

  Args:
    idx: The index of the event.
    depth: The depth of the event.
    module: The module of the event.
    grad_output: The gradient output of the event.
    grad_input: The gradient inputs of the event.

  Returns:
    An ActivationTracer backward log event.
  """
  if grad_input is None:
    return dict(idx=idx, depth=depth, module=module, grad_output=grad_output)
  return dict(
      idx=idx,
      depth=depth,
      module=module,
      grad_output=grad_output,
      grad_input=grad_input,
  )


class AllTest(absltest.TestCase):
  """Tests various functions in utils.py."""

  def setUp(self):
    super().setUp()
    self.device = torch.device("xla_cpu")

  def test_compare_strict_mode(self):
    t_tpu = torch.arange(8).reshape((4, 2)).to(torch.float32)
    t_cpu = torch.arange(8).reshape((4, 2)).to(torch.float32)
    utils.assert_close(t_tpu, t_cpu)

    t_cpu[0][-1] += 0.1
    with self.assertRaises(AssertionError) as cm:
      utils.assert_close(t_tpu, t_cpu, rtol=1e-6, atol=1e-6)
    expected_msg = """    at index (0, 1), expected=1.100000023841858, actual=1.0, relative diff=0.09090910851955414, diff=0.10000002384185791
    at index (0, 1), expected=1.100000023841858, actual=1.0, relative diff=0.09090910851955414, diff=0.10000002384185791

Expected tensor:
tensor([[0.0000, 1.1000],
        [2.0000, 3.0000],
        [4.0000, 5.0000],
        [6.0000, 7.0000]])

Actual tensor:
tensor([[0., 1.],
        [2., 3.],
        [4., 5.],
        [6., 7.]])

Tolerance Suggestions:
  Strict check failed.
  To pass in STRICT mode, you need BOTH:
    - rtol >= 0.09090910851955414 (9.1e-02)
    - atol >= 0.10000002384185791 (1.1e-01)"""
    assert expected_msg in str(cm.exception)

  def test_compare_loose_mode(self):
    t_tpu = torch.arange(8).reshape((4, 2)).to(torch.float32)
    t_cpu = torch.arange(8).reshape((4, 2)).to(torch.float32)
    utils.assert_close(t_tpu, t_cpu)

    t_cpu[0][-1] += 0.1
    with self.assertRaises(AssertionError) as cm:
      utils.assert_close(
          t_tpu,
          t_cpu,
          rtol=1e-6,
          atol=1e-6,
          check_value=utils.CheckValueMode.LOOSE,
      )
    expected_msg = """    at index (0, 1), expected=1.100000023841858, actual=1.0, relative diff=0.09090910851955414, diff=0.10000002384185791
    at index (0, 1), expected=1.100000023841858, actual=1.0, relative diff=0.09090910851955414, diff=0.10000002384185791

Expected tensor:
tensor([[0.0000, 1.1000],
        [2.0000, 3.0000],
        [4.0000, 5.0000],
        [6.0000, 7.0000]])

Actual tensor:
tensor([[0., 1.],
        [2., 3.],
        [4., 5.],
        [6., 7.]])

Tolerance Suggestions:
  Loose check failed.
  To pass in LOOSE mode, you need either:
    - rtol >= 0.09090910851955414 (9.1e-02) (with atol=0)
    - OR atol >= 0.10000002384185791 (1.1e-01) (with rtol=0)
    - OR a combination such that atol + rtol * |expected| covers the diffs."""
    assert expected_msg in str(cm.exception)

  def test_model_shlo_dyn(self):

    class AddModule(torch.nn.Module):

      def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y)

    model = AddModule().to(self.device)
    arg1 = torch.ones((128, 128))
    arg2 = torch.ones((128, 128))
    arg1 = arg1.to(self.device)
    arg2 = arg2.to(self.device)
    dynamism.mark_dynamic(arg1, 0, 2, 256)
    dynamism.mark_dynamic(arg2, 0, 2, 256)
    model_str = utils.format_model(model, arg1, arg2, shlo=True)
    self.assertIn("tensor<?x128xf32, #stablehlo.bounds<256, ?>>", model_str)

  def test_assert_close_with_callable_atol_success(self):
    t1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=self.device)
    t2 = torch.tensor([[1.1, 2.0, 3.0], [4.2, 5.0, 6.0]])
    utils.assert_close(
        actual=t1, expected=t2, atol=lambda x: 0.25 if x > 3 else 0.15
    )

  def test_assert_close_with_callable_atol_failure(self):
    t1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t2 = torch.tensor([[1.1, 2.0, 3.0], [4.2, 5.0, 6.0]], device=self.device)
    with self.assertRaises(AssertionError):
      utils.assert_close(
          actual=t1, expected=t2, atol=lambda x: 0.05 if x > 3 else 0.2
      )

  def test_model_shlo(self):
    class RoundModule(torch.nn.Module):

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x, decimals=1)

    model = RoundModule().to(self.device)
    args = torch.tensor(123.456, device=self.device)
    model_str = utils.format_model(
        model, args, params=True, pt=True, aten=True, shlo=True
    )
    self.assertIn("Parameters: 0", model_str)  # params=True
    self.assertIn("Estimated Memory", model_str)  # params=True
    self.assertIn("RoundModule()", model_str)  # pt=True
    self.assertIn("  aten.round.decimals: 1", model_str)  # aten=True
    self.assertIn("func.func @main", model_str)  # shlo=True
    self.assertIn("stablehlo.round_nearest", model_str)  # shlo=True

  def test_function_shlo(self):
    def foo(x):
      return x + 1

    args = torch.tensor(123.456)
    model_str = utils.format_model(
        foo, args, params=True, pt=True, aten=True, shlo=True
    )
    self.assertIn("Parameters: 0", model_str)  # params=True
    self.assertIn("Estimated Memory", model_str)  # params=True
    self.assertIn("class foo(torch.nn.Module)", model_str)  # pt=True
    self.assertIn("  aten.add.Tensor: 1", model_str)  # aten=True
    self.assertIn("func.func @main", model_str)  # shlo=True
    self.assertIn("stablehlo.add", model_str)  # shlo=True

  def test_function_list_inputs(self):
    def foo(x):
      return torch._foreach_sqrt(x)

    args = [torch.tensor(123.456), torch.tensor(456.789)]
    model_str = utils.format_model(
        foo, args, params=True, pt=True, aten=True, shlo=True
    )
    self.assertIn("torch.ops.aten._foreach_sqrt", model_str)  # aten=True
    self.assertIn("func.func @main", model_str)  # shlo=True
    self.assertIn("stablehlo.sqrt", model_str)  # shlo=True

  def test_input_metadata(self):
    inputs = [
        torch.tensor(1.0),
        torch.arange(10),
        torch.zeros(10, 10),
        None,
        "string_val",
        10,
    ]
    metadata = utils.InputMetadata(inputs)
    metadata_str = str(metadata)
    self.assertIn(
        "torch.tensor(...(), dtype=torch.float32, device='cpu')", metadata_str
    )
    self.assertIn(
        "torch.tensor(...(10,), dtype=torch.int64, device='cpu')", metadata_str
    )
    self.assertIn(
        "torch.tensor(...(10, 10), dtype=torch.float32, device='cpu')",
        metadata_str,
    )
    self.assertIn("None", metadata_str)
    self.assertIn("string_val", metadata_str)
    self.assertIn("10", metadata_str)

  def test_format_tensor(self):
    x = torch.Tensor([[0, 1], [2, 3]])
    tensor_str = utils.format_tensor(x)
    self.assertIn("Shape (Size):     torch.Size([2, 2])", tensor_str)
    self.assertIn("Number of Dims:   2", tensor_str)
    self.assertIn("Data Type (dtype):torch.float32", tensor_str)
    self.assertIn("Device:           cpu", tensor_str)
    self.assertIn("Requires Grad:    False", tensor_str)
    self.assertIn("Memory Layout:    torch.strided", tensor_str)
    self.assertIn("Is a Leaf:        True", tensor_str)
    self.assertIn("Memory Usage (B): 16", tensor_str)
    self.assertIn("Strides:          (2, 1)", tensor_str)
    self.assertIn("Gradient Function: None (or is a leaf tensor)", tensor_str)

    tensor_str = utils.format_tensor(x, data=True)
    self.assertIn("[0., 1.]", tensor_str)
    self.assertIn("[2., 3.]", tensor_str)

  def test_print_multiple_tensor_inputs(self):
    class MultiModule(torch.nn.Module):

      def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y)

    model = MultiModule()
    x = torch.Tensor([[0, 1], [2, 3]])
    y = torch.Tensor([[0, 1], [2, 3]])
    model_str = utils.format_model(model, x, y, params=True)
    self.assertIn("Parameters: 0, Trainable: 0", model_str)

  def test_op_tracer_as_decorator(self):
    """Tests the decorator example code in the docstring of OpTracer."""
    tracer = utils.OpTracer()

    @tracer
    def f(x):
      return x**0.5

    _ = f(torch.tensor(2.0))

    self.assertLen(tracer.ops_log, 1)
    self.assertIn("aten", tracer.ops_log)
    self.assertIn("pow", "".join(tracer.ops_log["aten"].keys()))

  def test_op_tracer_as_wrapper(self):
    """Tests the wrapper example code in the docstring of OpTracer."""

    def f(x):
      return x**0.5

    tracer = utils.OpTracer()
    wrapper = tracer(f)
    _ = wrapper(torch.tensor(2.0))

    self.assertLen(tracer.ops_log, 1)
    self.assertIn("aten", tracer.ops_log)
    self.assertIn("pow", "".join(tracer.ops_log["aten"].keys()))

  def test_op_tracer_as_context_manager(self):
    """Tests the context manager example code in the docstring of OpTracer."""
    x = torch.tensor(2.0)
    with utils.OpTracer() as tracer:
      _ = x**0.5

    self.assertLen(tracer.ops_log, 1)
    self.assertIn("aten", tracer.ops_log)
    self.assertIn("pow", "".join(tracer.ops_log["aten"].keys()))

  def test_aten_count(self):
    """Tests that the number of aten ops is counted correctly."""

    class MultiModule(torch.nn.Module):

      def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for _ in range(1000):
          x = torch.add(x, y)
        return x

    with utils.OpTracer() as tracer:
      model = MultiModule()
      model(torch.Tensor(1), torch.Tensor(1))

    # 2 empty tensor inits and 1000 adds:
    self.assertEqual(tracer.num_atens(), 1002)

  def test_print_tensor_summary(self):
    """Tests the get_tensor_summary method."""
    tensor = torch.tensor(1.0, dtype=torch.bfloat16, requires_grad=True)
    summary = utils.get_tensor_summary(tensor)
    self.assertIn("()", summary)  # Empty parens is shape of a scalar.
    self.assertIn(
        "Min: 1.00  Max: 1.00  Mean: 1.00  Var: 0.00  Grad:Y", summary
    )

  def test_print_tensor_summary_on_empty_tensor(self):
    """Tests the get_tensor_summary method."""
    tensor = torch.tensor(())
    summary = utils.get_tensor_summary(tensor)
    self.assertIn("Min:n/a", summary)

  def test_print_tensor_summary_no_data(self):
    """Tests the get_tensor_summary method."""
    tensor = torch.tensor(1.0, dtype=torch.bfloat16, requires_grad=True)
    summary = utils.get_tensor_summary(tensor, data=False)
    self.assertEqual(
        "torch.tensor(...(), dtype=torch.bfloat16, device='cpu',"
        " requires_grad=True)",
        summary,
    )

  def assert_logs_equal(
      self,
      actual: Sequence[utils.Event],
      expected: Sequence[utils.Event],
  ) -> None:
    """Helper to check equality of ActivationTracer logs.

    Args:
      actual: The actual logs.
      expected: The expected logs.

    Raises:
      absl.testing.absltest.TestCase.failureException: If the logs differ.
    """
    self.assertEqual(len(actual), len(expected))
    for actual_event, expected_event in zip(actual, expected):
      self.assert_events_equal(actual_event, expected_event)

  def assert_events_equal(
      self, actual: utils.Event, expected: utils.Event
  ) -> None:
    """Helper to check equality of ActivationTracer log events.

    Ignores the 'time' key, and for compactness, allows a missing 'kwargs' key
    to be equivalent to {'kwargs': {}}.

    Args:
      actual: The actual event to check.
      expected: The expected event to check against.

    Raises:
      absl.testing.absltest.TestCase.failureException: If the events differ.
    """
    # Ignore "time" and handle "kwargs" specially.
    self.assertEqual(
        actual.keys() - {"time", "kwargs"}, expected.keys() - {"kwargs"}
    )
    self.assertIs(actual["module"], expected["module"])
    self.assertEqual(actual["depth"], expected["depth"])
    self.assertEqual(actual["idx"], expected["idx"])
    for key in ("args", "kwargs", "output", "grad_input", "grad_output"):
      if key not in actual:
        continue
      if key == "kwargs" and "kwargs" not in expected:
        self.assertEqual(actual["kwargs"], {})
      else:
        try:
          utils.assert_close(actual[key], expected[key])
        except Exception as e:
          raise self.failureException(
              f"{key=}\n{actual["idx"]=}\n{actual[key]=}\n{expected[key]=}\n"
          ) from e

  def test_activation_tracer_on_zero_child_module_creates_log(self):
    """Tests the ActivationTracer context manager on zero-child module."""
    # Arrange
    model = torch.nn.Linear(10, 1)
    arg = torch.randn(1, 10)

    # Act
    with utils.ActivationTracer(model) as tracer:
      output = model(arg)

    # Assert
    expected = [fwd(0, 0, model, (arg,))]
    self.assert_logs_equal(tracer.forward_pre_log, expected)

    expected = [fwd(1, 0, model, (arg,), output)]
    self.assert_logs_equal(tracer.forward_log, expected)

  def test_activation_tracer_on_one_child_module_creates_log(self):
    """Tests the ActivationTracer context manager with one-child module."""
    # Arrange
    model = test_fixtures.NestedAddModule()
    arg1 = torch.randn(1, 10)
    arg2 = torch.randn(1, 10)

    # Act
    with utils.ActivationTracer(model) as tracer:
      output = model(arg1, arg2)

    # Assert
    expected = [
        fwd(0, 0, model, (arg1, arg2)),
        fwd(1, 1, model.add1, (arg1, arg2)),
    ]
    self.assert_logs_equal(tracer.forward_pre_log, expected)

    expected = [
        fwd(2, 1, model.add1, (arg1, arg2), output),
        fwd(3, 0, model, (arg1, arg2), output),
    ]
    self.assert_logs_equal(tracer.forward_log, expected)

  def test_activation_tracer_on_complex_module_creates_log(self):
    """Tests the ActivationTracer context manager with a complex model."""
    # Arrange
    model = test_fixtures.ContainerModule()
    arg = torch.randn(1, 10)

    # Act
    with utils.ActivationTracer(model) as tracer:
      output = model(arg)

    # Assert
    # The following would fail if the same module is called multiple times.
    fwd_map = {item["module"]: item for item in tracer.forward_log}
    # Chain the outputs of each layer to the inputs of the next.
    linear1_out = fwd_map[model.simple.relu]["args"][0]
    relu_arg = (fwd_map[model.simple.linear1]["output"],)
    relu_out = fwd_map[model.simple.adder]["args"][0]
    adder_arg = (
        fwd_map[model.simple.relu]["output"],
        torch.ones_like(fwd_map[model.simple.relu]["output"]),
    )
    adder_out = fwd_map[model.simple.linear2]["args"][0]
    linear2_arg = (fwd_map[model.simple.adder]["output"],)
    # Simple's first output comes from linear2.
    linear2_out = fwd_map[model.simple]["output"][0]
    # Simple layer returns two outputs: linear2's, linear2's * 2.
    simple_out = (
        fwd_map[model.linear3]["args"][0],
        fwd_map[model.linear3]["args"][0] * 2,
    )
    linear3_arg = fwd_map[model.simple]["output"][0:1]

    expected = [
        fwd(0, 0, model, (arg,)),
        fwd(1, 1, model.simple, (arg,)),
        fwd(2, 2, model.simple.linear1, (arg,)),
        fwd(4, 2, model.simple.relu, relu_arg),
        fwd(6, 2, model.simple.adder, adder_arg),
        fwd(8, 2, model.simple.linear2, linear2_arg),
        fwd(11, 1, model.linear3, linear3_arg),
    ]
    self.assert_logs_equal(tracer.forward_pre_log, expected)

    expected = [
        fwd(3, 2, model.simple.linear1, (arg,), linear1_out),
        fwd(5, 2, model.simple.relu, relu_arg, relu_out),
        fwd(7, 2, model.simple.adder, adder_arg, adder_out),
        fwd(9, 2, model.simple.linear2, linear2_arg, linear2_out),
        fwd(10, 1, model.simple, (arg,), simple_out),
        fwd(12, 1, model.linear3, linear3_arg, output),
        fwd(13, 0, model, (arg,), output),
    ]
    self.assert_logs_equal(tracer.forward_log, expected)

  def test_activation_tracer_on_zero_child_module_backward_call_creates_log(
      self,
  ):
    """Tests the ActivationTracer on backward on module with zero children."""
    # Arrange
    model = torch.nn.Linear(10, 1)
    arg = torch.randn(1, 10, requires_grad=True)

    # Act
    with utils.ActivationTracer(model) as tracer:
      output = model(arg)
      output.retain_grad()
      loss = output.sum()
      loss.backward(retain_graph=True)  # Gradient of sum is 1.0.

    # Assert
    expected = [bwd(2, 0, model, (output.grad,))]
    self.assert_logs_equal(tracer.backward_pre_log, expected)

    expected = [bwd(3, 0, model, (output.grad,), (arg.grad,))]
    self.assert_logs_equal(tracer.backward_log, expected)

  def test_activation_tracer_on_one_child_module_backward_call_creates_log(
      self,
  ):
    """Tests the ActivationTracer on backward on one-child module."""
    # Arrange
    model = test_fixtures.NestedAddModule()
    arg_left = torch.randn(1, 10, requires_grad=True)
    arg_right = torch.randn(1, 10, requires_grad=True)

    # Act
    with utils.ActivationTracer(model) as tracer:
      output = model(arg_left, arg_right)
      output.retain_grad()
      loss = output.sum()
      loss.backward(retain_graph=True)

    # Assert on all attributes of all items in the logs.
    grad_input = (arg_left.grad, arg_right.grad)
    expected = [
        bwd(4, 0, model, (output.grad,)),
        bwd(5, 1, model.add1, (output.grad,)),
    ]
    self.assert_logs_equal(tracer.backward_pre_log, expected)
    expected = [
        bwd(
            6,
            1,
            model.add1,
            (output.grad,),
            grad_input,
        ),
        bwd(7, 0, model, (output.grad,), grad_input),
    ]

    self.assert_logs_equal(tracer.backward_log, expected)

  def test_activation_tracer_on_complex_module_backward_call_creates_log(self):
    """Tests the ActivationTracer on backward on complex."""
    # Arrange
    model = test_fixtures.ContainerModule()
    arg = torch.randn(1, 10, requires_grad=True)

    # Act
    with utils.ActivationTracer(model) as tracer:
      output = model(arg)
      output.retain_grad()
      loss = output.sum()
      loss.backward(retain_graph=True)

    # Assert.
    # The following would fail if the same module is called multiple times.
    bwd_map = {item["module"]: item for item in tracer.backward_log}
    # Chain the grad_outputs of each layer to the grad_inputs of the next.
    # Linear3's input is the first output of Simple.
    grad_in_linear3 = bwd_map[model.simple]["grad_output"][0:1]
    # The second output of simple is ignored, so no grads.
    grad_out_simple = bwd_map[model.linear3]["grad_input"] + (None,)
    grad_out_linear2 = grad_out_simple[0:1]  # Simple's second input is ignored.
    grad_in_linear2 = bwd_map[model.simple.adder]["grad_output"]
    grad_out_adder = bwd_map[model.simple.linear2]["grad_input"]
    # The second input to the adder is a constant, so its grad is zero.
    grad_in_adder = (
        bwd_map[model.simple.relu]["grad_output"][0],
        torch.zeros(1, 5),
    )
    grad_out_relu = bwd_map[model.simple.adder]["grad_input"][0:1]
    grad_in_relu = bwd_map[model.simple.linear1]["grad_output"]
    grad_out_linear1 = bwd_map[model.simple.relu]["grad_input"]

    expected = [
        bwd(14, 0, model, (output.grad,)),
        bwd(15, 1, model.linear3, (output.grad,)),
        bwd(17, 1, model.simple, grad_out_simple),
        bwd(18, 2, model.simple.linear2, grad_out_linear2),
        bwd(20, 2, model.simple.adder, grad_out_adder),
        bwd(22, 2, model.simple.relu, grad_out_relu),
        bwd(24, 2, model.simple.linear1, grad_out_linear1),
    ]
    self.assert_logs_equal(tracer.backward_pre_log, expected)

    expected = [
        bwd(16, 1, model.linear3, (output.grad,), grad_in_linear3),
        bwd(19, 2, model.simple.linear2, grad_out_linear2, grad_in_linear2),
        bwd(21, 2, model.simple.adder, grad_out_adder, grad_in_adder),
        bwd(23, 2, model.simple.relu, grad_out_relu, grad_in_relu),
        bwd(25, 2, model.simple.linear1, grad_out_linear1, (arg.grad,)),
        bwd(26, 1, model.simple, grad_out_simple, (arg.grad,)),
        bwd(27, 0, model, (output.grad,), (arg.grad,)),
    ]
    self.assert_logs_equal(tracer.backward_log, expected)


if __name__ == "__main__":
  absltest.main()

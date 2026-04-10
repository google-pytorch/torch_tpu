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

"""Small graph test for TPU backend."""

import os
import pickle
from typing import List
from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu import api
from torch_tpu._internal import compile as compile_lib
# leaking the private function for testing.
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import utils


class FunctionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"

  def _run_and_compare(
      self,
      func,
      inputs: List[torch.Tensor],
      debug: bool = True,
  ) -> compile_lib.TpuBackend:
    """Runs the given function on CPU, TPU eager mode, and TPU compiled mode and compares the results.

    Args:
      func: The function to test.
      inputs: A list of tensor inputs for the function.
      debug: Whether to enable debug mode for the TPU backend.

    Returns:
      The TPU backend in case we need to run more test on it.
    """
    # CPU
    result_cpu = func(*inputs)

    # TPU eager
    inputs_tpu = _backend.to_device(inputs, api.tpu_device())
    tpu_eager_result = _backend.to_device(func(*inputs_tpu), "cpu")
    if isinstance(result_cpu, torch.Tensor):
      assert isinstance(tpu_eager_result, torch.Tensor)
      utils.assert_close(tpu_eager_result, result_cpu)
    elif isinstance(result_cpu, tuple):
      assert len(result_cpu) == len(tpu_eager_result)
      for result_cpu_value, tpu_eager_result_value in zip(
          result_cpu, tpu_eager_result
      ):
        if isinstance(result_cpu_value, torch.Tensor):
          assert isinstance(tpu_eager_result_value, torch.Tensor)
          utils.assert_close(tpu_eager_result_value, result_cpu_value)
    else:
      raise ValueError(f"Unsupported result type: {type(result_cpu)}")

    # TPU compiled
    tpu_backend = compile_lib.TpuBackend(debug=debug)
    compiled = torch.compile(func, backend=tpu_backend)
    tpu_compiled_result = _backend.to_device(compiled(*inputs_tpu), "cpu")
    if isinstance(result_cpu, torch.Tensor):
      assert isinstance(tpu_compiled_result, torch.Tensor)
      utils.assert_close(tpu_compiled_result, result_cpu)
    elif isinstance(result_cpu, tuple):
      assert len(result_cpu) == len(tpu_compiled_result)
      for result_cpu_value, tpu_compiled_result_value in zip(
          result_cpu, tpu_compiled_result
      ):
        if isinstance(result_cpu_value, torch.Tensor):
          assert isinstance(tpu_compiled_result_value, torch.Tensor)
          utils.assert_close(tpu_compiled_result_value, result_cpu_value)
    else:
      raise ValueError(f"Unsupported result type: {type(result_cpu)}")

    return tpu_backend._compiled_executables

  @mock.patch.object(_backend, "trace_structured")
  def test_trace_structured_called(self, mock_trace_structured):
    def simple(x, y):
      return x + y

    inputs = [
        torch.tensor([0.1, 0.2]),
        torch.tensor([0.4, 0.5]),
    ]
    with mock.patch(
        "torch._logging._internal.trace_log.handlers", [mock.Mock(level=0)]
    ):
      self._run_and_compare(simple, inputs)

    # Check if trace_structured was called
    self.assertTrue(mock_trace_structured.called)

    names = [call.args[0] for call in mock_trace_structured.call_args_list]
    self.assertTrue(all(name == "artifact" for name in names))

    # Extract metadata and evaluate payloads to verify artifacts are actually
    # created
    payloads = {
        call.kwargs["metadata_fn"]()["name"]: call.kwargs["payload_fn"]()
        for call in mock_trace_structured.call_args_list
    }

    self.assertIn("torchtpu_fx_graph", payloads)
    self.assertIn("torchtpu_stablehlo_graph", payloads)

    # Verify the FX graph artifact
    fx_payload = payloads["torchtpu_fx_graph"]
    self.assertIsInstance(fx_payload, str)
    self.assertNotEmpty(fx_payload)
    self.assertIn("add", fx_payload)  # simple() does x + y

    # Verify the StableHLO artifact
    mlir_payload = payloads["torchtpu_stablehlo_graph"]
    self.assertIsInstance(mlir_payload, str)
    self.assertNotEmpty(mlir_payload)
    self.assertIn("module", mlir_payload)

  def test_super_simple(self):
    def simple(x, y):
      a = 0.3 * x + 0.5 * y
      return a

    inputs_val = [
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    ]
    self._run_and_compare(simple, inputs_val)

  def test_to_copy(self):
    """Test that we can handle to_copy ops in compiled mode.

    Unclear why this is inserting a new op in the deferred graph, but
    generally ops that take in explicit devices have caused problems in
    compile. _to_copy should be a no-op in this case is both args are on
    the same device.

    Note: The test requires both (1) embedded constant and (2) to_copy op to
    exercise the issue.
    """

    def simple(x):
      c = torch.tensor([1, 2, 3, 4, 5], device=x.device)
      c_copy: "f32[64]" = torch.ops.aten._to_copy.default(c, device=x.device)
      return c_copy + x

    x = torch.tensor([1, 2, 3, 4, 5], device=api.tpu_device())
    self._run_and_compare(simple, [x], debug=True)

  def test_simple_handle_input_flip(self):
    # Without CL/794139909, the eager model will follow the invoke order
    # and flip the input from x, y to y, x. See this doc for more details:
    # https://docs.google.com/document/d/1D-Ltx94oZRwxnNvajME2c5CCtPJAle_WV1tU8p4sly0/edit?tab=t.0
    def simple(x, y):
      a = x + 0.5 * y
      return a

    inputs_val = [
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    ]
    self._run_and_compare(simple, inputs_val)

  def test_mlir_generates_one_more_input(self):
    # torch.ones_likes(x) used to introduce a phantom input to the MLIRgraph.
    # This test is to ensure it does not happen again.
    def func(x, y):
      a = 0.3 * x
      b = a + 0.5 * y
      c = torch.ones_like(x) + b
      return c

    inputs_val = [
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    ]
    self._run_and_compare(func, inputs_val)

  def test_compile_duplicate_inputs(self):
    def func(x, y):
      a = 0.3 * x
      b = a + 0.5 * y
      c = torch.ones_like(x) + b
      return c

    # Call with repeated and different inputs to ensure no torch_tpu caching
    # gets in the way.
    input_a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).to(api.tpu_device())
    input_b = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1]).to(api.tpu_device())
    compiled = torch.compile(func, backend=compile_lib.TpuBackend())
    res_a = compiled(input_a, input_a).to("cpu")
    res_ab = compiled(input_a, input_b).to("cpu")
    utils.assert_close(res_a.sum(), res_ab.sum(), rtol=1e-3, atol=1e-5)

  def test_fx_graph_generates_more_than_one_graph_to_compile(self):
    # The if case trigger a graph break, so tpu_backend will generate two graphs
    # to compile.
    def func(a, b):
      x = a / (torch.abs(a) + 1)
      if b.sum() < 0:
        b = b * -1
      return x * b

    inputs_val = [
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    ]

    v = self._run_and_compare(func, inputs_val)

    # This test has a graph break, so we are expecting two graphs in the cache.
    self.assertLen(v, 2)

    # debug mode enabled so expect graphs to be set and in plaintext
    self.assertIn("torch.ops.aten.abs", v[0].graph_module_debug_str)
    self.assertIn("stablehlo.abs", str(v[0].mlir_graph))
    self.assertNotEqual(
        v[0].graph_module_debug_str, v[1].graph_module_debug_str
    )
    self.assertNotEqual(v[0].mlir_graph, v[1].mlir_graph)

  def test_data_dependent_dynamic_op(self):
    """Test that a dynamo will break on data dependent ops.

    TorchTPU does not support data dependent ops, so we rely on dynamo falling
    back to eager executaion for these ops. This test to use torch_tpu once we
    have support for bincount.
    """

    def func(a, b):
      x = a + b
      y = torch.bincount(x)
      z = y.sum()
      return z

    inputs_val = [
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([6, 7, 8, 9, 10]),
    ]

    # Check if dynamo breaks on data dependent ops:
    compiled = torch.compile(func, backend="aot_eager")
    exp = torch._dynamo.explain(compiled)(*inputs_val)
    self.assertEqual(exp.graph_count, 2)
    self.assertNotIn("bincount", str(exp.graphs[0]))
    self.assertNotIn("bincount", str(exp.graphs[1]))

    # Execute a function with data dep op
    self._run_and_compare(func, inputs_val)

  def test_multi_output_aten_op(self):
    func = torch.ops.aten.native_batch_norm

    x = torch.ones(5, 5, 5)
    weight = torch.ones(5)
    bias = torch.ones(5)
    running_mean = torch.ones(5)
    running_var = torch.ones(5)
    training = False
    momentum = 0.5
    eps = 0.6
    inputs_val = [
        x,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    ]

    self._run_and_compare(func, inputs_val)

  def test_input_has_view(self):
    """Tests that we can handle input tensors that are view ops."""

    def simple(x):
      a = 0.3 * x
      return a

    device = api.tpu_device()
    input_1 = torch.arange(2, device=device).view(1, 2)

    backend = _backend.TpuBackend()
    compiled = torch.compile(simple, dynamic=False, backend=backend)
    actual = compiled(input_1).cpu()
    expected = torch.tensor([[0.0, 0.3]], device="cpu")
    utils.assert_close(actual, expected)

  def test_zero_sized_tensor_in_body(self):
    def func(a):
      zero_sized = a[..., 2:]
      b = a + a
      return (zero_sized, b)

    inputs = [torch.ones(1, 1, 2)]

    self._run_and_compare(func, inputs)

  def test_zero_sized_slice_input(self):
    def func(a, b):
      c = b.view((2, 0))
      d = a + a
      return (c, d)

    a = torch.ones(1, 1, 2)
    zero_sized = a[..., 2:]
    inputs = [a, zero_sized]

    self._run_and_compare(func, inputs)

  def test_zero_sized_inputs(self):
    def func(x, y):
      return x.mm(y)

    inputs = [
        torch.ones(5, 0),
        torch.ones(0, 10),
    ]
    self._run_and_compare(func, inputs)

  def test_empty_size_zero_tensor(self):
    """Test that we can handle empty size zero tensor."""

    def func(x):
      return torch.acos(x)

    inputs = [
        torch.empty(0).reshape(0, 1, 3),
    ]
    self._run_and_compare(func, inputs)

  @absltest.skip("Need to fix TPU pass error.")
  def test_addcdiv_c64(self):
    # TPU error: Unsupported CVT X64 expansion from c64[] to c128[]
    def func(arg0_1, arg1_1, arg2_1):
      div = torch.ops.aten.div.Tensor(arg1_1, arg2_1)
      mul = torch.ops.aten.mul.Tensor(div, 3.14)
      add = torch.ops.aten.add.Tensor(arg0_1, mul)
      return (add,)

    inputs = [
        torch.tensor(1.0, dtype=torch.complex64, device=api.tpu_device()),
        torch.tensor(1.0, dtype=torch.complex64, device=api.tpu_device()),
        torch.tensor(1.0, dtype=torch.complex64, device=api.tpu_device()),
    ]
    self._run_and_compare(func, inputs)

  def test_mixed_eager_compiled_mode(self):
    def func(x, y):
      a = 0.3 * x
      b = a + 0.5 * y
      c = torch.ones_like(x) + b
      return c

    # inputs on CPU
    x_cpu = (
        torch.arange(start=1, end=6, dtype=torch.float32, device="cpu") * 0.1
    )
    y_cpu = (
        torch.arange(start=6, end=11, dtype=torch.float32, device="cpu") * 0.1
    )
    cpu_eager_result = func(x_cpu, y_cpu)

    # inputs have deferred ops
    x_tpu = (
        torch.arange(
            start=1, end=6, dtype=torch.float32, device=api.tpu_device()
        )
        * 0.1
    )
    y_tpu = (
        torch.arange(
            start=6, end=11, dtype=torch.float32, device=api.tpu_device()
        )
        * 0.1
    )
    # Eager mode runs as normal
    tpu_eager_result = func(x_tpu, y_tpu)
    utils.assert_close(tpu_eager_result.to("cpu"), cpu_eager_result)

    # Compiled mode will compile only the part after the inputs_val
    tpu_backend = compile_lib.TpuBackend(debug=True)
    compiled = torch.compile(func, backend=tpu_backend)
    compiled_result = compiled(x_tpu, y_tpu)
    utils.assert_close(compiled_result.to("cpu"), cpu_eager_result)

    # Rerunning with a non-deferred input will not trigger a recompile
    input_nondeferred = [
        torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            dtype=torch.float32,
            device=api.tpu_device(),
        ),
        torch.tensor(
            [0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=torch.float32,
            device=api.tpu_device(),
        ),
    ]
    reexecuted_result = compiled(*input_nondeferred)
    utils.assert_close(reexecuted_result.to("cpu"), cpu_eager_result)
    v = tpu_backend._compiled_executables
    self.assertLen(v, 1)
    self.assertIn("torch.ops.aten.ones_like", v[0].graph_module_debug_str)
    self.assertIn("stablehlo.multiply", str(v[0].mlir_graph))

  def test_embedded_non_scalar_tensor(self):
    def simple(x):
      c = torch.tensor([1, 2, 3, 4, 5], device=x.device)
      return c + x

    x = torch.tensor([1, 2, 3, 4, 5], device=api.tpu_device())
    self._run_and_compare(simple, [x], debug=True)

  def test_embedded_scalar_tensor_constants(self):
    # Need to explicitly specify device to get eager eval result
    def where_const(x):
      return torch.where(
          x,
          torch.tensor(1.0, device=x.device),
          torch.tensor(2, device=x.device),
      )

    args = [torch.tensor([False, True], dtype=torch.bool)]
    self._run_and_compare(where_const, args)

  def test_inplace_updates(self):
    # Inplace updates generates a graph with no result tensors.
    # Before we add support to that case, this will fail.
    def inplace_add(x, y):
      x.add_(y)
      z = y + 1.0
      return z

    args = [torch.tensor([4.0, 5.0]), torch.tensor([10.0, 11.0])]
    args_tpu = _backend.to_device(args, api.tpu_device())
    tpu_backend = compile_lib.TpuBackend(debug=True)
    compiled = torch.compile(inplace_add, backend=tpu_backend)
    result_tpu = compiled(*args_tpu).to("cpu")
    utils.assert_close(result_tpu, torch.tensor([11.0, 12.0]))
    x_tpu_cpu = args_tpu[0].to("cpu")
    utils.assert_close(x_tpu_cpu, torch.tensor([14.0, 16.0]))

  def test_embedded_empty_constants(self):
    """Test that we can handle embedded zero-sized tensor constants."""

    def where_const(x):
      return torch.where(
          x,
          torch.tensor([], device=x.device),
          torch.tensor([], device=x.device),
      )

    args = [torch.tensor([], dtype=torch.bool).reshape(0, 1)]
    self._run_and_compare(where_const, args)

  def test_module_with_integer_input(self):
    """Test that TpuBackend handles integer inputs."""

    class ScaleModule(torch.nn.Module):

      def forward(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        return x * scale

    model = ScaleModule()
    x = torch.randn(4, 8)
    scale = 3

    x_tpu = x.to(api.tpu_device())
    compiled = torch.compile(model, backend="tpu")
    result_tpu = compiled(x_tpu, scale).cpu()

    utils.assert_close(result_tpu, x * scale)

  def test_input_baked_into_graph(self):
    """Test that input are baked into the graph.

    In this example, both `i` and `device` are baked in the graph module,
    effectively generating a graph with no inputs.
    """

    def func_with_int_put(i: int, device):
      value_range = torch.arange(0, 9, device=device)
      return value_range[i]

    tpu_backend = compile_lib.TpuBackend(debug=True)
    compiled = torch.compile(func_with_int_put, backend=tpu_backend)
    result_tpu = compiled(3, api.tpu_device()).to("cpu")
    self.assertEqual(result_tpu, 3)

  def test_compiled_executable_is_picklable(self):
    """Test that TpuBackend's compiled output can be pickled and unpickled.

    This is required for vLLM's compilation caching which serialize compiled
    functions to disk.
    """

    class SimpleModel(torch.nn.Module):

      def forward(self, x):
        return (x * 3 + 1,)

    model = SimpleModel().to(api.tpu_device())
    x = torch.randn(4, 8).to(api.tpu_device())

    backend = compile_lib.TpuBackend()
    gm = torch.fx.symbolic_trace(model)
    # Get the raw executable
    compiled_fn = backend._compile_graph_module(gm, [x])

    # Run before pickle
    result_before = compiled_fn(x)  # pylint: disable=unused-variable

    # Pickle roundtrip
    data = pickle.dumps(compiled_fn)
    restored = pickle.loads(data)

    result = restored(x)
    utils.assert_close(result[0].cpu(), x * 3 + 1)


class ModuleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCH_LOGS"] = "+dynamo"

  def _run_and_compare(
      self, module_class, inputs: List[torch.Tensor], debug: bool = True
  ) -> compile_lib.TpuBackend:
    """Runs the given function on CPU, TPU eager mode, and TPU compiled mode and compares the results.

    Args:
      module_class: The module to test.
      inputs: A list of tensor inputs for the function.
      debug: Whether to enable debug mode for the TPU backend.

    Returns:
      The TPU backend in case we need to run more test on it.
    """

    # CPU
    m = module_class()
    result_cpu = m(*inputs)

    # Fail when inputs on TPU but module on CPU
    inputs_tpu = _backend.to_device(inputs, api.tpu_device())
    with self.assertRaises(Exception) as err:
      _ = m(*inputs_tpu).to("cpu")
    self.assertIn("same device", str(err.exception))

    # Fail when inputs on TPU but module on CPU
    tpu_backend = compile_lib.TpuBackend(debug=debug)
    compiled = torch.compile(m, backend=tpu_backend)
    with self.assertRaises(Exception) as err:
      _ = compiled(*inputs_tpu).to("cpu")

    # TPU eager
    m = m.to(api.tpu_device())
    tpu_eager_result = m(*inputs_tpu).to("cpu")
    utils.assert_close(tpu_eager_result, result_cpu, rtol=1e-3, atol=1e-5)

    # TPU compiled
    compiled = torch.compile(m, backend=tpu_backend)
    tpu_compiled_result = compiled(*inputs_tpu).to("cpu")

    utils.assert_close(tpu_compiled_result, result_cpu, rtol=1e-3, atol=1e-5)
    return tpu_backend._compiled_executables

  def test_simple_module_compile(self):
    # The if case trigger a graph break, so tpu_backend will generate two graphs
    # to compile.
    class SimpleModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.Tensor(5))
        self.b = torch.nn.Parameter(torch.Tensor(5))
        self.c = torch.nn.Parameter(torch.Tensor(5))

      def forward(self, x, y):
        out = self.a + 0.8 * self.b + x
        out = out + 0.7 * self.c + y
        return out

    inputs = (
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        torch.tensor([0.4, 0.5, 0.6, 0.7, 0.6]),
    )
    v = self._run_and_compare(SimpleModule, inputs)
    self.assertLen(v, 1)
    self.assertIn("stablehlo.multiply", str(v[0].mlir_graph))
    self.assertIn("def forward(self,", v[0].graph_module_debug_str)

  def test_module_with_constants(self):
    class ModuleWithConstants(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.const1 = torch.randn(5, 5)
        self.const2 = torch.randn(5, 5)
        self.linear = torch.nn.Linear(5, 5)

      def forward(self, x):
        x = self.linear(x)
        x = x + self.const1
        x = x * self.const2
        return x

    inputs = [torch.randn(5, 5)]
    self._run_and_compare(ModuleWithConstants, inputs)

  def test_module_with_non_tensor_consts(self):
    class ModuleWithNonTensorConsts(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.const1 = torch.randn(5, 5)
        self.some_int = 123
        self.linear = torch.nn.Linear(5, 5)

      def forward(self, x):
        x = self.linear(x)
        return x + self.const1 + self.some_int

    inputs = [torch.randn(5, 5)]
    self._run_and_compare(ModuleWithNonTensorConsts, inputs)

  def test_nested_module_with_constants(self):
    class InnerModuleWithConst(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.inner_const = torch.arange(5).float()

      def forward(self, x):
        return x + self.inner_const

    class OuterModuleWithConst(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.inner = InnerModuleWithConst()
        self.outer_const = torch.ones(5) * 2

      def forward(self, x):
        x = self.inner(x)
        return x * self.outer_const

    inputs = [torch.randn(5, 5)]
    self._run_and_compare(OuterModuleWithConst, inputs)

  def test_module_no_tensor_consts(self):
    class ModuleNoTensorConsts(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
        self.val = 10

      def forward(self, x):
        return self.linear(x) * self.val

    inputs = [torch.randn(5, 5)]
    self._run_and_compare(ModuleNoTensorConsts, inputs)

  def test_module_const_control_flow(self):
    class ModuleConstControlFlow(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.threshold = torch.tensor(0.5)
        self.linear = torch.nn.Linear(5, 5)

      def forward(self, x):
        y = self.linear(x)
        # This will likely cause a graph break, but the constant
        # should still be lifted correctly in the subgraph.
        if y.mean() > self.threshold:
          return y * 2
        else:
          return y

    inputs = [torch.randn(5, 5)]
    self._run_and_compare(ModuleConstControlFlow, inputs)

  def test_module_with_buffer(self):
    class ModuleWithBuffer(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.register_buffer("my_buffer", torch.ones(5, 1))

      def forward(self, x):
        return x + self.my_buffer

    inputs = [torch.randn(5, 5)]
    self._run_and_compare(ModuleWithBuffer, inputs)


if __name__ == "__main__":
  absltest.main()

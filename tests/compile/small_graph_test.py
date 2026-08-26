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

import dataclasses
import os
import pickle
from typing import List
from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu._internal import compile as compile_lib
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile.torch_tpu_compiled_executable import NoOpCompiledArtifact
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


@dataclasses.dataclass
class CtcLossInputs:
  """Inputs for CTC loss tests.

  Attributes:
    log_probs: A Tensor of log probabilities.
    targets: A Tensor of targets.
    input_lengths: A Tensor of input lengths.
    target_lengths: A Tensor of target lengths.
  """

  log_probs: torch.Tensor
  targets: torch.Tensor
  input_lengths: torch.Tensor
  target_lengths: torch.Tensor


def _generate_ctc_loss_inputs(*, requires_grad: bool = False) -> CtcLossInputs:
  """Generates deterministic inputs for CTC loss tests.

  Args:
    requires_grad: Whether to set requires_grad=True on log_probs.

  Returns:
    A CtcLossInputs dataclass instance.
  """
  # T=10, N=2, C=5.
  log_probs = (
      torch.linspace(-2.0, 2.0, 100)
      .view(10, 2, 5)
      .requires_grad_(requires_grad)
  )
  targets = torch.tensor([[1, 2, 3, 4, 1], [2, 3, 4, 1, 2]], dtype=torch.long)
  input_lengths = torch.full((2,), 10, dtype=torch.long)
  target_lengths = torch.full((2,), 5, dtype=torch.long)
  return CtcLossInputs(log_probs, targets, input_lengths, target_lengths)


class FunctionTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
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
    inputs_tpu = _backend.to_device(inputs, torch.device("tpu"))
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

  @mock.patch.object(compiler, "trace_structured", autospec=True)
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

  def test_scaled_mm_v2_standard(self):
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:

      def run_scaled_mm_v2_standard(self_fp8, mat2_fp8, scale_a, scale_b):
        if self_fp8.device.type == "cpu":
          s_f32 = self_fp8.float()
          m_f32 = mat2_fp8.float()
          sa = scale_a[0]
          sb = scale_b[0]
          s_scaled = s_f32 * sa
          m_scaled = m_f32 * sb
          out_f32 = torch.mm(s_scaled, m_scaled)
          return out_f32.to(self_fp8.dtype)
        else:
          recipe_a = [0]
          recipe_b = [0]
          swizzle_a = [0]
          swizzle_b = [0]
          return torch._scaled_mm_v2(
              self_fp8,
              mat2_fp8,
              scale_a,
              recipe_a,
              swizzle_a,
              scale_b,
              recipe_b,
              swizzle_b,
              None,
              None,
              contraction_dim=[1, 0],
          )

      m, n, k = 16, 16, 16
      dtype = torch.float8_e4m3fn
      self_float = (
          torch.arange(m * k, dtype=torch.float32).view(m, k) % 10
      ) * 0.1
      self_fp8 = self_float.to(dtype)
      mat2_float = (
          torch.arange(k * n, dtype=torch.float32).view(k, n) % 10
      ) * 0.1
      mat2_fp8 = mat2_float.to(dtype)

      scale_a = [torch.tensor([1.5], dtype=torch.float32)]
      scale_b = [torch.tensor([2.0], dtype=torch.float32)]

      inputs = [self_fp8, mat2_fp8, scale_a, scale_b]
      self._run_and_compare(run_scaled_mm_v2_standard, inputs)
    finally:
      torch.set_float32_matmul_precision(old_precision)

  def test_scaled_mm_v2_blockwise(self):
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:

      def swizzle_lhs_scale(logical_scale):
        m, k_logical = logical_scale.shape
        m_blocks = m // 128
        k_blocks = k_logical // 4
        x = logical_scale.view(m_blocks, 128, k_blocks, 4)
        x = x.transpose(1, 2)
        x = x.view(m_blocks, k_blocks, 4, 32, 4)
        x = x.transpose(2, 3)
        x = x.reshape(m_blocks, k_blocks, 32, 16)
        x = x.transpose(1, 2)
        return x.reshape(m_blocks * 32, k_blocks * 16)

      def swizzle_rhs_scale(logical_scale):
        return swizzle_lhs_scale(logical_scale.t()).t()

      def run_scaled_mm_v2_blockwise(
          self_fp8,
          mat2_fp8,
          scale_a_tpu,
          scale_b_tpu,
          scale_a_logical,
          scale_b_logical,
      ):
        if self_fp8.device.type == "cpu":
          s_f32 = self_fp8.float()
          m_f32 = mat2_fp8.float()
          m, k = self_fp8.shape
          k, n = mat2_fp8.shape
          sa_expanded = (
              scale_a_logical.unsqueeze(-1).expand(m, k // 32, 32).reshape(m, k)
          )
          s_dequant = s_f32 * sa_expanded
          sb_expanded = (
              scale_b_logical.unsqueeze(1).expand(k // 32, 32, n).reshape(k, n)
          )
          m_dequant = m_f32 * sb_expanded
          out_f32 = torch.mm(s_dequant, m_dequant)
          return out_f32.to(self_fp8.dtype)
        else:
          recipe_a = [3]
          recipe_b = [3]
          swiz_a = [1]
          swiz_b = [1]
          return torch._scaled_mm_v2(
              self_fp8,
              mat2_fp8,
              [scale_a_tpu],
              recipe_a,
              swiz_a,
              [scale_b_tpu],
              recipe_b,
              swiz_b,
              None,
              None,
          )

      m, n, k = 128, 128, 128
      dtype = torch.float8_e4m3fn
      self_float = (
          torch.arange(m * k, dtype=torch.float32).view(m, k) % 10
      ) * 0.1
      self_fp8 = self_float.to(dtype)
      mat2_float = (
          torch.arange(k * n, dtype=torch.float32).view(k, n) % 10
      ) * 0.1
      mat2_fp8 = mat2_float.to(dtype)

      scale_a_logical = (
          torch.arange(m * (k // 32), dtype=torch.float32).view(m, k // 32) % 5
      ) * 0.1 + 0.1
      scale_b_logical = (
          torch.arange((k // 32) * n, dtype=torch.float32).view(k // 32, n) % 5
      ) * 0.1 + 0.1

      scale_a_tpu = swizzle_lhs_scale(scale_a_logical)
      scale_b_tpu = swizzle_rhs_scale(scale_b_logical)

      inputs = [
          self_fp8,
          mat2_fp8,
          scale_a_tpu,
          scale_b_tpu,
          scale_a_logical,
          scale_b_logical,
      ]
      self._run_and_compare(run_scaled_mm_v2_blockwise, inputs)
    finally:
      torch.set_float32_matmul_precision(old_precision)

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
      c_copy = torch.ops.aten._to_copy.default(c, device=x.device)
      return c_copy + x

    x = torch.tensor([1, 2, 3, 4, 5], device=torch.device("tpu"))
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
    input_a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).to(torch.device("tpu"))
    input_b = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1]).to(torch.device("tpu"))
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
    self.assertIn("torch.ops.aten.abs", v[0].graph_module_debug_strs[0])
    self.assertIn("stablehlo.abs", v[0].mlir_texts[0])
    self.assertNotEqual(
        v[0].graph_module_debug_strs[0], v[1].graph_module_debug_strs[0]
    )
    self.assertNotEqual(v[0].mlir_texts[0], v[1].mlir_texts[0])

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
    training = True
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

    device = torch.device("tpu")
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
        torch.tensor(
            1.0,
            dtype=torch.complex64,
            device=torch.device("tpu"),
        ),
        torch.tensor(
            1.0,
            dtype=torch.complex64,
            device=torch.device("tpu"),
        ),
        torch.tensor(
            1.0,
            dtype=torch.complex64,
            device=torch.device("tpu"),
        ),
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
            start=1,
            end=6,
            dtype=torch.float32,
            device=torch.device("tpu"),
        )
        * 0.1
    )
    y_tpu = (
        torch.arange(
            start=6,
            end=11,
            dtype=torch.float32,
            device=torch.device("tpu"),
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
            device=torch.device("tpu"),
        ),
        torch.tensor(
            [0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=torch.float32,
            device=torch.device("tpu"),
        ),
    ]
    reexecuted_result = compiled(*input_nondeferred)
    utils.assert_close(reexecuted_result.to("cpu"), cpu_eager_result)
    v = tpu_backend._compiled_executables
    self.assertLen(v, 1)
    self.assertIn("torch.ops.aten.ones_like", v[0].graph_module_debug_strs[0])
    self.assertIn("stablehlo.multiply", v[0].mlir_texts[0])

  def test_embedded_non_scalar_tensor(self):
    def simple(x):
      return torch.tensor([1, 2, 3, 4, 5], device=x.device) + x

    x = torch.tensor([1, 2, 3, 4, 5], device=torch.device("tpu"))
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
    args_tpu = _backend.to_device(args, torch.device("tpu"))
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

    x_tpu = x.to(torch.device("tpu"))
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
    result_tpu = compiled(3, torch.device("tpu")).to("cpu")
    self.assertEqual(result_tpu, 3)

  def test_compiled_executable_is_picklable(self):
    """Test that TpuBackend's compiled output can be pickled and unpickled.

    This is required for vLLM's compilation caching which serialize compiled
    functions to disk.
    """

    class SimpleModel(torch.nn.Module):

      def forward(self, x):
        return (x * 3 + 1,)

    model = SimpleModel().to(torch.device("tpu"))
    x = torch.randn(4, 8).to(torch.device("tpu"))

    compiler_instance = compiler.StaticCompiler()
    gm = torch.fx.symbolic_trace(model)
    # Get the raw executable
    compiled_fn = compiler_instance(gm, [x])

    # Run before pickle
    result_before = compiled_fn([x])
    utils.assert_close(result_before[0].cpu(), x * 3 + 1)

    # Pickle roundtrip
    data = pickle.dumps(compiled_fn)
    restored = pickle.loads(data)

    result = restored([x])
    utils.assert_close(result[0].cpu(), x * 3 + 1)

  def test_executable_boxed_calling_convention(self):
    """Test that _TorchTpuCompiledExecutable uses boxed calling convention."""

    def add_mul(x, y):
      return x + y, x * y

    x = torch.randn(4, 4).to(torch.device("tpu"))
    y = torch.randn(4, 4).to(torch.device("tpu"))

    gm = torch.fx.symbolic_trace(add_mul)
    executable = compiler.StaticCompiler()(gm, [x, y])

    expected_sum = (x + y).cpu()
    expected_prod = (x * y).cpu()

    result = executable([x, y])
    utils.assert_close(result[0].cpu(), expected_sum)
    utils.assert_close(result[1].cpu(), expected_prod)

  def test_executable_boxed_single_input(self):
    """Test boxed calling convention with a single tensor input."""

    class ScaleModel(torch.nn.Module):

      def forward(self, x):
        return (x * 3 + 1,)

    model = ScaleModel().to(torch.device("tpu"))
    x = torch.randn(4, 8).to(torch.device("tpu"))
    expected = (x * 3 + 1).cpu()

    gm = torch.fx.symbolic_trace(model)
    executable = compiler.StaticCompiler()(gm, [x])

    result = executable([x])
    utils.assert_close(result[0].cpu(), expected)

  def test_input_is_view(self):
    """Test that compile works even when an argument is a view."""

    def expects_transposed_input(x):
      # If x is in a transposed state, this will be valid; it will transpose
      # back to a contiguous layout, and the flattening view() will be valid.
      # But if x is in a contiguous layout, then the transpose will cause it to
      # be non-contiguous, and the flattening view() will fail.
      return x.t().view(-1)

    tpu_backend = compile_lib.TpuBackend(debug=True)
    compiled = torch.compile(expects_transposed_input, backend=tpu_backend)

    x = torch.arange(6).reshape(2, 3).to(torch.device("tpu"))
    x_t = x.t()

    result = compiled(x_t)
    expected = torch.arange(6, device="cpu")
    utils.assert_close(actual=result.cpu(), expected=expected)

  def test_output_is_non_overlapping_view(self):
    """Test that compile works even when an output is a non-overlapping view."""

    def returns_non_overlapping_output(x):
      # Returns every odd-offset element of the input.
      return x.view(-1, 2).t()[1]

    tpu_backend = compile_lib.TpuBackend(debug=True)
    compiled = torch.compile(
        returns_non_overlapping_output, backend=tpu_backend
    )

    x = torch.arange(6).to(torch.device("tpu"))

    result = compiled(x)
    # Logical values: torch.tensor([1, 3, 5])
    # Physical values: [0, 1, 2, 3, 4, 5]
    expected = torch.arange(6, device="cpu").view(-1, 2).t()[1]

    # The layout on-device should be the same as the expected layout on CPU.
    self.assertEqual(result.dtype, expected.dtype)
    self.assertEqual(result.shape, expected.shape)
    self.assertEqual(result.stride(), expected.stride())
    self.assertEqual(result.storage_offset(), expected.storage_offset())

    # The values should be the same as well.
    utils.assert_close(actual=result.cpu(), expected=expected)

  def test_output_is_broadcast_view(self):
    """Test that compile works even when an output is a broadcast view."""

    def returns_broadcast_output(x):
      return x.expand(2, *x.shape)

    tpu_backend = compile_lib.TpuBackend(debug=True)
    compiled = torch.compile(returns_broadcast_output, backend=tpu_backend)

    x = torch.arange(6).to(torch.device("tpu"))

    result = compiled(x)
    # Logical values: torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    # Physical values: [0, 1, 2, 3, 4, 5]
    expected = torch.arange(6, device="cpu").expand(2, *x.shape)

    # The layout on-device should be the same as the expected layout on CPU.
    self.assertEqual(result.dtype, expected.dtype)
    self.assertEqual(result.shape, expected.shape)
    self.assertEqual(result.stride(), expected.stride())
    self.assertEqual(result.storage_offset(), expected.storage_offset())

    # The values should be the same as well.
    utils.assert_close(actual=result.cpu(), expected=expected)

  def test_output_is_overlapping_view(self):
    """Test that compile works even when an output is an overlapping view."""

    def returns_overlapping_output(x):
      return x.as_strided(size=(2, 4), stride=(1, 1), storage_offset=1)

    tpu_backend = compile_lib.TpuBackend(debug=True)
    compiled = torch.compile(returns_overlapping_output, backend=tpu_backend)

    x = torch.arange(6).to(torch.device("tpu"))

    result = compiled(x)
    # Logical values: torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    # Physical values: [0, 1, 2, 3, 4, 5]
    expected = torch.arange(6, device="cpu").as_strided(
        size=(2, 4), stride=(1, 1), storage_offset=1
    )

    # The layout on-device should be the same as the expected layout on CPU.
    self.assertEqual(result.dtype, expected.dtype)
    self.assertEqual(result.shape, expected.shape)
    self.assertEqual(result.stride(), expected.stride())
    self.assertEqual(result.storage_offset(), expected.storage_offset())

    # The values should be the same as well.
    utils.assert_close(actual=result.cpu(), expected=expected)


class ModuleTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

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

    m_cpu = module_class()
    cpu_results = m_cpu(*inputs)

    inputs_tpu = _backend.to_device(inputs, torch.device("tpu"))
    # We create a new instance of the module inside a context manager with the
    # tpu device in order for any constants created inside the module's
    # `__init__` method to get allocated on TPU. Without this Dynamo throws a
    # fit.
    with torch.device("tpu"):
      m_tpu = module_class()
    # Load parameters from CPU so that any weights with random values are
    # exactly the same. Also make sure they're placed on the TPU device.
    m_tpu.load_state_dict(m_cpu.state_dict())
    m_tpu.to("tpu")

    tpu_backend = compile_lib.TpuBackend(debug=debug)
    compiled = torch.compile(m_tpu, backend=tpu_backend)
    tpu_results = _backend.to_device(compiled(*inputs_tpu), "cpu")

    utils.assert_close(tpu_results, cpu_results, rtol=1e-3, atol=1e-5)
    return tpu_backend._compiled_executables

  def test_simple_module_compile(self):
    # The if case trigger a graph break, so tpu_backend will generate two graphs
    # to compile.
    class SimpleModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(5))
        self.b = torch.nn.Parameter(torch.zeros(5))
        self.c = torch.nn.Parameter(torch.zeros(5))

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
    self.assertIn("stablehlo.multiply", v[0].mlir_texts[0])
    self.assertIn("def forward(self,", v[0].graph_module_debug_strs[0])

  def test_module_with_constants(self):
    class ModuleWithConstants(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.const1 = torch.ones(5, 5)
        self.const2 = torch.full([5, 5], 2)

      def forward(self, x):
        return (x + self.const1) * self.const2

    inputs = [torch.arange(25).reshape(5, 5)]
    self._run_and_compare(ModuleWithConstants, inputs)

  def test_module_with_non_tensor_consts(self):
    class ModuleWithNonTensorConsts(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.const1 = torch.ones(5, 5)
        self.some_int = 123

      def forward(self, x):
        return x + self.const1 + self.some_int

    inputs = [torch.arange(25).reshape(5, 5)]
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

    inputs = [torch.arange(25).reshape(5, 5)]
    self._run_and_compare(OuterModuleWithConst, inputs)

  def test_module_no_tensor_consts(self):
    class ModuleNoTensorConsts(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.val = 10

      def forward(self, x):
        return x * self.val

    inputs = [torch.arange(25).reshape(5, 5)]
    self._run_and_compare(ModuleNoTensorConsts, inputs)

  def test_module_const_control_flow(self):
    class ModuleConstControlFlow(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.threshold = torch.tensor(0.5)

      def forward(self, x):
        # This causes a graph break, but the constant should still be lifted
        # correctly in the subgraph.
        y = x.mean()
        if y > self.threshold:
          return y * 2
        return y

    inputs = [torch.arange(4, dtype=torch.float32).reshape(2, 2)]
    self._run_and_compare(ModuleConstControlFlow, inputs)

  def test_module_with_buffer(self):
    class ModuleWithBuffer(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.register_buffer("my_buffer", torch.ones(5, 1))

      def forward(self, x):
        return x + self.my_buffer

    inputs = [torch.arange(25).reshape(5, 5)]
    self._run_and_compare(ModuleWithBuffer, inputs)

  def test_adam_optimizer(self):
    class SimpleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)

      def forward(self, x):
        return self.linear(x)

    device = torch.device("tpu")
    model = SimpleModel().to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, capturable=True)
    criterion = torch.nn.MSELoss()

    tpu_backend = compile_lib.TpuBackend(debug=True)
    compiled = torch.compile(model, backend=tpu_backend)

    input_tensor = torch.arange(50, dtype=torch.float32, device=device).reshape(
        10, 5
    )
    target = torch.ones(10, 2, device=device)

    optimizer.zero_grad()
    outputs = compiled(input_tensor)
    loss = criterion(outputs, target)
    loss.backward()

    for i, p in enumerate(model.parameters()):
      with self.subTest(name=f"GradientCheck_Param_{i}"):
        self.assertIsNotNone(p.grad, "Gradients should have been computed.")
        self.assertFalse(
            torch.all(p.grad == 0), "Gradients should not be all zeros."
        )

    params_before = [p.detach().clone() for p in model.parameters()]

    optimizer.step()

    for i, (p_before, p_after) in enumerate(
        zip(params_before, model.parameters())
    ):
      with self.subTest(name=f"ParameterUpdateCheck_Param_{i}"):
        # We expect the parameters to be different after the step.
        # allclose returns True if they are within tolerance; we want False.
        self.assertFalse(
            torch.allclose(p_before, p_after, rtol=1e-4, atol=1e-4),
            "Parameters did not change after optimizer.step()",
        )

  def test_ctc_loss_bfloat16(self):
    ctc_inputs = _generate_ctc_loss_inputs()
    log_probs = ctc_inputs.log_probs.log_softmax(2).to(torch.bfloat16)
    inputs = [
        log_probs,
        ctc_inputs.targets,
        ctc_inputs.input_lengths,
        ctc_inputs.target_lengths,
    ]

    inputs_cpu_f32 = [
        log_probs.to(torch.float32),
        ctc_inputs.targets,
        ctc_inputs.input_lengths,
        ctc_inputs.target_lengths,
    ]
    cpu_result = torch.nn.CTCLoss()(*inputs_cpu_f32).to(torch.bfloat16)

    inputs_tpu = _backend.to_device(inputs, torch.device("tpu"))
    m_tpu = torch.nn.CTCLoss().to("tpu")

    tpu_backend = compile_lib.TpuBackend(debug=True)
    compiled = torch.compile(m_tpu, backend=tpu_backend)
    tpu_compiled_result = _backend.to_device(compiled(*inputs_tpu), "cpu")
    utils.assert_close(tpu_compiled_result, cpu_result, rtol=1e-2, atol=2e-2)

  def test_ctc_loss_bfloat16_eager_reproducer(self):
    ctc_inputs = _generate_ctc_loss_inputs()
    log_probs = ctc_inputs.log_probs.log_softmax(2)
    inputs = [
        log_probs,
        ctc_inputs.targets,
        ctc_inputs.input_lengths,
        ctc_inputs.target_lengths,
    ]

    m_cpu = torch.nn.CTCLoss()
    cpu_result = m_cpu(*inputs)

    log_probs_tpu_dev, targets_tpu, input_lengths_tpu, target_lengths_tpu = (
        _backend.to_device(inputs, torch.device("tpu"))
    )
    log_probs_tpu = log_probs_tpu_dev.to(torch.bfloat16)
    inputs_tpu = [
        log_probs_tpu,
        targets_tpu,
        input_lengths_tpu,
        target_lengths_tpu,
    ]
    m_tpu = torch.nn.CTCLoss().to("tpu")
    tpu_eager_result = _backend.to_device(m_tpu(*inputs_tpu), "cpu")

    utils.assert_close(
        tpu_eager_result, cpu_result, rtol=1e-2, atol=2e-2, check_dtype=False
    )

  def test_ctc_loss_bfloat16_eager_backward(self):
    ctc_inputs = _generate_ctc_loss_inputs(requires_grad=True)

    log_probs_cpu = ctc_inputs.log_probs.clone().detach().requires_grad_(True)
    log_probs_cpu_softmax = log_probs_cpu.log_softmax(2)
    loss_cpu = torch.nn.functional.ctc_loss(
        log_probs_cpu_softmax,
        ctc_inputs.targets,
        ctc_inputs.input_lengths,
        ctc_inputs.target_lengths,
        reduction="mean",
    )
    loss_cpu.backward()
    cpu_grad = log_probs_cpu.grad

    log_probs_tpu_dev, targets_tpu, input_lengths_tpu, target_lengths_tpu = (
        _backend.to_device(
            [
                ctc_inputs.log_probs,
                ctc_inputs.targets,
                ctc_inputs.input_lengths,
                ctc_inputs.target_lengths,
            ],
            torch.device("tpu"),
        )
    )
    log_probs_tpu = (
        log_probs_tpu_dev.to(torch.bfloat16).detach().requires_grad_(True)
    )
    log_probs_tpu_softmax = log_probs_tpu.log_softmax(2)
    loss_tpu = torch.nn.functional.ctc_loss(
        log_probs_tpu_softmax,
        targets_tpu,
        input_lengths_tpu,
        target_lengths_tpu,
        reduction="mean",
    )
    loss_tpu.backward()
    tpu_grad = _backend.to_device(log_probs_tpu.grad, "cpu")

    utils.assert_close(
        tpu_grad,
        cpu_grad,
        rtol=1e-2,
        atol=2e-2,
        check_dtype=False,
    )

  def test_geqrf_empty(self):
    # geqrf under torch.compile with empty inputs is not tested in the standard
    # ops_test_compiled target because the GPU golden files lack
    # geqrf samples. This test provides explicit compile-mode coverage for
    # geqrf with empty inputs.
    class GeqrfModule(torch.nn.Module):

      def forward(self, x):
        return torch.geqrf(x)

    for shape in [(4, 0), (0, 4), (0, 0)]:
      with self.subTest(shape=shape):
        inputs = [torch.randn(shape)]
        self._run_and_compare(GeqrfModule, inputs)


def _reconstruct_none_tuple(inputs, outputs):
  """Module-level (picklable) reconstruct fn returning the (None,) structure."""
  del inputs, outputs
  return (None,)


class NoOutputGraphTest(seed_test_utils.RepeatableTest):
  """A torch.compile graph with no output tensors runs as a no-op, not an abort.

  ``def f(x): x.sum(); return None`` traces to a graph with a real input but no
  computed output tensors (FX output ``()``). traverse_and_compile requires >=1
  result tensor and used to abort the process ("no result tensors provided");
  the backend now returns a NoOpCompiledArtifact that runs nothing and
  reconstructs the (all-None) output. The same all-None segments arise from
  torch.compile(fullgraph=False) partitioning on TP>1 models, where collectives
  force graph breaks.
  """

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def test_no_output_graph_runs_as_noop(self):
    def f(x):
      x.sum()  # computed but discarded: graph has an input, no output tensor
      return None

    compiled = torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
    x = torch.randn(4).to("tpu")
    # Before the fix this aborted the process inside traverse_and_compile.
    self.assertIsNone(compiled(x))


class NoOpCompiledArtifactTest(seed_test_utils.RepeatableTest):
  """The do-nothing callable the backend returns for a no-output graph."""

  def test_call_reconstructs_output(self):
    artifact = NoOpCompiledArtifact(_reconstruct_none_tuple)
    self.assertEqual(artifact([torch.randn(4)]), (None,))

  def test_picklable_for_dynamo_cache(self):
    # CompiledArtifact must pickle for the Dynamo/AOT caching layers; the
    # torch.compile path above cannot observe this directly.
    artifact = NoOpCompiledArtifact(_reconstruct_none_tuple)
    restored = pickle.loads(pickle.dumps(artifact))
    self.assertIsInstance(restored, NoOpCompiledArtifact)
    self.assertEqual(restored([]), (None,))


if __name__ == "__main__":
  absltest.main()

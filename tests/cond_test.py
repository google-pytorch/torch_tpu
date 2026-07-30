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

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.fx.experimental.proxy_tensor import make_fx
import torch.utils._pytree as pytree
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.utils import utils


def _compile_and_run(fn, *args, **kwargs):
  compiled = torch.compile(fn, backend="tpu", dynamic=False)
  return compiled(*args, **kwargs)


def _compile_and_get_stablehlo(model, *args):
  torch.compiler.reset()
  backend = _backend.TpuBackend(debug=True)
  compiled = torch.compile(model, backend=backend, dynamic=False)
  compiled(*args)

  texts = [
      e.mlir_text
      for e in backend._compiled_executables
      if getattr(e, "mlir_text", None)
  ]
  if not texts:
    raise AssertionError("no StableHLO captured")

  return "\n".join(texts)


def _to_cpu(tree):
  return pytree.tree_map_only(torch.Tensor, lambda t: t.detach().cpu(), tree)


def _assert_tree_close(actual, expected):
  actual_flat, actual_spec = pytree.tree_flatten(actual)
  expected_flat, expected_spec = pytree.tree_flatten(expected)
  if actual_spec != expected_spec:
    raise AssertionError(f"tree specs differ: {actual_spec} vs {expected_spec}")
  for actual_leaf, expected_leaf in zip(actual_flat, expected_flat):
    utils.assert_close(actual_leaf, expected_leaf)


def _has_higher_order_cond(gm):
  for node in gm.graph.nodes:
    if node.op != "call_function":
      continue
    if node.target is torch.ops.higher_order.cond:
      return True
    if "higher_order.cond" in str(node.target):
      return True
  return False


class CondOpTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.compiler.reset()

  @parameterized.named_parameters(("true", True), ("false", False))
  def test_basic_cond(self, pred_value):

    def model(pred, x, y):

      def true_fn(a, b):
        return a + b

      def false_fn(a, b):
        return a - b

      return torch.cond(pred, true_fn, false_fn, (x, y))

    x_cpu = torch.tensor([1.0, 2.0, 3.0])
    y_cpu = torch.tensor([10.0, 20.0, 30.0])
    pred_cpu = torch.tensor(pred_value)

    result = _compile_and_run(
        model,
        pred_cpu.to("tpu"),
        x_cpu.to("tpu"),
        y_cpu.to("tpu"),
    )
    expected = model(pred_cpu, x_cpu, y_cpu)

    utils.assert_close(result.cpu(), expected)

  def test_cond_with_scalars_multiple_results_and_nested_pytrees(self):

    def model(pred, x, y, z, scale):

      def true_fn(payload, s):
        a, (b, c) = payload
        return {
            "left": a + b * s,
            "right": (torch.sum(c), a * c),
        }

      def false_fn(payload, s):
        a, (b, c) = payload
        return {
            "left": a - b * s,
            "right": (torch.sum(c + 1.0), a + c),
        }

      return torch.cond(pred, true_fn, false_fn, ((x, (y, z)), scale))

    pred_cpu = torch.tensor(True)
    x_cpu = torch.tensor([1.0, 2.0, 3.0])
    y_cpu = torch.tensor([4.0, 5.0, 6.0])
    z_cpu = torch.tensor([7.0, 8.0, 9.0])
    scale_cpu = torch.tensor(2.0)

    result = _compile_and_run(
        model,
        pred_cpu.to("tpu"),
        x_cpu.to("tpu"),
        y_cpu.to("tpu"),
        z_cpu.to("tpu"),
        scale_cpu.to("tpu"),
    )
    expected = model(pred_cpu, x_cpu, y_cpu, z_cpu, scale_cpu)

    _assert_tree_close(_to_cpu(result), expected)

  def test_hlo_contains_if_and_branch_work(self):

    def model(pred, x):

      def true_fn(v):
        return v + 1.0

      def false_fn(v):
        return torch.exp(v) + 7.0

      return torch.cond(pred, true_fn, false_fn, (x,))

    pred = torch.tensor(True, device="tpu")
    x = torch.tensor([1.0, 2.0, 3.0], device="tpu")

    result = _compile_and_run(model, pred, x)
    utils.assert_close(result.cpu(), torch.tensor([2.0, 3.0, 4.0]))

    hlo = _compile_and_get_stablehlo(model, pred, x)
    self.assertIn("stablehlo.if", hlo)
    self.assertIn("stablehlo.exponential", hlo)
    self.assertNotIn("stablehlo.select", hlo)

  @parameterized.named_parameters(("true", True), ("false", False))
  def test_stock_make_fx_graph_compiles_without_torch_compile(self, pred_value):

    def model(pred, x, y):

      def true_fn(a, b):
        return a + b, torch.sum(a * b)

      def false_fn(a, b):
        return a - b, torch.sum(a / (b + 1.0))

      return torch.cond(pred, true_fn, false_fn, (x, y))

    pred_cpu = torch.tensor(pred_value)
    x_cpu = torch.tensor([1.0, 2.0, 3.0])
    y_cpu = torch.tensor([4.0, 5.0, 6.0])
    pred_tpu = pred_cpu.to("tpu")
    x_tpu = x_cpu.to("tpu")
    y_tpu = y_cpu.to("tpu")

    gm = make_fx(model)(pred_tpu, x_tpu, y_tpu)
    self.assertTrue(_has_higher_order_cond(gm), gm.graph)

    executable = compiler.StaticCompiler(debug=True)(gm, [pred_tpu, x_tpu, y_tpu])
    result = executable([pred_tpu, x_tpu, y_tpu])
    expected = model(pred_cpu, x_cpu, y_cpu)

    _assert_tree_close(_to_cpu(result), expected)
    self.assertIn("stablehlo.if", executable.mlir_text)

  @parameterized.named_parameters(("true", True), ("false", False))
  def test_aot_autograd_backward(self, pred_value):

    def model(pred, x):

      def true_fn(v):
        return (torch.sin(v) * v).sum()

      def false_fn(v):
        return (torch.cos(v) * v).sum()

      return torch.cond(pred, true_fn, false_fn, (x,))

    x_cpu = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
    pred_cpu = torch.tensor(pred_value)
    expected = model(pred_cpu, x_cpu)
    expected.backward()

    x_tpu = x_cpu.detach().to("tpu").requires_grad_(True)
    pred_tpu = pred_cpu.to("tpu")
    actual = _compile_and_run(model, pred_tpu, x_tpu)
    actual.backward()

    utils.assert_close(actual.detach().cpu(), expected.detach())
    utils.assert_close(x_tpu.grad.cpu(), x_cpu.grad)

  def test_shape_mismatch_error(self):

    def model(pred, x):

      def true_fn(v):
        return v

      def false_fn(v):
        return v[:1]

      return torch.cond(pred, true_fn, false_fn, (x,))

    with self.assertRaisesRegex(Exception, "shape mismatch|same metadata"):
      _compile_and_run(
          model,
          torch.tensor(True, device="tpu"),
          torch.ones(3, device="tpu"),
      )

  def test_dtype_mismatch_error(self):

    def model(pred, x):

      def true_fn(v):
        return v

      def false_fn(v):
        return v.to(torch.int32)

      return torch.cond(pred, true_fn, false_fn, (x,))

    with self.assertRaisesRegex(Exception, "dtype mismatch|same dtype"):
      _compile_and_run(
          model,
          torch.tensor(True, device="tpu"),
          torch.ones(3, device="tpu"),
      )

  def test_pytree_mismatch_error(self):

    def model(pred, x):

      def true_fn(v):
        return (v, v + 1.0)

      def false_fn(v):
        return {"x": v, "y": v + 1.0}

      return torch.cond(pred, true_fn, false_fn, (x,))

    with self.assertRaisesRegex(Exception, "pytree|tree_spec|output spec"):
      _compile_and_run(
          model,
          torch.tensor(True, device="tpu"),
          torch.ones(3, device="tpu"),
      )


if __name__ == "__main__":
  absltest.main()

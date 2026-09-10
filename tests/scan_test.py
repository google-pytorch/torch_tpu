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
from torch._higher_order_ops import scan
import torch.utils._pytree as pytree
from torch_tpu._internal.compile.debug import TpuCompileDebug
from torch_tpu._internal.utils import test_utils as utils
from tests import oss_utils
from tests import seed_test_utils


def _compile_and_run(fn, *args, **kwargs):
  compiled = torch.compile(fn, backend="tpu", dynamic=False)
  return compiled(*args, **kwargs)


def _compile_and_get_stablehlo(model, *args):
  torch.compiler.reset()
  debugs: list[TpuCompileDebug] = []
  compiled = torch.compile(
      model,
      backend="tpu",
      dynamic=False,
      options={"serializable": False, "debug_callback": debugs.append},
  )
  compiled(*args)  # Trigger compilation + lowering
  texts = [text for debug in debugs for text in debug.stablehlo_forward_text]

  if not texts:
    raise AssertionError("no StableHLO captured")

  return texts[0]


class ScanOpTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch.compiler.reset()

  @parameterized.named_parameters(
      (
          "forward",
          False,  # reverse
          torch.tensor([10.0]),  # expected_carry
          torch.tensor([[1.0], [3.0], [6.0], [10.0]]),  # expected_ys
      ),
      (
          "reverse",
          True,  # reverse
          torch.tensor([10.0]),  # expected_carry
          torch.tensor([[10.0], [9.0], [7.0], [4.0]]),  # expected_ys
      ),
  )
  def test_basic_scan(self, reverse, expected_carry, expected_ys):
    def fn(carry, x):
      return carry + x, carry + x

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="tpu")
    init = torch.tensor([0.0], device="tpu")

    out_carry, out_ys = _compile_and_run(
        scan, fn, init, x, dim=0, reverse=reverse
    )

    utils.assert_close(out_carry.cpu(), expected_carry)
    utils.assert_close(out_ys.cpu(), expected_ys)

  def test_scan_with_nested_pytree(self):
    def fn(carry, x):
      c1, c2 = carry
      return (c1 + x, c2 * x), (c1 + x, c2 * x)

    x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
    init = (
        torch.tensor([0.0], device="tpu"),
        torch.tensor([1.0], device="tpu"),
    )

    expected_carry = (torch.tensor([6.0]), torch.tensor([6.0]))
    expected_ys = (
        torch.tensor([[1.0], [3.0], [6.0]]),
        torch.tensor([[1.0], [2.0], [6.0]]),
    )

    out_carry, out_ys = _compile_and_run(scan, fn, init, x, dim=0)

    utils.assert_close(
        pytree.tree_map(lambda t: t.cpu(), out_carry), expected_carry
    )
    utils.assert_close(pytree.tree_map(lambda t: t.cpu(), out_ys), expected_ys)

  def test_scan_with_multiple_inputs(self):
    def fn(carry, x):
      x1, x2 = x
      return carry + x1 + x2, carry + x1 + x2

    xs1 = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device="tpu")
    xs2 = torch.tensor([[10.0], [20.0], [30.0], [40.0]], device="tpu")
    init = torch.tensor([0.0], device="tpu")

    expected_carry = torch.tensor([110.0])
    expected_ys = torch.tensor([[11.0], [33.0], [66.0], [110.0]])

    # Compile a helper that calls scan with a tuple of xs.
    def run_scan(init_val, xs1_val, xs2_val):
      return scan(fn, init_val, (xs1_val, xs2_val))

    out_carry, out_ys = _compile_and_run(run_scan, init, xs1, xs2)

    utils.assert_close(out_carry.cpu(), expected_carry)
    utils.assert_close(out_ys.cpu(), expected_ys)

  def test_scan_3d_inputs(self):
    def fn(carry, x):
      return carry + x, carry + x

    init = torch.tensor([[0.0, 0.0]], device="tpu")
    xs = torch.tensor(
        [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]], device="tpu"
    )

    expected_carry = torch.tensor([[16.0, 20.0]])
    expected_ys = torch.tensor(
        [[[1.0, 2.0]], [[4.0, 6.0]], [[9.0, 12.0]], [[16.0, 20.0]]]
    )

    out_carry, out_ys = _compile_and_run(scan, fn, init, xs)

    utils.assert_close(out_carry.cpu(), expected_carry)
    utils.assert_close(out_ys.cpu(), expected_ys)

  def test_scan_with_different_dtypes(self):
    def fn(carry, x):
      return carry + x.to(torch.float32), carry.to(torch.int32)

    x = torch.tensor([1, 2, 3], dtype=torch.int32, device="tpu")
    init = torch.tensor([2.5], device="tpu")

    expected_carry = torch.tensor([8.5])
    expected_ys = torch.tensor([[2], [3], [5]], dtype=torch.int32)

    out_carry, out_ys = _compile_and_run(scan, fn, init, x, dim=0)

    utils.assert_close(out_carry.cpu(), expected_carry)
    utils.assert_close(out_ys.cpu(), expected_ys)

  def test_scan_with_extra_input(self):
    extra = torch.tensor([10.0], device="tpu")

    def fn(carry, x):
      return carry + x + extra, carry + x

    x = torch.tensor([1.0, 2.0], device="tpu")
    init = torch.tensor([0.0], device="tpu")

    expected_carry = torch.tensor([23.0])
    expected_ys = torch.tensor([[1.0], [13.0]])

    out_carry, out_ys = _compile_and_run(scan, fn, init, x, dim=0)

    utils.assert_close(out_carry.cpu(), expected_carry)
    utils.assert_close(out_ys.cpu(), expected_ys)

  def test_scan_with_complex_body(self):
    extra_input = torch.tensor([0.1, 0.2], device="tpu")

    def complex_body_fn(carry, x):
      y = torch.cos(
          torch.sum(torch.sin(x))
          + torch.sum(torch.cos(carry))
          + torch.sum(extra_input)
      )
      next_carry = torch.sin(carry * y)
      return next_carry, y

    def scan_reference(init_carry_val, xs_val):
      current_carry = init_carry_val
      outputs = []
      for i in range(xs_val.shape[0]):
        current_carry, y = complex_body_fn(current_carry, xs_val[i])
        outputs.append(y.unsqueeze(0))
      return current_carry, torch.cat(outputs, dim=0)

    xs = torch.randn(5, 3, device="tpu")
    init_carry = torch.randn(4, device="tpu")
    expected_carry, expected_outputs = scan_reference(init_carry, xs)

    out_carry, out_outputs = _compile_and_run(
        scan, complex_body_fn, init_carry, xs
    )

    utils.assert_close(out_carry.cpu(), expected_carry.cpu())
    utils.assert_close(out_outputs.cpu(), expected_outputs.cpu())

  def test_scan_with_seq_len_0(self):
    def fn(carry, x):
      return carry + x, carry + x

    x = torch.empty(0, 1, device="tpu")
    init = torch.tensor([5.0], device="tpu")

    if oss_utils.is_oss():
      # The wrapper's validation should raise this error during tracing.
      with self.assertRaisesRegex(
          RuntimeError,
          "All xs leaves must at least have 'dim' number of dimensions and scan"
          " dimension > 0",
      ):
        _compile_and_run(scan, fn, init, x, dim=0)
    else:
      out_carry, out_outputs = _compile_and_run(scan, fn, init, x, dim=0)

      utils.assert_close(out_carry.cpu(), init.cpu())
      utils.assert_close(out_outputs.cpu(), x.cpu())

  def test_scan_with_rnn(self):
    hidden_size = 8
    input_size = 4

    w_h = torch.randn(hidden_size, hidden_size, device="tpu")
    w_x = torch.randn(hidden_size, input_size, device="tpu")

    def rnn_step(h, x):
      new_h = torch.tanh(w_h @ h + w_x @ x)
      return new_h, new_h.clone()

    seq_len = 5
    xs = torch.randn(seq_len, input_size, device="tpu")
    h0 = torch.zeros(hidden_size, device="tpu")

    out_h, hs = _compile_and_run(scan, rnn_step, h0, xs, dim=0)

    h = h0
    for i in range(seq_len):
      h, _ = rnn_step(h, xs[i])
      utils.assert_close(h.cpu(), hs[i].cpu())
    utils.assert_close(h.cpu(), out_h.cpu())

  def test_multiple_scans(self):
    def double_scan(init, xs):
      def fn1(carry, x):
        return carry + x, carry + x

      def fn2(carry, x):
        return carry * x, carry * x

      c1, ys1 = scan(fn1, init, xs)
      c2, ys2 = scan(fn2, init, ys1)
      return c1, c2, ys2

    xs = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device="tpu")
    init = torch.tensor([1.0], device="tpu")

    expected_c1 = torch.tensor([11.0])
    expected_c2 = torch.tensor([616.0])
    expected_ys2 = torch.tensor([[2.0], [8.0], [56.0], [616.0]])

    out_c1, out_c2, out_ys2 = _compile_and_run(double_scan, init, xs)

    utils.assert_close(out_c1.cpu(), expected_c1)
    utils.assert_close(out_c2.cpu(), expected_c2)
    utils.assert_close(out_ys2.cpu(), expected_ys2)

  def test_scan_with_outside_computations(self):

    def full_computation(init, xs):
      def fn(carry, x):
        return carry + x, carry + x

      xs_processed = xs * 3.0
      c, ys = scan(fn, init, xs_processed)
      return c + 10.0, ys * 4.0

    xs = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device="tpu")
    init = torch.tensor([0.0], device="tpu")

    expected_c = torch.tensor([40.0])
    expected_ys = torch.tensor([[12.0], [36.0], [72.0], [120.0]])

    out_c, out_ys = _compile_and_run(full_computation, init, xs)

    utils.assert_close(out_c.cpu(), expected_c)
    utils.assert_close(out_ys.cpu(), expected_ys)


class ScanDeferredOperandsTest(seed_test_utils.RepeatableTest):
  """Scan whose carry and captured tensor are computed inside the graph."""

  def setUp(self):
    super().setUp()
    torch.compiler.reset()

  def test_scan_with_computed_carry_and_captured_tensor(self):
    def fn(init, xs, w):
      carry = init * 2.0
      scale = w + 1.0

      def combine_fn(c, x):
        new_c = c + x * scale
        return new_c, new_c.clone()

      return scan(combine_fn, carry, xs)

    init = torch.tensor([1.0, 2.0], device="tpu")
    xs = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device="tpu")
    w = torch.tensor([1.0, 2.0], device="tpu")

    expected_carry = torch.tensor([14.0, 22.0])
    expected_ys = torch.tensor([[4.0, 7.0], [8.0, 13.0], [14.0, 22.0]])

    out_carry, out_ys = _compile_and_run(fn, init, xs, w)

    utils.assert_close(out_carry.cpu(), expected_carry)
    utils.assert_close(out_ys.cpu(), expected_ys)

  def test_scan_with_computed_operands_that_require_grad(self):
    """The same scan, asked for its gradients.

    The motivating case is a joint graph, where AOTAutograd emits a second scan
    for the backward and every operand of it is computed rather than a graph
    input.
    """

    def fn(init, xs, w):
      carry = init * 2.0
      scale = w + 1.0

      def combine_fn(c, x):
        new_c = c + x * scale
        return new_c, new_c.clone()

      out_carry, out_ys = scan(combine_fn, carry, xs)
      return (out_carry * out_carry).sum() + out_ys.sum()

    def operands(device):
      return (
          torch.tensor([1.0, 2.0], device=device, requires_grad=True),
          torch.tensor(
              [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
              device=device,
              requires_grad=True,
          ),
          torch.tensor([1.0, 2.0], device=device, requires_grad=True),
      )

    expected = operands("cpu")
    fn(*expected).backward()

    actual = operands("tpu")
    _compile_and_run(fn, *actual).backward()

    for name, a, e in zip(("init", "xs", "w"), actual, expected):
      utils.assert_close(a.grad.cpu(), e.grad, preamble=f"grad of {name}")


class ScanLoweringTest(seed_test_utils.RepeatableTest):

  def test_scan_is_not_unrolled(self):

    class SimpleScanModel(torch.nn.Module):

      def forward(self, init, xs):
        def combine_fn(carry, x):
          return carry + x, carry + x

        return scan(combine_fn, init, xs)

    device = torch.device("tpu")
    init = torch.tensor([0.0], device=device)
    xs = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], device=device)

    model = SimpleScanModel()
    hlo = _compile_and_get_stablehlo(model, init, xs)

    self.assertIn(
        "stablehlo.while",
        hlo,
        msg="scan was unexpectedly unrolled: stablehlo.while not found",
    )


if __name__ == "__main__":
  absltest.main()

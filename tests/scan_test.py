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

import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch._higher_order_ops import associative_scan as _upstream_associative_scan
from torch._higher_order_ops import scan
import torch.fx
import torch.fx.traceback
import torch.utils._pytree as pytree
from torch_tpu._internal.compile import scan as tpu_scan
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.debug import TpuCompileDebug
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

try:
  # Importing `offload_ops` registers the `ao.offload` and `ao.reload` custom
  # ops that the offloaded scan bodies below are built from.
  # pylint: disable=g-import-not-at-top
  from torch._functorch._activation_offloading import offload_ops

  _HAS_AO_OPS = offload_ops is not None
except (ImportError, AttributeError):
  _HAS_AO_OPS = False


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

  def test_annotated_body_scans_in_reverse(self):
    # The annotation has to survive Dynamo and the `scan` HOP re-tracing the
    # body, so this matches the "reverse" case of `test_basic_scan` without
    # passing `reverse=True`.
    def fn(carry, x):
      with torch.fx.traceback.annotate(
          {tpu_scan.SCAN_REVERSE_ANNOTATION: True}
      ):
        return carry + x, carry + x

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="tpu")
    init = torch.tensor([0.0], device="tpu")

    out_carry, out_ys = _compile_and_run(
        scan, fn, init, x, dim=0, reverse=False
    )

    utils.assert_close(out_carry.cpu(), torch.tensor([10.0]))
    utils.assert_close(
        out_ys.cpu(), torch.tensor([[10.0], [9.0], [7.0], [4.0]])
    )

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

    # PyTorch < 2.14 rejects seq_len == 0 in scan during tracing, whereas
    # PyTorch >= 2.14 (used in Google3 and newest-deps OSS CI) supports it.
    if utils.parse_version(torch.__version__) < (2, 14, 0):
      self.skipTest(
          "PyTorch < 2.14 rejects seq_len == 0 in scan during tracing."
      )

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


def _body_with_custom_meta(custom=None):
  """Builds a one-op scan body, optionally with `custom` node metadata."""
  graph = torch.fx.Graph()
  x = graph.placeholder("x")
  y = graph.call_function(torch.ops.aten.neg.default, (x,))
  if custom is not None:
    y.meta["custom"] = custom
  graph.output((y,))
  return torch.fx.GraphModule(torch.nn.Module(), graph)


class BodyAsksForReverseTest(seed_test_utils.RepeatableTest):
  """Tests detection of the reverse-scan annotation on body nodes."""

  def test_annotated_body_asks_for_reverse(self):
    body = _body_with_custom_meta({tpu_scan.SCAN_REVERSE_ANNOTATION: True})
    self.assertTrue(tpu_scan._body_asks_for_reverse(body))

  def test_unannotated_body_does_not_ask_for_reverse(self):
    self.assertFalse(tpu_scan._body_asks_for_reverse(_body_with_custom_meta()))

  def test_annotation_set_to_false_does_not_ask_for_reverse(self):
    body = _body_with_custom_meta({tpu_scan.SCAN_REVERSE_ANNOTATION: False})
    self.assertFalse(tpu_scan._body_asks_for_reverse(body))

  def test_other_custom_keys_do_not_ask_for_reverse(self):
    body = _body_with_custom_meta({"some_other_key": True})
    self.assertFalse(tpu_scan._body_asks_for_reverse(body))

  def test_eager_body_does_not_ask_for_reverse(self):
    # An eager combine_fn exposes no graph to inspect.
    self.assertFalse(tpu_scan._body_asks_for_reverse(lambda *a: a))


def _offloaded_body(*, reload_slice=True):
  """Builds a scan body that reads one row of a stack kept on the host.

  Matches the shape produced by torchtitan's offload pass: the full stack is a
  single body operand, and the body indexes one row out of it with the loop
  counter. The reload therefore applies to the row, not to the operand.

  Args:
    reload_slice: Whether to copy the row back with `ao.reload`.
  """
  graph = torch.fx.Graph()
  stack = graph.placeholder("stack")
  row = graph.placeholder("row")
  sliced = graph.call_function(
      torch.ops.aten.index_select.default, (stack, 0, row)
  )
  out = sliced
  if reload_slice:
    out = graph.call_function(
        torch.ops.ao.reload.default, (sliced, torch.device("cpu"), [4], [1])
    )
  graph.output((out,))
  return torch.fx.GraphModule(torch.nn.Module(), graph)


@absltest.skipIf(
    not _HAS_AO_OPS,
    "`torch._functorch._activation_offloading.offload_ops` is not available in "
    "this PyTorch version",
)
class PinOffloadedBodyArgsTest(seed_test_utils.RepeatableTest):
  """Tests which body operands get marked as host memory."""

  def test_reloaded_operand_is_pinned_to_host(self):
    args = [torch.zeros(8, 4), torch.zeros(1, dtype=torch.int64)]
    pinned = tpu_scan._pin_offloaded_body_args(_offloaded_body(), args)
    self.assertTrue(tpu_torch_compile.is_pinned_host(pinned[0]))
    self.assertEqual(pinned[0].shape, args[0].shape)
    self.assertEqual(pinned[0].dtype, args[0].dtype)

  def test_loop_counter_is_not_pinned(self):
    # `index_select(stack, 0, row)` takes the stack and the loop counter. Only
    # the stack reaches the reload; following the other arguments would pin the
    # counter to host memory as well.
    args = [torch.zeros(8, 4), torch.zeros(1, dtype=torch.int64)]
    pinned = tpu_scan._pin_offloaded_body_args(_offloaded_body(), args)
    self.assertIs(pinned[1], args[1])

  def test_body_with_no_reload_is_left_alone(self):
    args = [torch.zeros(8, 4), torch.zeros(1, dtype=torch.int64)]
    pinned = tpu_scan._pin_offloaded_body_args(
        _offloaded_body(reload_slice=False), args
    )
    self.assertIs(pinned[0], args[0])
    self.assertIs(pinned[1], args[1])

  def test_operand_offloaded_inside_the_body_is_not_pinned(self):
    # The body offloads the operand itself, so it arrived on device.
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    offloaded = graph.call_function(torch.ops.ao.offload.default, (x,))
    reloaded = graph.call_function(
        torch.ops.ao.reload.default, (offloaded, torch.device("cpu"), [4], [1])
    )
    graph.output((reloaded,))
    body = torch.fx.GraphModule(torch.nn.Module(), graph)

    args = [torch.zeros(8, 4)]
    pinned = tpu_scan._pin_offloaded_body_args(body, args)
    self.assertIs(pinned[0], args[0])

  def test_saved_for_offloading_meta_pins_without_a_reload(self):
    args = [torch.zeros(8, 4), torch.zeros(1, dtype=torch.int64)]
    body = _offloaded_body(reload_slice=False)
    stack = next(n for n in body.graph.nodes if n.op == "placeholder")
    stack.meta["saved_for_offloading"] = True
    pinned = tpu_scan._pin_offloaded_body_args(body, args)
    self.assertTrue(tpu_torch_compile.is_pinned_host(pinned[0]))

  def test_eager_body_is_passed_through(self):
    # An eager or exported combine_fn exposes no graph to inspect, so the args
    # are returned unchanged. Callers need not distinguish the two cases.
    args = [torch.zeros(8, 4), torch.zeros(1, dtype=torch.int64)]
    self.assertIs(tpu_scan._pin_offloaded_body_args(lambda *a: a, args), args)

  def test_to_placeholder_preserves_pinned_host(self):
    pinned_in = tpu_torch_compile.placeholder(
        [8, 4], torch.float32, is_pinned_host=True
    )
    unpinned_in = tpu_torch_compile.placeholder(
        [8, 4], torch.float32, is_pinned_host=False
    )
    self.assertTrue(
        tpu_torch_compile.is_pinned_host(
            tpu_scan._to_placeholder(pinned_in, shape=[4])
        )
    )
    self.assertFalse(
        tpu_torch_compile.is_pinned_host(
            tpu_scan._to_placeholder(unpinned_in, shape=[4])
        )
    )

  def test_mixed_pre_pinned_and_reloaded_operands(self):
    graph = torch.fx.Graph()
    reloaded_unpinned = graph.placeholder("reloaded_unpinned")
    reloaded_prepinned = graph.placeholder("reloaded_prepinned")
    unreloaded_prepinned = graph.placeholder("unreloaded_prepinned")
    unreloaded_unpinned = graph.placeholder("unreloaded_unpinned")

    r1 = graph.call_function(
        torch.ops.ao.reload.default,
        (reloaded_unpinned, torch.device("cpu"), [4], [1]),
    )
    r2 = graph.call_function(
        torch.ops.ao.reload.default,
        (reloaded_prepinned, torch.device("cpu"), [4], [1]),
    )
    graph.output((r1, r2, unreloaded_prepinned, unreloaded_unpinned))
    body = torch.fx.GraphModule(torch.nn.Module(), graph)

    args = [
        tpu_torch_compile.placeholder([4], torch.float32, is_pinned_host=False),
        tpu_torch_compile.placeholder([4], torch.float32, is_pinned_host=True),
        tpu_torch_compile.placeholder([4], torch.float32, is_pinned_host=True),
        tpu_torch_compile.placeholder([4], torch.float32, is_pinned_host=False),
    ]
    pinned = tpu_scan._pin_offloaded_body_args(body, args)

    # Unpinned + reloaded -> replaced with new pinned placeholder.
    self.assertIsNot(pinned[0], args[0])
    self.assertTrue(tpu_torch_compile.is_pinned_host(pinned[0]))
    # Pre-pinned + reloaded -> skipped (already pinned), identity preserved.
    self.assertIs(pinned[1], args[1])
    self.assertTrue(tpu_torch_compile.is_pinned_host(pinned[1]))
    # Pre-pinned + unreloaded -> identity preserved, remains pinned.
    self.assertIs(pinned[2], args[2])
    self.assertTrue(tpu_torch_compile.is_pinned_host(pinned[2]))
    # Unpinned + unreloaded -> identity preserved, remains unpinned.
    self.assertIs(pinned[3], args[3])
    self.assertFalse(tpu_torch_compile.is_pinned_host(pinned[3]))


_SCAN_OFFSET = 1.0
_SCAN_BIAS = torch.ones(4)


def associative_scan(
    combine_fn, xs, dim, reverse=False, combine_mode="pointwise"
):
  """Upstream `associative_scan`, with CPU references run in `'generic'` mode.

  Some PyTorch releases (e.g. 2.11) reject `combine_mode='pointwise'` for CPU
  tensors. `'generic'` computes the same result, so the CPU references use it.
  TPU calls are passed through unchanged.
  """
  if all(t.device.type == "cpu" for t in pytree.tree_leaves(xs)):
    combine_mode = "generic"
  return _upstream_associative_scan(
      combine_fn, xs, dim, reverse=reverse, combine_mode=combine_mode
  )


def _add_with_global_offset(a, b):
  # Associative for any constant offset: (a + b + c) + d + c == a + (b + d + c)
  # + c. Reads `_SCAN_OFFSET` from the module globals at trace time.
  return a + b + _SCAN_OFFSET


def _add_with_global_bias(a, b):
  return a + b + _SCAN_BIAS


def _calls_tensor_helper(a, b):
  return _add_with_global_bias(a, b)


class CallableCacheKeyTest(seed_test_utils.RepeatableTest):
  """Tests which combine fns the eager fast path may cache a traced body for."""

  # pylint: disable=protected-access
  def test_pure_callables_have_keys(self):
    scale = 2.0
    self.assertIsNotNone(tpu_scan._callable_cache_key(torch.add))
    self.assertIsNotNone(tpu_scan._callable_cache_key(lambda a, b: a + b))
    self.assertIsNotNone(tpu_scan._callable_cache_key(lambda a, b: a * scale))
    self.assertIsNotNone(tpu_scan._callable_cache_key(_add_with_global_offset))

  def test_keys_track_scalar_values(self):
    def make(scale):
      return lambda a, b: a + b * scale

    self.assertEqual(
        tpu_scan._callable_cache_key(make(1.0)),
        tpu_scan._callable_cache_key(make(1.0)),
    )
    self.assertNotEqual(
        tpu_scan._callable_cache_key(make(1.0)),
        tpu_scan._callable_cache_key(make(2.0)),
    )
    # `1`, `1.0`, and `True` compare equal but trace to different dtypes.
    self.assertNotEqual(
        tpu_scan._callable_cache_key(make(1)),
        tpu_scan._callable_cache_key(make(1.0)),
    )

  def test_tensor_state_has_no_key(self):
    bias = torch.ones(4)
    self.assertIsNone(tpu_scan._callable_cache_key(lambda a, b: a + b + bias))
    self.assertIsNone(tpu_scan._callable_cache_key(_add_with_global_bias))
    self.assertIsNone(tpu_scan._callable_cache_key(_calls_tensor_helper))
    self.assertIsNone(tpu_scan._callable_cache_key(bias.add))
    self.assertIsNone(tpu_scan._callable_cache_key(torch.nn.ReLU()))


class AssociativeScanOpTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch.compiler.reset()

  @parameterized.named_parameters(
      ("eager_forward_dim0", False, False, 0),
      ("eager_reverse_dim0", False, True, 0),
      ("eager_forward_dim1", False, False, 1),
      ("eager_reverse_dim1", False, True, 1),
      ("compiled_forward_dim0", True, False, 0),
      ("compiled_reverse_dim0", True, True, 0),
      ("compiled_forward_dim1", True, False, 1),
      ("compiled_reverse_dim1", True, True, 1),
  )
  def test_associative_scan_pointwise_ops(self, compiled, reverse, dim):
    x_cpu = torch.randn(8, 16, dtype=torch.float32) + 0.5
    x_tpu = x_cpu.to("tpu")

    def run_fn(t):
      return (
          associative_scan(torch.add, t, dim=dim, reverse=reverse),
          associative_scan(torch.mul, t, dim=dim, reverse=reverse),
          associative_scan(torch.maximum, t, dim=dim, reverse=reverse),
          associative_scan(torch.minimum, t, dim=dim, reverse=reverse),
      )

    expected = run_fn(x_cpu)
    actual = _compile_and_run(run_fn, x_tpu) if compiled else run_fn(x_tpu)
    for act, exp in zip(actual, expected, strict=True):
      utils.assert_close(act.cpu(), exp, rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      ("eager_bfloat16", False, torch.bfloat16),
      ("compiled_bfloat16", True, torch.bfloat16),
      ("eager_int32", False, torch.int32),
      ("compiled_int32", True, torch.int32),
  )
  def test_associative_scan_dtypes(self, compiled, dtype):
    if dtype == torch.int32:
      x_cpu = torch.randint(-5, 6, (4, 16, 8), dtype=dtype)
    else:
      x_cpu = torch.randn(4, 16, 8, dtype=dtype)
    x_tpu = x_cpu.to("tpu")

    def run_fn(t):
      return (
          associative_scan(torch.add, t, dim=1),
          associative_scan(torch.maximum, t, dim=1),
          associative_scan(torch.minimum, t, dim=1),
      )

    # The TPU scan combines partial sums in a different order than the CPU
    # sequential loop, so low-precision results are compared against a
    # float32 reference with tolerances of a few bf16 ulps.
    expected = run_fn(x_cpu if dtype == torch.int32 else x_cpu.float())
    actual = _compile_and_run(run_fn, x_tpu) if compiled else run_fn(x_tpu)
    for act, exp in zip(actual, expected, strict=True):
      utils.assert_close(act.cpu(), exp.to(dtype), rtol=2e-2, atol=5e-2)

  @parameterized.named_parameters(
      ("eager_forward_dim1", False, False, 1),
      ("eager_reverse_dim1", False, True, 1),
      ("eager_forward_dim_neg1", False, False, -1),
      ("compiled_forward_dim1", True, False, 1),
      ("compiled_reverse_dim1", True, True, 1),
      ("compiled_forward_dim_neg1", True, False, -1),
  )
  def test_associative_scan_mamba_ssm_tuple(self, compiled, reverse, dim):
    def ssm_combine(left, right):
      a_l, b_l = left
      a_r, b_r = right
      return a_l * a_r, a_r * b_l + b_r

    gates_cpu = torch.sigmoid(torch.randn(2, 8, 16, dtype=torch.float32))
    tokens_cpu = torch.randn(2, 8, 16, dtype=torch.float32)
    gates_tpu = gates_cpu.to("tpu")
    tokens_tpu = tokens_cpu.to("tpu")

    def run_ssm(g, u):
      return associative_scan(
          ssm_combine,
          (g, u),
          dim=dim,
          reverse=reverse,
          combine_mode="pointwise",
      )

    exp_g, exp_u = run_ssm(gates_cpu, tokens_cpu)
    act_g, act_u = (
        _compile_and_run(run_ssm, gates_tpu, tokens_tpu)
        if compiled
        else run_ssm(gates_tpu, tokens_tpu)
    )
    utils.assert_close(act_g.cpu(), exp_g, rtol=1e-4, atol=1e-4)
    utils.assert_close(act_u.cpu(), exp_u, rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      ("eager", False),
      ("compiled", True),
  )
  def test_associative_scan_high_rank_and_trailing_one(self, compiled):
    x4d_cpu = torch.randn(2, 3, 8, 4, dtype=torch.float32)
    x3d_t1_cpu = torch.randn(2, 8, 1, dtype=torch.float32)
    x4d_tpu = x4d_cpu.to("tpu")
    x3d_t1_tpu = x3d_t1_cpu.to("tpu")

    def run_fn(a, b):
      return (
          associative_scan(torch.add, a, dim=2),
          associative_scan(torch.add, b, dim=1),
      )

    exp_4d, exp_3d = run_fn(x4d_cpu, x3d_t1_cpu)
    act_4d, act_3d = (
        _compile_and_run(run_fn, x4d_tpu, x3d_t1_tpu)
        if compiled
        else run_fn(x4d_tpu, x3d_t1_tpu)
    )
    utils.assert_close(act_4d.cpu(), exp_4d, rtol=1e-4, atol=1e-4)
    utils.assert_close(act_3d.cpu(), exp_3d, rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      ("eager", False),
      ("compiled", True),
  )
  def test_associative_scan_seq_len_one(self, compiled):
    x_cpu = torch.randn(2, 1, 8, dtype=torch.float32)
    x_tpu = x_cpu.to("tpu")

    def run_fn(t):
      return associative_scan(torch.add, t, dim=1)

    expected = run_fn(x_cpu)
    actual = _compile_and_run(run_fn, x_tpu) if compiled else run_fn(x_tpu)
    utils.assert_close(actual.cpu(), expected)

  @parameterized.named_parameters(
      ("eager_forward", False, False),
      ("eager_reverse", False, True),
      ("compiled_forward", True, False),
      ("compiled_reverse", True, True),
  )
  def test_associative_scan_non_identity_combiner(self, compiled, reverse):
    # `affine_add` contains a constant (lowers via Tier 2 `stablehlo.while`),
    # while `bilinear_combine` is pure elementwise with no constant (lowers via
    # Tier 2 `Slice-Init` + `chlo.scan`).
    def affine_add(a, b):
      return a + b + 1.0

    def bilinear_combine(a, b):
      return a + b + a * b

    x_cpu = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4) * 0.1
    x_tpu = x_cpu.to("tpu")

    def run_fn(t):
      return (
          associative_scan(affine_add, t, dim=1, reverse=reverse),
          associative_scan(bilinear_combine, t, dim=1, reverse=reverse),
      )

    exp_affine, exp_bilinear = run_fn(x_cpu)
    act_affine, act_bilinear = (
        _compile_and_run(run_fn, x_tpu) if compiled else run_fn(x_tpu)
    )
    utils.assert_close(act_affine.cpu(), exp_affine, rtol=1e-4, atol=1e-4)
    utils.assert_close(act_bilinear.cpu(), exp_bilinear, rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      ("eager", False),
      ("compiled", True),
  )
  def test_associative_scan_captured_tensor_closure(self, compiled):
    bias_cpu = torch.tensor([0.5, -0.5, 1.0, -1.0], dtype=torch.float32)
    bias_tpu = bias_cpu.to("tpu")
    x_cpu = torch.randn(2, 6, 4, dtype=torch.float32)
    x_tpu = x_cpu.to("tpu")

    def run_cpu(t):
      return associative_scan(lambda a, b: a + b + bias_cpu, t, dim=1)

    def run_tpu(t):
      return associative_scan(lambda a, b: a + b + bias_tpu, t, dim=1)

    expected = run_cpu(x_cpu)
    actual = _compile_and_run(run_tpu, x_tpu) if compiled else run_tpu(x_tpu)
    utils.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      ("eager", False),
      ("compiled", True),
  )
  def test_associative_scan_autograd(self, compiled):
    def fn(x):
      y = associative_scan(lambda a, b: a + b, x, dim=1)
      return (y * y).sum()

    x_cpu = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    x_tpu = x_cpu.detach().clone().to("tpu").requires_grad_(True)

    fn(x_cpu).backward()
    out_tpu = _compile_and_run(fn, x_tpu) if compiled else fn(x_tpu)
    out_tpu.backward()

    utils.assert_close(x_tpu.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      ("compiled", True, False),
      ("compiled_reverse", True, True),
      ("eager_autograd", False, False),
  )
  def test_associative_scan_pointwise_bypasses_upstream_frontend(
      self, compiled, reverse
  ):
    # Some PyTorch releases (e.g. 2.11) reject `combine_mode='pointwise'` in
    # upstream's frontend for anything but CUDA and XPU, so TPU calls that the
    # eager fast path does not handle must reach the HOP without it.
    x_cpu = torch.randn(2, 6, 4, dtype=torch.float32)
    x_tpu = x_cpu.to("tpu").requires_grad_(not compiled)

    def run_fn(t):
      return associative_scan(torch.add, t, dim=1, reverse=reverse)

    expected = run_fn(x_cpu)
    with mock.patch.object(
        tpu_scan,
        "_SAVED_ORIG_ASSOCIATIVE_SCAN",
        side_effect=AssertionError("upstream frontend used for a TPU scan"),
    ):
      actual = _compile_and_run(run_fn, x_tpu) if compiled else run_fn(x_tpu)
    utils.assert_close(actual.detach().cpu(), expected, rtol=1e-4, atol=1e-4)

  def test_associative_scan_eager_cache_tracks_global_scalars(self):
    # The eager fast path caches the traced body. Changing a global scalar
    # that `combine_fn` reads must not reuse a body with the old value baked
    # in.
    x_cpu = torch.randn(2, 6, 4, dtype=torch.float32)
    x_tpu = x_cpu.to("tpu")
    steps = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    for offset in (1.0, 2.0):
      # Upstream rejects float lifted arguments, so use the closed form:
      # y_k = x_0 + ... + x_k + k * offset.
      expected = x_cpu.cumsum(dim=1) + steps * offset
      with mock.patch.object(sys.modules[__name__], "_SCAN_OFFSET", offset):
        actual = associative_scan(_add_with_global_offset, x_tpu, dim=1)
      utils.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)

  @parameterized.named_parameters(
      ("eager", False),
      ("compiled", True),
  )
  def test_associative_scan_bitwise_and_signed(self, compiled):
    # Negative values catch a wrong AND identity (e.g. INT32_MAX instead of -1).
    x_cpu = torch.randint(-8, 8, (2, 6, 4), dtype=torch.int32)
    x_tpu = x_cpu.to("tpu")

    def run_fn(t):
      return associative_scan(torch.bitwise_and, t, dim=1)

    expected = run_fn(x_cpu)
    actual = _compile_and_run(run_fn, x_tpu) if compiled else run_fn(x_tpu)
    utils.assert_close(actual.cpu(), expected)

  @parameterized.named_parameters(
      ("forward_dim_neg1", False, -1),
      ("reverse_dim1", True, 1),
  )
  def test_associative_scan_lowers_to_chlo_scan_without_transpose(
      self, reverse, dim
  ):
    def ssm_combine(left, right):
      a_l, b_l = left
      a_r, b_r = right
      return a_l * a_r, a_r * b_l + b_r

    class SsmScanModel(torch.nn.Module):

      def forward(self, gates, tokens):
        return associative_scan(
            ssm_combine,
            (gates, tokens),
            dim=dim,
            reverse=reverse,
            combine_mode="pointwise",
        )

    gates = torch.rand(2, 16, 64, device="tpu")
    tokens = torch.randn(2, 16, 64, device="tpu")
    hlo = _compile_and_get_stablehlo(SsmScanModel(), gates, tokens)

    self.assertIn("chlo.scan", hlo)
    self.assertNotIn("stablehlo.while", hlo)
    self.assertNotIn("stablehlo.transpose", hlo)


if __name__ == "__main__":
  absltest.main()

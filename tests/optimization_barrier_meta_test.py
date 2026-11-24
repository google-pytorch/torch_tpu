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

from collections.abc import Callable
from typing import Any
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.autograd import forward_ad
from torch.utils import _python_dispatch
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class _BarrierOperandCounts(_python_dispatch.TorchDispatchMode):
  """Records operand counts seen by `tpu::_optimization_barrier`."""

  def __init__(self):
    super().__init__()
    self.counts: list[int] = []

  def __torch_dispatch__(
      self,
      func: Any,
      types: tuple[type[Any], ...],
      args: tuple[Any, ...] = (),
      kwargs: dict[str, Any] | None = None,
  ) -> Any:
    # pylint: disable=protected-access
    if func is torch.ops.tpu._optimization_barrier.default:
      self.counts.append(len(args[0]))
    # pylint: enable=protected-access
    return func(*args, **(kwargs or {}))


def _barrier_loss(x: torch.Tensor) -> torch.Tensor:
  (y,) = torch.tpu.ops.optimization_barrier([x * x])
  return (y * y).sum()


class OptimizationBarrierMetaTest(seed_test_utils.RepeatableTest):

  def test_backward_mixed_inputs_and_partial_grad(self) -> None:
    frozen = torch.empty((3, 2), device="meta").t()
    a = torch.empty((2, 3), device="meta", requires_grad=True)
    index = torch.empty((2,), dtype=torch.int32, device="meta")
    b = torch.empty((2, 3), device="meta", requires_grad=True)

    with warnings.catch_warnings():
      warnings.filterwarnings(
          "error", message=".*autograd kernel was not registered"
      )
      outputs = torch.tpu.ops.optimization_barrier([frozen, a, index, b])
      self.assertEqual(
          [t.requires_grad for t in outputs], [False, True, False, True]
      )
      self.assertEqual(outputs[0].stride(), frozen.stride())
      self.assertEqual(
          outputs[1].grad_fn.name(), "OptimizationBarrierBackward0"
      )

      with _BarrierOperandCounts() as partial_calls:
        (grad_a,) = torch.autograd.grad(
            (outputs[1] + outputs[3]).sum(), [a], retain_graph=True
        )
      self.assertEqual(partial_calls.counts, [1])
      self.assertEqual(grad_a.device.type, "meta")
      self.assertEqual(grad_a.shape, a.shape)

      with _BarrierOperandCounts() as full_calls:
        (outputs[1] + outputs[3]).sum().backward()
      self.assertEqual(full_calls.counts, [2])

    self.assertIsNone(frozen.grad)
    self.assertIsNotNone(a.grad)
    self.assertIsNotNone(b.grad)
    self.assertEqual(a.grad.device.type, "meta")
    self.assertEqual(a.grad.shape, a.shape)
    self.assertEqual(b.grad.device.type, "meta")
    self.assertEqual(b.grad.shape, b.shape)

  def test_double_backward_fences_gradients_together(self) -> None:
    a = torch.empty((4,), device="meta", requires_grad=True)
    b = torch.empty((4,), device="meta", requires_grad=True)

    with _BarrierOperandCounts() as calls:
      y_a, y_b = torch.tpu.ops.optimization_barrier([a * a, b * b])
      grad_a, grad_b = torch.autograd.grad(
          (y_a * y_b).sum(), (a, b), create_graph=True
      )
      (grad_a + grad_b).sum().backward()

    # Forward, 1st backward, and 2nd backward through the 1st-bwd and fwd nodes.
    self.assertEqual(calls.counts, [2, 2, 2, 2])
    self.assertIsNotNone(a.grad)
    self.assertIsNotNone(b.grad)
    self.assertEqual(a.grad.shape, a.shape)
    self.assertEqual(b.grad.shape, b.shape)

  def test_fires_node_creation_hooks(self) -> None:
    if utils.parse_version(torch.__version__) < (2, 14, 0):
      self.skipTest("node_creation_hook requires PyTorch >= 2.14.")
    x = torch.empty((2,), device="meta", requires_grad=True)
    created_nodes = []

    with torch.autograd.graph.node_creation_hook(created_nodes.append):
      (y,) = torch.tpu.ops.optimization_barrier([x])

    self.assertIn(y.grad_fn, created_nodes)

  def test_compiled_autograd_is_traceable(self) -> None:
    a = torch.empty((4,), device="meta", requires_grad=True)
    frozen = torch.empty((4,), device="meta")
    compiler_fn = torch.compile(backend="eager", fullgraph=True)

    # pylint: disable=protected-access
    enable_compiled_autograd = torch._dynamo.compiled_autograd._enable
    # pylint: enable=protected-access
    with enable_compiled_autograd(compiler_fn):
      a_fenced, frozen_fenced = torch.tpu.ops.optimization_barrier(
          [a * 2.0, frozen]
      )
      (a_fenced + frozen_fenced).sum().backward()

    self.assertIsNotNone(a.grad)
    self.assertEqual(a.grad.device.type, "meta")
    self.assertEqual(a.grad.shape, a.shape)
    self.assertIsNone(frozen.grad)

  def test_forward_ad_mixed_list_passes_tangents_through_one_barrier(
      self,
  ) -> None:
    a = torch.empty((4,), device="meta")
    da = torch.empty((4,), device="meta")
    b = torch.empty((2, 3), device="meta")
    db = torch.empty((2, 3), device="meta")
    constant = torch.empty((4,), device="meta")
    index = torch.empty((4,), dtype=torch.int32, device="meta")

    with forward_ad.dual_level():
      inputs = [
          forward_ad.make_dual(a, da),
          constant,
          forward_ad.make_dual(b, db),
          index,
      ]
      with _BarrierOperandCounts() as calls:
        outputs = torch.tpu.ops.optimization_barrier(inputs)
      tangents = [forward_ad.unpack_dual(out).tangent for out in outputs]

    self.assertEqual(calls.counts, [4, 2])
    self.assertIsNotNone(tangents[0])
    self.assertEqual(tangents[0].device.type, "meta")
    self.assertEqual(tangents[0].shape, a.shape)
    self.assertIsNone(tangents[1])
    self.assertIsNotNone(tangents[2])
    self.assertEqual(tangents[2].device.type, "meta")
    self.assertEqual(tangents[2].shape, b.shape)
    self.assertIsNone(tangents[3])

  @parameterized.named_parameters(
      dict(
          testcase_name="grad",
          transform=lambda x, v: torch.func.grad(_barrier_loss)(x),
      ),
      dict(
          testcase_name="vjp",
          transform=lambda x, v: torch.func.vjp(_barrier_loss, x)[1](v.sum())[
              0
          ],
      ),
      dict(
          testcase_name="jvp",
          transform=lambda x, v: torch.func.jvp(_barrier_loss, (x,), (v,))[1],
      ),
      dict(
          testcase_name="jvp_of_grad",
          transform=lambda x, v: torch.func.jvp(
              torch.func.grad(_barrier_loss), (x,), (v,)
          )[1],
      ),
      dict(
          testcase_name="grad_of_jvp",
          transform=lambda x, v: torch.func.grad(
              lambda t: torch.func.jvp(_barrier_loss, (t,), (v,))[1]
          )(x),
      ),
  )
  def test_func_transforms(
      self,
      transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
  ) -> None:
    x = torch.empty((6,), device="meta")
    v = torch.empty((6,), device="meta")

    result = transform(x, v)

    self.assertEqual(result.device.type, "meta")
    self.assertIn(result.shape, (x.shape, torch.Size([])))


if __name__ == "__main__":
  absltest.main()

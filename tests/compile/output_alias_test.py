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

"""Tests that compiled graph outputs keep the aliasing of eager mode."""

from collections.abc import Callable, Sequence
import pickle

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.utils import _pytree as pytree
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.export import export as torch_tpu_export
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


def _views_and_slice(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
  y = x * 2
  return y, y.view(2, 8), y[1:]


def _transpose(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
  y = x + 1
  return y, y.t()


def _views_only(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
  y = x * 3
  return y[0], y[1:]


def _split(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
  return tuple(torch.split(x * 2, 1))


def _duplicate(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
  y = x * 2
  return y, y


def _independent(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
  return x * 2, x + 1


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
  return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _aliasing(outputs: Sequence[torch.Tensor]) -> list[list[bool]]:
  return [[_shares_storage(a, b) for b in outputs] for a in outputs]


class OutputAliasTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    torch.compiler.reset()

  def _run(
      self,
      fn: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
      device: str,
      backend: str | None,
      requires_grad: bool,
  ) -> tuple[list[list[bool]], list[torch.Tensor]]:
    """Returns the output aliasing and the outputs after a write to output 0."""
    x = torch.arange(16.0).view(4, 4).to(device)
    x.requires_grad_(requires_grad)
    compiled_fn = torch.compile(fn, backend=backend) if backend else fn
    outputs = compiled_fn(x)
    aliasing = _aliasing(outputs)
    with torch.no_grad():
      outputs[0].add_(100)
    # Autograd checks the views when they are read, so read them before detach.
    return aliasing, [output.cpu().detach() for output in outputs]

  @parameterized.product(
      fn=[
          _views_and_slice,
          _transpose,
          _views_only,
          _split,
          _duplicate,
          _independent,
      ],
      requires_grad=[False, True],
  )
  def test_compiled_outputs_alias_like_eager(self, fn, requires_grad):
    expected_aliasing, expected_outputs = self._run(
        fn, "cpu", backend=None, requires_grad=requires_grad
    )
    aliasing, outputs = self._run(
        fn, "tpu", backend="tpu", requires_grad=requires_grad
    )

    self.assertEqual(aliasing, expected_aliasing)
    self.assertEqual(len(outputs), len(expected_outputs))
    for output, expected_output in zip(outputs, expected_outputs, strict=True):
      utils.assert_close(output, expected_output)

  def test_aliased_output_reconstructor_survives_pickle(self):
    base = torch.arange(16.0).view(4, 4).clone()
    _, spec = pytree.tree_flatten((base, base[1:]))
    layout = torch_tpu_export._TensorLayout(  # pylint: disable=protected-access
        dtype=torch.float32,
        shape=torch.Size([3, 4]),
        stride=(4, 1),
        storage_offset=4,
    )
    reconstructor = torch_tpu_export._AliasedOutputReconstructor(  # pylint: disable=protected-access
        leaf_reconstructor=tpu_torch_compile.FxOutputReconstructor(
            [torch_tpu_export._DedupedTensorOutput(tensor_idx=0), None],  # pylint: disable=protected-access
            [
                torch_tpu_export._TensorLayout(  # pylint: disable=protected-access
                    dtype=torch.float32,
                    shape=torch.Size([4, 4]),
                    stride=(4, 1),
                    storage_offset=0,
                )
            ],
            pytree.tree_structure([0, 0]),
        ),
        aliased_layouts={1: (0, layout)},
        spec=spec,
    )

    restored = pickle.loads(pickle.dumps(reconstructor))
    out_base, out_view = restored(inputs=[], deduped_outputs=[base])

    self.assertIs(out_base, base)
    self.assertIs(out_view._base, base)  # pylint: disable=protected-access
    utils.assert_close(out_view, base[1:])


if __name__ == "__main__":
  absltest.main()

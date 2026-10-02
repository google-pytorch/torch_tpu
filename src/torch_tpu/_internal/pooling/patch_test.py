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

"""Unit tests for max pooling monkey patch module.

The test target does not run on a TPU, so the TPU-only behavior (argument
normalization and route selection) is covered through the pure helpers
`_normalize_pool_args` and `_select_route`. The normalized arguments are
checked numerically by feeding them to the CPU `aten::max_pool*d` kernel,
which has the same signature as `tpu::max_pool*d`, and comparing against
the original `F.max_pool*d` called with the raw user arguments.
"""

# TODO(wan): revisit DECISION (b/568593050): patch_test has no TPU, so TPU
# routing is covered by table tests of the pure helpers rather than numeric
# checks of the tpu::max_pool*d ops; add a TPU numeric test if one is wanted.

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single, _triple
from torch_tpu._internal.pooling import patch as pooling_patch
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils

_Route = pooling_patch._Route

_ORIG_MAX_POOL = {
    1: pooling_patch._ORIG_F_MAX_POOL1D,
    2: pooling_patch._ORIG_F_MAX_POOL2D,
    3: pooling_patch._ORIG_F_MAX_POOL3D,
}
_ATEN_MAX_POOL = {
    1: torch.ops.aten.max_pool1d,
    2: torch.ops.aten.max_pool2d,
    3: torch.ops.aten.max_pool3d,
}
_TO_TUPLE = {1: _single, 2: _pair, 3: _triple}
_PATCHED_MAX_POOL = {
    1: pooling_patch.patched_max_pool1d,
    2: pooling_patch.patched_max_pool2d,
    3: pooling_patch.patched_max_pool3d,
}


def _input(ndim: int) -> torch.Tensor:
  """Returns a deterministic (N, C, *spatial) CPU input without ties."""
  spatial = {1: (11,), 2: (9, 10), 3: (5, 6, 7)}[ndim]
  numel = 2 * 3 * torch.Size(spatial).numel()
  gen = torch.Generator().manual_seed(0)
  return torch.randperm(numel, generator=gen).float().reshape(2, 3, *spatial)


class PatchTest(seed_test_utils.RepeatableTest):

  def tearDown(self):
    super().tearDown()
    pooling_patch.unpatch_max_pool()

  def test_is_tpu_tensor(self):
    self.assertFalse(pooling_patch._is_tpu_tensor(torch.zeros(2)))
    self.assertFalse(
        pooling_patch._is_tpu_tensor(torch.zeros(2, device="meta"))
    )
    self.assertFalse(pooling_patch._is_tpu_tensor(123))
    self.assertFalse(pooling_patch._is_tpu_tensor(None))

  def test_patch_and_unpatch_lifecycle(self):
    orig_f1 = F.max_pool1d
    orig_f2 = F.max_pool2d
    orig_f3 = F.max_pool3d

    pooling_patch.patch_max_pool()
    self.assertIs(F.max_pool1d, pooling_patch.patched_max_pool1d)
    self.assertIs(F.max_pool2d, pooling_patch.patched_max_pool2d)
    self.assertIs(F.max_pool3d, pooling_patch.patched_max_pool3d)

    pooling_patch.unpatch_max_pool()
    self.assertIs(F.max_pool1d, orig_f1)
    self.assertIs(F.max_pool2d, orig_f2)
    self.assertIs(F.max_pool3d, orig_f3)

  @parameterized.named_parameters(
      dict(
          testcase_name="1d_ints",
          to_tuple=_single,
          kernel_size=3,
          stride=2,
          padding=1,
          dilation=1,
          expected=([3], [2], [1], [1]),
      ),
      dict(
          testcase_name="2d_ints",
          to_tuple=_pair,
          kernel_size=3,
          stride=2,
          padding=1,
          dilation=1,
          expected=([3, 3], [2, 2], [1, 1], [1, 1]),
      ),
      dict(
          testcase_name="3d_ints",
          to_tuple=_triple,
          kernel_size=3,
          stride=2,
          padding=1,
          dilation=1,
          expected=([3, 3, 3], [2, 2, 2], [1, 1, 1], [1, 1, 1]),
      ),
      dict(
          testcase_name="2d_stride_none_defaults_to_kernel",
          to_tuple=_pair,
          kernel_size=(3, 2),
          stride=None,
          padding=0,
          dilation=1,
          expected=([3, 2], [3, 2], [0, 0], [1, 1]),
      ),
      dict(
          testcase_name="3d_stride_none_defaults_to_kernel",
          to_tuple=_triple,
          kernel_size=2,
          stride=None,
          padding=0,
          dilation=1,
          expected=([2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1]),
      ),
      dict(
          testcase_name="2d_mixed_tuples",
          to_tuple=_pair,
          kernel_size=(3, 2),
          stride=(2, 1),
          padding=(1, 0),
          dilation=(2, 1),
          expected=([3, 2], [2, 1], [1, 0], [2, 1]),
      ),
      dict(
          testcase_name="1d_list_args",
          to_tuple=_single,
          kernel_size=[4],
          stride=[3],
          padding=[2],
          dilation=[2],
          expected=([4], [3], [2], [2]),
      ),
  )
  def test_normalize_pool_args(
      self, to_tuple, kernel_size, stride, padding, dilation, expected
  ):
    self.assertEqual(
        pooling_patch._normalize_pool_args(
            to_tuple, kernel_size, stride, padding, dilation
        ),
        expected,
    )

  @parameterized.product(
      ndim=(1, 2, 3),
      args=(
          dict(kernel_size=3, stride=2, padding=1, dilation=1),
          dict(kernel_size=2, stride=None, padding=0, dilation=1),
          dict(kernel_size=3, stride=1, padding=1, dilation=2),
          dict(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True),
      ),
  )
  def test_normalized_args_match_cpu_reference(self, ndim, args):
    """Normalized args give the same pooling result as the raw user args."""
    args = dict(args)
    ceil_mode = args.pop("ceil_mode", False)
    x = _input(ndim)
    k, s, p, d = pooling_patch._normalize_pool_args(_TO_TUPLE[ndim], **args)

    # aten::max_pool*d takes the same positional arguments as tpu::max_pool*d.
    actual = _ATEN_MAX_POOL[ndim](x, k, s, p, d, ceil_mode)
    expected = _ORIG_MAX_POOL[ndim](x, ceil_mode=ceil_mode, **args)

    test_utils.assert_close(actual, expected, rtol=0, atol=0)

  @parameterized.named_parameters(
      dict(
          testcase_name="cpu_falls_back",
          is_tpu=False,
          return_indices=False,
          dilation=[1, 1],
          needs_grad=False,
          expected=_Route.FALLBACK,
      ),
      dict(
          testcase_name="cpu_with_grad_falls_back",
          is_tpu=False,
          return_indices=False,
          dilation=[1, 1],
          needs_grad=True,
          expected=_Route.FALLBACK,
      ),
      dict(
          testcase_name="tpu_return_indices_falls_back",
          is_tpu=True,
          return_indices=True,
          dilation=[1, 1],
          needs_grad=False,
          expected=_Route.FALLBACK,
      ),
      dict(
          testcase_name="tpu_dilated_falls_back",
          is_tpu=True,
          return_indices=False,
          dilation=[1, 2],
          needs_grad=False,
          expected=_Route.FALLBACK,
      ),
      dict(
          testcase_name="tpu_dilated_with_grad_falls_back",
          is_tpu=True,
          return_indices=False,
          dilation=[2, 2, 2],
          needs_grad=True,
          expected=_Route.FALLBACK,
      ),
      dict(
          testcase_name="tpu_eval_is_native",
          is_tpu=True,
          return_indices=False,
          dilation=[1],
          needs_grad=False,
          expected=_Route.NATIVE,
      ),
      dict(
          testcase_name="tpu_train_is_autograd",
          is_tpu=True,
          return_indices=False,
          dilation=[1, 1, 1],
          needs_grad=True,
          expected=_Route.AUTOGRAD,
      ),
  )
  def test_select_route(
      self, is_tpu, return_indices, dilation, needs_grad, expected
  ):
    self.assertIs(
        pooling_patch._select_route(
            is_tpu=is_tpu,
            return_indices=return_indices,
            dilation=dilation,
            needs_grad=needs_grad,
        ),
        expected,
    )

  @parameterized.parameters(1, 2, 3)
  def test_patched_max_pool_cpu_fallback_matches_reference(self, ndim):
    x = _input(ndim)

    out = _PATCHED_MAX_POOL[ndim](x, kernel_size=3, stride=2, padding=1)

    self.assertEqual(out.device.type, "cpu")
    test_utils.assert_close(
        out,
        _ORIG_MAX_POOL[ndim](x, kernel_size=3, stride=2, padding=1),
        rtol=0,
        atol=0,
    )

  def test_patched_max_pool2d_return_indices_fallback(self):
    x = _input(2)

    out, idx = pooling_patch.patched_max_pool2d(
        x, kernel_size=2, stride=2, return_indices=True
    )

    expected_out, expected_idx = pooling_patch._ORIG_F_MAX_POOL2D(
        x, kernel_size=2, stride=2, return_indices=True
    )
    test_utils.assert_close(out, expected_out, rtol=0, atol=0)
    test_utils.assert_close(idx, expected_idx, rtol=0, atol=0)

  def test_patched_max_pool2d_cpu_backward_matches_reference(self):
    x = _input(2).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()

    pooling_patch.patched_max_pool2d(
        x, kernel_size=3, stride=2
    ).sum().backward()
    pooling_patch._ORIG_F_MAX_POOL2D(
        x_ref, kernel_size=3, stride=2
    ).sum().backward()

    test_utils.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)


if __name__ == "__main__":
  absltest.main()

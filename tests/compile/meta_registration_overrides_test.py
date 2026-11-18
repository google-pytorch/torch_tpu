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

"""Unit tests for Meta kernel overrides in meta_registration_overrides."""

import inspect
from unittest import mock

from absl.testing import absltest
import torch
from torch import _decomp
from torch._decomp import decompositions
from torch._subclasses import fake_tensor
from torch_tpu._internal.compile import meta_registration_overrides as overrides
from tests import seed_test_utils

# pylint: disable=protected-access
_META_KEY = torch._C.DispatchKey.Meta
_BMM_DTYPE = torch.ops.aten.bmm.dtype
_BMM_DTYPE_OUT = torch.ops.aten.bmm.dtype_out


def _restore_op(op, orig_kernel, had_table, orig_table) -> None:
  op.py_kernels.pop(_META_KEY, None)
  if orig_kernel is not None:
    op.py_kernels[_META_KEY] = orig_kernel
  if had_table:
    _decomp.meta_table[op] = orig_table
  else:
    _decomp.meta_table.pop(op, None)
  op._dispatch_cache.clear()


class MetaRegistrationOverridesTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    for op in (_BMM_DTYPE, _BMM_DTYPE_OUT):
      orig_kernel = op.py_kernels.get(_META_KEY)
      had_table = op in _decomp.meta_table
      orig_table = _decomp.meta_table.get(op)
      self.addCleanup(_restore_op, op, orig_kernel, had_table, orig_table)
    self._meta_lib = torch.library.Library("aten", "IMPL", "Meta")
    self.addCleanup(self._meta_lib._destroy)

  def test_bmm_dtype_meta_resizes_empty_out_tensor(self):
    with fake_tensor.FakeTensorMode():
      lhs = torch.empty((2, 4, 6), dtype=torch.bfloat16)
      rhs = torch.empty((2, 6, 8), dtype=torch.bfloat16)
      out = torch.empty((0,), dtype=torch.float32)
      result = overrides._bmm_dtype_meta(lhs, rhs, torch.float32, out=out)
      self.assertIs(result, out)
      self.assertEqual(out.shape, (2, 4, 8))
      self.assertEqual(out.dtype, torch.float32)

  def test_bmm_dtype_meta_rejects_mismatched_out_dtype(self):
    with fake_tensor.FakeTensorMode():
      lhs = torch.empty((2, 4, 6), dtype=torch.bfloat16)
      rhs = torch.empty((2, 6, 8), dtype=torch.bfloat16)
      out = torch.empty((0,), dtype=torch.bfloat16)
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing exact_dtype check.
          RuntimeError, r"Expected out tensor to have dtype torch\.float32"
      ):
        overrides._bmm_dtype_meta(lhs, rhs, torch.float32, out=out)

  def test_patch_bmm_dtype_meta_patches_unwrapped_and_missing_overloads(self):
    _BMM_DTYPE.py_kernels[_META_KEY] = lambda x, y, out_dtype: (x, y, out_dtype)
    _BMM_DTYPE_OUT.py_kernels.pop(_META_KEY, None)
    _decomp.meta_table.pop(_BMM_DTYPE_OUT, None)
    _BMM_DTYPE._dispatch_cache.clear()
    _BMM_DTYPE_OUT._dispatch_cache.clear()
    with mock.patch.object(
        torch._C, "_dispatch_has_kernel_for_dispatch_key", return_value=False
    ):
      overrides._patch_bmm_dtype_meta(self._meta_lib)

    patched = overrides._bmm_dtype_meta
    self.assertIs(_BMM_DTYPE.py_kernels[_META_KEY], patched)
    self.assertIs(_BMM_DTYPE_OUT.py_kernels[_META_KEY], patched)
    self.assertIs(_decomp.meta_table[_BMM_DTYPE], patched)
    self.assertIs(_decomp.meta_table[_BMM_DTYPE_OUT], patched)
    self.assertIn("out", inspect.signature(patched).parameters)

    with fake_tensor.FakeTensorMode():
      lhs = torch.empty((2, 4, 6), dtype=torch.float32)
      rhs = torch.empty((2, 6, 8), dtype=torch.float32)
      out = torch.empty((0,), dtype=torch.float32)
      self.assertIs(_BMM_DTYPE_OUT(lhs, rhs, torch.float32, out=out), out)
      self.assertEqual(out.shape, (2, 4, 8))
      self.assertEqual(_BMM_DTYPE(lhs, rhs, torch.float32).shape, (2, 4, 8))

  def test_patch_bmm_dtype_meta_skips_when_already_out_wrapped(self):
    @decompositions.out_wrapper(exact_dtype=True)
    def _existing_meta(self_tensor, mat2, out_dtype):
      return (self_tensor, mat2, out_dtype)

    _BMM_DTYPE.py_kernels[_META_KEY] = _existing_meta
    _BMM_DTYPE_OUT.py_kernels[_META_KEY] = _existing_meta
    _BMM_DTYPE._dispatch_cache.clear()
    _BMM_DTYPE_OUT._dispatch_cache.clear()
    overrides._patch_bmm_dtype_meta(self._meta_lib)
    self.assertIs(_BMM_DTYPE.py_kernels[_META_KEY], _existing_meta)
    self.assertIs(_BMM_DTYPE_OUT.py_kernels[_META_KEY], _existing_meta)

  def test_patch_bmm_dtype_meta_skips_cpp_meta_kernel(self):
    _BMM_DTYPE_OUT.py_kernels.pop(_META_KEY, None)
    _BMM_DTYPE_OUT._dispatch_cache.clear()
    with mock.patch.object(
        torch._C, "_dispatch_has_kernel_for_dispatch_key", return_value=True
    ):
      overrides._patch_bmm_dtype_meta(self._meta_lib)
    self.assertNotIn(_META_KEY, _BMM_DTYPE_OUT.py_kernels)


# pylint: enable=protected-access

if __name__ == "__main__":
  absltest.main()

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

"""Unit tests for RoPE slice concat fusion."""

from absl.testing import absltest
import torch
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils


class RotateHalfCanonicalDim(torch.nn.Module):

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=3)


class RotateHalfNegativeDim(torch.nn.Module):

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class IdentitySliceConcat(torch.nn.Module):

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1, x2), dim=3)


class RopeSliceFusionTest(seed_test_utils.RepeatableTest):

  def test_rope_rotate_half_fusion_canonical_dim(self):
    batch, heads, seq, head_dim = 2, 4, 8, 16
    mod = RotateHalfCanonicalDim()
    x = torch.randn(batch, heads, seq, head_dim)
    traced = torch.export.export(mod, (x,)).module()
    changed = _backend._fuse_rope_slice_concat(traced)
    self.assertTrue(changed)

    orig_out = mod(x)
    new_out = traced(x)
    test_utils.assert_close(orig_out, new_out)

  def test_rope_rotate_half_fusion_negative_dim(self):
    batch, heads, seq, head_dim = 2, 4, 8, 16
    mod = RotateHalfNegativeDim()
    x = torch.randn(batch, heads, seq, head_dim)
    traced = torch.export.export(mod, (x,)).module()
    changed = _backend._fuse_rope_slice_concat(traced)
    self.assertTrue(changed)

    orig_out = mod(x)
    new_out = traced(x)
    test_utils.assert_close(orig_out, new_out)

  def test_rope_identity_slice_concat_fusion(self):
    batch, heads, seq, head_dim = 2, 4, 8, 16
    mod = IdentitySliceConcat()
    x = torch.randn(batch, heads, seq, head_dim)
    traced = torch.export.export(mod, (x,)).module()
    changed = _backend._fuse_rope_slice_concat(traced)
    self.assertTrue(changed)

    orig_out = mod(x)
    new_out = traced(x)
    test_utils.assert_close(orig_out, new_out)


if __name__ == "__main__":
  absltest.main()

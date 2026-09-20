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

"""Unit tests for fuse_slice_concat FX pass and layout unconstraining."""

from absl.testing import absltest
import torch
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile.fx_passes import fuse_slice_concat
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils


class RotateHalfCanonicalDim(torch.nn.Module):
  """RoPE rotate_half with canonical positive dim=3."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=3)


class RotateHalfNegativeDim(torch.nn.Module):
  """RoPE rotate_half with negative dim=-1."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class IdentitySliceConcatCanonicalDim(torch.nn.Module):
  """Identity slice-concat along dim=3."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1, x2), dim=3)


class IdentitySliceConcatNegativeDim(torch.nn.Module):
  """Identity slice-concat along dim=-1."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x1, x2), dim=-1)


class SwappedSliceConcat(torch.nn.Module):
  """Swapped slice-concat without negation."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x2, x1), dim=3)


class SwappedNegSecondSliceConcat(torch.nn.Module):
  """Slice-concat with negation on second slice: (x2, -x1)."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((x2, -x1), dim=-1)


class NonMinorDimSliceConcat(torch.nn.Module):
  """Slice-concat along a non-minor dimension (dim=1 on 4D tensor)."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    half = x.shape[1] // 2
    x1 = x[:, :half, ...]
    x2 = x[:, half:, ...]
    return torch.cat((-x2, x1), dim=1)


class DifferentBaseSliceConcat(torch.nn.Module):
  """Slice-concat where inputs come from different base tensors."""

  def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    y2 = y[..., half:]
    return torch.cat((-y2, x1), dim=-1)


class FuseSliceConcatTest(seed_test_utils.RepeatableTest):

  def test_rope_rotate_half_canonical_dim(self):
    batch, heads, seq, head_dim = 2, 4, 8, 16
    mod = RotateHalfCanonicalDim()
    x = torch.randn(batch, heads, seq, head_dim)
    traced = torch.export.export(mod, (x,)).module()

    changed = fuse_slice_concat.apply(traced)
    self.assertTrue(changed)

    # Verify that the cat on minor dimension is rewritten to view / outer slicing
    orig_out = mod(x)
    new_out = traced(x)
    test_utils.assert_close(orig_out, new_out)

    # Verify cat is now on outer rank - 1 dimension
    cat_nodes = [
        n
        for n in traced.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.cat.default
    ]
    self.assertLen(cat_nodes, 1)
    self.assertEqual(cat_nodes[0].args[1], 3)

  def test_rope_rotate_half_negative_dim(self):
    batch, heads, seq, head_dim = 2, 4, 8, 16
    mod = RotateHalfNegativeDim()
    x = torch.randn(batch, heads, seq, head_dim)
    traced = torch.export.export(mod, (x,)).module()

    changed = fuse_slice_concat.apply(traced)
    self.assertTrue(changed)

    orig_out = mod(x)
    new_out = traced(x)
    test_utils.assert_close(orig_out, new_out)

  def test_identity_slice_concat_canonical_dim(self):
    batch, heads, seq, head_dim = 2, 4, 8, 16
    mod = IdentitySliceConcatCanonicalDim()
    x = torch.randn(batch, heads, seq, head_dim)
    traced = torch.export.export(mod, (x,)).module()

    changed = fuse_slice_concat.apply(traced)
    self.assertTrue(changed)

    # Cat should be eliminated completely
    cat_nodes = [
        n
        for n in traced.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.cat.default
    ]
    self.assertEmpty(cat_nodes)

    orig_out = mod(x)
    new_out = traced(x)
    test_utils.assert_close(orig_out, new_out)

  def test_identity_slice_concat_negative_dim(self):
    batch, heads, seq, head_dim = 2, 4, 8, 16
    mod = IdentitySliceConcatNegativeDim()
    x = torch.randn(batch, heads, seq, head_dim)
    traced = torch.export.export(mod, (x,)).module()

    changed = fuse_slice_concat.apply(traced)
    self.assertTrue(changed)

    cat_nodes = [
        n
        for n in traced.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.cat.default
    ]
    self.assertEmpty(cat_nodes)

    orig_out = mod(x)
    new_out = traced(x)
    test_utils.assert_close(orig_out, new_out)

  def test_swapped_slice_concat(self):
    batch, heads, seq, head_dim = 2, 4, 8, 16
    mod = SwappedSliceConcat()
    x = torch.randn(batch, heads, seq, head_dim)
    traced = torch.export.export(mod, (x,)).module()

    changed = fuse_slice_concat.apply(traced)
    self.assertTrue(changed)

    orig_out = mod(x)
    new_out = traced(x)
    test_utils.assert_close(orig_out, new_out)

  def test_swapped_neg_second_slice_concat(self):
    batch, heads, seq, head_dim = 2, 4, 8, 16
    mod = SwappedNegSecondSliceConcat()
    x = torch.randn(batch, heads, seq, head_dim)
    traced = torch.export.export(mod, (x,)).module()

    changed = fuse_slice_concat.apply(traced)
    self.assertTrue(changed)

    orig_out = mod(x)
    new_out = traced(x)
    test_utils.assert_close(orig_out, new_out)

  def test_non_minor_dim_slice_concat_no_op(self):
    batch, heads, seq, head_dim = 2, 4, 8, 16
    mod = NonMinorDimSliceConcat()
    x = torch.randn(batch, heads, seq, head_dim)
    traced = torch.export.export(mod, (x,)).module()

    changed = fuse_slice_concat.apply(traced)
    self.assertFalse(changed)

  def test_different_base_slice_concat_no_op(self):
    batch, heads, seq, head_dim = 2, 4, 8, 16
    mod = DifferentBaseSliceConcat()
    x = torch.randn(batch, heads, seq, head_dim)
    y = torch.randn(batch, heads, seq, head_dim)
    traced = torch.export.export(mod, (x, y)).module()

    changed = fuse_slice_concat.apply(traced)
    self.assertFalse(changed)

  def test_should_skip_argument_layout_constraint(self):
    # Gemma 2 [1024, 2304] shape: minor dim 2304 is not power-of-2
    gemma_param = torch.empty(1024, 2304)
    self.assertTrue(
        compiler._should_skip_argument_layout_constraint(gemma_param, [1, 0])
    )

    # Transposed Gemma 2 shape [2304, 1024]: dimension 0 is not power-of-2
    gemma_param_transposed = torch.empty(2304, 1024)
    self.assertTrue(
        compiler._should_skip_argument_layout_constraint(
            gemma_param_transposed, [1, 0]
        )
    )

    # Standard power-of-2 2D shape [1024, 1024]: all dims power-of-2
    pow2_param = torch.empty(1024, 1024)
    self.assertFalse(
        compiler._should_skip_argument_layout_constraint(pow2_param, [1, 0])
    )

    # 4D tensor with power-of-2 dims: [2, 4, 8, 16]
    pow2_4d = torch.empty(2, 4, 8, 16)
    self.assertFalse(
        compiler._should_skip_argument_layout_constraint(pow2_4d, [3, 2, 1, 0])
    )

    # 4D tensor with non-power-of-2 minor dim: [2, 4, 8, 18]
    non_pow2_4d = torch.empty(2, 4, 8, 18)
    self.assertTrue(
        compiler._should_skip_argument_layout_constraint(
            non_pow2_4d, [3, 2, 1, 0]
        )
    )

    # Scalar tensor: 0D
    scalar = torch.empty(())
    self.assertFalse(
        compiler._should_skip_argument_layout_constraint(scalar, [])
    )

  def test_static_compiler_wiring(self):
    # Verify that StaticCompiler has compile alias and applies the pass
    self.assertEqual(
        compiler.StaticCompiler.compile, compiler.StaticCompiler.__call__
    )


if __name__ == "__main__":
  absltest.main()

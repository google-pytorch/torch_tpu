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

"""Unit tests for C++ FxOutputReconstructor."""

import pickle

from absl.testing import absltest
import torch
from torch.utils import _pytree as pytree
import torch_tpu._internal.compile.tpu_torch_compile as tpu_torch_compile
from torch_tpu._internal.export import export as torch_tpu_export
from tests import seed_test_utils


class FxOutputReconstructorTest(seed_test_utils.RepeatableTest):
  """Unit tests for C++ FxOutputReconstructor verifying parity with Python."""

  def _make_layout(
      self,
      dtype=torch.float32,
      shape=(4,),
      stride=(1,),
      storage_offset=0,
  ):
    return torch_tpu_export._TensorLayout(  # pylint: disable=protected-access
        dtype=dtype,
        shape=torch.Size(shape),
        stride=stride,
        storage_offset=storage_offset,
    )

  def test_leaf_output_and_pickle(self):
    t0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    _, spec = pytree.tree_flatten(t0)
    output_map = [
        torch_tpu_export._DedupedTensorOutput(tensor_idx=0)  # pylint: disable=protected-access
    ]
    layouts = [self._make_layout()]

    reconstructor = tpu_torch_compile.FxOutputReconstructor(
        output_map, layouts, spec
    )
    # Verify pickle serialization round-trip (__reduce__).
    restored = pickle.loads(pickle.dumps(reconstructor))

    out = restored(inputs=[], deduped_outputs=[t0])
    self.assertIsInstance(out, torch.Tensor)
    self.assertTrue(torch.equal(out, t0))

  def test_flat_tuple_duplicate_cloning(self):
    t0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    _, spec = pytree.tree_flatten((t0, t0))
    output_map = [
        torch_tpu_export._DedupedTensorOutput(tensor_idx=0),  # pylint: disable=protected-access
        torch_tpu_export._DedupedTensorOutput(tensor_idx=0),  # pylint: disable=protected-access
    ]
    layouts = [self._make_layout()]

    reconstructor = tpu_torch_compile.FxOutputReconstructor(
        output_map, layouts, spec
    )
    out0, out1 = reconstructor(inputs=[], deduped_outputs=[t0])

    # First occurrence reuses t0; second occurrence must be cloned so in-place
    # mutation of one does not affect the other.
    self.assertIs(out0, t0)
    self.assertIsNot(out1, t0)
    self.assertTrue(torch.equal(out0, out1))
    out1.add_(10.0)
    self.assertFalse(torch.equal(out0, out1))

  def test_flat_list_with_none_and_passthrough(self):
    t0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    passthrough_obj = "passthrough_metadata"
    _, spec = pytree.tree_flatten([t0, None, passthrough_obj])
    output_map = [
        torch_tpu_export._DedupedTensorOutput(tensor_idx=0),  # pylint: disable=protected-access
        None,
        torch_tpu_export._NonTensorPassthrough(input_idx=1),  # pylint: disable=protected-access
    ]
    layouts = [self._make_layout()]

    reconstructor = tpu_torch_compile.FxOutputReconstructor(
        output_map, layouts, spec
    )
    out = reconstructor(
        inputs=["unused_arg0", passthrough_obj], deduped_outputs=[t0]
    )

    self.assertIsInstance(out, list)
    self.assertLen(out, 3)
    self.assertTrue(torch.equal(out[0], t0))
    self.assertIsNone(out[1])
    self.assertEqual(out[2], passthrough_obj)

  def test_general_nested_pytree(self):
    t0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t1 = torch.tensor([5.0, 6.0, 7.0, 8.0])
    nested_structure = {
        "first": (t0, None),
        "second": [t1, t0, "meta_val"],
    }
    _, spec = pytree.tree_flatten(nested_structure)
    output_map = [
        torch_tpu_export._DedupedTensorOutput(tensor_idx=0),  # pylint: disable=protected-access
        None,
        torch_tpu_export._DedupedTensorOutput(tensor_idx=1),  # pylint: disable=protected-access
        torch_tpu_export._DedupedTensorOutput(tensor_idx=0),  # pylint: disable=protected-access
        torch_tpu_export._NonTensorPassthrough(input_idx=0),  # pylint: disable=protected-access
    ]
    layouts = [self._make_layout(), self._make_layout()]

    reconstructor = tpu_torch_compile.FxOutputReconstructor(
        output_map, layouts, spec
    )
    out = reconstructor(inputs=["meta_val"], deduped_outputs=[t0, t1])

    self.assertEqual(set(out.keys()), {"first", "second"})
    self.assertIsNone(out["first"][1])
    self.assertTrue(torch.equal(out["first"][0], t0))
    self.assertTrue(torch.equal(out["second"][0], t1))
    self.assertTrue(torch.equal(out["second"][1], t0))
    # Verify duplicate tensor in nested structure was cloned.
    self.assertIsNot(out["first"][0], out["second"][1])
    self.assertEqual(out["second"][2], "meta_val")

  def test_non_contiguous_stride_forcing_and_dynamic_bypass(self):
    base = torch.arange(6, dtype=torch.float32).reshape(3, 2).to("tpu")
    _, spec = pytree.tree_flatten(base)
    output_map = [
        torch_tpu_export._DedupedTensorOutput(tensor_idx=0)  # pylint: disable=protected-access
    ]
    # Expect a (3, 2) view with transposed non-contiguous stride (1, 3).
    expected_layout = self._make_layout(
        dtype=torch.float32, shape=(3, 2), stride=(1, 3), storage_offset=0
    )

    # Static output (is_dynamic=False): should force target strides & offset.
    static_reconstructor = tpu_torch_compile.FxOutputReconstructor(
        output_map, [expected_layout], spec, dynamic_outputs=[False]
    )
    static_out = static_reconstructor(inputs=[], deduped_outputs=[base])
    self.assertEqual(static_out.shape, torch.Size([3, 2]))
    self.assertEqual(static_out.stride(), (1, 3))
    self.assertEqual(static_out.storage_offset(), 0)

    # Dynamic output (is_dynamic=True): should bypass stride forcing.
    dynamic_reconstructor = tpu_torch_compile.FxOutputReconstructor(
        output_map, [expected_layout], spec, dynamic_outputs=[True]
    )
    dynamic_out = dynamic_reconstructor(inputs=[], deduped_outputs=[base])
    self.assertEqual(dynamic_out.shape, base.shape)
    self.assertEqual(dynamic_out.stride(), base.stride())
    self.assertEqual(dynamic_out.storage_offset(), base.storage_offset())

  def test_size_one_dimension_stride_fast_path(self):
    # Shape (1, 4, 8) where standard contiguous strides are (32, 8, 1),
    # but FX graph expects stride (8, 8, 1) along the leading size-1 dimension.
    base = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8).to("tpu")
    _, spec = pytree.tree_flatten(base)
    output_map = [
        torch_tpu_export._DedupedTensorOutput(tensor_idx=0)  # pylint: disable=protected-access
    ]
    expected_layout = self._make_layout(
        dtype=torch.float32, shape=(1, 4, 8), stride=(8, 8, 1), storage_offset=0
    )
    reconstructor = tpu_torch_compile.FxOutputReconstructor(
        output_map, [expected_layout], spec, dynamic_outputs=[False]
    )
    out = reconstructor(inputs=[], deduped_outputs=[base])
    self.assertEqual(out.shape, torch.Size([1, 4, 8]))
    self.assertEqual(out.stride(), (8, 8, 1))
    self.assertEqual(out.storage_offset(), 0)
    self.assertTrue(torch.equal(out.cpu(), base.cpu()))


if __name__ == "__main__":
  absltest.main()

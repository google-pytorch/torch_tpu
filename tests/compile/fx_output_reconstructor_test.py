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

import collections
import pickle
from typing import Any, Callable
import unittest.mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.utils import _pytree as pytree
import torch_tpu._internal.compile._backend as _backend
import torch_tpu._internal.compile.tpu_torch_compile as tpu_torch_compile
from torch_tpu._internal.export import export as torch_tpu_export
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


def _compile_double_executable(shape: tuple[int, ...]):
  """Compiles a PJRT executable computing `2 * x` for a float32 `shape`."""
  tensor_type = f"tensor<{'x'.join(map(str, shape))}xf32>"
  module = tpu_torch_compile.parse_mlir_text(f"""
module {{
  func.func @main(%arg0: {tensor_type}) -> {tensor_type} {{
    %two = stablehlo.constant dense<2.0> : {tensor_type}
    %0 = stablehlo.multiply %arg0, %two : {tensor_type}
    return %0 : {tensor_type}
  }}
}}
""")
  return tpu_torch_compile.compile_mlir(module)


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

  @parameterized.named_parameters(
      ("cpu_trailing_zero", "cpu", lambda x: x[:, ::2][:, :0]),
      ("cpu_leading_zero", "cpu", lambda x: x[:, ::2][:0, :]),
      ("tpu_trailing_zero", "tpu", lambda x: x[:, ::2][:, :0]),
      ("tpu_leading_zero", "tpu", lambda x: x[:, ::2][:0, :]),
  )
  def test_zero_size_dimension_stride_fast_path(
      self,
      device: str,
      slice_fn,
  ):
    # Slicing every 2nd column gives shape (4, 3) with stride (6, 2); taking an
    # empty slice along dim 0 or dim 1 produces a 0-element view of shape (0, 3)
    # or (4, 0) that retains the non-standard stride (6, 2). Testing on CPU also
    # verifies that ForceStrides takes the metadata-only fast path rather than
    # falling through to TPU-only ContiguousToView.
    full = torch.empty((4, 6), dtype=torch.float32, device=device)
    expected_view = slice_fn(full)
    base = torch.empty_like(
        expected_view, memory_format=torch.contiguous_format
    )
    _, spec = pytree.tree_flatten(expected_view)
    output_map = [
        torch_tpu_export._DedupedTensorOutput(tensor_idx=0)  # pylint: disable=protected-access
    ]
    expected_layout = self._make_layout(
        dtype=expected_view.dtype,
        shape=expected_view.shape,
        stride=expected_view.stride(),
        storage_offset=expected_view.storage_offset(),
    )
    reconstructor = tpu_torch_compile.FxOutputReconstructor(
        output_map, [expected_layout], spec, dynamic_outputs=[False]
    )
    out = reconstructor(inputs=[], deduped_outputs=[base])
    self.assertEqual(out.shape, expected_view.shape)
    self.assertEqual(out.stride(), expected_view.stride())
    self.assertEqual(out.storage_offset(), 0)

  @parameterized.named_parameters(
      (
          "namedtuple",
          collections.namedtuple("Inputs", ["first", "second"])("a", "meta"),
      ),
      ("user_list", collections.UserList(["a", "meta"])),
  )
  def test_passthrough_from_non_exact_sequence(self, inputs):
    t0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    _, spec = pytree.tree_flatten([t0, "meta"])
    reconstructor = tpu_torch_compile.FxOutputReconstructor(
        [
            torch_tpu_export._DedupedTensorOutput(tensor_idx=0),  # pylint: disable=protected-access
            torch_tpu_export._NonTensorPassthrough(input_idx=1),  # pylint: disable=protected-access
        ],
        [self._make_layout()],
        spec,
    )

    out = reconstructor(inputs=inputs, deduped_outputs=[t0])

    self.assertIsInstance(out, list)
    self.assertLen(out, 2)
    self.assertIs(out[0], t0)
    self.assertEqual(out[1], "meta")

  @parameterized.named_parameters(
      (
          "namedtuple",
          collections.namedtuple("Inputs", ["x", "y"]),
      ),
      ("generator_of_views", lambda x, y: (t.detach() for t in (x, y))),
  )
  def test_compiled_runner_unpacks_non_exact_sequence_inputs(self, make_inputs):
    module = tpu_torch_compile.parse_mlir_text("""
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}
""")
    executable = tpu_torch_compile.compile_mlir(module)
    runner = tpu_torch_compile.make_compiled_runner(executable)

    x = torch.arange(4, dtype=torch.float32, device="tpu")
    y = torch.ones(4, dtype=torch.float32, device="tpu")
    outputs = runner(make_inputs(x, y))

    self.assertLen(outputs, 1)
    self.assertTrue(torch.equal(outputs[0].cpu(), x.cpu()))

  def test_compiled_runner_without_reconstruct_fn_returns_output_list(self):
    x = torch.randn(4, 4, device="tpu")
    runner = tpu_torch_compile.make_compiled_runner(
        _compile_double_executable(x.shape)
    )

    outputs = runner([x])

    self.assertIsInstance(outputs, list)
    self.assertLen(outputs, 1)
    self.assertTrue(torch.equal(outputs[0].cpu(), (x * 2.0).cpu()))

  @parameterized.named_parameters(
      ("list", list),
      ("tuple", tuple),
      ("generic_sequence", collections.UserList),
  )
  def test_compiled_runner_does_not_unbox_single_tensor_argument(
      self, make_args: Callable[[list[Any]], Any]
  ):
    x = torch.randn(4, 4, device="tpu")
    runner = tpu_torch_compile.make_compiled_runner(
        _compile_double_executable(x.shape)
    )

    # The sole element is a tensor argument rather than a boxed argument list,
    # so the 1-element sequence itself holds the arguments.
    (out,) = runner(make_args([x]))

    self.assertTrue(torch.equal(out.cpu(), (x * 2.0).cpu()))

  def test_compiled_runner_reconstructs_duplicate_size_one_strided_outputs(
      self,
  ):
    x = torch.randn(1, 4, 4, device="tpu")
    _, spec = pytree.tree_flatten((x, x))
    reconstructor = tpu_torch_compile.FxOutputReconstructor(
        [
            torch_tpu_export._DedupedTensorOutput(tensor_idx=0),  # pylint: disable=protected-access
            torch_tpu_export._DedupedTensorOutput(tensor_idx=0),  # pylint: disable=protected-access
        ],
        [self._make_layout(shape=(1, 4, 4), stride=(4, 4, 1))],
        spec,
    )
    runner = tpu_torch_compile.make_compiled_runner(
        _compile_double_executable(x.shape), reconstruct_fn=reconstructor
    )

    out0, out1 = runner([x])

    expected = (x * 2.0).cpu()
    for out in (out0, out1):
      self.assertEqual(out.shape, torch.Size((1, 4, 4)))
      self.assertEqual(out.stride(), (4, 4, 1))
      self.assertTrue(torch.equal(out.cpu(), expected))
    # The duplicate output must be cloned so in-place mutation of one does not
    # affect the other.
    self.assertIsNot(out0, out1)
    out1.add_(10.0)
    self.assertTrue(torch.equal(out0.cpu(), expected))

  def test_compiled_runner_lazy_force_strides_materializes_on_demand(self):
    x = torch.arange(16, dtype=torch.float32, device="tpu").reshape(4, 4)
    expected_values = x.cpu() * 2.0
    expected_view = expected_values.t()
    _, spec = pytree.tree_flatten((expected_view, expected_view))
    layout = self._make_layout(
        shape=expected_view.shape,
        stride=expected_view.stride(),
        storage_offset=expected_view.storage_offset(),
    )
    reconstructor = tpu_torch_compile.FxOutputReconstructor(
        [
            # pylint: disable-next=protected-access
            torch_tpu_export._DedupedTensorOutput(tensor_idx=0),
            # pylint: disable-next=protected-access
            torch_tpu_export._DedupedTensorOutput(tensor_idx=0),
        ],
        [layout],
        spec,
    )
    runner = tpu_torch_compile.make_compiled_runner(
        _compile_double_executable(x.shape), reconstruct_fn=reconstructor
    )

    unused_out, used_out = runner([x])
    del unused_out
    torch.accelerator.synchronize()

    self.assertEqual(used_out.shape, expected_view.shape)
    self.assertEqual(used_out.stride(), expected_view.stride())
    utils.assert_close(used_out.cpu(), expected_values)
    utils.assert_close((used_out + 1.0).cpu(), expected_values + 1.0)

  def test_compile_canonicalizes_saved_backward_view_activations(self):
    def fn(x, w):
      h = torch.matmul(x, w).view(2, 6, 4, 8).transpose(1, 2)[:, :, 1:, :]
      attn = torch.nn.functional.scaled_dot_product_attention(h, h, h)
      merged = attn.transpose(1, 2).reshape(2, 5, 32)
      return (merged * torch.sigmoid(merged)).sum()

    saved_partitions = []
    orig_partition = _backend._partition_with_saved_view_canonicalization

    def partition_hook(joint_gm, joint_inputs, **kwargs):
      fw_gm, bw_gm = orig_partition(joint_gm, joint_inputs, **kwargs)
      saved_partitions.append((fw_gm, bw_gm))
      return fw_gm, bw_gm

    with unittest.mock.patch.object(
        _backend,
        "_partition_with_saved_view_canonicalization",
        side_effect=partition_hook,
    ):
      x_cpu = torch.randn(2, 6, 32)
      w_cpu = torch.randn(32, 32)
      x_ref = x_cpu.to("tpu").requires_grad_(True)
      w_ref = w_cpu.to("tpu").requires_grad_(True)
      loss_ref = fn(x_ref, w_ref)
      loss_ref.backward()

      x_tpu = x_cpu.to("tpu").requires_grad_(True)
      w_tpu = w_cpu.to("tpu").requires_grad_(True)
      compiled_fn = torch.compile(fn, backend="tpu")
      loss_tpu = compiled_fn(x_tpu, w_tpu)
      loss_tpu.backward()

    self.assertNotEmpty(saved_partitions)
    fw_module, bw_module = saved_partitions[0]
    fw_output_node = fw_module.graph.find_nodes(op="output")[0]
    fw_outs = list(fw_output_node.args[0])
    bw_placeholders = {
        n.name: n for n in bw_module.graph.find_nodes(op="placeholder")
    }

    self.assertTrue(fw_outs[1].target == torch.ops.aten.clone.default)
    bw_ph = bw_placeholders.get(
        "transpose", bw_placeholders.get(fw_outs[1].name)
    )
    self.assertIsNotNone(bw_ph)
    self.assertTrue(bw_ph.meta["val"].is_contiguous())

    utils.assert_close(loss_tpu.cpu(), loss_ref.cpu(), atol=1e-4, rtol=1e-4)
    utils.assert_close(x_tpu.grad.cpu(), x_ref.grad.cpu(), atol=1e-4, rtol=1e-4)
    utils.assert_close(w_tpu.grad.cpu(), w_ref.grad.cpu(), atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
  absltest.main()

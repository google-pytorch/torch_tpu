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
from collections.abc import Callable
import pickle
from typing import Any
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

  def test_compiled_runner_force_strides_materialization(self):
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

    # Eager vs compiled TPU MXU fusion uses atol=1e-4, rtol=1e-4 due to
    # systolic array dot-product accumulation and SDPA contraction reordering.
    utils.assert_close(loss_tpu.cpu(), loss_ref.cpu(), atol=1e-4, rtol=1e-4)
    utils.assert_close(x_tpu.grad.cpu(), x_ref.grad.cpu(), atol=1e-4, rtol=1e-4)
    utils.assert_close(w_tpu.grad.cpu(), w_ref.grad.cpu(), atol=1e-4, rtol=1e-4)

  def test_canonicalize_saved_backward_views_downstream_rewrite_only(
      self,
  ):
    def fn(x, w, unrelated):
      h = torch.matmul(x, w).view(2, 6, 4, 8).transpose(1, 2)[:, :, 1:, :]
      attn = torch.nn.functional.scaled_dot_product_attention(h, h, h)
      merged = attn.transpose(1, 2).reshape(2, 5, 32)
      loss_main = (merged * torch.sigmoid(merged)).sum()

      unrelated_view = unrelated.view(16)
      loss_unrelated = (unrelated_view * 2.0).sum()
      return loss_main + loss_unrelated

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
      u_cpu = torch.randn(4, 4)

      x_ref = x_cpu.to("tpu").requires_grad_(True)
      w_ref = w_cpu.to("tpu").requires_grad_(True)
      u_ref = u_cpu.to("tpu").requires_grad_(True)
      loss_ref = fn(x_ref, w_ref, u_ref)
      loss_ref.backward()

      x_tpu = x_cpu.to("tpu").requires_grad_(True)
      w_tpu = w_cpu.to("tpu").requires_grad_(True)
      u_tpu = u_cpu.to("tpu").requires_grad_(True)
      compiled_fn = torch.compile(fn, backend="tpu")
      loss_tpu = compiled_fn(x_tpu, w_tpu, u_tpu)
      loss_tpu.backward()

    self.assertNotEmpty(saved_partitions)
    fw_module, bw_module = saved_partitions[0]

    fw_output_node = fw_module.graph.find_nodes(op="output")[0]
    fw_outs = list(fw_output_node.args[0])
    cloned_outs = [
        n
        for n in fw_outs
        if isinstance(n, torch.fx.Node)
        and n.target == torch.ops.aten.clone.default
    ]
    self.assertNotEmpty(cloned_outs)

    unrelated_views = [
        n
        for n in bw_module.graph.nodes
        if n.op == "call_function"
        and n.target in (
            torch.ops.aten.view.default,
            torch.ops.aten._unsafe_view.default,
        )
    ]
    self.assertNotEmpty(
        unrelated_views,
        "Expected unrelated backward view to remain aten.view, but none found.",
    )

    reshapes = [
        n
        for n in bw_module.graph.nodes
        if n.op == "call_function"
        and n.target == torch.ops.aten.reshape.default
    ]
    self.assertNotEmpty(
        reshapes,
        "Expected downstream backward views to be rewritten to aten.reshape.",
    )

    # Eager vs compiled TPU MXU fusion uses atol=1e-4, rtol=1e-4 due to
    # systolic array dot-product accumulation and SDPA contraction reordering.
    utils.assert_close(loss_tpu.cpu(), loss_ref.cpu(), atol=1e-4, rtol=1e-4)
    utils.assert_close(x_tpu.grad.cpu(), x_ref.grad.cpu(), atol=1e-4, rtol=1e-4)
    utils.assert_close(w_tpu.grad.cpu(), w_ref.grad.cpu(), atol=1e-4, rtol=1e-4)
    utils.assert_close(u_tpu.grad.cpu(), u_ref.grad.cpu(), atol=1e-4, rtol=1e-4)

  def test_saved_contiguous_slice_positive_storage_offset_canonicalization(
      self,
  ):
    def fn(x, w):
      h = torch.matmul(x, w)[1:, :]
      attn = torch.nn.functional.scaled_dot_product_attention(h, h, h)
      merged = attn.reshape(attn.shape[0], -1)
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
      x_cpu = torch.randn(4, 16)
      w_cpu = torch.randn(16, 16)
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
    cloned_outs = [
        n
        for n in fw_outs
        if isinstance(n, torch.fx.Node)
        and n.target == torch.ops.aten.clone.default
    ]
    self.assertNotEmpty(cloned_outs)
    contig_node = cloned_outs[0]
    self.assertEqual(contig_node.meta["val"].storage_offset(), 0)

    bw_placeholders = {
        n.name: n for n in bw_module.graph.find_nodes(op="placeholder")
    }
    bw_ph = bw_placeholders.get(contig_node.name)
    if bw_ph is None:
      bw_ph = [
          p
          for p in bw_placeholders.values()
          if p.meta.get("val") is not None
          and p.meta["val"].shape == contig_node.meta["val"].shape
      ][0]
    self.assertEqual(bw_ph.meta["val"].storage_offset(), 0)

    # Eager vs compiled TPU MXU fusion uses atol=1e-4, rtol=1e-4 due to
    # systolic array dot-product accumulation and SDPA contraction reordering.
    utils.assert_close(loss_tpu.cpu(), loss_ref.cpu(), atol=1e-4, rtol=1e-4)
    utils.assert_close(x_tpu.grad.cpu(), x_ref.grad.cpu(), atol=1e-4, rtol=1e-4)
    utils.assert_close(w_tpu.grad.cpu(), w_ref.grad.cpu(), atol=1e-4, rtol=1e-4)

  def test_has_stride_sensitive_view_user_direct_and_transitive_bailouts(self):
    stride_sensitive_targets = [
        torch.ops.aten.as_strided.default,
        torch.ops.aten.as_strided_scatter.default,
        torch.ops.aten.view.dtype,
        torch.ops.aten.view_as_complex.default,
    ]
    if hasattr(torch.ops, "inductor") and hasattr(
        torch.ops.inductor, "reinterpret_tensor"
    ):
      stride_sensitive_targets.append(
          getattr(
              torch.ops.inductor.reinterpret_tensor,
              "default",
              torch.ops.inductor.reinterpret_tensor,
          )
      )

    for target in stride_sensitive_targets:
      with self.subTest(target=str(target)):
        # Direct user test
        g_direct = torch.fx.Graph()
        ph = g_direct.placeholder("x")
        g_direct.call_function(target, (ph,))
        self.assertTrue(
            _backend._has_stride_sensitive_view_user(ph),
            f"Expected {target} to trigger stride-sensitive bailout directly.",
        )

        # Transitive user through clone / stride-preserving op
        g_transitive = torch.fx.Graph()
        ph_t = g_transitive.placeholder("x")
        clone_node = g_transitive.call_function(
            torch.ops.aten.clone.default, (ph_t,)
        )
        g_transitive.call_function(target, (clone_node,))
        self.assertTrue(
            _backend._has_stride_sensitive_view_user(ph_t),
            (
                f"Expected {target} to trigger stride-sensitive bailout"
                " transitively."
            ),
        )

    # Negative test: ordinary view or non-sensitive user does not bail out
    g_safe = torch.fx.Graph()
    ph_safe = g_safe.placeholder("x")
    view_node = g_safe.call_function(
        torch.ops.aten.view.default, (ph_safe, [4, 4])
    )
    g_safe.call_function(torch.ops.aten.sin.default, (view_node,))
    self.assertFalse(_backend._has_stride_sensitive_view_user(ph_safe))

  def test_view_dtype_in_backward_not_rewritten_and_does_not_crash(self):
    fw_g = torch.fx.Graph()
    ph_f = fw_g.placeholder("f_in")
    view_f = fw_g.call_function(torch.ops.aten.transpose.int, (ph_f, 0, 1))
    fw_g.output((ph_f, view_f))
    fw_mod = torch.fx.GraphModule({}, fw_g)

    bw_g = torch.fx.Graph()
    ph_b0 = bw_g.placeholder("b_in0")
    ph_b1 = bw_g.placeholder("transpose")
    view_dtype_node = bw_g.call_function(
        torch.ops.aten.view.dtype, (ph_b1, torch.int32)
    )
    bw_g.output((ph_b0, view_dtype_node))
    bw_mod = torch.fx.GraphModule({}, bw_g)

    t = torch.randn(4, 8)
    t_trans = t.t()
    ph_f.meta["val"] = t
    view_f.meta["val"] = t_trans
    ph_b0.meta["val"] = torch.randn(4, 8)
    ph_b1.meta["val"] = t_trans

    _backend._canonicalize_saved_backward_views(
        fw_mod, bw_mod, num_fwd_outputs=1
    )

    self.assertEqual(view_dtype_node.target, torch.ops.aten.view.dtype)

  def test_canonicalization_bailouts_for_ineligible_storages(self):
    def make_modules(saved_nodes_meta, is_primal=False, is_user_out=False):
      fw_g = torch.fx.Graph()
      p_in = fw_g.placeholder("x_in")
      outs = []
      if is_primal:
        p_in.meta["val"] = saved_nodes_meta[0]
      else:
        p_in.meta["val"] = torch.randn(16, 16)

      if is_user_out:
        uo = fw_g.call_function(torch.ops.aten.clone.default, (p_in,))
        uo.meta["val"] = saved_nodes_meta[0]
        outs.append(uo)
      else:
        uo = fw_g.call_function(torch.ops.aten.sin.default, (p_in,))
        uo.meta["val"] = torch.randn(4, 4)
        outs.append(uo)

      for i, meta in enumerate(saved_nodes_meta):
        n = fw_g.call_function(torch.ops.aten.transpose.int, (p_in, 0, 1))
        n.name = f"saved_{i}"
        n.meta["val"] = meta
        outs.append(n)
      fw_g.output(tuple(outs))
      fw_m = torch.fx.GraphModule({}, fw_g)

      bw_g = torch.fx.Graph()
      for i, meta in enumerate(saved_nodes_meta):
        bp = bw_g.placeholder(f"saved_{i}")
        bp.meta["val"] = meta
        bw_g.call_function(torch.ops.aten.sin.default, (bp,))
      bw_g.output(())
      bw_m = torch.fx.GraphModule({}, bw_g)
      return fw_m, bw_m

    base = torch.randn(8, 8)
    v1 = base[1:, :].t()
    v2 = base[:2, :].t()

    # Shared storage (saved_storage_counts != 1)
    fw_m, bw_m = make_modules([v1, v2])
    _backend._canonicalize_saved_backward_views(fw_m, bw_m, num_fwd_outputs=1)
    fw_outs = list(fw_m.graph.find_nodes(op="output")[0].args[0])
    self.assertFalse(
        any(n.target == torch.ops.aten.clone.default for n in fw_outs[1:])
    )

    # View of primal/placeholder
    fw_m2, bw_m2 = make_modules([v1], is_primal=True)
    _backend._canonicalize_saved_backward_views(fw_m2, bw_m2, num_fwd_outputs=1)
    fw_outs2 = list(fw_m2.graph.find_nodes(op="output")[0].args[0])
    self.assertFalse(
        any(n.target == torch.ops.aten.clone.default for n in fw_outs2[1:])
    )

    # View of user forward output
    fw_m3, bw_m3 = make_modules([v1], is_user_out=True)
    _backend._canonicalize_saved_backward_views(fw_m3, bw_m3, num_fwd_outputs=1)
    fw_outs3 = list(fw_m3.graph.find_nodes(op="output")[0].args[0])
    self.assertFalse(
        any(n.target == torch.ops.aten.clone.default for n in fw_outs3[1:])
    )

    # Dynamic SymInt shapes bailout
    fw_m4, bw_m4 = make_modules([v1])
    with unittest.mock.patch.object(
        _backend.compiler, "has_dynamic_symints", return_value=True
    ):
      _backend._canonicalize_saved_backward_views(
          fw_m4, bw_m4, num_fwd_outputs=1
      )
    fw_outs4 = list(fw_m4.graph.find_nodes(op="output")[0].args[0])
    self.assertFalse(
        any(n.target == torch.ops.aten.clone.default for n in fw_outs4[1:])
    )

    # Internally overlapping view bailout (unfold)
    unfolded = torch.randn(10).unfold(0, 3, 1)
    fw_m5, bw_m5 = make_modules([unfolded])
    _backend._canonicalize_saved_backward_views(fw_m5, bw_m5, num_fwd_outputs=1)
    fw_outs5 = list(fw_m5.graph.find_nodes(op="output")[0].args[0])
    self.assertFalse(
        any(n.target == torch.ops.aten.clone.default for n in fw_outs5[1:])
    )

  def test_partition_with_saved_view_canonicalization_matches_min_cut_partition(
      self,
  ):
    def fn(x, w):
      # Pure elementwise + transpose without systolic array contractions
      h = (x * 2.0).t()[:, 1:]
      y = torch.sin(h) + torch.cos(w)
      return (y * torch.relu(y)).sum()

    x_cpu = torch.randn(4, 8)
    w_cpu = torch.randn(8, 3)

    x_mincut = x_cpu.to("tpu").requires_grad_(True)
    w_mincut = w_cpu.to("tpu").requires_grad_(True)
    with unittest.mock.patch.object(
        _backend,
        "_partition_with_saved_view_canonicalization",
        side_effect=_backend.min_cut_rematerialization_partition,
    ):
      compiled_mincut = torch.compile(fn, backend="tpu")
      loss_mincut = compiled_mincut(x_mincut, w_mincut)
      loss_mincut.backward()

    x_canon = x_cpu.to("tpu").requires_grad_(True)
    w_canon = w_cpu.to("tpu").requires_grad_(True)
    compiled_canon = torch.compile(fn, backend="tpu")
    loss_canon = compiled_canon(x_canon, w_canon)
    loss_canon.backward()

    utils.assert_close(loss_canon.cpu(), loss_mincut.cpu())
    utils.assert_close(x_canon.grad.cpu(), x_mincut.grad.cpu())
    utils.assert_close(w_canon.grad.cpu(), w_mincut.grad.cpu())


if __name__ == "__main__":
  absltest.main()

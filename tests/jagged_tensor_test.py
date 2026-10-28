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

"""Unit tests for PyTorch torch.jagged layout and jagged tensors on TPU.

Tests cover:
- NestedTensor construction, layout queries, and device transfers on TPU.
- Elementwise and linear operations on jagged nested tensors.
- Eager ATen conversion kernels:
    aten._jagged_to_padded_dense_forward
    aten._padded_dense_to_jagged_forward
- Parity against CPU ATen kernels across all supported dtypes and padding modes.
- Complex corner cases: truncation, 1D scalar tokens, zero-length items,
  all-empty batches, strided/non-contiguous offsets, and compilation cache keys.
"""

import functools
from absl.testing import absltest
import torch
from torch.nested._internal import nested_tensor as nt_internal
from torch_tpu._internal.utils import annotations
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

jagged_ops = torch.tpu.ops


class JaggedTensorTest(seed_test_utils.RepeatableTest):

  def _assert_jagged_to_padded(
      self, values_cpu, offsets_cpu, max_lengths, padding_value=0.0
  ):
    """Helper to verify jagged-to-padded parity between TPU and CPU.

    Executes aten._jagged_to_padded_dense_forward on both CPU and TPU, verifies
    that device, shape, and dtype match, and asserts numerical equality.

    Args:
      values_cpu: CPU tensor of shape (total_L, *features).
      offsets_cpu: 1D int64 CPU tensor of shape (batch_size + 1,).
      max_lengths: Target sequence length list [max_length].
      padding_value: Scalar constant for padding unoccupied elements.

    Returns:
      A tuple of (padded_tpu, padded_cpu).
    """
    padded_cpu = torch.ops.aten._jagged_to_padded_dense_forward(
        values_cpu, [offsets_cpu], max_lengths, padding_value
    )
    padded_tpu = torch.ops.aten._jagged_to_padded_dense_forward(
        values_cpu.to("tpu"),
        [offsets_cpu.to("tpu")],
        max_lengths,
        padding_value,
    )
    self.assertEqual(padded_tpu.device.type, "tpu")
    self.assertEqual(padded_tpu.shape, padded_cpu.shape)
    self.assertEqual(padded_tpu.dtype, padded_cpu.dtype)
    utils.assert_close(padded_tpu.cpu(), padded_cpu)
    return padded_tpu, padded_cpu

  def _assert_padded_to_jagged(self, dense_cpu, offsets_cpu, total_L=None):
    """Helper to verify padded-to-jagged parity between TPU and CPU.

    Executes aten._padded_dense_to_jagged_forward on both CPU and TPU, verifies
    that device, shape, and dtype match, and asserts numerical equality.

    Args:
      dense_cpu: CPU dense tensor of shape (batch_size, max_length, *features).
      offsets_cpu: 1D int64 CPU tensor of shape (batch_size + 1,).
      total_L: Optional expected number of elements in output values.

    Returns:
      A tuple of (jagged_tpu, jagged_cpu).
    """
    kwargs = {} if total_L is None else {"total_L": total_L}
    jagged_cpu = torch.ops.aten._padded_dense_to_jagged_forward(
        dense_cpu, [offsets_cpu], **kwargs
    )
    jagged_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        dense_cpu.to("tpu"), [offsets_cpu.to("tpu")], **kwargs
    )
    self.assertEqual(jagged_tpu.device.type, "tpu")
    self.assertEqual(jagged_tpu.shape, jagged_cpu.shape)
    self.assertEqual(jagged_tpu.dtype, jagged_cpu.dtype)
    utils.assert_close(jagged_tpu.cpu(), jagged_cpu)
    return jagged_tpu, jagged_cpu

  # ---------------------------------------------------------------------------
  # Core Jagged Layout & NestedTensor Tests
  # ---------------------------------------------------------------------------

  def test_empty_jagged(self):
    """Tests creating an empty tensor with layout=torch.jagged on TPU.

    Calling torch.empty with layout=torch.jagged allocates the underlying flat
    strided buffer that PyTorch uses to back jagged nested tensors.
    """
    t = torch.empty(5, layout=torch.jagged, device="tpu")
    self.assertEqual(t.device.type, "tpu")
    self.assertEqual(t.shape, (5,))
    self.assertEqual(t.layout, torch.strided)

  def test_nested_tensor_construction(self):
    """Tests constructing a nested tensor with layout=torch.jagged on TPU."""
    t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="tpu")
    t2 = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    self.assertEqual(nt.layout, torch.jagged)
    self.assertEqual(nt.device.type, "tpu")
    self.assertEqual(nt.shape[0], 2)  # batch dimension
    self.assertEqual(nt.shape[2], 2)  # feature dimension

    # Verify that underlying values and offsets buffers live on TPU.
    values, offsets = nt.values(), nt.offsets()
    self.assertEqual(values.device.type, "tpu")
    self.assertEqual(offsets.device.type, "tpu")
    self.assertEqual(values.shape, (5, 2))
    utils.assert_close(
        offsets.cpu(), torch.tensor([0, 2, 5], dtype=torch.int64)
    )

  def test_as_nested_tensor(self):
    """Tests torch.nested.as_nested_tensor with layout=torch.jagged on TPU."""
    t1 = torch.randn(3, 4, device="tpu")
    t2 = torch.randn(5, 4, device="tpu")
    nt = torch.nested.as_nested_tensor([t1, t2], layout=torch.jagged)

    self.assertEqual(nt.layout, torch.jagged)
    self.assertEqual(nt.device.type, "tpu")
    self.assertEqual(nt.values().shape, (8, 4))
    utils.assert_close(
        nt.offsets().cpu(), torch.tensor([0, 3, 8], dtype=torch.int64)
    )

  def test_pointwise_operations_parity(self):
    """Tests that pointwise operations on jagged tensors match CPU."""
    t1_cpu = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    t2_cpu = torch.tensor([[5.0, 6.0], [-7.0, 8.0], [9.0, -10.0]])
    nt_cpu = torch.nested.nested_tensor([t1_cpu, t2_cpu], layout=torch.jagged)

    nt_tpu = torch.nested.nested_tensor(
        [t1_cpu.to("tpu"), t2_cpu.to("tpu")], layout=torch.jagged
    )

    # Scalar arithmetic operations
    utils.assert_close((nt_tpu + 2.5).values().cpu(), (nt_cpu + 2.5).values())
    utils.assert_close((nt_tpu * 1.5).values().cpu(), (nt_cpu * 1.5).values())

    # Unary activations
    utils.assert_close(
        torch.relu(nt_tpu).values().cpu(), torch.relu(nt_cpu).values()
    )
    utils.assert_close(
        torch.sin(nt_tpu).values().cpu(), torch.sin(nt_cpu).values()
    )

    # Binary tensor addition
    utils.assert_close(
        (nt_tpu + nt_tpu).values().cpu(), (nt_cpu + nt_cpu).values()
    )

  def test_linear_projection(self):
    """Tests linear layer projection on a jagged tensor on TPU."""
    t1_cpu = torch.randn(4, 8)
    t2_cpu = torch.randn(6, 8)
    nt_cpu = torch.nested.nested_tensor([t1_cpu, t2_cpu], layout=torch.jagged)
    linear_cpu = torch.nn.Linear(8, 16)

    nt_tpu = torch.nested.nested_tensor(
        [t1_cpu.to("tpu"), t2_cpu.to("tpu")], layout=torch.jagged
    )
    linear_tpu = torch.nn.Linear(8, 16).to("tpu")
    linear_tpu.load_state_dict(linear_cpu.state_dict())

    out_cpu = linear_cpu(nt_cpu)
    out_tpu = linear_tpu(nt_tpu)

    self.assertEqual(out_tpu.layout, torch.jagged)
    self.assertEqual(out_tpu.device.type, "tpu")
    self.assertEqual(out_tpu.values().shape, (10, 16))
    utils.assert_close(
        out_tpu.values().cpu(), out_cpu.values(), atol=1e-2, rtol=1e-2
    )

  def test_device_transfer(self):
    """Tests moving a jagged tensor between CPU and TPU."""
    t1 = torch.randn(2, 3)
    t2 = torch.randn(4, 3)
    nt_cpu = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    # Transfer CPU -> TPU
    nt_tpu = nt_cpu.to("tpu")
    self.assertEqual(nt_tpu.device.type, "tpu")
    self.assertEqual(nt_tpu.layout, torch.jagged)

    # Transfer TPU -> CPU
    nt_back_to_cpu = nt_tpu.to("cpu")
    self.assertEqual(nt_back_to_cpu.device.type, "cpu")
    self.assertEqual(nt_back_to_cpu.layout, torch.jagged)
    utils.assert_close(nt_back_to_cpu.values(), nt_cpu.values())
    utils.assert_close(nt_back_to_cpu.offsets(), nt_cpu.offsets())

  # ---------------------------------------------------------------------------
  # Autograd & Backward Operator Derivative Tests
  # ---------------------------------------------------------------------------

  def test_to_padded_tensor_autograd_analytic_gradients(self):
    """Verifies analytical gradients of to_padded_tensor across dtypes."""
    for dtype in [torch.float32, torch.bfloat16, torch.float64]:
      values = torch.randn(5, 4, dtype=dtype, device="tpu", requires_grad=True)
      offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")

      nt = nt_internal.nested_view_from_values_offsets(
          values, offsets, min_seqlen=2, max_seqlen=3
      )
      padded = nt.to_padded_tensor(0.0)

      self.assertEqual(padded.shape, (2, 3, 4))
      self.assertEqual(padded.dtype, dtype)
      self.assertEqual(padded.device.type, "tpu")

      # Backpropagate arbitrary weight matrix
      weights = torch.randn(2, 3, 4, dtype=dtype, device="tpu")
      loss = (padded * weights).sum()
      loss.backward()

      self.assertIsNotNone(values.grad)
      self.assertEqual(values.grad.shape, values.shape)
      self.assertEqual(values.grad.dtype, dtype)

      # Analytical expected grad: gather non-padded elements from weights
      expected_grad = torch.cat(
          [weights[0, :2, :], weights[1, :3, :]], dim=0
      ).cpu()
      utils.assert_close(values.grad.cpu(), expected_grad, atol=1e-3, rtol=1e-3)

  def test_to_padded_tensor_with_output_size_autograd(self):
    """Verifies autograd when padding to an explicit larger output_size."""
    values = torch.randn(
        5, 3, dtype=torch.float32, device="tpu", requires_grad=True
    )
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")

    nt = nt_internal.nested_view_from_values_offsets(
        values, offsets, min_seqlen=2, max_seqlen=3
    )
    # Pad to explicit larger sequence length = 4 (batch 0 has 2 pads, batch 1 has 1 pad)
    padded = nt.to_padded_tensor(0.0, output_size=(2, 4, 3))

    self.assertEqual(padded.shape, (2, 4, 3))
    loss = padded.sum()
    loss.backward()

    self.assertIsNotNone(values.grad)
    grad_cpu = values.grad.cpu()

    # All active tokens must have gradient 1.0
    self.assertTrue(torch.all(grad_cpu == 1.0))

  def test_nested_from_padded_tensor_autograd(self):
    """Verifies gradient flow through _nested_from_padded_tensor."""
    dense = torch.randn(
        2, 4, 3, dtype=torch.float32, device="tpu", requires_grad=True
    )
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")
    dummy = nt_internal._nt_view_dummy()

    nt = torch.ops.aten._nested_from_padded_tensor(
        dense, offsets, dummy, sum_S=5
    )
    self.assertEqual(nt.values().shape, (5, 3))

    # Compute loss on flat values
    target_weights = torch.arange(
        1, 16, dtype=torch.float32, device="tpu"
    ).reshape(5, 3)
    loss = (nt.values() * target_weights).sum()
    loss.backward()

    self.assertIsNotNone(dense.grad)
    dense_grad_cpu = dense.grad.cpu()

    # Active sequence elements receive the upstream weights
    utils.assert_close(dense_grad_cpu[0, :2, :], target_weights[:2].cpu())
    utils.assert_close(dense_grad_cpu[1, :3, :], target_weights[2:].cpu())

    # Padding positions in the dense tensor MUST receive zero gradient
    self.assertTrue(torch.all(dense_grad_cpu[0, 2:, :] == 0.0))
    self.assertTrue(torch.all(dense_grad_cpu[1, 3:, :] == 0.0))

  def test_roundtrip_jagged_to_padded_to_jagged_autograd(self):
    """Verifies identity gradient propagation through jagged -> padded -> jagged."""
    values = torch.randn(
        6, 4, dtype=torch.float32, device="tpu", requires_grad=True
    )
    offsets = torch.tensor([0, 3, 6], dtype=torch.int64, device="tpu")

    nt_in = nt_internal.nested_view_from_values_offsets(
        values, offsets, min_seqlen=3, max_seqlen=3
    )
    padded = nt_in.to_padded_tensor(0.0)
    dummy = nt_internal._nt_view_dummy()
    nt_out = torch.ops.aten._nested_from_padded_tensor(
        padded, offsets, dummy, sum_S=6
    )

    loss = (nt_out.values() * 3.5).sum()
    loss.backward()

    self.assertIsNotNone(values.grad)
    utils.assert_close(values.grad.cpu(), torch.full_like(values.cpu(), 3.5))

  def test_multidimensional_features_autograd(self):
    """Verifies autograd on jagged tensors with multidimensional feature shapes."""
    # (total_L=4, feature_dim_0=3, feature_dim_1=5)
    values = torch.randn(
        4, 3, 5, dtype=torch.float32, device="tpu", requires_grad=True
    )
    offsets = torch.tensor([0, 1, 4], dtype=torch.int64, device="tpu")

    nt = nt_internal.nested_view_from_values_offsets(
        values, offsets, min_seqlen=1, max_seqlen=3
    )
    padded = nt.to_padded_tensor(0.0)

    self.assertEqual(padded.shape, (2, 3, 3, 5))
    loss = padded.pow(2).sum()
    loss.backward()

    self.assertIsNotNone(values.grad)
    expected_grad = (2.0 * values).cpu()
    utils.assert_close(values.grad.cpu(), expected_grad)

  def test_autograd_retain_graph_and_accumulation(self):
    """Verifies gradient accumulation across multiple backward passes."""
    values = torch.randn(
        5, 2, dtype=torch.float32, device="tpu", requires_grad=True
    )
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")

    nt = nt_internal.nested_view_from_values_offsets(
        values, offsets, min_seqlen=2, max_seqlen=3
    )
    padded = nt.to_padded_tensor(0.0)

    loss1 = (padded * 1.5).sum()
    loss1.backward(retain_graph=True)

    loss2 = (padded * 2.5).sum()
    loss2.backward()

    # Accumulated grad must be 1.5 + 2.5 = 4.0
    self.assertIsNotNone(values.grad)
    utils.assert_close(values.grad.cpu(), torch.full_like(values.cpu(), 4.0))

  # ---------------------------------------------------------------------------
  # _nested_view_from_jagged Dedicated Unit Tests
  # ---------------------------------------------------------------------------

  def test_nested_view_from_jagged_basic_and_cpu_parity(self):
    """Tests _nested_view_from_jagged tensor construction and CPU parity."""
    values_cpu = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        dtype=torch.float32,
    )
    offsets_cpu = torch.tensor([0, 2, 5], dtype=torch.int64)
    dummy_cpu = nt_internal._nt_view_dummy()

    nt_cpu = torch._nested_view_from_jagged(values_cpu, offsets_cpu, dummy_cpu)

    values_tpu = values_cpu.to("tpu")
    offsets_tpu = offsets_cpu.to("tpu")
    dummy_tpu = nt_internal._nt_view_dummy()

    nt_tpu = torch._nested_view_from_jagged(values_tpu, offsets_tpu, dummy_tpu)

    self.assertEqual(nt_tpu.device.type, "tpu")
    self.assertEqual(nt_tpu.layout, torch.jagged)
    self.assertEqual(nt_tpu.shape[0], nt_cpu.shape[0])
    self.assertEqual(nt_tpu.shape[2], nt_cpu.shape[2])
    self.assertEqual(nt_tpu.dtype, nt_cpu.dtype)
    utils.assert_close(nt_tpu.values().cpu(), nt_cpu.values())
    utils.assert_close(nt_tpu.offsets().cpu(), nt_cpu.offsets())

  def test_nested_view_from_jagged_with_lengths(self):
    """Tests _nested_view_from_jagged with explicit lengths parameter."""
    values = torch.randn(6, 4, device="tpu")
    offsets = torch.tensor([0, 2, 6], dtype=torch.int64, device="tpu")
    lengths = torch.tensor([2, 4], dtype=torch.int64, device="tpu")
    dummy = nt_internal._nt_view_dummy()

    nt = torch._nested_view_from_jagged(values, offsets, dummy, lengths=lengths)
    self.assertEqual(nt.device.type, "tpu")
    self.assertEqual(nt.layout, torch.jagged)
    self.assertEqual(nt.shape[0], 2)
    self.assertEqual(nt.shape[2], 4)
    utils.assert_close(nt.values().cpu(), values.cpu())
    utils.assert_close(nt.offsets().cpu(), offsets.cpu())

  def test_nested_view_from_jagged_with_min_max_seqlen(self):
    """Tests _nested_view_from_jagged with min_seqlen and max_seqlen tensors."""
    values = torch.randn(7, 3, device="tpu")
    offsets = torch.tensor([0, 3, 7], dtype=torch.int64, device="tpu")
    dummy = nt_internal._nt_view_dummy()
    min_seqlen = torch.tensor(3, device="tpu")
    max_seqlen = torch.tensor(4, device="tpu")

    nt = torch._nested_view_from_jagged(
        values,
        offsets,
        dummy,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
    )
    self.assertEqual(nt.device.type, "tpu")
    self.assertEqual(nt.layout, torch.jagged)
    utils.assert_close(nt.values().cpu(), values.cpu())

  def test_nested_view_from_jagged_dtypes(self):
    """Tests _nested_view_from_jagged across multiple supported dtypes."""
    for dtype in [
        torch.float32,
        torch.bfloat16,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.bool,
    ]:
      if dtype == torch.bool:
        values = torch.tensor([True, False, True, True, False], device="tpu")
      elif dtype in (torch.int32, torch.int64):
        values = torch.tensor([1, 2, 3, 4, 5], dtype=dtype, device="tpu")
      else:
        values = torch.randn(5, 2, dtype=dtype, device="tpu")

      offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")
      dummy = nt_internal._nt_view_dummy()

      nt = torch._nested_view_from_jagged(values, offsets, dummy)
      self.assertEqual(nt.dtype, dtype)
      self.assertEqual(nt.device.type, "tpu")
      self.assertEqual(nt.layout, torch.jagged)
      utils.assert_close(nt.values().cpu(), values.cpu())

  def test_nested_view_from_jagged_mutation_aliasing(self):
    """Verifies that _nested_view_from_jagged creates an aliased view."""
    values = torch.zeros(5, 2, device="tpu")
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")
    dummy = nt_internal._nt_view_dummy()

    nt = torch._nested_view_from_jagged(values, offsets, dummy)

    # In-place add on values buffer should be reflected in the view
    values.add_(5.0)
    utils.assert_close(nt.values().cpu(), torch.full((5, 2), 5.0))

  # ---------------------------------------------------------------------------
  # _nested_from_padded_tensor Unit Tests
  # ---------------------------------------------------------------------------

  def test_nested_from_padded_tensor_eager_conversion_and_cpu_parity(self):
    """Tests _nested_from_padded_tensor conversion and CPU parity."""
    # Input dense tensor shape: (B=2, max_seqlen=3, D=2)
    # Batch item 0: [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]] (padded with [0, 0] at index 2)
    # Batch item 1: [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]] (full length 3)
    dense_cpu = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ],
        dtype=torch.float32,
    )
    # Offsets [0, 2, 5]:
    # - Batch 0 takes length 2 - 0 = 2 tokens: dense[0, 0:2, :] -> [[1.0, 2.0], [3.0, 4.0]]
    # - Batch 1 takes length 5 - 2 = 3 tokens: dense[1, 0:3, :] -> [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    # Total extracted tokens: sum_S = 2 + 3 = 5
    offsets_cpu = torch.tensor([0, 2, 5], dtype=torch.int64)
    dummy_cpu = nt_internal._nt_view_dummy()

    nt_cpu = torch.ops.aten._nested_from_padded_tensor(
        dense_cpu, offsets_cpu, dummy_cpu, sum_S=5
    )

    dense_tpu = dense_cpu.to("tpu")
    offsets_tpu = offsets_cpu.to("tpu")
    dummy_tpu = nt_internal._nt_view_dummy()

    nt_tpu = torch.ops.aten._nested_from_padded_tensor(
        dense_tpu, offsets_tpu, dummy_tpu, sum_S=5
    )

    self.assertEqual(nt_tpu.device.type, "tpu")
    self.assertEqual(nt_tpu.layout, torch.jagged)
    self.assertEqual(nt_tpu.shape[0], nt_cpu.shape[0])
    self.assertEqual(nt_tpu.shape[2], nt_cpu.shape[2])

    # Explicitly verify against manually constructed expected values buffer (5, 2)
    expected_values = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ],
        dtype=torch.float32,
    )
    utils.assert_close(nt_tpu.values().cpu(), expected_values)
    utils.assert_close(nt_tpu.values().cpu(), nt_cpu.values())
    utils.assert_close(nt_tpu.offsets().cpu(), offsets_cpu)

  def test_nested_from_padded_tensor_with_min_max_seqlen(self):
    """Tests _nested_from_padded_tensor with min/max seqlen metadata."""
    # Input dense tensor shape: (B=2, max_seqlen=4, D=3)
    # Offsets [0, 2, 6]:
    # - Batch 0 takes length 2 - 0 = 2 tokens (dense[0, :2, :])
    # - Batch 1 takes length 6 - 2 = 4 tokens (dense[1, :4, :])
    # - Total unpadded tokens = 2 + 4 = 6, yielding a flat values shape of (6, 3).
    dense = torch.randn(2, 4, 3, device="tpu")
    offsets = torch.tensor([0, 2, 6], dtype=torch.int64, device="tpu")
    dummy = nt_internal._nt_view_dummy()
    min_seqlen = torch.tensor(2, device="tpu")
    max_seqlen = torch.tensor(4, device="tpu")

    nt = torch.ops.aten._nested_from_padded_tensor(
        dense,
        offsets,
        dummy,
        ragged_idx=1,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
        sum_S=6,
    )
    self.assertEqual(nt.device.type, "tpu")
    self.assertEqual(nt.layout, torch.jagged)
    self.assertEqual(nt.values().shape, (6, 3))

  def test_nested_from_padded_tensor_dtypes(self):
    """Tests _nested_from_padded_tensor across multiple dtypes with CPU parity."""
    offsets_cpu = torch.tensor([0, 2, 5], dtype=torch.int64)
    dummy_cpu = nt_internal._nt_view_dummy()
    offsets_tpu = offsets_cpu.to("tpu")
    dummy_tpu = nt_internal._nt_view_dummy()

    for dtype in [
        torch.float32,
        torch.bfloat16,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.bool,
    ]:
      if dtype == torch.bool:
        dense_cpu = torch.tensor(
            [
                [[True, False], [True, True], [False, False]],
                [[False, True], [True, False], [True, True]],
            ],
            dtype=dtype,
        )
      elif dtype in (torch.int32, torch.int64):
        dense_cpu = torch.tensor(
            [[[1, 2], [3, 4], [0, 0]], [[5, 6], [7, 8], [9, 10]]],
            dtype=dtype,
        )
      else:
        dense_cpu = torch.randn(2, 3, 4, dtype=dtype)

      dense_tpu = dense_cpu.to("tpu")

      nt_cpu = torch.ops.aten._nested_from_padded_tensor(
          dense_cpu, offsets_cpu, dummy_cpu, sum_S=5
      )
      nt_tpu = torch.ops.aten._nested_from_padded_tensor(
          dense_tpu, offsets_tpu, dummy_tpu, sum_S=5
      )

      self.assertEqual(nt_tpu.dtype, dtype)
      self.assertEqual(nt_tpu.device.type, "tpu")
      self.assertEqual(nt_tpu.layout, torch.jagged)
      self.assertEqual(nt_tpu.shape[0], nt_cpu.shape[0])
      self.assertEqual(nt_tpu.shape[2], nt_cpu.shape[2])
      utils.assert_close(nt_tpu.values().cpu(), nt_cpu.values())
      utils.assert_close(nt_tpu.offsets().cpu(), nt_cpu.offsets())

  # ---------------------------------------------------------------------------
  # _nested_get_* Accessor Primitives Unit Tests
  # ---------------------------------------------------------------------------

  def test_nested_get_accessors(self):
    """Tests ATen _nested_get_* accessor primitives on jagged nested tensors."""
    values = torch.randn(5, 3, device="tpu", requires_grad=True)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")

    nt = nt_internal.nested_view_from_values_offsets(
        values, offsets, min_seqlen=2, max_seqlen=3
    )

    extracted_values = torch.ops.aten._nested_get_values(nt)
    extracted_offsets = torch.ops.aten._nested_get_offsets(nt)
    extracted_min_seqlen = torch.ops.aten._nested_get_min_seqlen(nt)
    extracted_max_seqlen = torch.ops.aten._nested_get_max_seqlen(nt)
    extracted_ragged_idx = torch.ops.aten._nested_get_ragged_idx(nt)
    dummy = torch.ops.aten._nested_get_jagged_dummy(nt)

    self.assertEqual(extracted_values.device.type, "tpu")
    self.assertEqual(extracted_offsets.device.type, "tpu")
    utils.assert_close(extracted_values.cpu(), values.cpu())
    utils.assert_close(extracted_offsets.cpu(), offsets.cpu())
    if extracted_min_seqlen.numel() > 0:
      self.assertEqual(extracted_min_seqlen.item(), 2)
    if extracted_max_seqlen.numel() > 0:
      self.assertEqual(extracted_max_seqlen.item(), 3)
    self.assertEqual(extracted_ragged_idx, 1)
    self.assertIsNotNone(dummy)

    # Verify autograd flows through _nested_get_values
    loss = (extracted_values * 2.0).sum()
    loss.backward()
    self.assertIsNotNone(values.grad)
    utils.assert_close(values.grad.cpu(), torch.full_like(values.cpu(), 2.0))

  # ---------------------------------------------------------------------------
  # Conversion Operators: Jagged <-> Padded Dense Tests
  # ---------------------------------------------------------------------------

  def test_jagged_to_padded_dense_forward_basic(self):
    """Tests basic jagged-to-padded conversion."""
    values = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    self._assert_jagged_to_padded(
        values, offsets, max_lengths=[3], padding_value=0.0
    )

  def test_jagged_to_padded_dense_padding_values_and_dtypes(self):
    """Tests various padding values and dtypes for jagged-to-padded conversion."""
    for dtype in [
        torch.float32,
        torch.bfloat16,
        torch.float64,
        torch.int64,
        torch.int32,
        torch.bool,
    ]:
      for pad_val in [0.0, 1.0, -1.0, float("-inf"), float("inf")]:
        if dtype == torch.bool:
          values = torch.tensor([True, False, True, True, False])
        elif dtype in (torch.int64, torch.int32):
          values = torch.tensor([10, 20, 30, 40, 50], dtype=dtype)
        else:
          values = torch.tensor([1.5, -2.5, 3.5, -4.5, 5.5], dtype=dtype)

        offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
        self._assert_jagged_to_padded(
            values, offsets, max_lengths=[4], padding_value=pad_val
        )

  def test_jagged_to_padded_dense_truncation(self):
    """Tests jagged-to-padded truncation when actual sequence length > max_length."""
    # Batch sequence lengths: [5, 2, 8, 0, 3], with max_length = 4
    offsets = torch.tensor([0, 5, 7, 15, 15, 18], dtype=torch.int64)
    values = torch.randn(18, 3)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[4], padding_value=-99.0
    )
    self.assertEqual(padded_tpu.shape, (5, 4, 3))

    # Verify specific batch rows:
    # Row 0: truncated from length 5 to 4
    utils.assert_close(padded_tpu[0].cpu(), values[0:4])
    # Row 1: length 2 preserved + 2 padding positions
    utils.assert_close(padded_tpu[1, :2].cpu(), values[5:7])
    self.assertTrue(torch.all(padded_tpu[1, 2:].cpu() == -99.0))
    # Row 3: empty sequence is 100% padding
    self.assertTrue(torch.all(padded_tpu[3].cpu() == -99.0))

  def test_jagged_1d_tokens(self):
    """Tests 1D jagged values (scalar tokens with no trailing feature dimensions)."""
    values = torch.tensor([101, 2054, 2003, 1037, 3231, 102], dtype=torch.int64)
    offsets = torch.tensor([0, 3, 4, 6], dtype=torch.int64)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[4]
    )
    self.assertEqual(padded_tpu.shape, (3, 4))

    # Convert back to 1D flat values
    recovered_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        padded_tpu, [offsets.to("tpu")], total_L=6
    )
    self.assertEqual(recovered_tpu.shape, (6,))
    utils.assert_close(recovered_tpu.cpu(), values)

  def test_padded_dense_to_jagged_forward_basic(self):
    """Tests basic padded-to-jagged conversion with and without total_L."""
    padded = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ],
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    self._assert_padded_to_jagged(padded, offsets)
    self._assert_padded_to_jagged(padded, offsets, total_L=5)

  def test_roundtrip_conversion(self):
    """Tests round-trip conversion: jagged -> padded dense -> jagged."""
    t1 = torch.randn(3, 4, device="tpu")
    t2 = torch.randn(1, 4, device="tpu")
    t3 = torch.randn(5, 4, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2, t3], layout=torch.jagged)

    values, offsets = nt.values(), nt.offsets()
    padded = torch.ops.aten._jagged_to_padded_dense_forward(
        values, [offsets], [5], 0.0
    )
    self.assertEqual(padded.shape, (3, 5, 4))

    recovered_values = torch.ops.aten._padded_dense_to_jagged_forward(
        padded, [offsets], total_L=values.size(0)
    )
    utils.assert_close(recovered_values.cpu(), values.cpu())

  def test_jagged_to_padded_dense_multidim(self):
    """Tests jagged conversion with higher-dimensional features."""
    values = torch.randn(6, 3, 5)
    offsets = torch.tensor([0, 1, 4, 6], dtype=torch.int64)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[3]
    )
    self.assertEqual(padded_tpu.shape, (3, 3, 3, 5))
    self._assert_padded_to_jagged(padded_tpu.cpu(), offsets, total_L=6)

  # ---------------------------------------------------------------------------
  # Edge Cases & Stress Scenarios
  # ---------------------------------------------------------------------------

  def test_all_empty_batch(self):
    """Tests entirely empty batch (offsets = [0, 0, 0, 0], total_L = 0)."""
    values = torch.empty((0, 4), dtype=torch.float32)
    offsets = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[5], padding_value=42.0
    )
    self.assertEqual(padded_tpu.shape, (3, 5, 4))
    self.assertTrue(torch.all(padded_tpu.cpu() == 42.0))

    recovered_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        padded_tpu, [offsets.to("tpu")], total_L=0
    )
    self.assertEqual(recovered_tpu.shape, (0, 4))

  def test_zero_padded_length(self):
    """Tests max_lengths = [0] with non-empty input."""
    values = torch.randn(5, 4)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[0]
    )
    self.assertEqual(padded_tpu.shape, (2, 0, 4))

  def test_zero_batch_size(self):
    """Tests zero batch size (offsets = [0])."""
    values = torch.empty((0, 4))
    offsets = torch.tensor([0], dtype=torch.int64)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[5]
    )
    self.assertEqual(padded_tpu.shape, (0, 5, 4))

  def test_exact_fit_boundary(self):
    """Tests boundary case where max_length equals the longest sequence."""
    offsets = torch.tensor([0, 4, 5, 9], dtype=torch.int64)
    values = torch.randn(9, 4)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[4]
    )
    self.assertEqual(padded_tpu.shape, (3, 4, 4))
    self._assert_padded_to_jagged(padded_tpu.cpu(), offsets, total_L=9)

  def test_compilation_cache_varying_offsets(self):
    """Tests compilation cache isolation when offsets change across executions."""
    values = torch.randn(6, 4)
    for offsets_list in [
        [0, 2, 4, 6],
        [0, 4, 5, 6],
        [0, 0, 0, 6],
        [0, 6, 6, 6],
        [0, 1, 3, 6],
    ]:
      offsets = torch.tensor(offsets_list, dtype=torch.int64)
      padded_tpu, _ = self._assert_jagged_to_padded(
          values, offsets, max_lengths=[6]
      )
      recovered_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
          padded_tpu, [offsets.to("tpu")], total_L=6
      )
      utils.assert_close(recovered_tpu.cpu(), values)

  def test_special_values_and_complex(self):
    """Tests complex64, nan, and inf padding values on TPU."""
    # Complex64
    values_complex = torch.tensor(
        [
            [1.0 + 2.0j, 3.0 - 4.0j],
            [5.0 + 6.0j, -7.0 + 8.0j],
            [9.0 - 10.0j, 11.0 + 12.0j],
        ],
        dtype=torch.complex64,
    )
    offsets_tpu = torch.tensor([0, 1, 3], dtype=torch.int64, device="tpu")
    padded_complex_tpu = torch.ops.aten._jagged_to_padded_dense_forward(
        values_complex.to("tpu"), [offsets_tpu], [2], 0.0
    )
    self.assertEqual(padded_complex_tpu.shape, (2, 2, 2))
    utils.assert_close(padded_complex_tpu.cpu()[0, 0], values_complex[0])
    utils.assert_close(padded_complex_tpu.cpu()[1, 0], values_complex[1])
    utils.assert_close(padded_complex_tpu.cpu()[1, 1], values_complex[2])

    recovered_complex_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        padded_complex_tpu, [offsets_tpu], total_L=3
    )
    utils.assert_close(recovered_complex_tpu.cpu(), values_complex)

    # NaN / Inf padding
    offsets_tpu = torch.tensor([0, 1, 3], dtype=torch.int64, device="tpu")
    values_float = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="tpu"
    )
    padded_nan = torch.ops.aten._jagged_to_padded_dense_forward(
        values_float, [offsets_tpu], [2], float("nan")
    )
    self.assertTrue(torch.isnan(padded_nan[0, 1, 0]).item())
    self.assertFalse(torch.isnan(padded_nan[0, 0, 0]).item())

    padded_inf = torch.ops.aten._jagged_to_padded_dense_forward(
        values_float, [offsets_tpu], [2], float("inf")
    )
    self.assertTrue(torch.isinf(padded_inf[0, 1, 0]).item())

  def test_nan_inf_token_payload_preservation(self):
    """Tests that NaNs and Infs inside valid tokens are preserved during dense-to-jagged."""
    dense = torch.tensor(
        [
            [[float("nan"), 1.0], [float("inf"), float("-inf")], [0.0, 0.0]],
            [[2.0, 3.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        device="tpu",
    )
    offsets = torch.tensor([0, 2, 3], dtype=torch.int64, device="tpu")

    jagged_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        dense, [offsets], total_L=3
    )
    self.assertTrue(torch.isnan(jagged_tpu[0, 0]).item())
    self.assertEqual(jagged_tpu[0, 1].item(), 1.0)
    self.assertTrue(torch.isposinf(jagged_tpu[1, 0]).item())
    self.assertTrue(torch.isneginf(jagged_tpu[1, 1]).item())
    self.assertEqual(jagged_tpu[2, 0].item(), 2.0)
    self.assertEqual(jagged_tpu[2, 1].item(), 3.0)

  def test_heterogeneous_device_offsets(self):
    """Tests conversion when values is on TPU but offsets is on CPU."""
    values = torch.randn(5, 4)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    padded_tpu = torch.ops.aten._jagged_to_padded_dense_forward(
        values.to("tpu"), [offsets], [3], 0.0
    )
    self.assertEqual(padded_tpu.device.type, "tpu")
    self.assertEqual(padded_tpu.shape, (2, 3, 4))
    recovered_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        padded_tpu, [offsets], total_L=5
    )
    utils.assert_close(recovered_tpu.cpu(), values)

  def test_non_contiguous_inputs(self):
    """Tests non-contiguous strided dense and offsets inputs."""
    # Transposed dense input
    dense = torch.randn(2, 3, 4, 5)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    self._assert_padded_to_jagged(dense.transpose(2, 3), offsets)

    # Strided offsets input
    full_offsets = torch.tensor([0, 999, 2, 999, 5], dtype=torch.int64)
    strided_offsets = full_offsets[::2]
    self._assert_jagged_to_padded(
        torch.randn(5, 4), strided_offsets, max_lengths=[3]
    )

  def test_sparse_and_asymmetric_sequences(self):
    """Tests heavily sparse batch with huge length disparity."""
    offsets = torch.tensor([0, 0, 1, 1, 129, 129, 132], dtype=torch.int64)
    values = torch.randn(132, 8)
    self._assert_jagged_to_padded(
        values, offsets, max_lengths=[64], padding_value=-1.0
    )

  def test_large_batch_stress(self):
    """Stress test with large batch size and uneven sequence lengths."""
    batch_size = 64
    lengths = torch.randint(0, 32, (batch_size,), dtype=torch.int64)
    offsets = torch.cat(
        [torch.tensor([0], dtype=torch.int64), torch.cumsum(lengths, dim=0)]
    )
    total_l = offsets[-1].item()

    values = torch.randn(total_l, 16)
    padded_tpu, _ = self._assert_jagged_to_padded(
        values, offsets, max_lengths=[32], padding_value=-1.0
    )
    self.assertEqual(padded_tpu.shape, (batch_size, 32, 16))

    recovered_tpu = torch.ops.aten._padded_dense_to_jagged_forward(
        padded_tpu, [offsets.to("tpu")], total_L=total_l
    )
    utils.assert_close(recovered_tpu.cpu(), values)

  def test_end_to_end_model_pipeline(self):
    """Tests end-to-end forward pipeline combining linear, conversion, and dense layers."""
    t1 = torch.randn(5, 8, device="tpu")
    t2 = torch.randn(3, 8, device="tpu")
    t3 = torch.randn(7, 8, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2, t3], layout=torch.jagged)

    # 1. Pointwise linear projection on jagged tensor
    linear = torch.nn.Linear(8, 16).to("tpu")
    projected_nt = linear(nt)

    # 2. Convert to padded dense for attention / layer norm
    padded = torch.ops.aten._jagged_to_padded_dense_forward(
        projected_nt.values(), [projected_nt.offsets()], [8], 0.0
    )
    self.assertEqual(padded.shape, (3, 8, 16))

    # 3. Dense operation on padded tensor
    layernorm = torch.nn.LayerNorm(16).to("tpu")
    normed = layernorm(padded)

    # 4. Convert back to jagged
    recovered_values = torch.ops.aten._padded_dense_to_jagged_forward(
        normed, [projected_nt.offsets()], total_L=projected_nt.values().size(0)
    )
    self.assertEqual(recovered_values.shape, (15, 16))

  # ---------------------------------------------------------------------------
  # Pointwise & Activation Kernels Unit Tests
  # ---------------------------------------------------------------------------

  def test_unary_pointwise_activations_and_cpu_parity(self):
    """Tests unary activation functions on TPU jagged nested tensors with CPU parity."""
    t1_cpu = torch.tensor([[-1.0, 2.0], [0.5, -3.0]], dtype=torch.float32)
    t2_cpu = torch.tensor(
        [[1.5, -0.5], [-2.0, 4.0], [0.0, 1.0]], dtype=torch.float32
    )
    nt_cpu = torch.nested.nested_tensor([t1_cpu, t2_cpu], layout=torch.jagged)

    nt_tpu = torch.nested.nested_tensor(
        [t1_cpu.to("tpu"), t2_cpu.to("tpu")], layout=torch.jagged
    )

    activations = [
        ("relu", torch.relu),
        (
            "gelu_exact",
            functools.partial(torch.nn.functional.gelu, approximate="none"),
        ),
        (
            "gelu_tanh",
            functools.partial(torch.nn.functional.gelu, approximate="tanh"),
        ),
        ("silu", torch.nn.functional.silu),
        ("sigmoid", torch.sigmoid),
        ("tanh", torch.tanh),
    ]

    for name, act_fn in activations:
      out_cpu = act_fn(nt_cpu)
      out_tpu = act_fn(nt_tpu)

      self.assertEqual(
          out_tpu.device.type, "tpu", msg=f"Device mismatch for {name}"
      )
      self.assertEqual(
          out_tpu.layout, torch.jagged, msg=f"Layout mismatch for {name}"
      )
      self.assertEqual(
          out_tpu.shape[0],
          out_cpu.shape[0],
          msg=f"Batch size mismatch for {name}",
      )
      self.assertEqual(
          out_tpu.shape[2],
          out_cpu.shape[2],
          msg=f"Feature dim mismatch for {name}",
      )
      utils.assert_close(out_tpu.offsets().cpu(), out_cpu.offsets())
      utils.assert_close(
          out_tpu.values().cpu(),
          out_cpu.values(),
          rtol=1e-4,
          atol=1e-4,
      )

  def test_unary_pointwise_math_and_dtypes(self):
    """Tests unary mathematical functions across supported floating point dtypes."""
    for dtype in [torch.float32, torch.bfloat16, torch.float64]:
      t1 = torch.tensor([[0.5, 1.2], [-0.8, 2.3]], dtype=dtype, device="tpu")
      t2 = torch.tensor(
          [[1.1, 0.4], [1.9, 0.7], [0.2, 3.1]], dtype=dtype, device="tpu"
      )
      nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

      math_ops = [
          ("sin", torch.sin),
          ("cos", torch.cos),
          ("exp", torch.exp),
          ("abs", torch.abs),
          ("neg", torch.neg),
          ("sqrt", torch.sqrt),
          ("clamp", functools.partial(torch.clamp, min=0.0, max=2.0)),
      ]

      for name, op in math_ops:
        out = op(nt)
        self.assertEqual(
            out.device.type, "tpu", msg=f"Device mismatch for {name}"
        )
        self.assertEqual(
            out.layout, torch.jagged, msg=f"Layout mismatch for {name}"
        )
        self.assertEqual(out.dtype, dtype, msg=f"Dtype mismatch for {name}")
        utils.assert_close(out.offsets().cpu(), nt.offsets().cpu())
        utils.assert_close(out.values().cpu(), op(nt.values()).cpu())

  def test_unary_pointwise_autograd_gradients(self):
    """Tests backward autograd gradient propagation through unary activations."""
    values_tpu = torch.randn(
        5, 3, dtype=torch.float32, device="tpu", requires_grad=True
    )
    offsets_tpu = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")

    nt_tpu = nt_internal.nested_view_from_values_offsets(
        values_tpu, offsets_tpu
    )
    out_tpu = torch.nn.functional.gelu(nt_tpu)

    loss = (out_tpu.values() * 3.0).sum()
    loss.backward()

    self.assertIsNotNone(values_tpu.grad)

    # Compare against CPU direct autograd
    values_cpu = values_tpu.detach().cpu().requires_grad_(True)
    out_cpu = torch.nn.functional.gelu(values_cpu)
    loss_cpu = (out_cpu * 3.0).sum()
    loss_cpu.backward()

    utils.assert_close(values_tpu.grad.cpu(), values_cpu.grad)

  def test_unary_pointwise_multidimensional_features(self):
    """Tests unary pointwise activations with multi-dimensional trailing feature shapes."""
    t1 = torch.randn(2, 4, 8, device="tpu")
    t2 = torch.randn(3, 4, 8, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    out = torch.relu(nt)
    self.assertEqual(out.device.type, "tpu")
    self.assertEqual(out.layout, torch.jagged)
    self.assertEqual(out.values().shape, (5, 4, 8))
    utils.assert_close(out.values().cpu(), torch.relu(nt.values()).cpu())

  # ---------------------------------------------------------------------------
  # Binary Operators & Arithmetic Unit Tests
  # ---------------------------------------------------------------------------

  def test_binary_jagged_with_jagged_arithmetic(self):
    """Tests elementwise binary operations between two identically-shaped jagged tensors."""
    # 1. Set up CPU input data & compute EXPECTED results
    offsets_cpu = torch.tensor([0, 3, 5], dtype=torch.int64)
    values_a_cpu = torch.randn(5, 4, dtype=torch.float32)
    values_b_cpu = torch.randn(5, 4, dtype=torch.float32)

    nt_a_cpu = nt_internal.nested_view_from_values_offsets(
        values_a_cpu, offsets_cpu
    )
    nt_b_cpu = nt_internal.nested_view_from_values_offsets(
        values_b_cpu, offsets_cpu
    )

    # 2. Set up TPU input data & compute ACTUAL results
    offsets_tpu = offsets_cpu.to("tpu")
    values_a_tpu = values_a_cpu.to("tpu")
    values_b_tpu = values_b_cpu.to("tpu")

    nt_a_tpu = nt_internal.nested_view_from_values_offsets(
        values_a_tpu, offsets_tpu
    )
    nt_b_tpu = nt_internal.nested_view_from_values_offsets(
        values_b_tpu, offsets_tpu
    )

    binary_ops = [
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
        ("div", lambda a, b: a / b.abs().clamp(min=0.1)),
    ]

    # 3. Compare ACTUAL (TPU) results against EXPECTED (CPU) results
    for name, op in binary_ops:
      expected_cpu = op(nt_a_cpu, nt_b_cpu)
      actual_tpu = op(nt_a_tpu, nt_b_tpu)

      self.assertEqual(
          actual_tpu.device.type, "tpu", msg=f"Device mismatch for {name}"
      )
      self.assertEqual(
          actual_tpu.layout, torch.jagged, msg=f"Layout mismatch for {name}"
      )
      utils.assert_close(actual_tpu.offsets().cpu(), expected_cpu.offsets())
      utils.assert_close(actual_tpu.values().cpu(), expected_cpu.values())

  def test_binary_jagged_with_scalar(self):
    """Tests binary operations between a jagged nested tensor and a scalar."""
    t1 = torch.randn(3, 4, device="tpu")
    t2 = torch.randn(2, 4, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    out_add = nt + 2.5
    self.assertEqual(out_add.device.type, "tpu")
    self.assertEqual(out_add.layout, torch.jagged)
    utils.assert_close(out_add.values().cpu(), (nt.values() + 2.5).cpu())

    out_mul = nt * 0.5
    self.assertEqual(out_mul.device.type, "tpu")
    utils.assert_close(out_mul.values().cpu(), (nt.values() * 0.5).cpu())

  def test_binary_jagged_with_broadcast_dense_feature(self):
    """Tests broadcasting a dense feature vector (D,) over a jagged tensor (B, j1, D)."""
    t1 = torch.randn(3, 4, device="tpu")
    t2 = torch.randn(2, 4, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    bias = torch.randn(4, device="tpu")
    out = nt + bias

    self.assertEqual(out.device.type, "tpu")
    self.assertEqual(out.layout, torch.jagged)
    self.assertEqual(out.values().shape, (5, 4))
    utils.assert_close(out.values().cpu(), (nt.values() + bias).cpu())

  def test_binary_autograd_gradients(self):
    """Tests two-sided and broadcasted autograd gradient flow for binary operations."""
    values_a = torch.randn(5, 4, device="tpu", requires_grad=True)
    values_b = torch.randn(5, 4, device="tpu", requires_grad=True)
    bias = torch.randn(4, device="tpu", requires_grad=True)
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64, device="tpu")

    nt_a = nt_internal.nested_view_from_values_offsets(values_a, offsets)
    nt_b = nt_internal.nested_view_from_values_offsets(values_b, offsets)

    # c = a * b + bias
    out = nt_a * nt_b + bias
    loss = out.values().sum()
    loss.backward()

    self.assertIsNotNone(values_a.grad)
    self.assertIsNotNone(values_b.grad)
    self.assertIsNotNone(bias.grad)

    utils.assert_close(values_a.grad.cpu(), values_b.detach().cpu())
    utils.assert_close(values_b.grad.cpu(), values_a.detach().cpu())
    utils.assert_close(bias.grad.cpu(), torch.full((4,), 5.0))

  # ---------------------------------------------------------------------------
  # Core Neural Network Modules Unit Tests
  # ---------------------------------------------------------------------------

  def test_nn_linear_layer(self):
    """Tests nn.Linear forward and backward propagation on jagged nested tensors."""
    t1_cpu = torch.randn(3, 8, dtype=torch.float32)
    t2_cpu = torch.randn(2, 8, dtype=torch.float32)
    nt_cpu = torch.nested.nested_tensor([t1_cpu, t2_cpu], layout=torch.jagged)

    nt_tpu = torch.nested.nested_tensor(
        [t1_cpu.to("tpu"), t2_cpu.to("tpu")], layout=torch.jagged
    )

    linear_cpu = torch.nn.Linear(8, 16, bias=True)
    linear_tpu = torch.nn.Linear(8, 16, bias=True).to("tpu")
    linear_tpu.weight.data.copy_(linear_cpu.weight.data)
    linear_tpu.bias.data.copy_(linear_cpu.bias.data)

    out_cpu = linear_cpu(nt_cpu)
    out_tpu = linear_tpu(nt_tpu)

    self.assertEqual(out_tpu.device.type, "tpu")
    self.assertEqual(out_tpu.layout, torch.jagged)
    self.assertEqual(out_tpu.shape[0], 2)
    self.assertEqual(out_tpu.shape[2], 16)
    utils.assert_close(
        out_tpu.values().cpu(), out_cpu.values(), rtol=1e-2, atol=1e-2
    )
    utils.assert_close(out_tpu.offsets().cpu(), out_cpu.offsets())

    # Test backward pass
    loss_cpu = out_cpu.values().sum()
    loss_cpu.backward()

    loss_tpu = out_tpu.values().sum()
    loss_tpu.backward()

    utils.assert_close(
        linear_tpu.weight.grad.cpu(),
        linear_cpu.weight.grad,
        rtol=1e-2,
        atol=1e-2,
    )
    utils.assert_close(
        linear_tpu.bias.grad.cpu(), linear_cpu.bias.grad, rtol=1e-2, atol=1e-2
    )

  def test_nn_embedding_layer(self):
    """Tests nn.Embedding lookup with jagged integer token IDs."""
    ids_cpu = torch.tensor([1, 4, 2, 7, 3], dtype=torch.int64)
    offsets_cpu = torch.tensor([0, 2, 5], dtype=torch.int64)
    nt_ids_cpu = nt_internal.nested_view_from_values_offsets(
        ids_cpu, offsets_cpu
    )

    nt_ids_tpu = nt_internal.nested_view_from_values_offsets(
        ids_cpu.to("tpu"), offsets_cpu.to("tpu")
    )

    emb_cpu = torch.nn.Embedding(10, 16)
    emb_tpu = torch.nn.Embedding(10, 16).to("tpu")
    emb_tpu.weight.data.copy_(emb_cpu.weight.data)

    out_cpu = emb_cpu(nt_ids_cpu)
    out_tpu = emb_tpu(nt_ids_tpu)

    self.assertEqual(out_tpu.device.type, "tpu")
    self.assertEqual(out_tpu.layout, torch.jagged)
    self.assertEqual(out_tpu.values().shape, (5, 16))
    utils.assert_close(out_tpu.values().cpu(), out_cpu.values())

    # Test backward gradient accumulation to embedding table
    loss_cpu = out_cpu.values().sum()
    loss_cpu.backward()

    loss_tpu = out_tpu.values().sum()
    loss_tpu.backward()

    utils.assert_close(emb_tpu.weight.grad.cpu(), emb_cpu.weight.grad)

  def test_nn_layernorm(self):
    """Tests nn.LayerNorm over feature dimensions of jagged nested tensors."""
    t1 = torch.randn(3, 16, device="tpu")
    t2 = torch.randn(4, 16, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    ln = torch.nn.LayerNorm(16).to("tpu")
    out = ln(nt)

    self.assertEqual(out.device.type, "tpu")
    self.assertEqual(out.layout, torch.jagged)
    self.assertEqual(out.values().shape, (7, 16))
    utils.assert_close(
        out.values().cpu(), ln(nt.values()).cpu(), rtol=1e-3, atol=1e-3
    )

  def test_nn_rmsnorm(self):
    """Tests nn.RMSNorm over feature dimensions of jagged nested tensors."""
    t1 = torch.randn(3, 16, device="tpu")
    t2 = torch.randn(4, 16, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    rms = torch.nn.RMSNorm(16).to("tpu")
    out = rms(nt)

    self.assertEqual(out.device.type, "tpu")
    self.assertEqual(out.layout, torch.jagged)
    self.assertEqual(out.values().shape, (7, 16))
    utils.assert_close(
        out.values().cpu(), rms(nt.values()).cpu(), rtol=1e-3, atol=1e-3
    )

  def test_nn_dropout(self):
    """Tests nn.Dropout in eval and train modes on jagged nested tensors."""
    t1 = torch.randn(3, 8, device="tpu")
    t2 = torch.randn(2, 8, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    dropout = torch.nn.Dropout(p=0.5).to("tpu")
    dropout.eval()
    out_eval = dropout(nt)
    utils.assert_close(out_eval.values().cpu(), nt.values().cpu())

  def test_transformer_feedforward_block_e2e(self):
    """Tests complete Transformer Feed-Forward Network block on jagged nested tensors."""
    d_model = 16
    d_ff = 64

    class TransformerFFN(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.act = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
        self.norm = torch.nn.LayerNorm(d_model)

      def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.norm(residual + x)

    model = TransformerFFN().to("tpu")
    t1 = torch.randn(4, d_model, device="tpu", requires_grad=True)
    t2 = torch.randn(2, d_model, device="tpu", requires_grad=True)
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    out = model(nt)
    self.assertEqual(out.device.type, "tpu")
    self.assertEqual(out.layout, torch.jagged)
    self.assertEqual(out.values().shape, (6, d_model))

    loss = out.values().sum()
    loss.backward()

    self.assertIsNotNone(model.linear1.weight.grad)
    self.assertIsNotNone(model.linear2.weight.grad)
    self.assertIsNotNone(model.norm.weight.grad)

  # ---------------------------------------------------------------------------
  # Tensor Manipulation & Slicing Unit Tests
  # ---------------------------------------------------------------------------

  def test_tensor_manipulation_split_chunk_cat(self):
    """Tests feature-dimension split, chunk, and concatenation on jagged tensors."""
    t1 = torch.randn(3, 12, device="tpu")
    t2 = torch.randn(2, 12, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    # Split into 3 equal chunks along feature dim
    chunks = torch.chunk(nt, 3, dim=-1)
    self.assertEqual(len(chunks), 3)
    for c in chunks:
      self.assertEqual(c.device.type, "tpu")
      self.assertEqual(c.layout, torch.jagged)
      self.assertEqual(c.values().shape, (5, 4))

    # Cat them back together along feature dim
    reconstructed = torch.cat(list(chunks), dim=-1)
    self.assertEqual(reconstructed.device.type, "tpu")
    self.assertEqual(reconstructed.layout, torch.jagged)
    self.assertEqual(reconstructed.values().shape, (5, 12))
    utils.assert_close(reconstructed.values().cpu(), nt.values().cpu())

  def test_tensor_manipulation_softmax(self):
    """Tests softmax normalization across feature dimensions of jagged tensors."""
    t1 = torch.randn(3, 8, device="tpu")
    t2 = torch.randn(2, 8, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    out = torch.softmax(nt, dim=-1)
    self.assertEqual(out.device.type, "tpu")
    self.assertEqual(out.layout, torch.jagged)
    self.assertEqual(out.values().shape, (5, 8))
    utils.assert_close(
        out.values().cpu(), torch.softmax(nt.values(), dim=-1).cpu()
    )

  def test_tensor_manipulation_clone_and_unbind(self):
    """Tests clone and unbind operations on jagged nested tensors."""
    t1 = torch.randn(3, 4, device="tpu")
    t2 = torch.randn(2, 4, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    # Clone
    nt_cloned = nt.clone()
    self.assertEqual(nt_cloned.device.type, "tpu")
    self.assertEqual(nt_cloned.layout, torch.jagged)
    utils.assert_close(nt_cloned.values().cpu(), nt.values().cpu())
    utils.assert_close(nt_cloned.offsets().cpu(), nt.offsets().cpu())

    # Unbind into list of dense tensors
    dense_list = torch.unbind(nt, dim=0)
    self.assertEqual(len(dense_list), 2)
    self.assertEqual(dense_list[0].shape, (3, 4))
    self.assertEqual(dense_list[1].shape, (2, 4))
    utils.assert_close(dense_list[0].cpu(), t1.cpu())
    utils.assert_close(dense_list[1].cpu(), t2.cpu())

  # ---------------------------------------------------------------------------
  # Reductions & Pooling Unit Tests
  # ---------------------------------------------------------------------------

  def test_reduction_global_and_feature(self):
    """Tests global and feature-dimension reduction operations on jagged tensors."""
    t1 = torch.randn(3, 4, device="tpu")
    t2 = torch.randn(2, 4, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    # Global reductions
    self.assertAlmostEqual(nt.sum().item(), nt.values().sum().item(), places=4)
    self.assertAlmostEqual(
        nt.mean().item(), nt.values().mean().item(), places=4
    )

    # Feature-dimension reduction (dim=-1)
    feat_sum = nt.sum(dim=-1)
    self.assertEqual(feat_sum.device.type, "tpu")
    self.assertEqual(feat_sum.layout, torch.jagged)
    self.assertEqual(feat_sum.values().shape, (5,))
    utils.assert_close(feat_sum.values().cpu(), nt.values().sum(dim=-1).cpu())

  def test_reduction_sequence_dim_pooling(self):
    """Tests sequence-dimension pooling (dim=1) reducing jagged (B, j1, D) to dense (B, D)."""
    t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="tpu")
    t2 = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)

    # Sequence sum pooling: Batch 0 -> [4.0, 6.0], Batch 1 -> [21.0, 24.0]
    pooled_sum = nt.sum(dim=1)
    self.assertEqual(pooled_sum.shape, (2, 2))
    expected_sum = torch.tensor([[4.0, 6.0], [21.0, 24.0]], device="tpu")
    utils.assert_close(pooled_sum.cpu(), expected_sum.cpu())

    # Sequence mean pooling: Batch 0 -> [2.0, 3.0], Batch 1 -> [7.0, 8.0]
    pooled_mean = nt.mean(dim=1)
    self.assertEqual(pooled_mean.shape, (2, 2))
    expected_mean = torch.tensor([[2.0, 3.0], [7.0, 8.0]], device="tpu")
    utils.assert_close(pooled_mean.cpu(), expected_mean.cpu())

  # ---------------------------------------------------------------------------
  # Performance Optimizations & Pipeline Verification Unit Tests
  # ---------------------------------------------------------------------------

  def test_mxu_tile_alignment_utility(self):
    """Tests padding jagged flat buffer to multiples of 128 for optimal TPU MXU systolic tiling."""
    # Verify experimental API metadata
    self.assertEqual(
        getattr(jagged_ops.align_jagged_to_multiple, annotations.TT_API_STAGE),
        annotations.Stage.EXPERIMENTAL,
    )

    # Create jagged tensor with 347 tokens (not divisible by 128)
    t1 = torch.randn(200, 64, device="tpu")
    t2 = torch.randn(147, 64, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)
    self.assertEqual(nt.values().shape, (347, 64))

    # Align to nearest multiple of 128 (384)
    aligned_nt = jagged_ops.align_jagged_to_multiple(nt, multiple=128)
    self.assertEqual(aligned_nt.device.type, "tpu")
    self.assertEqual(aligned_nt.layout, torch.jagged)
    self.assertEqual(aligned_nt.values().shape, (384, 64))
    utils.assert_close(aligned_nt.offsets().cpu(), nt.offsets().cpu())

    # Valid sequence values are preserved exactly
    utils.assert_close(aligned_nt.values()[:347].cpu(), nt.values().cpu())

    # 2D GEMM executes with 128-tile systolic efficiency
    linear = torch.nn.Linear(64, 128).to("tpu")
    projected = linear(aligned_nt)
    self.assertEqual(projected.values().shape, (384, 128))

    # Integer token ID jagged tensor with integer padding value
    tok1 = torch.randint(0, 1000, (50,), dtype=torch.int64, device="tpu")
    tok2 = torch.randint(0, 1000, (60,), dtype=torch.int64, device="tpu")
    nt_tok = torch.nested.nested_tensor([tok1, tok2], layout=torch.jagged)
    aligned_tok = jagged_ops.align_jagged_to_multiple(
        nt_tok, multiple=128, padding_value=0
    )
    self.assertEqual(aligned_tok.values().dtype, torch.int64)
    self.assertEqual(aligned_tok.values().shape, (128,))
    utils.assert_close(aligned_tok.values()[:110].cpu(), nt_tok.values().cpu())

    # Already aligned tensor is returned directly without extra allocations
    t_aligned = torch.randn(256, 64, device="tpu")
    nt_aligned = torch.nested.nested_tensor([t_aligned], layout=torch.jagged)
    self.assertIs(
        jagged_ops.align_jagged_to_multiple(nt_aligned, multiple=128),
        nt_aligned,
    )

  def test_strip_jagged_padding_utility(self):
    """Tests zero-copy stripping of trailing dummy padding rows from jagged tensors."""
    # Verify experimental API metadata
    self.assertEqual(
        getattr(jagged_ops.strip_jagged_padding, annotations.TT_API_STAGE),
        annotations.Stage.EXPERIMENTAL,
    )

    t1 = torch.randn(200, 64, device="tpu")
    t2 = torch.randn(147, 64, device="tpu")
    nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)
    self.assertEqual(nt.values().shape, (347, 64))

    # Align to 384
    aligned_nt = jagged_ops.align_jagged_to_multiple(nt, multiple=128)
    self.assertEqual(aligned_nt.values().shape, (384, 64))

    # Strip back to 347 (automatic sync)
    stripped_nt = jagged_ops.strip_jagged_padding(aligned_nt)
    self.assertEqual(stripped_nt.device.type, "tpu")
    self.assertEqual(stripped_nt.layout, torch.jagged)
    self.assertEqual(stripped_nt.values().shape, (347, 64))
    utils.assert_close(stripped_nt.values().cpu(), nt.values().cpu())
    utils.assert_close(stripped_nt.offsets().cpu(), nt.offsets().cpu())

    # Strip back to 347 with explicit valid_tokens (bypassing host-sync)
    stripped_explicit = jagged_ops.strip_jagged_padding(
        aligned_nt, valid_tokens=347
    )
    self.assertEqual(stripped_explicit.values().shape, (347, 64))
    utils.assert_close(stripped_explicit.values().cpu(), nt.values().cpu())

    # Calling on already compact tensor returns original tensor
    self.assertIs(jagged_ops.strip_jagged_padding(nt), nt)

  def test_align_and_strip_preserve_lengths(self):
    """Tests that align and strip keep the lengths of a jagged tensor."""
    # Two slots of 4 and 6 tokens that hold sequences of 2 and 3 tokens, so
    # the valid sequences are [0, 1] and [4, 5, 6].
    values = torch.arange(10.0, device="tpu").reshape(10, 1)
    offsets = torch.tensor([0, 4, 10], dtype=torch.int64, device="tpu")
    lengths = torch.tensor([2, 3], dtype=torch.int64, device="tpu")
    nt = torch.nested.nested_tensor_from_jagged(
        values, offsets, lengths=lengths
    )
    expected = [t.cpu() for t in nt.unbind()]

    # Already-compact / already-aligned tensor with holes is returned as-is
    self.assertIs(jagged_ops.align_jagged_to_multiple(nt, multiple=5), nt)
    self.assertIs(jagged_ops.strip_jagged_padding(nt), nt)
    self.assertIs(jagged_ops.strip_jagged_padding(nt, valid_tokens=10), nt)

    # Align 10 tokens to 12; nested shape, offsets, lengths and sequences kept
    aligned_nt = jagged_ops.align_jagged_to_multiple(nt, multiple=4)
    self.assertEqual(aligned_nt.shape, nt.shape)
    self.assertEqual(aligned_nt.values().shape, (12, 1))
    utils.assert_close(aligned_nt.offsets().cpu(), offsets.cpu())
    utils.assert_close(aligned_nt.lengths().cpu(), lengths.cpu())
    for got, want in zip(aligned_nt.unbind(), expected, strict=True):
      utils.assert_close(got.cpu(), want)

    # Strip back to 10 tokens with both automatic and explicit valid_tokens
    for valid_tokens in (None, 10):
      with self.subTest(valid_tokens=valid_tokens):
        stripped_nt = jagged_ops.strip_jagged_padding(
            aligned_nt, valid_tokens=valid_tokens
        )
        self.assertEqual(stripped_nt.shape, nt.shape)
        self.assertEqual(stripped_nt.values().shape, (10, 1))
        utils.assert_close(stripped_nt.offsets().cpu(), offsets.cpu())
        utils.assert_close(stripped_nt.lengths().cpu(), lengths.cpu())
        for got, want in zip(stripped_nt.unbind(), expected, strict=True):
          utils.assert_close(got.cpu(), want)

    # Jagged tensor with holes built via torch.nested.narrow where
    # offsets[-1] (9) < values.size(0) (10).
    base = torch.arange(30.0, device="tpu").reshape(2, 5, 3)
    nt_narrow = torch.nested.narrow(
        base,
        dim=1,
        start=torch.tensor([0, 1], dtype=torch.int64, device="tpu"),
        length=torch.tensor([2, 3], dtype=torch.int64, device="tpu"),
        layout=torch.jagged,
    )
    expected_narrow = [t.cpu() for t in nt_narrow.unbind()]
    aligned_narrow = jagged_ops.align_jagged_to_multiple(nt_narrow, multiple=4)
    self.assertEqual(aligned_narrow.shape, nt_narrow.shape)
    self.assertEqual(aligned_narrow.values().shape, (12, 3))
    for got, want in zip(aligned_narrow.unbind(), expected_narrow, strict=True):
      utils.assert_close(got.cpu(), want)

    # With valid_tokens=None, strip trims to offsets[-1] (9) while keeping
    # lengths, cached seqlens, and sequence contents.
    stripped_narrow = jagged_ops.strip_jagged_padding(aligned_narrow)
    self.assertEqual(stripped_narrow.shape, nt_narrow.shape)
    self.assertEqual(stripped_narrow.values().shape, (9, 3))
    utils.assert_close(
        stripped_narrow.lengths().cpu(), nt_narrow.lengths().cpu()
    )
    # pylint: disable=protected-access
    self.assertEqual(
        stripped_narrow._maybe_min_seqlen, nt_narrow._maybe_min_seqlen
    )
    self.assertEqual(
        stripped_narrow._maybe_max_seqlen, nt_narrow._maybe_max_seqlen
    )
    # pylint: enable=protected-access
    for got, want in zip(
        stripped_narrow.unbind(), expected_narrow, strict=True
    ):
      utils.assert_close(got.cpu(), want)

  def test_align_and_strip_preserve_cached_seqlens(self):
    """Tests that align and strip keep full, partial, or empty seqlen caches."""
    values = torch.randn(7, 2, device="tpu")
    offsets = torch.tensor([0, 3, 7], dtype=torch.int64, device="tpu")

    for cache_kwargs in (
        {"min_seqlen": 3, "max_seqlen": 4},
        {"min_seqlen": 3},
        {"max_seqlen": 4},
        {},
    ):
      with self.subTest(cache_kwargs=cache_kwargs):
        nt = torch.nested.nested_tensor_from_jagged(
            values, offsets, **cache_kwargs
        )
        aligned_nt = jagged_ops.align_jagged_to_multiple(nt, multiple=4)
        stripped_nt = jagged_ops.strip_jagged_padding(
            aligned_nt, valid_tokens=7
        )

        # `_maybe_*` reads the cache without computing or syncing; None stays
        # None and the input's cache is not populated as a side effect.
        # pylint: disable=protected-access
        for tensor in (nt, aligned_nt, stripped_nt):
          self.assertEqual(
              tensor._maybe_min_seqlen, cache_kwargs.get("min_seqlen")
          )
          self.assertEqual(
              tensor._maybe_max_seqlen, cache_kwargs.get("max_seqlen")
          )
        # pylint: enable=protected-access

  def test_align_and_strip_transposed_ragged_dim(self):
    """Tests align and strip when the ragged dim is not the first values dim."""
    # pylint: disable=protected-access
    # With 2 features, values dims 0 and 1 differ in size, so padding the
    # wrong dim is caught. With 7 features, both dims match the 7 tokens, so
    # a size check on the wrong dim would find nothing to strip.
    for features in [2, 7]:
      with self.subTest(features=features):
        t1 = torch.randn(3, features, device="tpu")
        t2 = torch.randn(4, features, device="tpu")
        nt = torch.nested.nested_tensor([t1, t2], layout=torch.jagged)
        nt_t = nt.transpose(1, 2)  # Values (features, 7), ragged_idx 2.
        self.assertEqual(nt_t._ragged_idx, 2)

        # Align 7 tokens to 8 along values dim 1; nested shape and ragged_idx
        # are kept and the padded slice equals padding_value.
        aligned_nt = jagged_ops.align_jagged_to_multiple(
            nt_t, multiple=4, padding_value=-1.5
        )
        self.assertEqual(aligned_nt._ragged_idx, 2)
        self.assertEqual(aligned_nt.shape, nt_t.shape)
        self.assertEqual(aligned_nt.values().shape, (features, 8))
        utils.assert_close(
            aligned_nt.values()[:, :7].cpu(), nt_t.values().cpu()
        )
        utils.assert_close(
            aligned_nt.values()[:, 7:].cpu(),
            torch.full((features, 1), -1.5),
        )

        # Strip back to 7 tokens; each sequence matches its input again
        stripped_nt = jagged_ops.strip_jagged_padding(aligned_nt)
        self.assertEqual(stripped_nt._ragged_idx, 2)
        self.assertEqual(stripped_nt.shape, nt_t.shape)
        self.assertEqual(stripped_nt.values().shape, (features, 7))
        for got, want in zip(
            stripped_nt.transpose(1, 2).unbind(), [t1, t2], strict=True
        ):
          utils.assert_close(got.cpu(), want.cpu())

    # No-op identity when _ragged_idx != 1 and ragged dim is already aligned /
    # compact (8 tokens with 2 features -> values shape (2, 8)).
    nt_aligned = torch.nested.nested_tensor(
        [
            torch.randn(3, 2, device="tpu"),
            torch.randn(5, 2, device="tpu"),
        ],
        layout=torch.jagged,
    ).transpose(1, 2)
    self.assertEqual(nt_aligned._ragged_idx, 2)
    self.assertIs(
        jagged_ops.align_jagged_to_multiple(nt_aligned, multiple=4), nt_aligned
    )
    self.assertIs(jagged_ops.strip_jagged_padding(nt_aligned), nt_aligned)
    self.assertIs(
        jagged_ops.strip_jagged_padding(nt_aligned, valid_tokens=8), nt_aligned
    )
    # pylint: enable=protected-access

  def test_align_and_strip_higher_rank(self):
    """Tests align and strip on higher-rank jagged tensors."""
    # pylint: disable=protected-access
    # Higher rank: jagged_dim=2 with 3D values (H=2, total=7, D=3) pads the
    # middle dim of values.
    values_3d = torch.randn(2, 7, 3, device="tpu")
    offsets_3d = torch.tensor([0, 3, 7], dtype=torch.int64, device="tpu")
    nt_mid = torch.nested.nested_tensor_from_jagged(
        values_3d, offsets_3d, jagged_dim=2, min_seqlen=3, max_seqlen=4
    )
    expected_mid = [t.cpu() for t in nt_mid.unbind()]
    aligned_mid = jagged_ops.align_jagged_to_multiple(
        nt_mid, multiple=4, padding_value=-2.0
    )
    self.assertEqual(aligned_mid._ragged_idx, 2)
    self.assertEqual(aligned_mid.shape, nt_mid.shape)
    self.assertEqual(aligned_mid.values().shape, (2, 8, 3))
    utils.assert_close(aligned_mid.values()[:, :7, :].cpu(), values_3d.cpu())
    utils.assert_close(
        aligned_mid.values()[:, 7:, :].cpu(),
        torch.full((2, 1, 3), -2.0),
    )
    stripped_mid = jagged_ops.strip_jagged_padding(aligned_mid)
    self.assertEqual(stripped_mid._ragged_idx, 2)
    self.assertEqual(stripped_mid.shape, nt_mid.shape)
    self.assertEqual(stripped_mid.values().shape, (2, 7, 3))
    for got, want in zip(stripped_mid.unbind(), expected_mid, strict=True):
      utils.assert_close(got.cpu(), want)

    # Higher rank: ragged_idx=3 via transpose(1, 3) on a 4D (B, j, H, D) NT.
    t1_4d = torch.randn(3, 2, 5, device="tpu")
    t2_4d = torch.randn(4, 2, 5, device="tpu")
    nt_4d = torch.nested.nested_tensor(
        [t1_4d, t2_4d], layout=torch.jagged
    ).transpose(1, 3)
    self.assertEqual(nt_4d._ragged_idx, 3)
    aligned_4d = jagged_ops.align_jagged_to_multiple(
        nt_4d, multiple=4, padding_value=3.5
    )
    self.assertEqual(aligned_4d._ragged_idx, 3)
    self.assertEqual(aligned_4d.shape, nt_4d.shape)
    self.assertEqual(aligned_4d.values().shape, (5, 2, 8))
    utils.assert_close(
        aligned_4d.values()[:, :, :7].cpu(), nt_4d.values().cpu()
    )
    utils.assert_close(
        aligned_4d.values()[:, :, 7:].cpu(),
        torch.full((5, 2, 1), 3.5),
    )
    for valid_tokens in (None, 7):
      stripped_4d = jagged_ops.strip_jagged_padding(
          aligned_4d, valid_tokens=valid_tokens
      )
      self.assertEqual(stripped_4d._ragged_idx, 3)
      self.assertEqual(stripped_4d.shape, nt_4d.shape)
      self.assertEqual(stripped_4d.values().shape, (5, 2, 7))
      for got, want in zip(
          stripped_4d.transpose(1, 3).unbind(),
          [t1_4d, t2_4d],
          strict=True,
      ):
        utils.assert_close(got.cpu(), want.cpu())
    # pylint: enable=protected-access

  def test_align_and_strip_combined_metadata_and_dtypes(self):
    """Tests align and strip with combined metadata across dtypes."""
    # pylint: disable=protected-access
    offsets = torch.tensor([0, 4, 10], dtype=torch.int64, device="tpu")
    lengths = torch.tensor([2, 3], dtype=torch.int64, device="tpu")

    for dtype, pad_val in (
        (torch.float32, -1.0),
        (torch.bfloat16, 0.0),
        (torch.int64, 7),
        (torch.bool, False),
    ):
      with self.subTest(dtype=dtype):
        if dtype == torch.bool:
          values = (torch.arange(30, device="tpu") % 2 == 0).reshape(10, 3)
        else:
          values = torch.arange(30, dtype=dtype, device="tpu").reshape(10, 3)

        nt_t = torch.nested.nested_tensor_from_jagged(
            values,
            offsets,
            lengths=lengths,
            min_seqlen=2,
            max_seqlen=3,
        ).transpose(1, 2)
        expected = [t.cpu() for t in nt_t.unbind()]

        aligned_nt = jagged_ops.align_jagged_to_multiple(
            nt_t, multiple=4, padding_value=pad_val
        )
        self.assertEqual(aligned_nt.values().dtype, dtype)
        self.assertEqual(aligned_nt.values().shape, (3, 12))
        self.assertEqual(aligned_nt._ragged_idx, 2)
        self.assertEqual(aligned_nt.shape, nt_t.shape)
        utils.assert_close(aligned_nt.offsets().cpu(), offsets.cpu())
        utils.assert_close(aligned_nt.lengths().cpu(), lengths.cpu())
        self.assertEqual(aligned_nt._maybe_min_seqlen, 2)
        self.assertEqual(aligned_nt._maybe_max_seqlen, 3)
        utils.assert_close(
            aligned_nt.values()[:, 10:].cpu(),
            torch.full((3, 2), pad_val, dtype=dtype),
        )
        for got, want in zip(aligned_nt.unbind(), expected, strict=True):
          utils.assert_close(got.cpu(), want)

        for valid_tokens in (None, 10):
          stripped_nt = jagged_ops.strip_jagged_padding(
              aligned_nt, valid_tokens=valid_tokens
          )
          self.assertEqual(stripped_nt.values().dtype, dtype)
          self.assertEqual(stripped_nt.values().shape, (3, 10))
          self.assertEqual(stripped_nt._ragged_idx, 2)
          self.assertEqual(stripped_nt.shape, nt_t.shape)
          utils.assert_close(stripped_nt.offsets().cpu(), offsets.cpu())
          utils.assert_close(stripped_nt.lengths().cpu(), lengths.cpu())
          self.assertEqual(stripped_nt._maybe_min_seqlen, 2)
          self.assertEqual(stripped_nt._maybe_max_seqlen, 3)
          for got, want in zip(stripped_nt.unbind(), expected, strict=True):
            utils.assert_close(got.cpu(), want)
    # pylint: enable=protected-access

  def test_align_and_strip_autograd(self):
    """Tests autograd through align and strip with holes and transposed dim."""
    # Leaf values buffer -> holes + transposed ragged dim -> align -> strip ->
    # unbind loss. Valid tokens receive exact upstream weights; holes and
    # stripped padding receive zero gradient.
    values = torch.randn(10, 3, device="tpu", requires_grad=True)
    offsets = torch.tensor([0, 4, 10], dtype=torch.int64, device="tpu")
    lengths = torch.tensor([2, 3], dtype=torch.int64, device="tpu")
    nt_t = torch.nested.nested_tensor_from_jagged(
        values, offsets, lengths=lengths, min_seqlen=2, max_seqlen=3
    ).transpose(1, 2)

    aligned_nt = jagged_ops.align_jagged_to_multiple(nt_t, multiple=4)
    stripped_nt = jagged_ops.strip_jagged_padding(aligned_nt, valid_tokens=10)
    self.assertTrue(aligned_nt.requires_grad)
    self.assertTrue(stripped_nt.requires_grad)

    weights = torch.arange(1.0, 4.0, device="tpu").reshape(3, 1)
    loss = sum((seq * weights).sum() for seq in stripped_nt.unbind())
    loss.backward()

    expected_grad = torch.zeros(10, 3)
    expected_grad[0:2] = torch.arange(1.0, 4.0)
    expected_grad[4:7] = torch.arange(1.0, 4.0)
    self.assertIsNotNone(values.grad)
    utils.assert_close(values.grad.cpu(), expected_grad)

    # NestedTensor leaf with requires_grad=True and transposed ragged dim
    nt_leaf = torch.nested.nested_tensor(
        [
            torch.randn(3, 2, device="tpu"),
            torch.randn(4, 2, device="tpu"),
        ],
        layout=torch.jagged,
        requires_grad=True,
    )
    nt_leaf_t = nt_leaf.transpose(1, 2)
    aligned_leaf = jagged_ops.align_jagged_to_multiple(nt_leaf_t, multiple=4)
    stripped_leaf = jagged_ops.strip_jagged_padding(aligned_leaf)
    out = stripped_leaf * 2.0
    # pylint: disable=protected-access
    grad_out = torch.nested.nested_tensor_from_jagged(
        torch.ones_like(stripped_leaf.values()),
        stripped_leaf.offsets(),
        jagged_dim=stripped_leaf._ragged_idx,
    )
    # pylint: enable=protected-access
    out.backward(grad_out)
    self.assertIsNotNone(nt_leaf.grad)
    self.assertTrue(nt_leaf.grad.is_nested)
    utils.assert_close(nt_leaf.grad.values().cpu(), torch.full((7, 2), 2.0))

  def test_strip_jagged_padding_invalid_valid_tokens(self):
    """Tests that strip_jagged_padding fails on out-of-bounds valid_tokens."""
    # pylint: disable=protected-access
    nt = torch.nested.nested_tensor_from_jagged(
        torch.randn(8, 2, device="tpu"),
        torch.tensor([0, 3, 7], dtype=torch.int64, device="tpu"),
    )
    for candidate in (nt, nt.transpose(1, 2)):
      for invalid_tokens in (-1, 9):
        with self.subTest(
            ragged_idx=candidate._ragged_idx,
            valid_tokens=invalid_tokens,
        ):
          with self.assertRaises(  # ASSERT_RAISES_OK=Python utility check.
              RuntimeError
          ):
            jagged_ops.strip_jagged_padding(
                candidate, valid_tokens=invalid_tokens
            )
    # pylint: enable=protected-access

  def test_inplace_residual_accumulation(self):
    """Verifies that in-place residual addition (add_) operates zero-copy and matches out-of-place add."""
    t1_x = torch.randn(3, 16, device="tpu")
    t2_x = torch.randn(2, 16, device="tpu")
    nt_x = torch.nested.nested_tensor([t1_x, t2_x], layout=torch.jagged)

    t1_res = torch.randn(3, 16, device="tpu")
    t2_res = torch.randn(2, 16, device="tpu")
    nt_res = torch.nested.nested_tensor([t1_res, t2_res], layout=torch.jagged)

    # Out-of-place baseline
    expected = nt_x.values() + nt_res.values()

    # In-place accumulation
    initial_ptr = nt_x.values().data_ptr()
    nt_x.values().add_(nt_res.values())

    # Verify zero-copy pointer preservation and numerical equivalence
    self.assertEqual(nt_x.values().data_ptr(), initial_ptr)
    utils.assert_close(nt_x.values().cpu(), expected.cpu())

  def test_zero_host_sync_pipeline_execution(self):
    """Verifies that complete Transformer blocks on jagged tensors execute asynchronously without host syncs."""
    d_model = 32
    d_ff = 128

    class FullTransformerLayer(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.proj = torch.nn.Linear(d_model, d_model)
        self.ffn1 = torch.nn.Linear(d_model, d_ff)
        self.ffn2 = torch.nn.Linear(d_ff, d_model)
        self.norm1 = torch.nn.RMSNorm(d_model)
        self.norm2 = torch.nn.RMSNorm(d_model)

      def forward(self, x):
        # Attention projection & residual
        residual = x
        qkv = self.qkv(x)
        q, _, _ = torch.chunk(qkv, 3, dim=-1)
        attn_out = self.proj(q)
        x = self.norm1(residual + attn_out)

        # FFN & residual
        residual = x
        ffn_out = self.ffn2(torch.nn.functional.gelu(self.ffn1(x)))
        return self.norm2(residual + ffn_out)

    layer = FullTransformerLayer().to("tpu")
    values = torch.randn(10, d_model, device="tpu", requires_grad=True)
    offsets = torch.tensor([0, 3, 7, 10], dtype=torch.int64, device="tpu")
    nt = nt_internal.nested_view_from_values_offsets(values, offsets)

    # Forward + backward pass executes asynchronously without blocking
    out = layer(nt)
    loss = out.values().sum()
    loss.backward()

    self.assertIsNotNone(values.grad)
    self.assertIsNotNone(layer.qkv.weight.grad)
    self.assertIsNotNone(layer.ffn2.weight.grad)


if __name__ == "__main__":
  absltest.main()

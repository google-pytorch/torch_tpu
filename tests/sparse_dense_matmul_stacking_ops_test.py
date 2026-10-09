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

"""Tests for sparse_dense_matmul_activation_unstack and sparse_dense_matmul_gradient_stack ops."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
from tests import op_testing


def reference_interleave_gradients(
    unstacked_gradients: list[torch.Tensor],
    stacked_batch_size: int,
    stacked_feature_dim: int,
    num_sc: int = 2,
) -> torch.Tensor:
  """CPU reference implementation for interleaving and padding gradients."""
  num_features = len(unstacked_gradients)
  feature_batch_size = unstacked_gradients[0].shape[0]
  per_sc_batch_size = feature_batch_size // num_sc
  padded_grads = []
  for g in unstacked_gradients:
    pad_dim = stacked_feature_dim - g.shape[1]
    if pad_dim > 0:
      padded_grads.append(F.pad(g, (0, pad_dim), value=0.0))
    else:
      padded_grads.append(g)
  stacked_grads_feat = torch.stack(padded_grads, dim=0)
  return (
      stacked_grads_feat.view(num_features, num_sc, per_sc_batch_size, -1)
      .transpose(0, 1)
      .reshape(stacked_batch_size, stacked_feature_dim)
  )


def reference_uninterleave_activations(
    stacked_activations: torch.Tensor,
    per_feature_batch_sizes: list[int],
    per_feature_dims: list[int],
    num_sc: int = 2,
) -> list[torch.Tensor]:
  """CPU reference implementation for uninterleaving activations."""
  num_features = len(per_feature_batch_sizes)
  feature_batch_size = per_feature_batch_sizes[0]
  per_sc_batch_size = feature_batch_size // num_sc
  unstacked = (
      stacked_activations.view(num_sc, num_features, per_sc_batch_size, -1)
      .transpose(0, 1)
      .reshape(num_features, feature_batch_size, -1)
  )
  outputs = []
  for i in range(num_features):
    outputs.append(unstacked[i][:, : per_feature_dims[i]])
  return outputs


class SparseDenseMatmulStackingOpsTest(op_testing.TorchTpuTestBase):
  """Tests for activation unstack and gradient stack operators."""

  def test_unstack_meta_shape(self):
    stacked = torch.empty(64, 32, device="meta", dtype=torch.float32)
    per_feature_batch_sizes = [32, 32]
    per_feature_dims = [16, 32]

    results = torch.tpu.ops.sparse_dense_matmul_activation_unstack(
        stacked, per_feature_batch_sizes, per_feature_dims
    )

    self.assertLen(results, 2)
    self.assertEqual(results[0].shape, torch.Size([32, 16]))
    self.assertEqual(results[1].shape, torch.Size([32, 32]))
    self.assertEqual(results[0].dtype, torch.float32)
    self.assertEqual(results[1].dtype, torch.float32)
    self.assertEqual(results[0].device.type, "meta")
    self.assertEqual(results[1].device.type, "meta")

  def test_unstack_meta_padded_stacked_dim(self):
    # Tests that stacked_embedding_dim can exceed max(per_feature_dims) (e.g. padded to 8).
    stacked = torch.empty(64, 16, device="meta", dtype=torch.float32)
    per_feature_batch_sizes = [32, 32]
    per_feature_dims = [8, 12]

    results = torch.tpu.ops.sparse_dense_matmul_activation_unstack(
        stacked, per_feature_batch_sizes, per_feature_dims
    )

    self.assertLen(results, 2)
    self.assertEqual(results[0].shape, torch.Size([32, 8]))
    self.assertEqual(results[1].shape, torch.Size([32, 12]))

  def test_unstack_meta_validation_errors(self):
    stacked_1d = torch.empty(64, device="meta", dtype=torch.float32)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected stacked_activations to be a 2D tensor"
    ):
      torch.tpu.ops.sparse_dense_matmul_activation_unstack(
          stacked_1d, [32, 32], [16, 32]
      )

    stacked_2d = torch.empty(64, 32, device="meta", dtype=torch.float32)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected at least one feature"
    ):
      torch.tpu.ops.sparse_dense_matmul_activation_unstack(stacked_2d, [], [])

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError,
        "expected per_feature_batch_sizes and per_feature_dims to have the same"
        " size",
    ):
      torch.tpu.ops.sparse_dense_matmul_activation_unstack(
          stacked_2d, [32], [16, 32]
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "to match total stacked batch size"
    ):
      torch.tpu.ops.sparse_dense_matmul_activation_unstack(
          stacked_2d, [32, 16], [16, 32]
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "to be at most stacked_embedding_dim"
    ):
      torch.tpu.ops.sparse_dense_matmul_activation_unstack(
          stacked_2d, [32, 32], [16, 48]
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected per_feature_batch_sizes\\[1\\] to be positive"
    ):
      torch.tpu.ops.sparse_dense_matmul_activation_unstack(
          stacked_2d, [64, 0], [16, 32]
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected per_feature_dims\\[0\\] to be positive"
    ):
      torch.tpu.ops.sparse_dense_matmul_activation_unstack(
          stacked_2d, [32, 32], [0, 32]
      )

  def test_stack_meta_shape(self):
    g1 = torch.empty(32, 16, device="meta", dtype=torch.float32)
    g2 = torch.empty(32, 32, device="meta", dtype=torch.float32)

    result = torch.tpu.ops.sparse_dense_matmul_gradient_stack([g1, g2], 64, 32)

    self.assertEqual(result.shape, torch.Size([64, 32]))
    self.assertEqual(result.dtype, torch.float32)
    self.assertEqual(result.device.type, "meta")

  def test_stack_meta_padded_stacked_dim(self):
    # Tests that stacked_feature_dim can exceed max(unstacked_gradients[i].size(1)).
    g1 = torch.empty(32, 8, device="meta", dtype=torch.float32)
    g2 = torch.empty(32, 12, device="meta", dtype=torch.float32)

    result = torch.tpu.ops.sparse_dense_matmul_gradient_stack([g1, g2], 64, 16)

    self.assertEqual(result.shape, torch.Size([64, 16]))

  def test_stack_meta_validation_errors(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        (RuntimeError, NotImplementedError),
        "(expected at least one unstacked gradient|no fallback function is"
        " registered)",
    ):
      torch.tpu.ops.sparse_dense_matmul_gradient_stack([], 64, 32)

    g_1d = torch.empty(32, device="meta", dtype=torch.float32)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected unstacked_gradients\\[0\\] to be a 2D tensor"
    ):
      torch.tpu.ops.sparse_dense_matmul_gradient_stack([g_1d], 32, 32)

    g1 = torch.empty(32, 16, device="meta", dtype=torch.float32)
    g2_int = torch.empty(32, 32, device="meta", dtype=torch.int32)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError,
        "expected unstacked_gradients\\[1\\] to have float32 dtype",
    ):
      torch.tpu.ops.sparse_dense_matmul_gradient_stack([g1, g2_int], 64, 32)

    g2 = torch.empty(32, 32, device="meta", dtype=torch.float32)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "to match stacked_batch_size"
    ):
      torch.tpu.ops.sparse_dense_matmul_gradient_stack([g1, g2], 60, 32)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "to be at most stacked_feature_dim"
    ):
      torch.tpu.ops.sparse_dense_matmul_gradient_stack([g1, g2], 64, 24)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected stacked_batch_size to be positive"
    ):
      torch.tpu.ops.sparse_dense_matmul_gradient_stack([g1, g2], 0, 32)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected stacked_feature_dim to be positive"
    ):
      torch.tpu.ops.sparse_dense_matmul_gradient_stack([g1, g2], 64, 0)

  @parameterized.parameters(
      (False, [8, 16], 16),
      (True, [8, 16], 16),
      (False, [8, 12], 16),
      (True, [8, 12], 16),
  )
  def test_sparse_dense_matmul_activation_unstack_on_tpu(
      self,
      compile_op: bool,
      per_feature_dims: list[int],
      stacked_dim: int,
  ):
    device = torch.device("tpu")
    torch.manual_seed(42)

    # 2 features; per-SC batch size is feature_bs // num_sc.
    num_sc = torch.tpu.get_device_properties().sparse_cores_per_device
    per_feature_batch_sizes = [16, 16]
    stacked_batch_size = 32

    stacked_activations_cpu = torch.randn(
        stacked_batch_size, stacked_dim, dtype=torch.float32
    )
    stacked_activations_tpu = stacked_activations_cpu.to(device)

    def unstack_fn(x):
      return torch.tpu.ops.sparse_dense_matmul_activation_unstack(
          x, per_feature_batch_sizes, per_feature_dims
      )

    if compile_op:
      unstack_fn = torch.compile(unstack_fn, fullgraph=True)

    results_tpu = unstack_fn(stacked_activations_tpu)

    self.assertLen(results_tpu, len(per_feature_dims))
    for i, dim in enumerate(per_feature_dims):
      self.assertEqual(results_tpu[i].shape, torch.Size([16, dim]))

    golden_results = reference_uninterleave_activations(
        stacked_activations_cpu,
        per_feature_batch_sizes,
        per_feature_dims,
        num_sc=num_sc,
    )

    for res_tpu, golden in zip(results_tpu, golden_results):
      self.assert_close(golden_result=golden, torch_tpu_result=res_tpu.cpu())

  @parameterized.parameters(
      (False, [8, 16], 16),
      (True, [8, 16], 16),
      (False, [8, 12], 16),
      (True, [8, 12], 16),
  )
  def test_sparse_dense_matmul_gradient_stack_on_tpu(
      self,
      compile_op: bool,
      feature_dims: list[int],
      stacked_dim: int,
  ):
    device = torch.device("tpu")
    torch.manual_seed(42)

    num_sc = torch.tpu.get_device_properties().sparse_cores_per_device
    stacked_batch_size = 32

    grads_cpu = [
        torch.randn(16, dim, dtype=torch.float32) for dim in feature_dims
    ]
    grads_tpu = [g.to(device) for g in grads_cpu]

    def stack_fn(grads):
      return torch.tpu.ops.sparse_dense_matmul_gradient_stack(
          grads, stacked_batch_size, stacked_dim
      )

    if compile_op:
      stack_fn = torch.compile(stack_fn, fullgraph=True)

    result_tpu = stack_fn(grads_tpu)

    self.assertEqual(result_tpu.shape, torch.Size([32, stacked_dim]))

    golden_result = reference_interleave_gradients(
        grads_cpu,
        stacked_batch_size,
        stacked_dim,
        num_sc=num_sc,
    )

    self.assert_close(
        golden_result=golden_result, torch_tpu_result=result_tpu.cpu()
    )

  @parameterized.parameters(
      ([8, 16], 16),
      ([8, 12], 16),
  )
  def test_unstack_stack_autograd_flow_on_tpu(
      self,
      per_feature_dims: list[int],
      stacked_dim: int,
  ):
    device = torch.device("tpu")
    torch.manual_seed(42)
    num_sc = torch.tpu.get_device_properties().sparse_cores_per_device

    stacked_activations_cpu = torch.randn(
        32, stacked_dim, dtype=torch.float32, requires_grad=True
    )
    stacked_activations_tpu = (
        stacked_activations_cpu.detach().clone().to(device).requires_grad_(True)
    )

    per_feature_batch_sizes = [16, 16]

    grads_cpu = [
        torch.randn(16, dim, dtype=torch.float32) for dim in per_feature_dims
    ]
    grads_tpu = [g.to(device) for g in grads_cpu]

    # CPU reference forward + backward
    golden_unstacked = reference_uninterleave_activations(
        stacked_activations_cpu,
        per_feature_batch_sizes,
        per_feature_dims,
        num_sc=num_sc,
    )
    golden_grad = torch.autograd.grad(
        outputs=golden_unstacked,
        inputs=stacked_activations_cpu,
        grad_outputs=grads_cpu,
    )[0]

    # TPU forward + backward using native custom op autograd
    tpu_unstacked = torch.tpu.ops.sparse_dense_matmul_activation_unstack(
        stacked_activations_tpu,
        per_feature_batch_sizes,
        per_feature_dims,
    )
    tpu_grad = torch.autograd.grad(
        outputs=tpu_unstacked,
        inputs=stacked_activations_tpu,
        grad_outputs=grads_tpu,
    )[0]

    self.assertEqual(tpu_grad.shape, torch.Size([32, stacked_dim]))
    self.assert_close(
        golden_result=golden_grad,
        torch_tpu_result=tpu_grad.cpu(),
    )

  @parameterized.parameters(
      ([8, 16], 16),
      ([8, 12], 16),
  )
  def test_unstack_autograd_partial_output_gradient(
      self,
      per_feature_dims: list[int],
      stacked_dim: int,
  ):
    """Verifies backward pass when downstream loss only uses a subset of features."""
    device = torch.device("tpu")
    torch.manual_seed(42)
    num_sc = torch.tpu.get_device_properties().sparse_cores_per_device

    stacked_cpu = torch.randn(
        32, stacked_dim, dtype=torch.float32, requires_grad=True
    )
    stacked_tpu = stacked_cpu.detach().clone().to(device).requires_grad_(True)

    per_feature_batch_sizes = [16, 16]

    golden_unstacked = reference_uninterleave_activations(
        stacked_cpu, per_feature_batch_sizes, per_feature_dims, num_sc=num_sc
    )
    golden_loss = golden_unstacked[0].sum()
    golden_loss.backward()

    tpu_unstacked = torch.tpu.ops.sparse_dense_matmul_activation_unstack(
        stacked_tpu, per_feature_batch_sizes, per_feature_dims
    )
    tpu_loss = tpu_unstacked[0].sum()
    tpu_loss.backward()

    self.assertIsNotNone(stacked_tpu.grad)
    self.assert_close(
        golden_result=stacked_cpu.grad,
        torch_tpu_result=stacked_tpu.grad.cpu(),
    )


if __name__ == "__main__":
  absltest.main()

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


class _TestAutogradUnstack(torch.autograd.Function):
  """Test autograd wrapper for unstack (FWD) and stack (BWD)."""

  @staticmethod
  def forward(
      ctx,
      stacked_activations: torch.Tensor,
      per_feature_batch_sizes: list[int],
      per_feature_dims: list[int],
  ) -> tuple[torch.Tensor, ...]:
    ctx.per_feature_batch_sizes = per_feature_batch_sizes
    ctx.per_feature_dims = per_feature_dims
    ctx.stacked_batch_size = stacked_activations.shape[0]
    ctx.stacked_feature_dim = stacked_activations.shape[1]
    results = torch.ops.tpu.sparse_dense_matmul_activation_unstack(
        stacked_activations, per_feature_batch_sizes, per_feature_dims
    )
    return tuple(results)

  @staticmethod
  def backward(ctx, *grad_outputs: torch.Tensor):
    grad_stacked = torch.ops.tpu.sparse_dense_matmul_gradient_stack(
        list(grad_outputs),
        ctx.stacked_batch_size,
        ctx.stacked_feature_dim,
    )
    return grad_stacked, None, None


class SparseDenseMatmulStackingOpsTest(op_testing.TorchTpuTestBase):
  """Tests for activation unstack and gradient stack operators."""

  def test_unstack_meta_shape(self):
    stacked = torch.empty(64, 32, device="meta", dtype=torch.float32)
    per_feature_batch_sizes = [32, 32]
    per_feature_dims = [16, 32]

    results = torch.ops.tpu.sparse_dense_matmul_activation_unstack(
        stacked, per_feature_batch_sizes, per_feature_dims
    )

    self.assertLen(results, 2)
    self.assertEqual(results[0].shape, torch.Size([32, 16]))
    self.assertEqual(results[1].shape, torch.Size([32, 32]))
    self.assertEqual(results[0].dtype, torch.float32)
    self.assertEqual(results[1].dtype, torch.float32)
    self.assertEqual(results[0].device.type, "meta")
    self.assertEqual(results[1].device.type, "meta")

  def test_unstack_meta_validation_errors(self):
    stacked_1d = torch.empty(64, device="meta", dtype=torch.float32)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected stacked_activations to be a 2D tensor"
    ):
      torch.ops.tpu.sparse_dense_matmul_activation_unstack(
          stacked_1d, [32, 32], [16, 32]
      )

    stacked_2d = torch.empty(64, 32, device="meta", dtype=torch.float32)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected at least one feature"
    ):
      torch.ops.tpu.sparse_dense_matmul_activation_unstack(stacked_2d, [], [])

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError,
        "expected per_feature_batch_sizes and per_feature_dims to have the same"
        " size",
    ):
      torch.ops.tpu.sparse_dense_matmul_activation_unstack(
          stacked_2d, [32], [16, 32]
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "to match total stacked batch size"
    ):
      torch.ops.tpu.sparse_dense_matmul_activation_unstack(
          stacked_2d, [32, 16], [16, 32]
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "to match maximum feature dimension"
    ):
      torch.ops.tpu.sparse_dense_matmul_activation_unstack(
          stacked_2d, [32, 32], [16, 48]
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected per_feature_batch_sizes\\[1\\] to be positive"
    ):
      torch.ops.tpu.sparse_dense_matmul_activation_unstack(
          stacked_2d, [64, 0], [16, 32]
      )

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected per_feature_dims\\[0\\] to be positive"
    ):
      torch.ops.tpu.sparse_dense_matmul_activation_unstack(
          stacked_2d, [32, 32], [0, 32]
      )

  def test_stack_meta_shape(self):
    g1 = torch.empty(32, 16, device="meta", dtype=torch.float32)
    g2 = torch.empty(32, 32, device="meta", dtype=torch.float32)

    result = torch.ops.tpu.sparse_dense_matmul_gradient_stack([g1, g2], 64, 32)

    self.assertEqual(result.shape, torch.Size([64, 32]))
    self.assertEqual(result.dtype, torch.float32)
    self.assertEqual(result.device.type, "meta")

  def test_stack_meta_validation_errors(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        (RuntimeError, NotImplementedError),
        "(expected at least one unstacked gradient|no fallback function is"
        " registered)",
    ):
      torch.ops.tpu.sparse_dense_matmul_gradient_stack([], 64, 32)

    g_1d = torch.empty(32, device="meta", dtype=torch.float32)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected unstacked_gradients\\[0\\] to be a 2D tensor"
    ):
      torch.ops.tpu.sparse_dense_matmul_gradient_stack([g_1d], 32, 32)

    g1 = torch.empty(32, 16, device="meta", dtype=torch.float32)
    g2_int = torch.empty(32, 32, device="meta", dtype=torch.int32)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError,
        "expected unstacked_gradients\\[1\\] to have float32 dtype",
    ):
      torch.ops.tpu.sparse_dense_matmul_gradient_stack([g1, g2_int], 64, 32)

    g2 = torch.empty(32, 32, device="meta", dtype=torch.float32)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "to match stacked_batch_size"
    ):
      torch.ops.tpu.sparse_dense_matmul_gradient_stack([g1, g2], 60, 32)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "to match stacked_feature_dim"
    ):
      torch.ops.tpu.sparse_dense_matmul_gradient_stack([g1, g2], 64, 24)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected stacked_batch_size to be positive"
    ):
      torch.ops.tpu.sparse_dense_matmul_gradient_stack([g1, g2], 0, 32)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Validates custom op schema errors.
        RuntimeError, "expected stacked_feature_dim to be positive"
    ):
      torch.ops.tpu.sparse_dense_matmul_gradient_stack([g1, g2], 64, 0)

  @parameterized.parameters(False, True)
  def test_sparse_dense_matmul_activation_unstack_on_tpu(
      self, compile_op: bool
  ):
    device = torch.device("tpu")
    torch.manual_seed(42)

    # 2 features, 2 SparseCores, per_sc_bs = 8, feature_bs = 16, stacked_bs = 32.
    num_sc = 2
    per_feature_batch_sizes = [16, 16]
    per_feature_dims = [8, 16]
    stacked_batch_size = 32
    stacked_dim = 16

    stacked_activations_cpu = torch.randn(
        stacked_batch_size, stacked_dim, dtype=torch.float32
    )
    stacked_activations_tpu = stacked_activations_cpu.to(device)

    def unstack_fn(x):
      return torch.ops.tpu.sparse_dense_matmul_activation_unstack(
          x, per_feature_batch_sizes, per_feature_dims
      )

    if compile_op:
      unstack_fn = torch.compile(unstack_fn, fullgraph=True)

    results_tpu = unstack_fn(stacked_activations_tpu)

    self.assertLen(results_tpu, 2)
    self.assertEqual(results_tpu[0].shape, torch.Size([16, 8]))
    self.assertEqual(results_tpu[1].shape, torch.Size([16, 16]))

    golden_results = reference_uninterleave_activations(
        stacked_activations_cpu,
        per_feature_batch_sizes,
        per_feature_dims,
        num_sc=num_sc,
    )

    for res_tpu, golden in zip(results_tpu, golden_results):
      self.assert_close(golden_result=golden, torch_tpu_result=res_tpu.cpu())

  @parameterized.parameters(False, True)
  def test_sparse_dense_matmul_gradient_stack_on_tpu(self, compile_op: bool):
    device = torch.device("tpu")
    torch.manual_seed(42)

    num_sc = 2
    stacked_batch_size = 32
    stacked_dim = 16

    g1_cpu = torch.randn(16, 8, dtype=torch.float32)
    g2_cpu = torch.randn(16, 16, dtype=torch.float32)

    g1_tpu = g1_cpu.to(device)
    g2_tpu = g2_cpu.to(device)

    def stack_fn(grads):
      return torch.ops.tpu.sparse_dense_matmul_gradient_stack(
          grads, stacked_batch_size, stacked_dim
      )

    if compile_op:
      stack_fn = torch.compile(stack_fn, fullgraph=True)

    result_tpu = stack_fn([g1_tpu, g2_tpu])

    self.assertEqual(result_tpu.shape, torch.Size([32, 16]))

    golden_result = reference_interleave_gradients(
        [g1_cpu, g2_cpu],
        stacked_batch_size,
        stacked_dim,
        num_sc=num_sc,
    )

    self.assert_close(
        golden_result=golden_result, torch_tpu_result=result_tpu.cpu()
    )

  def test_unstack_stack_autograd_flow_on_tpu(self):
    device = torch.device("tpu")
    torch.manual_seed(42)

    stacked_activations_cpu = torch.randn(
        32, 16, dtype=torch.float32, requires_grad=True
    )
    stacked_activations_tpu = (
        stacked_activations_cpu.detach().clone().to(device).requires_grad_(True)
    )

    per_feature_batch_sizes = [16, 16]
    per_feature_dims = [8, 16]

    g1_cpu = torch.randn(16, 8, dtype=torch.float32)
    g2_cpu = torch.randn(16, 16, dtype=torch.float32)

    g1_tpu = g1_cpu.to(device)
    g2_tpu = g2_cpu.to(device)

    # CPU reference forward + backward
    golden_unstacked = reference_uninterleave_activations(
        stacked_activations_cpu,
        per_feature_batch_sizes,
        per_feature_dims,
        num_sc=2,
    )
    golden_grad = torch.autograd.grad(
        outputs=[golden_unstacked[0], golden_unstacked[1]],
        inputs=stacked_activations_cpu,
        grad_outputs=[g1_cpu, g2_cpu],
    )[0]

    # TPU forward + backward using custom autograd function
    tpu_unstacked = _TestAutogradUnstack.apply(
        stacked_activations_tpu,
        per_feature_batch_sizes,
        per_feature_dims,
    )
    tpu_grad = torch.autograd.grad(
        outputs=[tpu_unstacked[0], tpu_unstacked[1]],
        inputs=stacked_activations_tpu,
        grad_outputs=[g1_tpu, g2_tpu],
    )[0]

    self.assertEqual(tpu_grad.shape, torch.Size([32, 16]))
    self.assert_close(
        golden_result=golden_grad,
        torch_tpu_result=tpu_grad.cpu(),
    )


if __name__ == "__main__":
  absltest.main()

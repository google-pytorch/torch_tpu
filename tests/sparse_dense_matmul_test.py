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


from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from torch_tpu._internal.utils.hardware import get_tpu_version
from torch_tpu._internal.utils.hardware import TpuVersion
from tests import op_testing


def _get_num_sc_per_device():
  tpu_version = get_tpu_version()
  if tpu_version in (TpuVersion.V7, TpuVersion.V6E):
    return 2
  elif tpu_version == TpuVersion.V5P:
    return 4
  else:
    raise ValueError(
        f"Unsupported TPU version for sparse_dense_matmul: {tpu_version}"
    )


class SparseDenseMatmulTest(
    op_testing.TorchTpuTestBase, parameterized.TestCase
):
  """Tests for sparse_dense_matmul op.

  This operator requires TPU v5e hardware because it uses SparseCore.
  """

  def _get_inputs(self, device):
    row_pointers = torch.tensor(
        [
            3,
            9,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,  # input for SC 0
            3,
            9,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            16,  # input for SC 1
        ],
        dtype=torch.int32,
        device=device,
    )
    embedding_ids = torch.tensor(
        [
            0,
            1,
            2,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            0,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            0,
            1,
            3,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            1,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
        ],
        dtype=torch.int32,
        device=device,
    )
    sample_ids = torch.tensor(
        [
            3,
            0,
            2,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            1,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2,
            3,
            1,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            0,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
            2147483647,
        ],
        dtype=torch.int32,
        device=device,
    )
    gains = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            1.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            1.0,
            1.0,
            1.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            1.0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        ],
        dtype=torch.float32,
        device=device,
    )
    embedding_table = (
        torch.arange(32, dtype=torch.float32, device=device)
        .unsqueeze(1)
        .repeat(1, 8)
    )
    return row_pointers, embedding_ids, sample_ids, gains, embedding_table

  @parameterized.parameters(False, True)
  def test_sparse_dense_matmul_on_tpu(self, compile_op):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )

    def matmul_fn(rp, e_ids, s_ids, g, et):
      return torch.ops.tpu.sparse_dense_matmul(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          device_batch_size=16,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
      )

    if compile_op:
      matmul_fn = torch.compile(matmul_fn, fullgraph=True)

    out = matmul_fn(
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        embedding_table,
    )

    expected = torch.tensor(
        [
            [1.0] * 8,
            [16.0] * 8,
            [2.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [17.0] * 8,
            [3.0] * 8,
            [0.0] * 8,
            [1.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
        ],
        dtype=torch.float32,
    )

    self.assert_close(golden_result=expected, torch_tpu_result=out.cpu())

  @parameterized.parameters(False, True)
  def test_sparse_dense_matmul_grad_with_sgd_on_tpu(self, compile_op):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )

    num_sc_per_device = _get_num_sc_per_device()

    # Shard the table for TPU (MOD sharding)
    vocab_size = embedding_table.shape[0]
    embedding_dim = embedding_table.shape[1]

    sharded_tables = []
    for core_id in range(num_sc_per_device):
      indices = [
          i for i in range(vocab_size) if i % num_sc_per_device == core_id
      ]
      sharded_tables.append(embedding_table[indices])

    embedding_table_sharded = torch.cat(sharded_tables, dim=0)

    batch_size = 16

    activations_grad = (
        torch.ones(
            batch_size, embedding_dim, dtype=torch.float32, device=device
        )
        * 0.01
    )
    learning_rate = torch.tensor(0.01, dtype=torch.float32, device=device)

    def grad_fn(rp, e_ids, s_ids, g, et, ag, lr):
      return torch.ops.tpu.sparse_dense_matmul_grad_with_sgd(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          ag,
          lr,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
      )

    if compile_op:
      grad_fn = torch.compile(grad_fn, fullgraph=True)

    updated_table_tpu = grad_fn(
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        embedding_table_sharded,
        activations_grad,
        learning_rate,
    )

    updated_table_tpu_np = updated_table_tpu.cpu().numpy()
    initial_table_sharded_np = embedding_table_sharded.cpu().numpy()

    updated_rows = [0, 1, 2, 3, 16, 17]

    # Verify that non-updated rows are indeed unchanged
    for i in range(vocab_size):
      if i not in updated_rows:
        self.assertTrue(
            np.allclose(
                updated_table_tpu_np[i], initial_table_sharded_np[i], atol=1e-5
            ),
            msg=f"Row {i} expected to be unchanged",
        )

    # Numerical check using finite differences of forward op
    def matmul_fn(rp, e_ids, s_ids, g, et):
      return torch.ops.tpu.sparse_dense_matmul(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
      )

    if compile_op:
      matmul_fn = torch.compile(matmul_fn, fullgraph=True)

    eps = 1e-3
    grad_approx = torch.zeros_like(embedding_table_sharded)

    for r in updated_rows:
      table_sharded_plus = embedding_table_sharded.clone()
      table_sharded_plus[r] += eps
      table_sharded_minus = embedding_table_sharded.clone()
      table_sharded_minus[r] -= eps

      out_plus = matmul_fn(
          row_pointers, embedding_ids, sample_ids, gains, table_sharded_plus
      )
      out_minus = matmul_fn(
          row_pointers, embedding_ids, sample_ids, gains, table_sharded_minus
      )

      dout = (out_plus - out_minus) / (2 * eps)
      grad_approx[r] = torch.sum(activations_grad * dout, dim=0)

    actual_update = initial_table_sharded_np - updated_table_tpu_np
    expected_update = (learning_rate * grad_approx).cpu().numpy()

    for r in updated_rows:
      self.assertTrue(
          np.allclose(actual_update[r], expected_update[r], atol=1e-4),
          msg=(
              f"Row {r} update mismatch. Actual: {actual_update[r]}, Expected:"
              f" {expected_update[r]}"
          ),
      )

  @parameterized.parameters(False, True)
  def test_sparse_dense_matmul_grad_with_adagrad_on_tpu(self, compile_op):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )
    num_sc_per_device = _get_num_sc_per_device()
    vocab_size = embedding_table.shape[0]
    embedding_dim = embedding_table.shape[1]
    sharded_tables = []
    for core_id in range(num_sc_per_device):
      indices = [
          i for i in range(vocab_size) if i % num_sc_per_device == core_id
      ]
      sharded_tables.append(embedding_table[indices])
    embedding_table_sharded = torch.cat(sharded_tables, dim=0)
    accumulator = (
        torch.ones(
            vocab_size, embedding_dim, dtype=torch.float32, device=device
        )
        * 0.1
    )
    sharded_accumulators = []
    for core_id in range(num_sc_per_device):
      indices = [
          i for i in range(vocab_size) if i % num_sc_per_device == core_id
      ]
      sharded_accumulators.append(accumulator[indices])
    accumulator_sharded = torch.cat(sharded_accumulators, dim=0)
    batch_size = 16
    activations_grad = (
        torch.ones(
            batch_size, embedding_dim, dtype=torch.float32, device=device
        )
        * 0.01
    )
    learning_rate = torch.tensor(0.01, dtype=torch.float32, device=device)
    epsilon = torch.tensor(1e-10, dtype=torch.float32, device=device)

    def grad_fn(rp, e_ids, s_ids, g, et, acc, ag, lr, eps):
      return torch.ops.tpu.sparse_dense_matmul_grad_with_adagrad(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          acc,
          ag,
          lr,
          eps,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=64,
      )

    if compile_op:
      grad_fn = torch.compile(grad_fn, fullgraph=True)
    updated_table_tpu, updated_acc_tpu = grad_fn(
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        embedding_table_sharded,
        accumulator_sharded,
        activations_grad,
        learning_rate,
        epsilon,
    )
    updated_table_tpu_np = updated_table_tpu.cpu().numpy()
    initial_table_sharded_np = embedding_table_sharded.cpu().numpy()
    updated_acc_tpu_np = updated_acc_tpu.cpu().numpy()
    initial_acc_sharded_np = accumulator_sharded.cpu().numpy()
    updated_rows = [0, 1, 2, 3, 16, 17]
    for i in range(vocab_size):
      if i not in updated_rows:
        self.assertTrue(
            np.allclose(
                updated_table_tpu_np[i], initial_table_sharded_np[i], atol=1e-5
            ),
            msg=f"Row {i} expected to be unchanged",
        )
        self.assertTrue(
            np.allclose(
                updated_acc_tpu_np[i], initial_acc_sharded_np[i], atol=1e-5
            ),
            msg=f"Accumulator row {i} expected to be unchanged",
        )

    def matmul_fn(rp, e_ids, s_ids, g, et):
      return torch.ops.tpu.sparse_dense_matmul(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
      )

    if compile_op:
      matmul_fn = torch.compile(matmul_fn, fullgraph=True)
    eps = 1e-3
    grad_approx = torch.zeros_like(embedding_table_sharded)
    for r in updated_rows:
      table_sharded_plus = embedding_table_sharded.clone()
      table_sharded_plus[r] += eps
      table_sharded_minus = embedding_table_sharded.clone()
      table_sharded_minus[r] -= eps
      out_plus = matmul_fn(
          row_pointers, embedding_ids, sample_ids, gains, table_sharded_plus
      )
      out_minus = matmul_fn(
          row_pointers, embedding_ids, sample_ids, gains, table_sharded_minus
      )
      dout = (out_plus - out_minus) / (2 * eps)
      grad_approx[r] = torch.sum(activations_grad * dout, dim=0)
    expected_acc_np = (
        initial_acc_sharded_np
        + grad_approx.cpu().numpy() * grad_approx.cpu().numpy()
    )
    expected_update = (
        learning_rate.cpu().numpy()
        * grad_approx.cpu().numpy()
        / (np.sqrt(expected_acc_np) + epsilon.cpu().numpy())
    )
    actual_update = initial_table_sharded_np - updated_table_tpu_np
    for r in updated_rows:
      self.assertTrue(
          np.allclose(actual_update[r], expected_update[r], atol=1e-4),
          msg=(
              f"Row {r} update mismatch. Actual: {actual_update[r]}, Expected:"
              f" {expected_update[r]}"
          ),
      )
      self.assertTrue(
          np.allclose(updated_acc_tpu_np[r], expected_acc_np[r], atol=1e-4),
          msg=(
              f"Accumulator row {r} mismatch. Actual: {updated_acc_tpu_np[r]},"
              f" Expected: {expected_acc_np[r]}"
          ),
      )

  @parameterized.parameters(False, True)
  def test_sparse_dense_matmul_grad_with_rowwise_adagrad_on_tpu(
      self, compile_op
  ):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )
    num_sc_per_device = _get_num_sc_per_device()
    vocab_size = embedding_table.shape[0]
    embedding_dim = embedding_table.shape[1]
    sharded_tables = []
    for core_id in range(num_sc_per_device):
      indices = [
          i for i in range(vocab_size) if i % num_sc_per_device == core_id
      ]
      sharded_tables.append(embedding_table[indices])
    embedding_table_sharded = torch.cat(sharded_tables, dim=0)
    # Row-wise accumulator: one scalar per row, so a 1-D [vocab] slot variable
    # (1/embedding_dim of the per-element Adagrad state).
    accumulator = torch.ones(vocab_size, dtype=torch.float32, device=device) * 0.1
    sharded_accumulators = []
    for core_id in range(num_sc_per_device):
      indices = [
          i for i in range(vocab_size) if i % num_sc_per_device == core_id
      ]
      sharded_accumulators.append(accumulator[indices])
    accumulator_sharded = torch.cat(sharded_accumulators, dim=0)
    batch_size = 16
    activations_grad = (
        torch.ones(
            batch_size, embedding_dim, dtype=torch.float32, device=device
        )
        * 0.01
    )
    learning_rate = torch.tensor(0.01, dtype=torch.float32, device=device)
    epsilon = torch.tensor(1e-10, dtype=torch.float32, device=device)

    def grad_fn(rp, e_ids, s_ids, g, et, acc, ag, lr, eps):
      return torch.ops.tpu.sparse_dense_matmul_grad_with_rowwise_adagrad(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          acc,
          ag,
          lr,
          eps,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=64,
      )

    if compile_op:
      grad_fn = torch.compile(grad_fn, fullgraph=True)
    updated_table_tpu, updated_acc_tpu = grad_fn(
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        embedding_table_sharded,
        accumulator_sharded,
        activations_grad,
        learning_rate,
        epsilon,
    )
    updated_table_tpu_np = updated_table_tpu.cpu().numpy()
    initial_table_sharded_np = embedding_table_sharded.cpu().numpy()
    updated_acc_tpu_np = updated_acc_tpu.cpu().numpy()
    initial_acc_sharded_np = accumulator_sharded.cpu().numpy()
    updated_rows = [0, 1, 2, 3, 16, 17]
    for i in range(vocab_size):
      if i not in updated_rows:
        self.assertTrue(
            np.allclose(
                updated_table_tpu_np[i], initial_table_sharded_np[i], atol=1e-5
            ),
            msg=f"Row {i} expected to be unchanged",
        )
        self.assertTrue(
            np.allclose(
                updated_acc_tpu_np[i], initial_acc_sharded_np[i], atol=1e-5
            ),
            msg=f"Accumulator row {i} expected to be unchanged",
        )

    def matmul_fn(rp, e_ids, s_ids, g, et):
      return torch.ops.tpu.sparse_dense_matmul(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
      )

    if compile_op:
      matmul_fn = torch.compile(matmul_fn, fullgraph=True)
    eps = 1e-3
    grad_approx = torch.zeros_like(embedding_table_sharded)
    for r in updated_rows:
      table_sharded_plus = embedding_table_sharded.clone()
      table_sharded_plus[r] += eps
      table_sharded_minus = embedding_table_sharded.clone()
      table_sharded_minus[r] -= eps
      out_plus = matmul_fn(
          row_pointers, embedding_ids, sample_ids, gains, table_sharded_plus
      )
      out_minus = matmul_fn(
          row_pointers, embedding_ids, sample_ids, gains, table_sharded_minus
      )
      dout = (out_plus - out_minus) / (2 * eps)
      grad_approx[r] = torch.sum(activations_grad * dout, dim=0)
    grad_approx_np = grad_approx.cpu().numpy()
    # Row-wise: the 1-D accumulator gets the per-row MEAN of squared gradient,
    # matching TorchRec/FBGEMM RowWiseAdagrad.
    grad_sq_row_mean = np.mean(grad_approx_np * grad_approx_np, axis=1)  # [vocab]
    expected_acc_np = initial_acc_sharded_np + grad_sq_row_mean  # 1-D [vocab]
    # The per-row scalar denominator broadcasts across the feature dimension.
    expected_update = (
        learning_rate.cpu().numpy()
        * grad_approx_np
        / (np.sqrt(expected_acc_np)[:, None] + epsilon.cpu().numpy())
    )
    actual_update = initial_table_sharded_np - updated_table_tpu_np
    for r in updated_rows:
      self.assertTrue(
          np.allclose(actual_update[r], expected_update[r], atol=1e-4),
          msg=(
              f"Row {r} update mismatch. Actual: {actual_update[r]}, Expected:"
              f" {expected_update[r]}"
          ),
      )
      self.assertTrue(
          np.allclose(updated_acc_tpu_np[r], expected_acc_np[r], atol=1e-4),
          msg=(
              f"Accumulator row {r} mismatch. Actual: {updated_acc_tpu_np[r]},"
              f" Expected: {expected_acc_np[r]}"
          ),
      )


if __name__ == "__main__":
  absltest.main()

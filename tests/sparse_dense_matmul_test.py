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


from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from torch._dynamo import testing as dynamo_testing
from torch._subclasses import fake_tensor
from torch.fx.experimental import symbolic_shapes
from torch_tpu._internal.compile import tpu_torch_compile
import torch_tpu._internal.device_utils.annotations as tpu_annotations
from tests import op_testing


def preprocess_sparse_dense_matmul_input(
    input_indices: torch.Tensor,
    input_offsets: torch.Tensor,
    max_ids_per_partition: int = 128,
    max_unique_ids_per_partition: int = 128,
    suggested_coo_buffer_size: int = 128,
    batch_size: int = 16,
    combiner: str = "sum",
    options: dict[str, Any] | None = None,
    table_name: str = "table_0",
) -> dict[str, torch.Tensor]:
  """Preprocesses sparse dense matmul input tensors for a table into CSR buffers and stats.

  Args:
    input_indices: Index tensor for the feature.
    input_offsets: Offset tensor for the feature.
    max_ids_per_partition: Max ID count per SparseCore partition.
    max_unique_ids_per_partition: Max unique ID count per partition.
    suggested_coo_buffer_size: Suggested COO buffer size.
    batch_size: Batch size per device.
    combiner: Reduction combiner for sample IDs ("sum", "mean", etc.).
    options: Optional dict specifying additional options (local_device_count,
      global_device_count, num_sc_per_device, sc_simd_width,
      hbm_word_size_in_4b, allow_id_dropping). num_sc_per_device and
      sc_simd_width default to the values of the attached TPU.
    table_name: Table name for logging and identification.

  Returns:
    Dict[str, Tensor] containing CSR buffers ("row_pointers", "embedding_ids",
    "sample_ids", "gains") and stats ("max_ids_per_partition",
    "max_unique_ids_per_partition", "required_buffer_sizes",
    "dropped_id_count").
  """
  if options is None:
    options = {}

  # Only query the attached TPU for the values that aren't given, since
  # `torch.tpu.get_device_properties()` raises without a TPU generation that
  # TorchTPU supports.
  num_sc_per_device = (
      options["num_sc_per_device"]
      if "num_sc_per_device" in options
      else torch.tpu.get_device_properties().sparse_cores_per_device
  )
  sc_simd_width = (
      options["sc_simd_width"]
      if "sc_simd_width" in options
      else torch.tpu.get_device_properties().sparse_core_simd_width
  )
  return torch.ops.tpu.preprocess_sparse_dense_matmul_input(
      input_indices,
      input_offsets,
      max_ids_per_partition,
      max_unique_ids_per_partition,
      suggested_coo_buffer_size,
      batch_size,
      combiner,
      options.get("local_device_count", 1),
      options.get("global_device_count", 1),
      num_sc_per_device,
      sc_simd_width,
      options.get("hbm_word_size_in_4b", _get_hbm_word_size_in_4b()),
      options.get("allow_id_dropping", True),
      table_name,
      options.get("enable_minibatching", False),
  )


def _get_hbm_word_size_in_4b():
  return 8


class SparseDenseMatmulTest(op_testing.TorchTpuTestBase):
  """Tests for sparse_dense_matmul op on SparseCore-enabled TPUs."""

  def test_input_preprocessing_on_tpu_numerical(self):
    device = torch.device("tpu")

    # Provide a deterministic dataset on CPU with exactly 32 values and 16
    # samples.
    # To make hand calculation trivial, embedding_table[id] = [id, id, ...].
    # Then the sum of any values is simply the sum of their IDs.
    values = torch.arange(32, dtype=torch.int32)
    # Samples distribution (lens: 1, 2, 3, 0, 2, 4, 1, 2, 0, 3, 1, 2, 3, 1, 4,
    # 3)
    offsets = torch.tensor(
        [0, 1, 3, 6, 6, 8, 12, 13, 15, 15, 18, 19, 21, 24, 25, 29, 32],
        dtype=torch.int32,
    )
    device_batch_size = 16

    # Compute the mathematical expectation native.
    # We hand-calculated the sum of the IDs for each sample bucket here:
    expected_sums = [
        0,  # s0: [0]
        1 + 2,  # s1: [1, 2]
        3 + 4 + 5,  # s2: [3, 4, 5]
        0,  # s3: []
        6 + 7,  # s4: [6, 7]
        8 + 9 + 10 + 11,  # s5: [8, 9, 10, 11]
        12,  # s6: [12]
        13 + 14,  # s7: [13, 14]
        0,  # s8: []
        15 + 16 + 17,  # s9: [15, 16, 17]
        18,  # s10: [18]
        19 + 20,  # s11: [19, 20]
        21 + 22 + 23,  # s12: [21, 22, 23]
        24,  # s13: [24]
        25 + 26 + 27 + 28,  # s14: [25, 26, 27, 28]
        29 + 30 + 31,  # s15: [29, 30, 31]
    ]

    vocab_size = 32
    embedding_dim = 8

    # exp has shape (device_batch_size, embedding_dim)
    exp = (
        torch.tensor(expected_sums, dtype=torch.float32)
        .unsqueeze(1)
        .repeat(1, embedding_dim)
    )

    table_0_outputs = preprocess_sparse_dense_matmul_input(
        values,
        offsets,
        max_ids_per_partition=128,
        max_unique_ids_per_partition=128,
        suggested_coo_buffer_size=128,
        batch_size=device_batch_size,
        combiner="sum",
    )

    rp = table_0_outputs["row_pointers"]
    e_ids = table_0_outputs["embedding_ids"]
    s_ids = table_0_outputs["sample_ids"]
    gains = table_0_outputs["gains"]

    logical_table = (
        torch.arange(vocab_size, dtype=torch.float32)
        .unsqueeze(1)
        .repeat(1, embedding_dim)
    )

    num_sc = torch.tpu.get_device_properties().sparse_cores_per_device
    sharded_tables = []
    for core_id in range(num_sc):
      chunk = logical_table[core_id::num_sc]
      sharded_tables.append(chunk)

    embedding_table = torch.cat(sharded_tables, dim=0).to(device)

    out_tpu = torch.tpu.ops.sparse_dense_matmul(
        rp.to(device),
        e_ids.to(device),
        s_ids.to(device),
        gains.to(device),
        embedding_table,
        device_batch_size=device_batch_size,
        max_ids_per_partition=32,
        max_unique_ids_per_partition=32,
    )

    self.assert_close(golden_result=exp, torch_tpu_result=out_tpu.cpu())

  def test_input_preprocessing_basic(self):
    val_f0 = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    off_f0 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)

    options = {
        "local_device_count": 1,
        "global_device_count": 1,
        "num_sc_per_device": 2,
        "allow_id_dropping": True,
    }

    table_0_outputs = preprocess_sparse_dense_matmul_input(
        val_f0,
        off_f0,
        max_ids_per_partition=8,
        max_unique_ids_per_partition=8,
        suggested_coo_buffer_size=16,
        batch_size=4,
        combiner="sum",
        options=options,
    )

    self.assertIn("row_pointers", table_0_outputs)
    self.assertIn("embedding_ids", table_0_outputs)
    self.assertIn("sample_ids", table_0_outputs)
    self.assertIn("gains", table_0_outputs)

    rp = table_0_outputs["row_pointers"]
    e_ids = table_0_outputs["embedding_ids"]
    s_ids = table_0_outputs["sample_ids"]
    gains = table_0_outputs["gains"]

    self.assertEqual(rp.dtype, torch.int32)
    self.assertEqual(e_ids.dtype, torch.int32)
    self.assertEqual(s_ids.dtype, torch.int32)
    self.assertEqual(gains.dtype, torch.float32)

  def test_input_preprocessing_stats(self):
    """Tests that preprocess_sparse_dense_matmul_input returns stats dict."""
    num_sc = torch.tpu.get_device_properties().sparse_cores_per_device
    ids_per_sample = 5 * num_sc
    val_f0 = torch.arange(ids_per_sample * num_sc, dtype=torch.int32)
    off_f0 = torch.arange(
        0, ids_per_sample * (num_sc + 1), ids_per_sample, dtype=torch.int32
    )

    options = {
        "local_device_count": 1,
        "global_device_count": 1,
        "num_sc_per_device": num_sc,
        "sc_simd_width": (
            torch.tpu.get_device_properties().sparse_core_simd_width
        ),
        "allow_id_dropping": True,
    }

    table_0_outputs = preprocess_sparse_dense_matmul_input(
        val_f0,
        off_f0,
        max_ids_per_partition=2,
        max_unique_ids_per_partition=2,
        suggested_coo_buffer_size=8 * num_sc * num_sc,
        batch_size=num_sc,
        combiner="sum",
        options=options,
    )

    self.assertIn("dropped_id_count", table_0_outputs)
    self.assertIn("max_ids_per_partition", table_0_outputs)
    self.assertIn("max_unique_ids_per_partition", table_0_outputs)
    self.assertIn("required_buffer_sizes", table_0_outputs)

    self.assert_close(
        golden_result=torch.tensor([3 * num_sc * num_sc], dtype=torch.int32),
        torch_tpu_result=table_0_outputs["dropped_id_count"],
    )
    self.assert_close(
        golden_result=torch.full((num_sc,), 5, dtype=torch.int32),
        torch_tpu_result=table_0_outputs["max_ids_per_partition"],
    )
    self.assert_close(
        golden_result=torch.full((num_sc,), 5, dtype=torch.int32),
        torch_tpu_result=table_0_outputs["max_unique_ids_per_partition"],
    )
    self.assert_close(
        golden_result=torch.full((num_sc,), 8 * num_sc, dtype=torch.int32),
        torch_tpu_result=table_0_outputs["required_buffer_sizes"],
    )

  def test_compute_stats_basic(self):
    """Verifies stats computation for a single batch."""
    indices = torch.tensor([0, 2, 0, 2, 1, 3, 1, 5], dtype=torch.int32)
    offsets = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32)

    max_ids, max_unique_ids = torch.ops.tpu.compute_sparse_dense_matmul_stats(
        indices,
        offsets,
        1,  # global_device_count
        2,  # num_sc_per_device
    )

    self.assertEqual(max_ids, 4)
    self.assertEqual(max_unique_ids, 3)

  def test_compute_stats_dedup_within_sample(self):
    """Verifies duplicate IDs in the same sample are counted once per sample."""
    indices = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
    offsets = torch.tensor([0, 3, 5], dtype=torch.int32)

    max_ids, max_unique_ids = torch.ops.tpu.compute_sparse_dense_matmul_stats(
        indices,
        offsets,
        1,
        2,
    )

    self.assertEqual(max_ids, 1)
    self.assertEqual(max_unique_ids, 1)

  @parameterized.named_parameters(
      (
          "sequential_basic",
          torch.arange(20, dtype=torch.int32),
          torch.tensor([0, 10, 20], dtype=torch.int32),
          1,
          2,
      ),
      (
          "uneven_lengths_and_duplicates",
          torch.tensor(
              [0, 1, 0, 4, 5, 2, 3, 3, 6, 7, 8, 9, 10, 11, 12, 13],
              dtype=torch.int32,
          ),
          torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32),
          1,
          2,
      ),
      (
          "four_sparsecores",
          torch.tensor(
              [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11],
              dtype=torch.int32,
          ),
          torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32),
          1,
          4,
      ),
      (
          "two_devices_two_sc",
          torch.arange(32, dtype=torch.int32),
          torch.tensor([0, 8, 16, 24, 32], dtype=torch.int32),
          2,
          2,
      ),
  )
  def test_compute_stats_parity_with_input_preprocessing(
      self,
      indices: torch.Tensor,
      offsets: torch.Tensor,
      global_device_count: int,
      num_sc_per_device: int,
  ):
    """Verifies that compute_sparse_dense_matmul_stats produces identical max stats to preprocess_sparse_dense_matmul_input."""
    batch_size = offsets.numel() - 1

    actual_max_ids, actual_max_unique_ids = (
        torch.ops.tpu.compute_sparse_dense_matmul_stats(
            indices,
            offsets,
            global_device_count,
            num_sc_per_device,
        )
    )

    dummy_limit = 1048576
    table_outputs = torch.ops.tpu.preprocess_sparse_dense_matmul_input(
        indices,
        offsets,
        dummy_limit,  # max_ids_per_partition
        dummy_limit,  # max_unique_ids_per_partition
        dummy_limit,  # suggested_coo_buffer_size
        batch_size,
        "sum",  # combiner
        1,  # local_device_count
        global_device_count,
        num_sc_per_device,
        torch.tpu.get_device_properties().sparse_core_simd_width,
        _get_hbm_word_size_in_4b(),  # hbm_word_size_in_4b
        True,  # allow_id_dropping
        "table_parity",
    )

    expected_max_ids = int(table_outputs["max_ids_per_partition"].max().item())
    expected_max_unique_ids = int(
        table_outputs["max_unique_ids_per_partition"].max().item()
    )

    self.assertEqual(
        actual_max_ids,
        expected_max_ids,
        msg=(
            f"max_ids mismatch: actual={actual_max_ids}, "
            f"expected={expected_max_ids}"
        ),
    )
    self.assertEqual(
        actual_max_unique_ids,
        expected_max_unique_ids,
        msg=(
            f"max_unique_ids mismatch: actual={actual_max_unique_ids}, "
            f"expected={expected_max_unique_ids}"
        ),
    )

  def _get_inputs(self, device, enable_minibatching: bool = False):
    num_sc = torch.tpu.get_device_properties().sparse_cores_per_device
    input_indices = torch.tensor([2, 1, 4, 0, 3, 6, 0, 2], dtype=torch.int32)
    input_offsets = torch.tensor(
        [0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7, 8, 8, 8, 8, 8],
        dtype=torch.int32,
    )
    preprocessed = preprocess_sparse_dense_matmul_input(
        input_indices,
        input_offsets,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        suggested_coo_buffer_size=8 * num_sc * num_sc,
        batch_size=16,
        combiner="sum",
        options={
            "local_device_count": 1,
            "global_device_count": 1,
            "num_sc_per_device": num_sc,
            "sc_simd_width": (
                torch.tpu.get_device_properties().sparse_core_simd_width
            ),
            "allow_id_dropping": False,
            "enable_minibatching": enable_minibatching,
        },
    )
    row_pointers = preprocessed["row_pointers"].to(device)
    embedding_ids = preprocessed["embedding_ids"].to(device)
    sample_ids = preprocessed["sample_ids"].to(device)
    gains = preprocessed["gains"].to(device)
    embedding_table = (
        torch.arange(32, dtype=torch.float32, device=device)
        .unsqueeze(1)
        .repeat(1, 8)
    )
    return row_pointers, embedding_ids, sample_ids, gains, embedding_table

  def _get_updated_rows(self, vocab_size: int = 32) -> list[int]:
    num_sc = torch.tpu.get_device_properties().sparse_cores_per_device
    shard_size = vocab_size // num_sc
    logical_ids = [0, 1, 2, 3, 4, 6]
    return sorted(
        (k % num_sc) * shard_size + (k // num_sc) for k in logical_ids
    )

  def _get_expected_forward_output(self, vocab_size: int = 32) -> torch.Tensor:
    num_sc = torch.tpu.get_device_properties().sparse_cores_per_device
    shard_size = vocab_size // num_sc

    def phys_row(k: int) -> float:
      return float((k % num_sc) * shard_size + (k // num_sc))

    return torch.tensor(
        [
            [phys_row(2)] * 8,
            [phys_row(1)] * 8,
            [phys_row(4)] * 8,
            [phys_row(0)] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [phys_row(3)] * 8,
            [phys_row(6)] * 8,
            [phys_row(0)] * 8,
            [phys_row(2)] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
            [0.0] * 8,
        ],
        dtype=torch.float32,
    )

  @parameterized.parameters(False, True)
  def test_sparse_dense_matmul_on_tpu(self, compile_op):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )

    def matmul_fn(rp, e_ids, s_ids, g, et):
      return torch.tpu.ops.sparse_dense_matmul(
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

    expected = self._get_expected_forward_output()

    self.assert_close(golden_result=expected, torch_tpu_result=out.cpu())

  def test_sparse_dense_matmul_eager_device_assignment(self):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )

    out = torch.tpu.ops.sparse_dense_matmul(
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        embedding_table,
        device_batch_size=16,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )

    expected = self._get_expected_forward_output()
    self.assert_close(golden_result=expected, torch_tpu_result=out.cpu())

  @parameterized.parameters("meta", "tpu")
  def test_sparse_dense_matmul_fake_symbolic_shapes(self, device):
    """Tests sparse_dense_matmul doesn't specialize symbolic FakeTensor dims."""
    device_batch_size = 16
    shape_env = symbolic_shapes.ShapeEnv()
    mode = fake_tensor.FakeTensorMode(shape_env=shape_env)

    def make_fake(*shape, dtype):
      return mode.from_tensor(
          torch.empty(shape, dtype=dtype, device=device),
          symbolic_context=symbolic_shapes.StatelessSymbolicContext(
              dynamic_sizes=[symbolic_shapes.DimDynamic.DYNAMIC] * len(shape)
          ),
      )

    row_pointers = make_fake(64, dtype=torch.int32)
    embedding_ids = make_fake(128, dtype=torch.int32)
    sample_ids = make_fake(128, dtype=torch.int32)
    gains = make_fake(128, dtype=torch.float32)
    embedding_table = make_fake(32, 8, dtype=torch.float32)
    with mode:
      out = torch.tpu.ops.sparse_dense_matmul(
          row_pointers,
          embedding_ids,
          sample_ids,
          gains,
          embedding_table,
          device_batch_size=device_batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
      )
    sizes = [*embedding_table.shape, out.shape[1]]
    for size in sizes:
      self.assertIsInstance(size, torch.SymInt)
    specialized = [
        size for size in sizes if shape_env.replace(size.node.expr).is_number
    ]
    self.assertEmpty(specialized)
    self.assertEqual(out.shape[0], device_batch_size)

  def test_sparse_dense_matmul_compile_dynamic_shapes(self):
    """Tests compiled sparse_dense_matmul recompiles once for dynamic shapes."""
    torch._dynamo.reset()
    self.addCleanup(torch._dynamo.reset)
    counter = dynamo_testing.CompileCounter()
    device_batch_size = 16

    @torch.compile(backend=counter, fullgraph=True)
    def fn(row_pointers, embedding_ids, sample_ids, gains, embedding_table):
      return torch.tpu.ops.sparse_dense_matmul(
          row_pointers,
          embedding_ids,
          sample_ids,
          gains,
          embedding_table,
          device_batch_size=device_batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
      )

    for n in (2, 3, 4):
      # All dims change between calls.
      row_pointers = torch.empty(n * 8, dtype=torch.int32, device="meta")
      embedding_ids = torch.empty(n * 16, dtype=torch.int32, device="meta")
      sample_ids = torch.empty(n * 16, dtype=torch.int32, device="meta")
      gains = torch.empty(n * 16, dtype=torch.float32, device="meta")
      embedding_table = torch.empty(n * 32, n + 4, device="meta")
      out = fn(row_pointers, embedding_ids, sample_ids, gains, embedding_table)
      self.assertEqual(out.shape, (device_batch_size, n + 4))
    # One static frame, then one dynamic frame that serves all later sizes.
    self.assertEqual(counter.frame_count, 2)

  def test_sparse_dense_matmul_grad_eager_device_assignment(self):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )
    num_sc_per_device = (
        torch.tpu.get_device_properties().sparse_cores_per_device
    )
    vocab_size, embedding_dim = embedding_table.shape

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

    # Execute backward SGD in eager mode without torch.compile
    # to verify eager DeviceAssignment
    updated_table_tpu = torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        embedding_table_sharded,
        activations_grad,
        learning_rate,
        device_batch_size=batch_size,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        computation_name="test_sgd_eager_da",
    )
    self.assertEqual(updated_table_tpu.shape, embedding_table_sharded.shape)

  def test_sparse_dense_matmul_grad_with_adagrad_eager_device_assignment(self):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )
    num_sc_per_device = (
        torch.tpu.get_device_properties().sparse_cores_per_device
    )
    vocab_size, embedding_dim = embedding_table.shape

    sharded_tables = []
    for core_id in range(num_sc_per_device):
      indices = [
          i for i in range(vocab_size) if i % num_sc_per_device == core_id
      ]
      sharded_tables.append(embedding_table[indices])
    embedding_table_sharded = torch.cat(sharded_tables, dim=0)
    accumulator_sharded = torch.ones_like(embedding_table_sharded) * 0.1

    batch_size = 16
    activations_grad = (
        torch.ones(
            batch_size, embedding_dim, dtype=torch.float32, device=device
        )
        * 0.01
    )
    learning_rate = torch.tensor(0.01, dtype=torch.float32, device=device)
    epsilon = torch.tensor(1e-8, dtype=torch.float32, device=device)

    # Execute backward Adagrad in eager mode without torch.compile
    # to verify eager DeviceAssignment
    updated_table, updated_acc = (
        torch.tpu.ops.sparse_dense_matmul_grad_with_adagrad(
            row_pointers,
            embedding_ids,
            sample_ids,
            gains,
            embedding_table_sharded,
            accumulator_sharded,
            activations_grad,
            learning_rate,
            epsilon,
            device_batch_size=batch_size,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="test_adagrad_eager_da",
        )
    )
    self.assertEqual(updated_table.shape, embedding_table_sharded.shape)
    self.assertEqual(updated_acc.shape, accumulator_sharded.shape)

  @parameterized.parameters(False, True)
  def test_sparse_dense_matmul_grad_with_sgd_on_tpu(self, compile_op):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )

    num_sc_per_device = (
        torch.tpu.get_device_properties().sparse_cores_per_device
    )

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
      return torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(
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
          computation_name="test_sgd_table",
      )

    if compile_op:
      grad_fn = torch.compile(grad_fn, fullgraph=True)

    initial_table_sharded_cpu = embedding_table_sharded.detach().cpu().clone()
    initial_table_sharded_np = initial_table_sharded_cpu.numpy()

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

    updated_rows = self._get_updated_rows(vocab_size)

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
      return torch.tpu.ops.sparse_dense_matmul(
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
    grad_approx = torch.zeros(
        (vocab_size, embedding_dim), dtype=torch.float32, device=device
    )

    for r in updated_rows:
      table_sharded_plus = initial_table_sharded_cpu.to(device).clone()
      table_sharded_plus[r] += eps
      table_sharded_minus = initial_table_sharded_cpu.to(device).clone()
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
    num_sc_per_device = (
        torch.tpu.get_device_properties().sparse_cores_per_device
    )
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
    epsilon = 1e-10

    def grad_fn(rp, e_ids, s_ids, g, et, acc, ag, lr, eps):
      return torch.tpu.ops.sparse_dense_matmul_grad_with_adagrad(
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
          computation_name="test_adagrad_table",
      )

    if compile_op:
      grad_fn = torch.compile(grad_fn, fullgraph=True)

    initial_table_sharded_cpu = embedding_table_sharded.detach().cpu().clone()
    initial_table_sharded_np = initial_table_sharded_cpu.numpy()
    initial_acc_sharded_cpu = accumulator_sharded.detach().cpu().clone()
    initial_acc_sharded_np = initial_acc_sharded_cpu.numpy()

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
    updated_acc_tpu_np = updated_acc_tpu.cpu().numpy()
    updated_rows = self._get_updated_rows(vocab_size)
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
      return torch.tpu.ops.sparse_dense_matmul(
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
    grad_approx = torch.zeros(
        (vocab_size, embedding_dim), dtype=torch.float32, device=device
    )
    for r in updated_rows:
      table_sharded_plus = initial_table_sharded_cpu.to(device).clone()
      table_sharded_plus[r] += eps
      table_sharded_minus = initial_table_sharded_cpu.to(device).clone()
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
        / (np.sqrt(expected_acc_np) + epsilon)
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
    num_sc_per_device = (
        torch.tpu.get_device_properties().sparse_cores_per_device
    )
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
        torch.ones(vocab_size, dtype=torch.float32, device=device) * 0.1
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
    epsilon = 1e-10

    def grad_fn(rp, e_ids, s_ids, g, et, acc, ag, lr, eps):
      return torch.tpu.ops.sparse_dense_matmul_grad_with_adagrad(
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
          computation_name="test_rowwise_adagrad_table",
      )

    if compile_op:
      grad_fn = torch.compile(grad_fn, fullgraph=True)

    initial_table_sharded_cpu = embedding_table_sharded.detach().cpu().clone()
    initial_table_sharded_np = initial_table_sharded_cpu.numpy()
    initial_acc_sharded_cpu = accumulator_sharded.detach().cpu().clone()
    initial_acc_sharded_np = initial_acc_sharded_cpu.numpy()

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
    updated_acc_tpu_np = updated_acc_tpu.cpu().numpy()
    updated_rows = self._get_updated_rows(vocab_size)
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
      return torch.tpu.ops.sparse_dense_matmul(
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
    grad_approx = torch.zeros(
        (vocab_size, embedding_dim), dtype=torch.float32, device=device
    )
    for r in updated_rows:
      table_sharded_plus = initial_table_sharded_cpu.to(device).clone()
      table_sharded_plus[r] += eps
      table_sharded_minus = initial_table_sharded_cpu.to(device).clone()
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
    expected_acc_np = initial_acc_sharded_np + np.mean(
        grad_approx_np * grad_approx_np, axis=1
    )
    expected_update = (
        learning_rate.cpu().numpy()
        * grad_approx_np
        / (np.sqrt(expected_acc_np)[:, np.newaxis] + epsilon)
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
  def test_sparse_dense_matmul_grad_with_adam_on_tpu(self, compile_op):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )

    num_sc_per_device = (
        torch.tpu.get_device_properties().sparse_cores_per_device
    )

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

    # Initialize momentum (m) and velocity (v)
    momentum = (
        torch.ones(
            vocab_size, embedding_dim, dtype=torch.float32, device=device
        )
        * 0.2
    )
    velocity = (
        torch.ones(
            vocab_size, embedding_dim, dtype=torch.float32, device=device
        )
        * 0.1
    )

    # Shard the slot variables
    sharded_momentums = []
    sharded_velocities = []
    for core_id in range(num_sc_per_device):
      indices = [
          i for i in range(vocab_size) if i % num_sc_per_device == core_id
      ]
      sharded_momentums.append(momentum[indices])
      sharded_velocities.append(velocity[indices])

    momentum_sharded = torch.cat(sharded_momentums, dim=0)
    velocity_sharded = torch.cat(sharded_velocities, dim=0)

    batch_size = 16

    activations_grad = (
        torch.ones(
            batch_size, embedding_dim, dtype=torch.float32, device=device
        )
        * 0.01
    )
    alpha_t = torch.tensor(0.01, dtype=torch.float32, device=device)
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    def grad_fn(rp, e_ids, s_ids, g, et, mom, vel, ag, lr, b1, b2, eps):
      return torch.tpu.ops.sparse_dense_matmul_grad_with_adam(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          mom,
          vel,
          ag,
          lr,
          b1,
          b2,
          eps,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=64,
          computation_name="test_adam_table",
      )

    if compile_op:
      grad_fn = torch.compile(grad_fn, fullgraph=True)

    initial_table_sharded_cpu = embedding_table_sharded.detach().cpu().clone()
    initial_table_sharded_np = initial_table_sharded_cpu.numpy()
    initial_mom_sharded_cpu = momentum_sharded.detach().cpu().clone()
    initial_mom_sharded_np = initial_mom_sharded_cpu.numpy()
    initial_vel_sharded_cpu = velocity_sharded.detach().cpu().clone()
    initial_vel_sharded_np = initial_vel_sharded_cpu.numpy()

    updated_table_tpu, updated_mom_tpu, updated_vel_tpu = grad_fn(
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        embedding_table_sharded,
        momentum_sharded,
        velocity_sharded,
        activations_grad,
        alpha_t,
        beta_1,
        beta_2,
        epsilon,
    )

    updated_table_tpu_np = updated_table_tpu.cpu().numpy()
    updated_mom_tpu_np = updated_mom_tpu.cpu().numpy()
    updated_vel_tpu_np = updated_vel_tpu.cpu().numpy()

    updated_rows = self._get_updated_rows(vocab_size)

    # Verify that non-updated rows are indeed unchanged
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
                updated_mom_tpu_np[i], initial_mom_sharded_np[i], atol=1e-5
            ),
            msg=f"Momentum row {i} expected to be unchanged",
        )
        self.assertTrue(
            np.allclose(
                updated_vel_tpu_np[i], initial_vel_sharded_np[i], atol=1e-5
            ),
            msg=f"Velocity row {i} expected to be unchanged",
        )

    # Numerical check using finite differences of forward op
    def matmul_fn(rp, e_ids, s_ids, g, et):
      return torch.tpu.ops.sparse_dense_matmul(
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
    grad_approx = torch.zeros(
        (vocab_size, embedding_dim), dtype=torch.float32, device=device
    )

    for r in updated_rows:
      table_sharded_plus = initial_table_sharded_cpu.to(device).clone()
      table_sharded_plus[r] += eps
      table_sharded_minus = initial_table_sharded_cpu.to(device).clone()
      table_sharded_minus[r] -= eps

      out_plus = matmul_fn(
          row_pointers, embedding_ids, sample_ids, gains, table_sharded_plus
      )
      out_minus = matmul_fn(
          row_pointers, embedding_ids, sample_ids, gains, table_sharded_minus
      )

      dout = (out_plus - out_minus) / (2 * eps)
      grad_approx[r] = torch.sum(activations_grad * dout, dim=0)

    # Calculate expected updates
    grad_approx_np = grad_approx.cpu().numpy()
    expected_mom_np = initial_mom_sharded_np + (1.0 - beta_1) * (
        grad_approx_np - initial_mom_sharded_np
    )
    expected_vel_np = initial_vel_sharded_np + (1.0 - beta_2) * (
        grad_approx_np * grad_approx_np - initial_vel_sharded_np
    )
    expected_update = (
        alpha_t.cpu().numpy()
        * expected_mom_np
        / (np.sqrt(expected_vel_np) + epsilon)
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
          np.allclose(updated_mom_tpu_np[r], expected_mom_np[r], atol=1e-4),
          msg=(
              f"Momentum row {r} mismatch. Actual: {updated_mom_tpu_np[r]},"
              f" Expected: {expected_mom_np[r]}"
          ),
      )
      self.assertTrue(
          np.allclose(updated_vel_tpu_np[r], expected_vel_np[r], atol=1e-4),
          msg=(
              f"Velocity row {r} mismatch. Actual: {updated_vel_tpu_np[r]},"
              f" Expected: {expected_vel_np[r]}"
          ),
      )

  @parameterized.parameters(False, True)
  def test_sparse_dense_matmul_grad_with_rowwise_adam_on_tpu(self, compile_op):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )

    num_sc_per_device = (
        torch.tpu.get_device_properties().sparse_cores_per_device
    )

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

    # Initialize momentum (m) and velocity (v)
    momentum = (
        torch.ones(
            vocab_size, embedding_dim, dtype=torch.float32, device=device
        )
        * 0.2
    )
    velocity = torch.ones(vocab_size, dtype=torch.float32, device=device) * 0.1

    # Shard the slot variables
    sharded_momentums = []
    sharded_velocities = []
    for core_id in range(num_sc_per_device):
      indices = [
          i for i in range(vocab_size) if i % num_sc_per_device == core_id
      ]
      sharded_momentums.append(momentum[indices])
      sharded_velocities.append(velocity[indices])

    momentum_sharded = torch.cat(sharded_momentums, dim=0)
    velocity_sharded = torch.cat(sharded_velocities, dim=0)

    batch_size = 16

    activations_grad = (
        torch.ones(
            batch_size, embedding_dim, dtype=torch.float32, device=device
        )
        * 0.01
    )
    alpha_t = torch.tensor(0.01, dtype=torch.float32, device=device)
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    def grad_fn(rp, e_ids, s_ids, g, et, mom, vel, ag, lr, b1, b2, eps):
      return torch.tpu.ops.sparse_dense_matmul_grad_with_adam(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          mom,
          vel,
          ag,
          lr,
          b1,
          b2,
          eps,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=64,
          computation_name="test_rowwise_adam_table",
      )

    if compile_op:
      grad_fn = torch.compile(grad_fn, fullgraph=True)

    initial_table_sharded_cpu = embedding_table_sharded.detach().cpu().clone()
    initial_table_sharded_np = initial_table_sharded_cpu.numpy()
    initial_mom_sharded_cpu = momentum_sharded.detach().cpu().clone()
    initial_mom_sharded_np = initial_mom_sharded_cpu.numpy()
    initial_vel_sharded_cpu = velocity_sharded.detach().cpu().clone()
    initial_vel_sharded_np = initial_vel_sharded_cpu.numpy()

    updated_table_tpu, updated_mom_tpu, updated_vel_tpu = grad_fn(
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        embedding_table_sharded,
        momentum_sharded,
        velocity_sharded,
        activations_grad,
        alpha_t,
        beta_1,
        beta_2,
        epsilon,
    )

    updated_table_tpu_np = updated_table_tpu.cpu().numpy()
    updated_mom_tpu_np = updated_mom_tpu.cpu().numpy()
    updated_vel_tpu_np = updated_vel_tpu.cpu().numpy()

    updated_rows = self._get_updated_rows(vocab_size)

    # Verify that non-updated rows are indeed unchanged
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
                updated_mom_tpu_np[i], initial_mom_sharded_np[i], atol=1e-5
            ),
            msg=f"Momentum row {i} expected to be unchanged",
        )
        self.assertTrue(
            np.allclose(
                updated_vel_tpu_np[i], initial_vel_sharded_np[i], atol=1e-5
            ),
            msg=f"Velocity row {i} expected to be unchanged",
        )

    # Numerical check using finite differences of forward op
    def matmul_fn(rp, e_ids, s_ids, g, et):
      return torch.tpu.ops.sparse_dense_matmul(
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
    grad_approx = torch.zeros(
        (vocab_size, embedding_dim), dtype=torch.float32, device=device
    )

    for r in updated_rows:
      table_sharded_plus = initial_table_sharded_cpu.to(device).clone()
      table_sharded_plus[r] += eps
      table_sharded_minus = initial_table_sharded_cpu.to(device).clone()
      table_sharded_minus[r] -= eps

      out_plus = matmul_fn(
          row_pointers, embedding_ids, sample_ids, gains, table_sharded_plus
      )
      out_minus = matmul_fn(
          row_pointers, embedding_ids, sample_ids, gains, table_sharded_minus
      )

      dout = (out_plus - out_minus) / (2 * eps)
      grad_approx[r] = torch.sum(activations_grad * dout, dim=0)

    # Calculate expected updates
    grad_approx_np = grad_approx.cpu().numpy()
    expected_mom_np = initial_mom_sharded_np + (1.0 - beta_1) * (
        grad_approx_np - initial_mom_sharded_np
    )
    expected_vel_np = initial_vel_sharded_np + (1.0 - beta_2) * (
        np.mean(grad_approx_np * grad_approx_np, axis=1)
        - initial_vel_sharded_np
    )
    expected_update = (
        alpha_t.cpu().numpy()
        * expected_mom_np
        / (np.sqrt(expected_vel_np)[:, np.newaxis] + epsilon)
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
          np.allclose(updated_mom_tpu_np[r], expected_mom_np[r], atol=1e-4),
          msg=(
              f"Momentum row {r} mismatch. Actual: {updated_mom_tpu_np[r]},"
              f" Expected: {expected_mom_np[r]}"
          ),
      )
      self.assertTrue(
          np.allclose(updated_vel_tpu_np[r], expected_vel_np[r], atol=1e-4),
          msg=(
              f"Velocity row {r} mismatch. Actual: {updated_vel_tpu_np[r]},"
              f" Expected: {expected_vel_np[r]}"
          ),
      )

  @parameterized.parameters(False, True)
  def test_sparse_dense_matmul_grad_preserves_t8_layout_on_tpu(
      self, compile_op
  ):
    device = torch.device("tpu")
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )
    t8_layout = tpu_annotations.TpuLayout(minor_to_major=[1, 0], tiles=[[8]])
    with tpu_annotations.LayoutContext(t8_layout):
      table_t8 = embedding_table.cpu().to(device)
      acc_t8 = torch.zeros_like(embedding_table.cpu()).to(device)
      mom_t8 = torch.zeros_like(embedding_table.cpu()).to(device)
      vel_t8 = torch.zeros_like(embedding_table.cpu()).to(device)

    batch_size = 16
    embedding_dim = embedding_table.shape[1]
    activations_grad = (
        torch.ones(
            batch_size, embedding_dim, dtype=torch.float32, device=device
        )
        * 0.01
    )
    lr = torch.tensor(0.01, dtype=torch.float32, device=device)

    grad_fn = torch.tpu.ops.sparse_dense_matmul_grad_with_sgd
    if compile_op:
      grad_fn = torch.compile(grad_fn, fullgraph=True)

    for step in range(2):
      table_t8 = grad_fn(
          row_pointers,
          embedding_ids,
          sample_ids,
          gains,
          table_t8,
          activations_grad,
          lr,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
          computation_name="t8_sgd_table",
      )
      _ = table_t8.cpu()
      layout_tuple = tpu_torch_compile.get_device_layout_if_materialized(
          table_t8
      )
      self.assertIsNotNone(
          layout_tuple, msg=f"Step {step}: layout should exist"
      )
      self.assertEqual(tpu_annotations.TpuLayout(*layout_tuple), t8_layout)

    upd_table, upd_acc = torch.tpu.ops.sparse_dense_matmul_grad_with_adagrad(
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        table_t8,
        acc_t8,
        activations_grad,
        lr,
        epsilon=1e-8,
        device_batch_size=batch_size,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        computation_name="t8_adagrad_table",
    )
    _ = upd_table.cpu(), upd_acc.cpu()
    for t in (upd_table, upd_acc):
      layout_tuple = tpu_torch_compile.get_device_layout_if_materialized(t)
      self.assertIsNotNone(layout_tuple)
      self.assertEqual(tpu_annotations.TpuLayout(*layout_tuple), t8_layout)

    upd_table, upd_mom, upd_vel = (
        torch.tpu.ops.sparse_dense_matmul_grad_with_adam(
            row_pointers,
            embedding_ids,
            sample_ids,
            gains,
            upd_table,
            mom_t8,
            vel_t8,
            activations_grad,
            lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            device_batch_size=batch_size,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="t8_adam_table",
        )
    )
    _ = upd_table.cpu(), upd_mom.cpu(), upd_vel.cpu()
    for t in (upd_table, upd_mom, upd_vel):
      layout_tuple = tpu_torch_compile.get_device_layout_if_materialized(t)
      self.assertIsNotNone(layout_tuple)
      self.assertEqual(tpu_annotations.TpuLayout(*layout_tuple), t8_layout)

  def _compile_sgd(
      self, device: torch.device, debugs: list[Any]
  ) -> tuple[Any, torch.Tensor]:
    """Returns a compiled SGD step on a table, and the table's initial value."""
    row_pointers, embedding_ids, sample_ids, gains, embedding_table = (
        self._get_inputs(device)
    )
    batch_size = 16
    activations_grad = (
        torch.ones(
            batch_size,
            embedding_table.shape[1],
            dtype=torch.float32,
            device=device,
        )
        * 0.01
    )
    lr = torch.tensor(0.01, dtype=torch.float32, device=device)

    @torch.compile(
        backend="tpu",
        fullgraph=True,
        options={"debug_callback": debugs.append},
    )
    def sgd(table):
      return torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(
          row_pointers,
          embedding_ids,
          sample_ids,
          gains,
          table,
          activations_grad,
          lr,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
          computation_name="sgd_table",
      )

    return sgd, embedding_table.cpu()

  def test_compiled_op_recompiles_when_table_loses_t8_layout(self):
    device = torch.device("tpu")
    t8_layout = tpu_annotations.TpuLayout(minor_to_major=[1, 0], tiles=[[8]])
    debugs = []
    sgd, initial_table = self._compile_sgd(device, debugs)

    def t8_table():
      with tpu_annotations.LayoutContext(t8_layout):
        return initial_table.to(device)

    # The first call compiles the program for the SparseCore layout.
    expected = sgd(t8_table()).cpu()
    self.assertLen(debugs, 1)

    # Like load_state_dict, copy_ gives a table the default layout, so the next
    # call compiles a program for it.
    table = t8_table()
    table.copy_(initial_table)
    self.assert_close(golden_result=expected, torch_tpu_result=sgd(table).cpu())
    self.assertLen(debugs, 2)

    # A table in the SparseCore layout reuses the first program.
    self.assert_close(
        golden_result=expected, torch_tpu_result=sgd(t8_table()).cpu()
    )
    self.assertLen(debugs, 2)

  def test_compiled_op_rejects_t8_table_after_default_layout(self):
    device = torch.device("tpu")
    t8_layout = tpu_annotations.TpuLayout(minor_to_major=[1, 0], tiles=[[8]])
    sgd, initial_table = self._compile_sgd(device, [])
    # The first call compiles the program for the default layout.
    sgd(initial_table.to(device)).cpu()

    # Programs are only compiled again for inputs that had non-default layouts.
    with tpu_annotations.LayoutContext(t8_layout):
      table = initial_table.to(device)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Tests when not to recompile.
        RuntimeError, "compiled for"
    ):
      sgd(table).cpu()

  def test_input_preprocessing_minibatching(self):
    val_f0 = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    off_f0 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    base_options = {
        "local_device_count": 1,
        "global_device_count": 1,
        "num_sc_per_device": 2,
        "allow_id_dropping": True,
    }
    outputs_disabled = preprocess_sparse_dense_matmul_input(
        val_f0,
        off_f0,
        max_ids_per_partition=8,
        max_unique_ids_per_partition=8,
        suggested_coo_buffer_size=16,
        batch_size=4,
        combiner="sum",
        options={**base_options, "enable_minibatching": False},
    )
    outputs_enabled = preprocess_sparse_dense_matmul_input(
        val_f0,
        off_f0,
        max_ids_per_partition=8,
        max_unique_ids_per_partition=8,
        suggested_coo_buffer_size=16,
        batch_size=4,
        combiner="sum",
        options={**base_options, "enable_minibatching": True},
    )
    # Device minibatching allocates 64 buckets per SparseCore partition.
    self.assertGreater(
        outputs_enabled["row_pointers"].numel(),
        outputs_disabled["row_pointers"].numel(),
    )
    self.assertEqual(outputs_enabled["row_pointers"].dtype, torch.int32)
    self.assertEqual(outputs_enabled["embedding_ids"].dtype, torch.int32)
    self.assertEqual(outputs_enabled["sample_ids"].dtype, torch.int32)
    self.assertEqual(outputs_enabled["gains"].dtype, torch.float32)

  @parameterized.parameters(False, True)
  def test_sparse_dense_matmul_with_minibatching_on_tpu(self, compile_op):
    device = torch.device("tpu")
    rp_no_mb, e_ids_no_mb, s_ids_no_mb, gains_no_mb, embedding_table = (
        self._get_inputs(device, enable_minibatching=False)
    )
    rp_mb, e_ids_mb, s_ids_mb, gains_mb, _ = self._get_inputs(
        device, enable_minibatching=True
    )

    def fwd_fn(rp, e_ids, s_ids, g, et):
      return torch.tpu.ops.sparse_dense_matmul(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          device_batch_size=16,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
          enable_minibatching=True,
      )

    if compile_op:
      fwd_fn = torch.compile(fwd_fn, fullgraph=True)

    out_mb = fwd_fn(rp_mb, e_ids_mb, s_ids_mb, gains_mb, embedding_table)
    self.assert_close(
        golden_result=self._get_expected_forward_output(),
        torch_tpu_result=out_mb.cpu(),
    )

    num_sc_per_device = (
        torch.tpu.get_device_properties().sparse_cores_per_device
    )
    vocab_size, embedding_dim = embedding_table.shape
    sharded_tables = [
        embedding_table[core_id::num_sc_per_device]
        for core_id in range(num_sc_per_device)
    ]
    table_init_cpu = torch.cat(sharded_tables, dim=0).detach().cpu()
    batch_size = 16
    activations_grad = (
        torch.ones(
            batch_size, embedding_dim, dtype=torch.float32, device=device
        )
        * 0.01
    )
    lr = torch.tensor(0.01, dtype=torch.float32, device=device)

    # 1. SGD backward parity between enable_minibatching=True and False
    sgd_ref_cpu = torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(
        rp_no_mb,
        e_ids_no_mb,
        s_ids_no_mb,
        gains_no_mb,
        table_init_cpu.to(device),
        activations_grad,
        lr,
        device_batch_size=batch_size,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        computation_name="test_mb_sgd_ref",
        enable_minibatching=False,
    ).cpu()

    def sgd_mb_fn(rp, e_ids, s_ids, g, et, ag, learning_rate):
      return torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          ag,
          learning_rate,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
          computation_name="test_mb_sgd",
          enable_minibatching=True,
      )

    if compile_op:
      sgd_mb_fn = torch.compile(sgd_mb_fn, fullgraph=True)

    sgd_mb_cpu = sgd_mb_fn(
        rp_mb,
        e_ids_mb,
        s_ids_mb,
        gains_mb,
        table_init_cpu.to(device),
        activations_grad,
        lr,
    ).cpu()
    self.assert_close(golden_result=sgd_ref_cpu, torch_tpu_result=sgd_mb_cpu)

    # 2. Adagrad backward parity between enable_minibatching=True and False
    acc_init_cpu = torch.ones_like(table_init_cpu) * 0.1
    adagrad_table_ref, adagrad_acc_ref = (
        torch.tpu.ops.sparse_dense_matmul_grad_with_adagrad(
            rp_no_mb,
            e_ids_no_mb,
            s_ids_no_mb,
            gains_no_mb,
            table_init_cpu.to(device),
            acc_init_cpu.to(device),
            activations_grad,
            lr,
            1e-8,
            device_batch_size=batch_size,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="test_mb_adagrad_ref",
            enable_minibatching=False,
        )
    )
    adagrad_table_ref_cpu = adagrad_table_ref.cpu()
    adagrad_acc_ref_cpu = adagrad_acc_ref.cpu()

    def adagrad_mb_fn(rp, e_ids, s_ids, g, et, acc, ag, learning_rate):
      return torch.tpu.ops.sparse_dense_matmul_grad_with_adagrad(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          acc,
          ag,
          learning_rate,
          1e-8,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
          computation_name="test_mb_adagrad",
          enable_minibatching=True,
      )

    if compile_op:
      adagrad_mb_fn = torch.compile(adagrad_mb_fn, fullgraph=True)

    adagrad_table_mb, adagrad_acc_mb = adagrad_mb_fn(
        rp_mb,
        e_ids_mb,
        s_ids_mb,
        gains_mb,
        table_init_cpu.to(device),
        acc_init_cpu.to(device),
        activations_grad,
        lr,
    )
    adagrad_table_mb_cpu = adagrad_table_mb.cpu()
    adagrad_acc_mb_cpu = adagrad_acc_mb.cpu()
    self.assert_close(
        golden_result=adagrad_table_ref_cpu,
        torch_tpu_result=adagrad_table_mb_cpu,
    )
    self.assert_close(
        golden_result=adagrad_acc_ref_cpu,
        torch_tpu_result=adagrad_acc_mb_cpu,
    )

    # 3. Adam backward parity between enable_minibatching=True and False
    mom_init_cpu = torch.ones_like(table_init_cpu) * 0.2
    vel_init_cpu = torch.ones_like(table_init_cpu) * 0.1
    adam_table_ref, adam_mom_ref, adam_vel_ref = (
        torch.tpu.ops.sparse_dense_matmul_grad_with_adam(
            rp_no_mb,
            e_ids_no_mb,
            s_ids_no_mb,
            gains_no_mb,
            table_init_cpu.to(device),
            mom_init_cpu.to(device),
            vel_init_cpu.to(device),
            activations_grad,
            lr,
            0.9,
            0.999,
            1e-8,
            device_batch_size=batch_size,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="test_mb_adam_ref",
            enable_minibatching=False,
        )
    )
    adam_table_ref_cpu = adam_table_ref.cpu()
    adam_mom_ref_cpu = adam_mom_ref.cpu()
    adam_vel_ref_cpu = adam_vel_ref.cpu()

    def adam_mb_fn(rp, e_ids, s_ids, g, et, mom, vel, ag, learning_rate):
      return torch.tpu.ops.sparse_dense_matmul_grad_with_adam(
          rp,
          e_ids,
          s_ids,
          g,
          et,
          mom,
          vel,
          ag,
          learning_rate,
          0.9,
          0.999,
          1e-8,
          device_batch_size=batch_size,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
          computation_name="test_mb_adam",
          enable_minibatching=True,
      )

    if compile_op:
      adam_mb_fn = torch.compile(adam_mb_fn, fullgraph=True)

    adam_table_mb, adam_mom_mb, adam_vel_mb = adam_mb_fn(
        rp_mb,
        e_ids_mb,
        s_ids_mb,
        gains_mb,
        table_init_cpu.to(device),
        mom_init_cpu.to(device),
        vel_init_cpu.to(device),
        activations_grad,
        lr,
    )
    adam_table_mb_cpu = adam_table_mb.cpu()
    adam_mom_mb_cpu = adam_mom_mb.cpu()
    adam_vel_mb_cpu = adam_vel_mb.cpu()
    self.assert_close(
        golden_result=adam_table_ref_cpu,
        torch_tpu_result=adam_table_mb_cpu,
    )
    self.assert_close(
        golden_result=adam_mom_ref_cpu,
        torch_tpu_result=adam_mom_mb_cpu,
    )
    self.assert_close(
        golden_result=adam_vel_ref_cpu,
        torch_tpu_result=adam_vel_mb_cpu,
    )


if __name__ == "__main__":
  absltest.main()

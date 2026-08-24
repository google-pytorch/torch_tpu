/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include "ATen/core/Dict.h"
#include "ATen/core/List.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/full.h"
#include "ATen/ops/zeros.h"
#include "Eigen/Core"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "torch/library.h"
#include "torch_tpu/csrc/common/error_utils.h"

// JAX input preprocessing library
// Note: Despite the jax_* prefix in paths and namespaces, the core C++
// preprocessing library is independent of JAX and framework-agnostic.
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/ragged_tensor_input_batch.h"

namespace torch_tpu {
namespace {

namespace sc_preprocessing = jax_sc_embedding;

}  // namespace

// TT_KERNEL is not required because this is a wrapper on the input
// preprocessing from jax_tpu_embedding (which is framework-independent).
TORCH_LIBRARY_FRAGMENT(tpu, m) {
  // Preprocesses sparse dense matmul inputs for TPU embedding execution per
  // table and single feature. Returns a Dict<std::string, Tensor> containing
  // CSR buffers and stats:
  //   * "row_pointers": 1D int32 Tensor of size (local_device_count *
  //     row_pointers_size_per_device). Row offsets per SparseCore partition.
  //   * "embedding_ids": 1D int32 Tensor of size (local_device_count *
  //     buffer_size). Flattened embedding IDs for lookup.
  //   * "sample_ids": 1D int32 Tensor of size (local_device_count *
  //     buffer_size). Sample/row index for each embedding ID.
  //   * "gains": 1D float32 Tensor of size (local_device_count *
  //     buffer_size). Scalar gain/weight for each embedding ID.
  //   * "max_ids_per_partition": 1D int32 Tensor (num_sc)
  //     Max count of embedding IDs routed to each SparseCore partition.
  //   * "max_unique_ids_per_partition": 1D int32 Tensor (num_sc)
  //     Max count of unique embedding IDs assigned to each SparseCore
  //     partition.
  //   * "required_buffer_sizes": 1D int32 Tensor (num_sc)
  //     Minimum buffer size required for each SparseCore partition to avoid
  //     overflow.
  //   * "dropped_id_count": 1D int32 Tensor (size 1)
  //     Total count of dropped IDs for the table when buffer capacity is
  //     exceeded.
  auto func =
      [](const at::Tensor& input_indices, const at::Tensor& input_offsets,
         int64_t max_ids_per_partition, int64_t max_unique_ids_per_partition,
         int64_t suggested_coo_buffer_size, int64_t batch_size,
         std::string_view combiner, int64_t local_device_count,
         int64_t global_device_count, int64_t num_sc_per_device,
         bool allow_id_dropping,
         std::string_view table_name) -> c10::Dict<std::string, at::Tensor> {
    TT_CHECK_THROW(input_indices.dtype() == at::kInt, error::kInvalidArgument)
        << "indices must be int32";
    TT_CHECK_THROW(input_offsets.dtype() == at::kInt, error::kInvalidArgument)
        << "offsets must be int32";

    std::string table_name_str(table_name);
    auto contiguous_indices = input_indices.contiguous();
    auto contiguous_offsets = input_offsets.contiguous();

    absl::Span<const int32_t> val_span(contiguous_indices.data_ptr<int32_t>(),
                                       contiguous_indices.numel());
    absl::Span<const int32_t> off_span(contiguous_offsets.data_ptr<int32_t>(),
                                       contiguous_offsets.numel());

    std::vector<std::unique_ptr<sc_preprocessing::AbstractInputBatch>>
        input_batches;
    input_batches.push_back(
        std::make_unique<sc_preprocessing::RaggedTensorInputBatch<
            absl::Span<const int32_t>, absl::Span<const int32_t>>>(
            val_span, off_span, table_name_str));

    sc_preprocessing::OutputCsrArrays sc_output_buffers;

    sc_preprocessing::PreprocessSparseDenseMatmulInputOptions sc_options{
        .local_device_count = static_cast<int>(local_device_count),
        .global_device_count = static_cast<int>(global_device_count),
        .num_sc_per_device = static_cast<int>(num_sc_per_device),
        .sharding_strategy = sc_preprocessing::ShardingStrategy::kMod,
        .allow_id_dropping = allow_id_dropping,
        .enable_minibatching = false,
    };
    const int row_pointers_size_per_device =
        sc_options.GetRowPointersSizePerDevice();

    int32_t required_buffer_size =
        static_cast<int32_t>(suggested_coo_buffer_size);
    if (required_buffer_size <= 0) {
      required_buffer_size = 128;  // Default fallback
    }

    int total_row_pointers_size =
        local_device_count * row_pointers_size_per_device;
    int total_coo_buffer_size = local_device_count * required_buffer_size;

    at::Tensor row_pointers = at::zeros({total_row_pointers_size}, at::kInt);
    at::Tensor embedding_ids = at::empty({total_coo_buffer_size}, at::kInt);
    at::Tensor sample_ids = at::empty({total_coo_buffer_size}, at::kInt);
    at::Tensor gains = at::empty({total_coo_buffer_size}, at::kFloat);

    // Wrap PyTorch data tensors DIRECTLY
    Eigen::Map<sc_preprocessing::MatrixXi> row_pointers_map(
        row_pointers.data_ptr<int32_t>(), local_device_count,
        row_pointers_size_per_device);
    Eigen::Map<sc_preprocessing::MatrixXi> embedding_ids_map(
        embedding_ids.data_ptr<int32_t>(), local_device_count,
        required_buffer_size);
    Eigen::Map<sc_preprocessing::MatrixXi> sample_ids_map(
        sample_ids.data_ptr<int32_t>(), local_device_count,
        required_buffer_size);
    Eigen::Map<sc_preprocessing::MatrixXf> gains_map(
        gains.data_ptr<float>(), local_device_count, required_buffer_size);

    sc_output_buffers.lhs_row_pointers.emplace(table_name_str,
                                               row_pointers_map);
    sc_output_buffers.lhs_embedding_ids.emplace(table_name_str,
                                                embedding_ids_map);
    sc_output_buffers.lhs_sample_ids.emplace(table_name_str, sample_ids_map);
    sc_output_buffers.lhs_gains.emplace(table_name_str, gains_map);

    TT_ASSIGN_OR_THROW(
        const auto preprocess_result,
        sc_preprocessing::PreprocessSparseDenseMatmulInput(
            absl::MakeSpan(input_batches),
            {{table_name_str,
              {sc_preprocessing::FeatureMetadataInStack(
                  absl::StrCat(table_name_str, "_feature_0"),
                  /*feature_index=*/0, static_cast<int>(max_ids_per_partition),
                  static_cast<int>(max_unique_ids_per_partition),
                  /*row_offset=*/0,
                  /*col_offset=*/0,
                  /*col_shift=*/0, static_cast<int>(batch_size),
                  suggested_coo_buffer_size > 0
                      ? std::make_optional(
                            static_cast<int>(suggested_coo_buffer_size))
                      : std::nullopt,
                  sc_preprocessing::GetRowCombiner(combiner))}}},
            sc_options, &sc_output_buffers));

    c10::Dict<std::string, at::Tensor> outputs;
    outputs.insert("row_pointers", row_pointers);
    outputs.insert("embedding_ids", embedding_ids);
    outputs.insert("sample_ids", sample_ids);
    outputs.insert("gains", gains);

    const auto& stats = preprocess_result.stats;
    auto insert_vector_stat = [&](std::string_view key, const auto& map) {
      if (auto it = map.find(table_name_str); it != map.end()) {
        const auto& vec = it->second;
        at::Tensor tensor =
            at::empty({static_cast<int64_t>(vec.size())}, at::kInt);
        std::memcpy(tensor.data_ptr<int32_t>(), vec.data(),
                    vec.size() * sizeof(int32_t));
        outputs.insert(key, tensor);
      }
    };

    insert_vector_stat("max_ids_per_partition", stats.max_ids_per_partition);
    insert_vector_stat("max_unique_ids_per_partition",
                       stats.max_unique_ids_per_partition);
    insert_vector_stat("required_buffer_sizes", stats.required_buffer_sizes);

    if (auto it = stats.dropped_id_count.find(table_name_str);
        it != stats.dropped_id_count.end()) {
      outputs.insert("dropped_id_count", at::full({1}, it->second, at::kInt));
    }

    return outputs;
  };

  // Registers preprocess_sparse_dense_matmul_input op.
  // Args:
  //   input_indices: Tensor - Feature indices tensor for the table.
  //   input_offsets: Tensor - Feature offsets tensor for the table.
  //   max_ids_per_partition: int - Maximum IDs per SparseCore partition.
  //   max_unique_ids_per_partition: int - Maximum unique IDs per SparseCore
  //   partition. suggested_coo_buffer_size: int - Suggested COO buffer size per
  //   device. batch_size: int - Batch size per device. combiner: str - Row
  //   combiner ("sum", "mean", etc.). local_device_count: int - Number of local
  //   TPU devices. global_device_count: int - Total TPU device count across
  //   hosts. num_sc_per_device: int - Number of SparseCore units per TPU
  //   device. allow_id_dropping: bool - Whether to allow dropping excess IDs.
  //   table_name: str - Table name for logging and identification.
  // Returns:
  //   Dict[str, Tensor] containing CSR buffers ("row_pointers",
  //   "embedding_ids", "sample_ids", "gains") and stats
  //   ("max_ids_per_partition", "max_unique_ids_per_partition",
  //   "required_buffer_sizes", "dropped_id_count").
  m.def(
      "preprocess_sparse_dense_matmul_input("
      "    Tensor input_indices, "
      "    Tensor input_offsets, "
      "    int max_ids_per_partition, "
      "    int max_unique_ids_per_partition, "
      "    int suggested_coo_buffer_size, "
      "    int batch_size, "
      "    str combiner, "
      "    int local_device_count, "
      "    int global_device_count, "
      "    int num_sc_per_device, "
      "    bool allow_id_dropping, "
      "    str table_name"
      ") -> Dict(str, Tensor)",
      func);
}

}  // namespace torch_tpu

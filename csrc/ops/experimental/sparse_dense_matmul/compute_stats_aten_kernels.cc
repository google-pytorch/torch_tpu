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

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/container/flat_hash_set.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/error_utils.h"
#include "torch/library.h"

namespace torch_tpu {
namespace {

using IntVector = std::vector<int64_t>;  // INT_VEC_OK

template <typename IndexT>
void AccumulateSampleStats(
    const IndexT* indices_ptr, int64_t start, int64_t end,
    int64_t num_partitions, absl::flat_hash_set<IndexT>& sample_unique_ids,
    IntVector& sc_ids_for_core,
    std::vector<absl::flat_hash_set<IndexT>>& sc_unique_for_core) {
  sample_unique_ids.clear();
  for (int64_t j = start; j < end; ++j) {
    sample_unique_ids.insert(indices_ptr[j]);
  }
  for (IndexT id : sample_unique_ids) {
    int64_t partition = static_cast<int64_t>(id) % num_partitions;
    if (partition < 0) {
      partition += num_partitions;
    }
    sc_ids_for_core[partition]++;
    sc_unique_for_core[partition].insert(id);
  }
}

template <typename IndexT>
std::tuple<int64_t, int64_t> SummarizePartitionStats(
    const std::vector<IntVector>& sc_ids_count,
    const std::vector<std::vector<absl::flat_hash_set<IndexT>>>& sc_unique_ids,
    int64_t num_sc_per_device, int64_t num_partitions) {
  int64_t max_ids_per_partition = 0;
  int64_t max_unique_ids_per_partition = 0;
  for (int64_t p = 0; p < num_partitions; ++p) {
    for (int64_t sc = 0; sc < num_sc_per_device; ++sc) {
      max_ids_per_partition =
          std::max(max_ids_per_partition, sc_ids_count[sc][p]);
      max_unique_ids_per_partition =
          std::max(max_unique_ids_per_partition,
                   static_cast<int64_t>(sc_unique_ids[sc][p].size()));
    }
  }
  return std::make_tuple(max_ids_per_partition, max_unique_ids_per_partition);
}

template <typename IndexT, typename OffsetT>
std::tuple<int64_t, int64_t> ComputeStatsImpl(const at::Tensor& indices,
                                              const at::Tensor& offsets,
                                              int64_t global_device_count,
                                              int64_t num_sc_per_device) {
  const IndexT* indices_ptr = indices.data_ptr<IndexT>();
  const OffsetT* offsets_ptr = offsets.data_ptr<OffsetT>();
  const int64_t num_indices = indices.numel();
  const int64_t num_samples = offsets.numel() - 1;
  const int64_t num_partitions = global_device_count * num_sc_per_device;
  const int64_t batch_size_per_sc =
      (num_samples + num_sc_per_device - 1) / num_sc_per_device;

  // Per-SC partition counts: [num_sc_per_device, num_partitions]
  std::vector<IntVector> sc_ids_count(num_sc_per_device,
                                      IntVector(num_partitions, 0));
  std::vector<std::vector<absl::flat_hash_set<IndexT>>> sc_unique_ids(
      num_sc_per_device,
      std::vector<absl::flat_hash_set<IndexT>>(num_partitions));

  absl::flat_hash_set<IndexT> sample_unique_ids;
  for (int64_t sc = 0; sc < num_sc_per_device; ++sc) {
    const int64_t sample_start = sc * batch_size_per_sc;
    const int64_t sample_end =
        std::min((sc + 1) * batch_size_per_sc, num_samples);

    for (int64_t i = sample_start; i < sample_end; ++i) {
      const int64_t start = static_cast<int64_t>(offsets_ptr[i]);
      const int64_t end = static_cast<int64_t>(offsets_ptr[i + 1]);
      TT_CHECK_THROW(start >= 0 && end >= start && end <= num_indices,
                     error::kInvalidArgument)
          << "Invalid offsets range [" << start << ", " << end << ") for "
          << "indices size " << num_indices;

      AccumulateSampleStats(indices_ptr, start, end, num_partitions,
                            sample_unique_ids, sc_ids_count[sc],
                            sc_unique_ids[sc]);
    }
  }

  return SummarizePartitionStats(sc_ids_count, sc_unique_ids, num_sc_per_device,
                                 num_partitions);
}

}  // namespace

// TT_KERNEL is not required because this is a host CPU statistics calculation
// op for sparse dense matmul that does not dispatch XLA/StableHLO TPU kernels.
TORCH_LIBRARY_FRAGMENT(tpu, m) {
  auto compute_stats_func =
      [](const at::Tensor& input_indices, const at::Tensor& input_offsets,
         int64_t global_device_count,
         int64_t num_sc_per_device) -> std::tuple<int64_t, int64_t> {
    TT_CHECK_THROW(
        input_indices.dtype() == at::kInt || input_indices.dtype() == at::kLong,
        error::kInvalidArgument)
        << "indices must be int32 or int64";
    TT_CHECK_THROW(
        input_offsets.dtype() == at::kInt || input_offsets.dtype() == at::kLong,
        error::kInvalidArgument)
        << "offsets must be int32 or int64";
    TT_CHECK_THROW(global_device_count > 0, error::kInvalidArgument)
        << "global_device_count must be positive";
    TT_CHECK_THROW(num_sc_per_device > 0, error::kInvalidArgument)
        << "num_sc_per_device must be positive";
    TT_CHECK_THROW(input_offsets.numel() > 0, error::kInvalidArgument)
        << "offsets cannot be empty";

    auto indices = input_indices.contiguous();
    auto offsets = input_offsets.contiguous();
    if (indices.dtype() == at::kLong) {
      if (offsets.dtype() == at::kLong) {
        return ComputeStatsImpl<int64_t, int64_t>(
            indices, offsets, global_device_count, num_sc_per_device);
      }
      return ComputeStatsImpl<int64_t, int32_t>(
          indices, offsets, global_device_count, num_sc_per_device);
    }
    if (offsets.dtype() == at::kLong) {
      return ComputeStatsImpl<int32_t, int64_t>(
          indices, offsets, global_device_count, num_sc_per_device);
    }
    return ComputeStatsImpl<int32_t, int32_t>(
        indices, offsets, global_device_count, num_sc_per_device);
  };

  m.def(
      "compute_sparse_dense_matmul_stats("
      "    Tensor input_indices, "
      "    Tensor input_offsets, "
      "    int global_device_count, "
      "    int num_sc_per_device"
      ") -> (int, int)",
      compute_stats_func);
}

}  // namespace torch_tpu

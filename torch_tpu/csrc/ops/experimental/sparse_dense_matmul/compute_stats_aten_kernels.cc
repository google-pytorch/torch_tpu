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
#include "torch/library.h"
#include "torch_tpu/csrc/common/error_utils.h"

namespace torch_tpu {

// TT_KERNEL is not required because this is a host CPU statistics calculation
// op for sparse dense matmul that does not dispatch XLA/StableHLO TPU kernels.
TORCH_LIBRARY_FRAGMENT(tpu, m) {
  auto compute_stats_func =
      [](const at::Tensor& input_indices, const at::Tensor& input_offsets,
         int64_t global_device_count,
         int64_t num_sc_per_device) -> std::tuple<int64_t, int64_t> {
    TT_CHECK_THROW(input_indices.dtype() == at::kInt, error::kInvalidArgument)
        << "indices must be int32";
    TT_CHECK_THROW(input_offsets.dtype() == at::kInt, error::kInvalidArgument)
        << "offsets must be int32";
    TT_CHECK_THROW(global_device_count > 0, error::kInvalidArgument)
        << "global_device_count must be positive";
    TT_CHECK_THROW(num_sc_per_device > 0, error::kInvalidArgument)
        << "num_sc_per_device must be positive";
    TT_CHECK_THROW(input_offsets.numel() > 0, error::kInvalidArgument)
        << "offsets cannot be empty";

    auto indices = input_indices.contiguous();
    auto offsets = input_offsets.contiguous();
    const int32_t* indices_ptr = indices.data_ptr<int32_t>();
    const int32_t* offsets_ptr = offsets.data_ptr<int32_t>();

    const int64_t num_samples = offsets.numel() - 1;
    const int32_t num_partitions = global_device_count * num_sc_per_device;

    const int64_t batch_size_per_sc =
        (num_samples + num_sc_per_device - 1) / num_sc_per_device;

    // Per-SC partition counts: [num_sc_per_device, num_partitions]
    std::vector<std::vector<int64_t>> sc_ids_count(  // INT_VEC_OK
        num_sc_per_device,
        std::vector<int64_t>(num_partitions, 0));  // INT_VEC_OK
    std::vector<std::vector<absl::flat_hash_set<int32_t>>> sc_unique_ids(
        num_sc_per_device,
        std::vector<absl::flat_hash_set<int32_t>>(num_partitions));

    absl::flat_hash_set<int32_t> sample_unique_ids;
    for (int64_t sc = 0; sc < num_sc_per_device; ++sc) {
      int64_t sample_start = sc * batch_size_per_sc;
      int64_t sample_end = std::min((sc + 1) * batch_size_per_sc, num_samples);

      for (int64_t i = sample_start; i < sample_end; ++i) {
        int32_t start = offsets_ptr[i];
        int32_t end = offsets_ptr[i + 1];
        TT_CHECK_THROW(start >= 0 && end >= start && end <= indices.numel(),
                       error::kInvalidArgument)
            << "Invalid offsets range [" << start << ", " << end << ") for "
            << "indices size " << indices.numel();

        sample_unique_ids.clear();
        for (int32_t j = start; j < end; ++j) {
          sample_unique_ids.insert(indices_ptr[j]);
        }

        for (int32_t id : sample_unique_ids) {
          int32_t partition = id % num_partitions;
          if (partition < 0) {
            partition += num_partitions;
          }
          sc_ids_count[sc][partition]++;
          sc_unique_ids[sc][partition].insert(id);
        }
      }
    }

    int64_t max_ids_per_partition = 0;
    int64_t max_unique_ids_per_partition = 0;

    for (int32_t p = 0; p < num_partitions; ++p) {
      for (int64_t sc = 0; sc < num_sc_per_device; ++sc) {
        max_ids_per_partition =
            std::max(max_ids_per_partition, sc_ids_count[sc][p]);
        max_unique_ids_per_partition =
            std::max(max_unique_ids_per_partition,
                     static_cast<int64_t>(sc_unique_ids[sc][p].size()));
      }
    }

    return std::make_tuple(max_ids_per_partition, max_unique_ids_per_partition);
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

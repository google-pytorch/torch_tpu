// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TORCH_TPU_RUNTIME_CEREBRO_CEREBRO_CLUSTER_PIPELINE_H_
#define TORCH_TPU_RUNTIME_CEREBRO_CEREBRO_CLUSTER_PIPELINE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/runtime/cerebro/cerebro_backend_impl.h"
#include "torch_tpu/runtime/cerebro/cerebro_device_buffer.h"
#include "torch_tpu/runtime/device_buffer.h"

namespace torch_tpu {

// Represents collective communication operations supported by Cerebro.
enum class CollectiveOp {
  kAllReduce,
  kAllGather,
  kBroadcast,
  kReduceScatter,
};

// Orchestrates distributed execution and multi-device coordination on Cerebro.
class CerebroClusterPipeline {
 public:
  explicit CerebroClusterPipeline(CerebroBackendImpl* backend);

  // Executes a simulated collective operation across a group of Cerebro device
  // buffers.
  absl::StatusOr<std::vector<std::unique_ptr<DeviceBuffer>>> ExecuteCollective(
      CollectiveOp op, absl::Span<DeviceBuffer* const> input_buffers);

  // Issues a cluster-wide barrier across all participating devices.
  absl::Status Barrier();

  // Resolves the 3D grid coordinate (node, chip, core) of a global device ID.
  absl::StatusOr<std::vector<int64_t> /* INT_VEC_OK */> GetDeviceCoordinate(
      int device_id);

  // Returns participating devices count in the active cluster.
  int GetClusterDeviceCount() const;

 private:
  CerebroBackendImpl* backend_;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_RUNTIME_CEREBRO_CEREBRO_CLUSTER_PIPELINE_H_

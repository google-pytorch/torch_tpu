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

#include "torch_tpu/runtime/cerebro/cerebro_cluster_pipeline.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/runtime/cerebro/cerebro_backend_impl.h"
#include "torch_tpu/runtime/cerebro/cerebro_device_buffer.h"
#include "torch_tpu/runtime/device_buffer.h"

namespace torch_tpu {

CerebroClusterPipeline::CerebroClusterPipeline(CerebroBackendImpl* backend)
    : backend_(backend) {}

absl::StatusOr<std::vector<std::unique_ptr<DeviceBuffer>>>
CerebroClusterPipeline::ExecuteCollective(
    CollectiveOp op, absl::Span<DeviceBuffer* const> input_buffers) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend pointer check
      backend_ != nullptr, error::kInternal)
      << "cerebro backend is null";
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Input buffers empty check
      !input_buffers.empty(), error::kInvalidArgument)
      << "no input buffers provided for collective operation";

  std::vector<std::unique_ptr<DeviceBuffer>> output_buffers;
  output_buffers.reserve(input_buffers.size());

  for (DeviceBuffer* in_buf : input_buffers) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Buffer null check
        in_buf != nullptr, error::kInvalidArgument)
        << "null buffer passed to collective operation";

    Dimensions shape(in_buf->shape().begin(), in_buf->shape().end());
    uint64_t handle = 3000 + output_buffers.size();
    output_buffers.push_back(std::make_unique<CerebroDeviceBuffer>(
        in_buf->dtype(), std::move(shape), in_buf->device(), handle,
        in_buf->GetOnDeviceSizeInBytes()));
  }

  return output_buffers;
}

absl::Status CerebroClusterPipeline::Barrier() {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend pointer check
      backend_ != nullptr, error::kInternal)
      << "cerebro backend is null";
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int64_t>>  // INT_VEC_OK
CerebroClusterPipeline::GetDeviceCoordinate(int device_id) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Backend pointer check
      backend_ != nullptr, error::kInternal)
      << "cerebro backend is null";

  auto devices = backend_->GetDevices();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Device bounds check
      device_id >= 0 && device_id < static_cast<int>(devices.size()),
      error::kInvalidArgument)
      << "invalid device_id: " << device_id;

  auto* dev = dynamic_cast<CerebroDevice*>(devices[device_id]);
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Device cast check
      dev != nullptr, error::kInternal)
      << "device is not a CerebroDevice";

  auto coords_opt = dev->coords();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Device coords check
      coords_opt.has_value(), error::kNotFound)
      << "device has no coordinate metadata";

  return *coords_opt;
}

int CerebroClusterPipeline::GetClusterDeviceCount() const {
  return (backend_ != nullptr) ? backend_->GetDevices().size() : 0;
}

}  // namespace torch_tpu

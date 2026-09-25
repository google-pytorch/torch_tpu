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

#include "csrc/ops/experimental/collectives/collectives_aten_kernels.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "ATen/ops/empty.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/distributed/allgather.h"
#include "csrc/distributed/process_group_tpu.h"
#include "csrc/distributed/reduce_scatter.h"
#include "csrc/distributed/types.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/GroupRegistry.hpp"
#include "torch/csrc/distributed/c10d/ProcessGroup.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {
namespace {

struct CollectiveShape {
  int64_t dim;
  Dimensions output_shape;
};

absl::StatusOr<CollectiveShape> ValidateArgs(const at::Tensor& input,
                                             int64_t dim, int64_t group_size) {
  const int64_t rank = input.dim();
  TT_RET_CHECK(rank > 0, error::kInvalidArgument)
      << "collectives with an explicit dimension require a non-scalar tensor";
  TT_RET_CHECK(dim >= -rank && dim < rank, error::kInvalidArgument)
      << "dimension " << dim << " is invalid for a tensor with rank " << rank;
  TT_RET_CHECK(group_size > 0, error::kInvalidArgument)
      << "group_size must be positive, got " << group_size;
  return CollectiveShape{.dim = dim < 0 ? dim + rank : dim,
                         .output_shape = CopyIntVector(input.sizes())};
}

absl::StatusOr<CollectiveShape> AllGatherShape(const at::Tensor& input,
                                               int64_t dim,
                                               int64_t group_size) {
  TT_ASSIGN_OR_RETURN(CollectiveShape shape,
                      ValidateArgs(input, dim, group_size));
  shape.output_shape[shape.dim] *= group_size;
  return shape;
}

absl::StatusOr<CollectiveShape> ReduceScatterShape(const at::Tensor& input,
                                                   int64_t dim,
                                                   int64_t group_size) {
  TT_ASSIGN_OR_RETURN(CollectiveShape shape,
                      ValidateArgs(input, dim, group_size));
  int64_t& size = shape.output_shape[shape.dim];
  TT_RET_CHECK(size % group_size == 0, error::kInvalidArgument)
      << "scatter dimension " << shape.dim << " of size " << size
      << " is not divisible by group size " << group_size;
  size /= group_size;
  return shape;
}

absl::StatusOr<c10::intrusive_ptr<ProcessGroupTpu>> GetProcessGroupTpu(
    std::string_view group_name) {
  c10::intrusive_ptr<c10d::ProcessGroup> pg =
      c10d::resolve_process_group(std::string(group_name));
  TT_RET_CHECK(pg != nullptr, error::kInternal)
      << "failed to resolve given process group";
  c10::intrusive_ptr<c10d::Backend> backend =
      pg->getBackend(c10::DeviceType::PrivateUse1);
  TT_RET_CHECK(backend != nullptr, error::kInternal)
      << "failed to get backend for tpu device";
  auto process_group_tpu =
      c10::dynamic_intrusive_pointer_cast<ProcessGroupTpu>(std::move(backend));
  TT_RET_CHECK(process_group_tpu != nullptr, error::kInternal)
      << "failed to cast c10d::Backend to ProcessGroupTpu";
  return process_group_tpu;
}

absl::Status ValidateGroupSize(const DeviceGroupList& device_groups,
                               int64_t group_size) {
  TT_RET_CHECK(!device_groups.empty(), error::kInternal)
      << "process group has no device groups";
  for (const DeviceGroup& device_group : device_groups) {
    TT_RET_CHECK(static_cast<int64_t>(device_group.size()) == group_size,
                 error::kInvalidArgument)
        << "group_size " << group_size << " does not match process group size "
        << device_group.size();
  }
  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildAllGather(
    mlir::MlirOp input, int64_t dim, const DeviceGroupList& device_groups) {
  TT_ASSIGN_OR_RETURN(
      auto outputs,
      BuildDistributedAllGatherShlo(input, device_groups,
                                    AllGatherOutputMode::kConcat, dim));
  return outputs[0];
}

absl::StatusOr<mlir::MlirOp> BuildReduceScatter(
    mlir::MlirOp input, int64_t dim, const DeviceGroupList& device_groups) {
  return BuildDistributedReduceScatterShlo(input, c10d::ReduceOp::SUM,
                                           device_groups, dim);
}

using CollectiveBuilder = absl::StatusOr<mlir::MlirOp> (*)(
    mlir::MlirOp, int64_t, const DeviceGroupList&);

absl::StatusOr<DeviceBufferRef> DispatchCollectiveWithDim(
    const at::Tensor& input, int64_t group_size, CollectiveShape shape,
    std::string_view group_name, CollectiveBuilder collective_builder,
    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(auto process_group, GetProcessGroupTpu(group_name));
  const DeviceGroupList& device_groups = process_group->GetSubgroupDeviceIds();
  TT_RETURN_IF_ERROR(ValidateGroupSize(device_groups, group_size));
  TT_ASSIGN_OR_RETURN(auto output_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));

  auto op_builder = [collective_builder, dim = shape.dim,
                     device_groups](mlir::MlirOp input_op) {
    return collective_builder(input_op, dim, device_groups);
  };

  return DispatchOp<1>(
      std::move(op_builder), input,
      {.out_dtype = output_dtype,
       .out_dims = std::move(shape.output_shape),
       .op_param_cache_keys = std::move(param_keys),
       .split_mode = GetCollectiveSplitMode(),
       .device_assignment = process_group->device_assignment()});
}

}  // namespace

at::Tensor ExperimentalAllGatherDim(const at::Tensor& input, int64_t dim,
                                    int64_t group_size,
                                    std::string_view group_name) {
  TT_KERNEL(OpName::kExperimentalAllGatherDim, param_keys,
            (input, dim, group_size, group_name), {
              TT_ASSIGN_OR_THROW(auto shape,
                                 AllGatherShape(input, dim, group_size));
              TT_ASSIGN_OR_THROW(
                  auto result,
                  DispatchCollectiveWithDim(input, group_size, std::move(shape),
                                            group_name, BuildAllGather,
                                            std::move(param_keys)));
              return MakeTensor(std::move(result));
            });
}

at::Tensor ExperimentalReduceScatterDim(const at::Tensor& input, int64_t dim,
                                        int64_t group_size,
                                        std::string_view group_name) {
  TT_KERNEL(OpName::kExperimentalReduceScatterDim, param_keys,
            (input, dim, group_size, group_name), {
              TT_ASSIGN_OR_THROW(auto shape,
                                 ReduceScatterShape(input, dim, group_size));
              TT_ASSIGN_OR_THROW(
                  auto result,
                  DispatchCollectiveWithDim(input, group_size, std::move(shape),
                                            group_name, BuildReduceScatter,
                                            std::move(param_keys)));
              return MakeTensor(std::move(result));
            });
}

at::Tensor MetaExperimentalAllGatherDim(const at::Tensor& input, int64_t dim,
                                        int64_t group_size,
                                        std::string_view /*group_name*/) {
  TT_ASSIGN_OR_THROW(auto shape, AllGatherShape(input, dim, group_size));
  return at::empty(shape.output_shape, input.options());
}

at::Tensor MetaExperimentalReduceScatterDim(const at::Tensor& input,
                                            int64_t dim, int64_t group_size,
                                            std::string_view /*group_name*/) {
  TT_ASSIGN_OR_THROW(auto shape, ReduceScatterShape(input, dim, group_size));
  return at::empty(shape.output_shape, input.options());
}

}  // namespace torch_tpu

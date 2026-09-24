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

#include "csrc/ops/experimental/dimensional_collectives/dimensional_collectives_aten_kernels.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "ATen/ops/empty.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/util/intrusive_ptr.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/distributed/process_group_tpu.h"
#include "csrc/distributed/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/distributed/c10d/Backend.hpp"
#include "torch/csrc/distributed/c10d/GroupRegistry.hpp"
#include "torch/csrc/distributed/c10d/ProcessGroup.hpp"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/DeviceType.h"

namespace torch_tpu {
namespace {

absl::StatusOr<int64_t> NormalizeDim(int64_t dim, int64_t rank) {
  TT_RET_CHECK(rank > 0, error::kInvalidArgument)
      << "dimensional collectives require a non-scalar tensor";
  TT_RET_CHECK(dim >= -rank && dim < rank, error::kInvalidArgument)
      << "dimension " << dim << " is invalid for a tensor with rank " << rank;
  return dim < 0 ? dim + rank : dim;
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
  TT_RET_CHECK(group_size > 0, error::kInvalidArgument)
      << "group_size must be positive, got " << group_size;
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

absl::StatusOr<mlir::MlirOp> BuildAllGatherDimShlo(
    mlir::MlirOp input, int64_t dim, const DeviceGroupList& device_groups) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  Dimensions output_shape = CopyIntVector(input_type.getShape());
  output_shape[dim] *= device_groups[0].size();
  auto output_type =
      mlir::RankedTensorType::get(output_shape, input_type.getElementType());

  auto& builder = input.getBuilder();
  auto outputs = mlir::stablehlo::AllGather(
      builder, output_type, input, dim,
      BuildReplicaGroupsAttr(builder, device_groups),
      BuildChannelHandleAttr(builder, /*handle=*/1, /*type=*/0),
      /*use_global_device_ids=*/true);
  TT_RET_CHECK(outputs.size() == 1, error::kInternal)
      << "all-gather produced " << outputs.size() << " outputs";
  return outputs[0];
}

absl::StatusOr<mlir::MlirOp> BuildReduceScatterDimShlo(
    mlir::MlirOp input, int64_t dim, const DeviceGroupList& device_groups) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  Dimensions output_shape = CopyIntVector(input_type.getShape());
  TT_RET_CHECK(output_shape[dim] % device_groups[0].size() == 0,
               error::kInvalidArgument)
      << "scatter dimension " << dim << " of size " << output_shape[dim]
      << " is not divisible by group size " << device_groups[0].size();
  output_shape[dim] /= device_groups[0].size();
  auto output_type =
      mlir::RankedTensorType::get(output_shape, input_type.getElementType());

  auto reduction = [element_type = input_type.getElementType()](
                       mlir::RegionBuilder& builder) {
    absl::Status status =
        BuildReduceBody(builder, element_type, c10d::ReduceOp::SUM);
    if (!status.ok()) {
      ABSL_LOG(FATAL) << "failed to build reduce-scatter sum: " << status;
    }
  };

  auto& builder = input.getBuilder();
  return mlir::stablehlo::ReduceScatter(
      output_type, input, reduction, dim,
      BuildReplicaGroupsAttr(builder, device_groups),
      BuildChannelHandleAttr(builder, /*handle=*/1, /*type=*/0),
      /*use_global_device_ids=*/true);
}

// Keep the lowering below the experimental API boundary. The bridge operators
// currently resolve a process-group name and expose only sum reduce-scatter.
// Once PyTorch has dimension-aware _c10d_functional operators, their
// PrivateUse1 adapters can decode the `Any` group representation, parse
// `reduce_op` for reduce-scatter, and delegate to this lowering without
// preserving the experimental schema.
absl::StatusOr<DeviceBufferRef> DimensionalCollective(
    const at::Tensor& input, int64_t dim, int64_t group_size,
    const ProcessGroupTpu& process_group, bool reduce_scatter,
    OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(dim, NormalizeDim(dim, input.dim()));
  TT_RET_CHECK(group_size > 0, error::kInvalidArgument)
      << "group_size must be positive, got " << group_size;

  Dimensions output_shape = CopyIntVector(input.sizes());
  if (reduce_scatter) {
    TT_RET_CHECK(output_shape[dim] % group_size == 0, error::kInvalidArgument)
        << "scatter dimension " << dim << " of size " << output_shape[dim]
        << " is not divisible by group size " << group_size;
    output_shape[dim] /= group_size;
  } else {
    output_shape[dim] *= group_size;
  }

  TT_ASSIGN_OR_RETURN(auto output_dtype,
                      ConvertTo<mlir::ElementType>(input.scalar_type()));
  const DeviceGroupList& device_groups = process_group.GetSubgroupDeviceIds();
  TT_RETURN_IF_ERROR(ValidateGroupSize(device_groups, group_size));

  auto op_builder = [dim, device_groups, reduce_scatter](
                        mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
    if (reduce_scatter) {
      return BuildReduceScatterDimShlo(input_op, dim, device_groups);
    }
    return BuildAllGatherDimShlo(input_op, dim, device_groups);
  };

  return DispatchOp<1>(
      std::move(op_builder), input,
      {.out_dtype = output_dtype,
       .out_dims = std::move(output_shape),
       .op_param_cache_keys = std::move(param_keys),
       .split_mode = GetCollectiveSplitMode(),
       .device_assignment = process_group.device_assignment()});
}

at::Tensor MetaDimensionalCollective(const at::Tensor& input, int64_t dim,
                                     int64_t group_size, bool reduce_scatter) {
  TT_ASSIGN_OR_THROW(dim, NormalizeDim(dim, input.dim()));
  TT_CHECK_THROW(group_size > 0, error::kInvalidArgument)
      << "group_size must be positive, got " << group_size;
  Dimensions output_shape = CopyIntVector(input.sizes());
  if (reduce_scatter) {
    TT_CHECK_THROW(output_shape[dim] % group_size == 0, error::kInvalidArgument)
        << "scatter dimension " << dim << " of size " << output_shape[dim]
        << " is not divisible by group size " << group_size;
    output_shape[dim] /= group_size;
  } else {
    output_shape[dim] *= group_size;
  }
  return at::empty(output_shape, input.options());
}

}  // namespace

at::Tensor ExperimentalAllGatherDim(const at::Tensor& input, int64_t dim,
                                    int64_t group_size,
                                    std::string_view group_name) {
  TT_KERNEL(
      OpName::kExperimentalAllGatherDim, param_keys,
      (input, dim, group_size, group_name), {
        TT_ASSIGN_OR_THROW(auto process_group, GetProcessGroupTpu(group_name));
        TT_ASSIGN_OR_THROW(
            auto result, DimensionalCollective(
                             input, dim, group_size, *process_group,
                             /*reduce_scatter=*/false, std::move(param_keys)));
        return MakeTensor(std::move(result));
      });
}

at::Tensor ExperimentalReduceScatterDim(const at::Tensor& input, int64_t dim,
                                        int64_t group_size,
                                        std::string_view group_name) {
  TT_KERNEL(
      OpName::kExperimentalReduceScatterDim, param_keys,
      (input, dim, group_size, group_name), {
        TT_ASSIGN_OR_THROW(auto process_group, GetProcessGroupTpu(group_name));
        TT_ASSIGN_OR_THROW(auto result,
                           DimensionalCollective(
                               input, dim, group_size, *process_group,
                               /*reduce_scatter=*/true, std::move(param_keys)));
        return MakeTensor(std::move(result));
      });
}

at::Tensor MetaExperimentalAllGatherDim(const at::Tensor& input, int64_t dim,
                                        int64_t group_size,
                                        std::string_view /*group_name*/) {
  return MetaDimensionalCollective(input, dim, group_size,
                                   /*reduce_scatter=*/false);
}

at::Tensor MetaExperimentalReduceScatterDim(const at::Tensor& input,
                                            int64_t dim, int64_t group_size,
                                            std::string_view /*group_name*/) {
  return MetaDimensionalCollective(input, dim, group_size,
                                   /*reduce_scatter=*/true);
}

}  // namespace torch_tpu

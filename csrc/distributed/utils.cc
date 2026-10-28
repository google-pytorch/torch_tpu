/*
 * Copyright 2025 Google LLC
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

#include "csrc/distributed/utils.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/context_manager.h"
#include "csrc/common/context_states.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/distributed/types.h"
#include "csrc/eager/events_queue.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

absl::StatusOr<VirtualMultiHostFold> FoldVirtualMultiHostGroup(
    const DeviceGroup& member_device_ids) {
  const int64_t size = member_device_ids.size();
  TT_RET_CHECK(size > 0, error::kInvalidArgument)
      << "virtual multi-host: a process group without members cannot be "
         "folded";
  // The physical members are the longest duplicate-free prefix.
  int64_t physical_size = 0;
  for (; physical_size < size; ++physical_size) {
    const int64_t id = member_device_ids[physical_size];
    bool seen = false;
    for (int64_t j = 0; j < physical_size && !seen; ++j) {
      seen = member_device_ids[j] == id;
    }
    if (seen) break;
  }
  VirtualMultiHostFold result;
  result.fold = size / physical_size;
  // The rest of the group must repeat that prefix exactly, tile after tile.
  bool tiled = size % physical_size == 0;
  for (int64_t i = physical_size; tiled && i < size; ++i) {
    tiled = member_device_ids[i] == member_device_ids[i % physical_size];
  }
  TT_RET_CHECK(tiled, error::kInvalidArgument)
      << "virtual multi-host: the process group's member device ids "
      << ToString(member_device_ids) << " are not the " << physical_size
      << " devices "
      << ToString(DeviceGroup(member_device_ids.begin(),
                              member_device_ids.begin() + physical_size))
      << " of one slice repeated per slice; only groups whose ranks visit the "
         "slices in the same device order can be emulated";
  result.physical_device_ids.assign(member_device_ids.begin(),
                                    member_device_ids.begin() + physical_size);
  return result;
}

DeviceGroup UniqueDeviceIds(const DeviceGroup& device_ids) {
  DeviceGroup unique;
  unique.reserve(device_ids.size());
  for (int64_t id : device_ids) {
    if (std::find(unique.begin(), unique.end(), id) == unique.end()) {
      unique.push_back(id);
    }
  }
  return unique;
}

DeviceGroupList FoldReplicaGroups(const DeviceGroupList& device_groups) {
  const int64_t fold = device_groups.virtual_fold();
  if (fold == 1) return device_groups;
  std::vector<DeviceGroup> folded;
  folded.reserve(device_groups.size());
  for (const DeviceGroup& group : device_groups) {
    ABSL_CHECK_EQ(group.size() % fold, 0)  // CRASH_OK
        << "virtual multi-host: group " << ToString(group)
        << " is not a multiple of the fold " << fold;
    folded.emplace_back(group.begin(), group.begin() + group.size() / fold);
  }
  return DeviceGroupList(std::move(folded), /*normalize=*/false);
}

int64_t PhysicalGroupSize(const DeviceGroupList& device_groups) {
  ABSL_CHECK(!device_groups.empty())  // CRASH_OK
      << "[PhysicalGroupSize]: device_groups is empty";
  return device_groups[0].size() / device_groups.virtual_fold();
}

mlir::DenseIntElementsAttr BuildReplicaGroupsAttr(
    mlir::MlirBuilder& builder, const DeviceGroupList& logical_groups) {
  const DeviceGroupList device_groups = FoldReplicaGroups(logical_groups);
  int64_t num_groups = device_groups.size();
  int64_t group_size = device_groups[0].size();
  auto tensor_type = mlir::makeTensorType(
      builder.getContext(), {num_groups, group_size}, mlir::ElementType::I64);

  // MLIR builder requires a representation using flat list + shape.
  // TODO: Simplify this after b/445264222
  DeviceGroup flattened_ids(num_groups * group_size);
  for (int64_t i = 0; i < num_groups; ++i) {
    for (int64_t j = 0; j < group_size; ++j) {
      flattened_ids[i * group_size + j] = device_groups[i][j];
    }
  }

  return llvm::cast<mlir::DenseIntElementsAttr>(
      mlir::makeConstant(llvm::ArrayRef<int64_t>(flattened_ids), tensor_type));
}

mlir::stablehlo::ChannelHandleAttr BuildChannelHandleAttr(
    mlir::MlirBuilder& builder, int handle, int type) {
  return mlir::stablehlo::ChannelHandleAttr::get(&builder.getContext(), handle,
                                                 type);
}

absl::Status ValidateReductionOp(c10d::ReduceOp reduce_op,
                                 c10::ScalarType scalar_type) {
  bool is_valid_standard_op =
      reduce_op == c10d::ReduceOp::SUM || reduce_op == c10d::ReduceOp::AVG ||
      reduce_op == c10d::ReduceOp::PRODUCT ||
      reduce_op == c10d::ReduceOp::MIN || reduce_op == c10d::ReduceOp::MAX;
  bool is_valid_bitwise_op = reduce_op == c10d::ReduceOp::BAND ||
                             reduce_op == c10d::ReduceOp::BOR ||
                             reduce_op == c10d::ReduceOp::BXOR;
  TT_RET_CHECK(is_valid_bitwise_op || is_valid_standard_op,
               error::kInvalidArgument)
      << "reduce option not supported: " << ToString(reduce_op.op_);
  if (is_valid_bitwise_op) {
    TT_RET_CHECK(IsIntegral(scalar_type), error::kInvalidArgument)
        << "expected the tensors dtype to be integer on bitwise reduction ops "
           "(BAND, BOR, BXOR), got "
        << ToString(scalar_type);
  }
  const bool is_64bit_dtype = scalar_type == c10::ScalarType::Long ||
                              scalar_type == c10::ScalarType::Double ||
                              scalar_type == c10::ScalarType::UInt64;
  if (is_64bit_dtype) {
    TT_RET_CHECK(
        reduce_op == c10d::ReduceOp::SUM || reduce_op == c10d::ReduceOp::AVG,
        error::kPythonNotImplementedError)
        << "reduce op " << ToString(reduce_op.op_) << " is not supported for "
        << ToString(scalar_type)
        << " tensors; only sum and avg are supported for 64-bit dtypes";
  }
  return absl::OkStatus();
}

void EnterSpmdSafeRegion() {
  PushContextState<IsSpmdSafeContextState>(true);
  ResetRepeatedOpsHeuristicState();
}

void ExitSpmdSafeRegion() { PopContextState<IsSpmdSafeContextState>(); }

bool IsSpmdSafe() {
  return GetContextState<IsSpmdSafeContextState>().value_or(false);
}

}  // namespace torch_tpu

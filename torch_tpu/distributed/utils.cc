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

#include "torch_tpu/distributed/utils.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "c10/core/ScalarType.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/distributed/types.h"

namespace torch_tpu {

mlir::DenseIntElementsAttr BuildReplicaGroupsAttr(
    mlir::MlirBuilder& builder, const DeviceGroupList& device_groups) {
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
      << "reduce option not supported: " << ToString(reduce_op);
  if (is_valid_bitwise_op) {
    TT_RET_CHECK(c10::isIntegralType(scalar_type, true),
                 error::kInvalidArgument)
        << "bitwise reduction ops (BAND, BOR, BXOR) are only supported for "
           "integer tensors, got "
        << scalar_type;
  }
  return absl::OkStatus();
}

void EnterSpmdSafeRegion() { PushContextState<IsSpmdSafeContextState>(true); }

void ExitSpmdSafeRegion() { PopContextState<IsSpmdSafeContextState>(); }

bool IsSpmdSafe() { return GetContextState<IsSpmdSafeContextState>(false); }

}  // namespace torch_tpu

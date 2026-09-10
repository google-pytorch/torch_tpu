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

#include "csrc/ops/channel_shuffle/channel_shuffle_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nullary_aten_kernels.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildChannelShuffleShlo(mlir::MlirOp input,
                                                     int64_t groups) {
  const auto input_type = GetTensorTypeOrDie(input);
  const Dimensions original_sizes = CopyIntVector(input_type.getShape());
  const int64_t rank = original_sizes.size();
  const int64_t c = original_sizes[1];
  const int64_t channels_per_group = c / groups;

  // 1. Intermediate shape: (N, groups, C / groups, D1, D2, ...)
  Dimensions reshaped_sizes;
  reshaped_sizes.reserve(rank + 1);
  reshaped_sizes.push_back(original_sizes[0]);
  reshaped_sizes.push_back(groups);
  reshaped_sizes.push_back(channels_per_group);
  for (int64_t i = 2; i < rank; ++i) {
    reshaped_sizes.push_back(original_sizes[i]);
  }

  // Note: mlir::MlirOp variables cannot be marked const because StableHLO
  // builder functions (e.g., Transpose, Reshape) take operands as MlirOp&.
  mlir::MlirOp reshaped = mlir::stablehlo::Reshape(input, reshaped_sizes);

  // 2. Permutation: swap group and per-group channels -> (0, 2, 1, 3, 4, ...)
  Dimensions permute_dims;
  permute_dims.reserve(rank + 1);
  permute_dims.push_back(0);
  permute_dims.push_back(2);
  permute_dims.push_back(1);
  for (int64_t i = 3; i <= rank; ++i) {
    permute_dims.push_back(i);
  }

  mlir::MlirOp permuted = mlir::stablehlo::Transpose(reshaped, permute_dims);

  // 3. Reshape back to original sizes: (N, C, D1, D2, ...)
  return mlir::stablehlo::Reshape(permuted, original_sizes);
}

}  // namespace

at::Tensor AtenChannelShuffle(const at::Tensor& self, int64_t groups) {
  TT_KERNEL(OpName::kChannelShuffle, param_keys, (self, groups), {
    TT_CHECK_THROW(self.dim() > 2, error::kInvalidArgument)
        << "expected input with > 2 dims, got input with sizes "
        << ToString(self.sizes());
    TT_CHECK_THROW(groups > 0, error::kInvalidArgument)
        << "expected number of groups to divide channels in to be positive, "
           "got "
        << groups;
    const int64_t c = self.size(1);
    TT_CHECK_THROW(c % groups == 0, error::kInvalidArgument)
        << "expected number of channels to be divisible by groups, got " << c
        << " channels and " << groups << " groups";

    if (self.numel() == 0) {
      return AtenEfficientZeroTensor(self.sizes(), self.scalar_type(),
                                     self.layout(), self.device(),
                                     std::nullopt);
    }

    TT_ASSIGN_OR_THROW(const auto element_type,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    auto op_builder =
        [groups](mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildChannelShuffleShlo(input, groups);
    };

    TT_ASSIGN_OR_THROW(
        auto out_buf,
        (DispatchOp<1>(std::move(op_builder), self,
                       {.out_dtype = element_type,
                        .out_dims = CopyIntVector(self.sizes()),
                        .op_param_cache_keys = std::move(param_keys)})));
    TT_ASSIGN_OR_THROW(
        at::Tensor out,
        MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
    TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(out_buf), out));
    return out;
  });
}

}  // namespace  torch_tpu

// Copyright 2025 Google LLC
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

#include "torch_tpu/ops/take/take_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/util/Exception.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace {

namespace stablehlo = mlir::stablehlo;

absl::StatusOr<mlir::MlirOp> BuildTakeShlo(const int64_t num_elements,
                                           mlir::MlirOp input_op,
                                           mlir::MlirOp index_op) {
  auto& builder = input_op.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  const mlir::RankedTensorType index_type = GetTensorTypeOrDie(index_op);

  // Reshape the input to a 1D tensor
  const auto input_element_type = input_type.getElementType();

  const mlir::RankedTensorType expected_type = mlir::makeTensorType(
      builder.getContext(), {num_elements}, input_element_type);
  auto reshaped_input = mlir::stablehlo::Reshape(expected_type, input_op);

  // PyTorch accepts indices in the range [-num_elements, num_elements) and
  // wraps them. SHLO expects indices to be in [0, num_elements). Normalize
  // the indices by index < 0 ? index + num_elements : index. We avoid the
  // % operation as it's very slow to compile for i64.
  auto num_elements_op = MakeConstantLike(index_op, num_elements);
  auto add = stablehlo::Add(index_op, num_elements_op);
  auto zero = MakeConstantLike(index_op, 0);
  auto is_negative =
      stablehlo::Compare(index_op, zero, stablehlo::ComparisonDirection::LT);
  auto normalized_index = stablehlo::Select(is_negative, add, index_op);

  const stablehlo::GatherDimensionNumbersAttr gather_dimension_numbers =
      stablehlo::GatherDimensionNumbersAttr::get(
          &builder.getContext(),
          /*offset_dims=*/{},
          /*collapsed_slice_dims=*/{0},
          /*operand_batching_dims=*/{},
          /*start_indices_batching_dims=*/{},
          /*start_index_map=*/{0},
          /*index_vector_dim=*/index_type.getRank());

  // When num_elements is 0, it's invalid to call Gather() with slice_sizes =
  // {1}, so we just return an empty 1D tensor in that case.
  auto result =
      num_elements == 0
          ? reshaped_input
          : stablehlo::Gather(reshaped_input, normalized_index,
                              gather_dimension_numbers, /*slice_sizes=*/{1},
                              /*indices_are_sorted=*/false);
  return result;
}

// Validates that the indices are in the range [-num_elements, num_elements).
absl::Status ValidateIndices(const int64_t num_elements,
                             const at::Tensor& index) {
  TT_ASSIGN_OR_RETURN(const int64_t num_indices, NumElements(index));
  if (num_indices == 0) {  // No indices to check.
    return absl::OkStatus();
  }
  TT_RET_CHECK(num_elements > 0, error::kOutOfRange)
      << "input tensor must be non-empty when the index tensor is non-empty";
  const auto max_index = index.max().cpu().item<int64_t>();
  TT_RET_CHECK(max_index < num_elements, error::kOutOfRange)
      << "expected indices to be in range [" << -num_elements << ", "
      << (num_elements - 1) << "], got " << max_index;
  const auto min_index = index.min().cpu().item<int64_t>();
  TT_RET_CHECK(min_index >= -num_elements, error::kOutOfRange)
      << "expected indices to be in range [" << -num_elements << ", "
      << (num_elements - 1) << "], got " << min_index;
  return absl::OkStatus();
}

}  // namespace

at::Tensor AtenTake(const at::Tensor& self, const at::Tensor& index) {
  TT_KERNEL(OpName::kTake, _, (self, index), {
    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));
    TT_ASSIGN_OR_THROW(auto index_dtype,
                       ConvertTo<mlir::ElementType>(index.scalar_type()));
    TT_CHECK_THROW(index_dtype == mlir::ElementType::I64,
                   error::kInvalidArgument)
        << "expected index dtype to be int64, got " << ToString(index_dtype);

    TT_ASSIGN_OR_THROW(const int64_t num_elements, NumElements(self));
    if (GetEnableDebugChecks()) {
      TT_THROW_IF_ERROR(ValidateIndices(num_elements, index));
    } else {
      TORCH_WARN_ONCE(
          "Unlike PyTorch on CUDA or CPU, TorchTPU doesn't by default check "
          "that the indices for take() are in range. Set the "
          "TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS environment variable to "
          "\"1\" to enable this check, which incurs a performance penalty.");
    }

    auto op_builder = [num_elements](FixedSizeSpan<mlir::MlirOp, 2> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [self, index] = inputs;
      return BuildTakeShlo(num_elements, self, index);
    };
    TT_ASSIGN_OR_THROW(
        auto result,
        DispatchOp<2>(std::move(op_builder),
                      /*inputs=*/{self, index},
                      {.out_dtype = out_dtype,
                       .out_dims = index.sizes(),
                       .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    return MakeTensor(std::move(result));
  });
}

at::Tensor& AtenTakeOut(const at::Tensor& self, const at::Tensor& index,
                        at::Tensor& out) {
  TT_KERNEL(OpName::kTakeOut, _, (self, index, out), {
    out.copy_(AtenTake(self, index));
    return out;
  });
}

}  // namespace torch_tpu

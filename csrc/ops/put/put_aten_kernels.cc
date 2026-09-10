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

#include "csrc/ops/put/put_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/util/accumulate.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> BuildPutShlo(mlir::MlirOp self,
                                          const Dimensions& self_shape,
                                          mlir::MlirOp index,
                                          mlir::MlirOp source,
                                          bool accumulate) {
  auto& builder = self.getBuilder();
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType index_type = GetTensorTypeOrDie(index);

  if (c10::multiply_integers(self_shape) == 0) {
    return self;
  }

  // 1. Flatten inputs to 1D (handles dynamic shapes)
  auto self_flat = Flatten(self);
  auto index_flat = Flatten(index);
  auto source_flat = Flatten(source);

  // 2. Normalize Negative Indices
  auto index_el_type = index_type.getElementType();
  auto numel_op_scalar = GetNumElements(self, index_el_type);
  TT_ASSIGN_OR_RETURN(auto numel_op,
                      BroadcastIfNeeded(numel_op_scalar, index_flat));

  auto add_numel = mlir::stablehlo::Add(index_flat, numel_op);
  auto zero_op = MakeConstantLike(index_flat, 0);
  auto is_negative = mlir::stablehlo::Compare(
      index_flat, zero_op, mlir::stablehlo::ComparisonDirection::LT);
  auto normalized_index_flat =
      mlir::stablehlo::Select(is_negative, add_numel, index_flat);

  // 3. Prepare Scatter Indices
  // Unsqueeze {N} to {N, 1} dynamically
  TT_ASSIGN_OR_RETURN(auto scatter_indices,
                      Unsqueeze(normalized_index_flat, 1));

  // 4. Configure Scatter Dimension Numbers
  mlir::stablehlo::ScatterDimensionNumbersAttr scatter_dimension_numbers =
      mlir::stablehlo::ScatterDimensionNumbersAttr::get(
          &builder.getContext(),
          /*update_window_dims=*/{},
          /*inserted_window_dims=*/{0},
          /*input_batching_dims=*/{},
          /*scatter_indices_batching_dims=*/{},
          /*scatter_dims_to_operand_dims=*/{0},
          /*index_vector_dim=*/1);

  // 5. Define Scatter Reduction Region
  const auto self_element_type = self_type.getElementType();
  const mlir::RankedTensorType self_flat_type = GetTensorTypeOrDie(self_flat);
  auto block_type = self_flat_type.clone({}, self_element_type);
  auto region_builder = [block_type, accumulate](mlir::RegionBuilder& builder) {
    mlir::MlirOp old_value = mlir::Argument(builder, block_type);
    mlir::MlirOp new_value = mlir::Argument(builder, block_type);
    if (accumulate) {
      mlir::MlirOp sum = mlir::stablehlo::Add(old_value, new_value);
      mlir::stablehlo::Return(builder, {sum});
    } else {
      mlir::stablehlo::Return(builder, {new_value});
    }
  };

  // 6. Apply StableHLO Scatter
  auto self_flat_updated =
      mlir::stablehlo::Scatter(self_flat, scatter_indices, source_flat,
                               region_builder, scatter_dimension_numbers)[0];

  // 7. Reshape Back
  int64_t self_numel = c10::multiply_integers(self_shape);
  TT_ASSIGN_OR_RETURN(
      auto result,
      ReshapeFromStaticDimensions(self_flat_updated, {self_numel}, self_shape));
  return result;
}

absl::Status ValidateInputs(const at::Tensor& self, const at::Tensor& index,
                            const at::Tensor& source) {
  TT_ASSIGN_OR_RETURN(auto self_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(auto index_dtype,
                      ConvertTo<mlir::ElementType>(index.scalar_type()));
  TT_ASSIGN_OR_RETURN(auto source_dtype,
                      ConvertTo<mlir::ElementType>(source.scalar_type()));

  TT_RET_CHECK(index_dtype == mlir::ElementType::I64, error::kInvalidArgument)
      << "expected a long tensor for index, got "
      << ToString(index.scalar_type());
  TT_RET_CHECK(self_dtype == source_dtype, error::kInvalidArgument)
      << "expected self and source to have the same dtype, got "
         "self dtype "
      << ToString(self.scalar_type()) << " and source dtype "
      << ToString(source.scalar_type());

  TT_ASSIGN_OR_RETURN(const int64_t self_numel, NumElements(self));
  TT_ASSIGN_OR_RETURN(const int64_t index_numel, NumElements(index));
  TT_ASSIGN_OR_RETURN(const int64_t source_numel, NumElements(source));

  TT_RET_CHECK(source_numel == index_numel, error::kIndexError)
      << "expected source and index to have the same number of "
         "elements, got source numel "
      << source_numel << " and index numel " << index_numel;

  TT_RET_CHECK(!(self_numel == 0 && index_numel != 0), error::kPythonIndexError)
      << "expected self to be non-empty, got self numel 0";

  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenPut_(at::Tensor& self, const at::Tensor& index,
                     const at::Tensor& source, bool accumulate) {
  TT_KERNEL(OpName::kPut_, param_keys, (self, index, source, accumulate), {
    TT_THROW_IF_ERROR(ValidateInputs(self, index, source));

    TT_ASSIGN_OR_THROW(auto out_dtype,
                       ConvertTo<mlir::ElementType>(self.scalar_type()));

    if (self.numel() == 0) {
      return self;
    }

    auto self_shape = CopyIntVector(self.sizes());
    auto op_builder = [self_shape,
                       accumulate](FixedSizeSpan<mlir::MlirOp, 3> inputs)
        -> absl::StatusOr<mlir::MlirOp> {
      auto& [self, index, source] = inputs;
      return BuildPutShlo(self, self_shape, index, source, accumulate);
    };

    DispatchOpOptions<1> options = {
        .out_dtype = out_dtype,
        .out_dims = self.sizes(),
        .op_param_cache_keys = std::move(param_keys),
    };
    TT_THROW_IF_ERROR(DispatchOpOut<3>(std::move(op_builder),
                                       /*inputs=*/{self, index, source},
                                       /*outputs=*/self, std::move(options)));
    return self;
  });
}

}  // namespace torch_tpu

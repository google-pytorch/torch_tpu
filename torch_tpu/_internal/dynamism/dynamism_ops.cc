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

#include "torch_tpu/_internal/dynamism/dynamism_ops.h"

#include <cstdint>
#include <string>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> Pad(mlir::MlirOp input_op,
                                 const Dimensions& target_dimensions) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  TT_RET_CHECK(input_type.hasStaticShape(), error::kInvalidArgument)
      << "input type must have static shape, but has dynamic shape.";
  TT_RET_CHECK(input_type.getRank() == target_dimensions.size(),
               error::kInvalidArgument)
      << "input type rank must match target dimensions size.";
  Dimensions zeros(input_type.getRank(), 0);
  Dimensions edge_padding_high(input_type.getRank(), 0);
  bool was_padded = false;
  for (int i = 0; i < input_type.getRank(); ++i) {
    int64_t upper_bound = target_dimensions[i];
    int padding_size = upper_bound - input_type.getDimSize(i);
    TT_RET_CHECK(padding_size >= 0, error::kInvalidArgument)
        << "padding size must be non-negative, but is " << padding_size;
    if (padding_size > 0) {
      was_padded = true;
    }
    edge_padding_high[i] = padding_size;
  }
  if (!was_padded) {
    return input_op;
  }
  auto padding_value =
      MakeScalarConstant(input_op.getBuilder(), 0, input_type.getElementType());
  auto pad_op =
      mlir::stablehlo::Pad(input_op, padding_value, /*edge_padding_low=*/zeros,
                           edge_padding_high, /*interior_padding=*/zeros);
  return pad_op;
}

absl::StatusOr<mlir::MlirOp> Slice(mlir::MlirOp input_op,
                                   const Dimensions& target_dimensions) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  TT_RET_CHECK(input_type.hasStaticShape(), error::kInvalidArgument)
      << "input type must have static shape, but has dynamic shape.";
  TT_RET_CHECK(input_type.getRank() == target_dimensions.size(),
               error::kInvalidArgument)
      << "input type rank must match target dimensions size.";
  bool needs_slice = false;
  for (int i = 0; i < input_type.getRank(); ++i) {
    int64_t target_size = target_dimensions[i];
    TT_RET_CHECK(target_size <= input_type.getDimSize(i),
                 error::kInvalidArgument)
        << "target dimension size must be smaller or equal than input "
           "dimension size.";
    if (target_size < input_type.getDimSize(i)) {
      needs_slice = true;
    }
  }
  if (!needs_slice) {
    return input_op;
  }

  Dimensions start_indices(target_dimensions.size(), 0);
  Dimensions strides(target_dimensions.size(), 1);
  return mlir::stablehlo::Slice(input_op, start_indices, target_dimensions,
                                strides);
}

}  // namespace

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GetPadModule(
    mlir::MLIRContext& mlir_context,
    absl::Span<const Dimensions> dimensions_vec,
    absl::Span<const Dimensions> padded_dimensions_vec,
    absl::Span<const mlir::ElementType> input_dtypes) {
  ABSL_VLOG(2) << "[GetPadModule] Creating a padding module for dynamism with "
               << dimensions_vec.size() << " shapes.";
  TT_RET_CHECK(!dimensions_vec.empty(), error::kInvalidArgument)
      << "PadModule requires at least one shape.";
  TT_RET_CHECK(dimensions_vec.size() == padded_dimensions_vec.size(),
               error::kInvalidArgument)
      << "dimensions_vec and padded_dimensions_vec must have the same size.";
  TT_RET_CHECK(dimensions_vec.size() == input_dtypes.size(),
               error::kInvalidArgument)
      << "dimensions_vec and input_dtypes must have the same size.";
  std::string module_name = "pad_module";
  mlir::ModuleBuilder mb(mlir_context, module_name);
  mlir::func::FunctionBuilder fb(mb, "main");
  DynamicMlirOpResults results;
  results.reserve(dimensions_vec.size());
  // Add a function parameter for each input shape.
  for (int i = 0; i < dimensions_vec.size(); ++i) {
    auto type =
        makeTensorType(mlir_context, dimensions_vec[i], input_dtypes[i]);
    // Zero-sized tensors are not passed to the executable.
    if (type.getNumElements() == 0) {
      continue;
    }
    auto input_op = mlir::func::Argument(fb, type);
    if (dimensions_vec[i] == padded_dimensions_vec[i]) {
      results.push_back(input_op);
      continue;
    }
    TT_ASSIGN_OR_RETURN(auto padded_op,
                        Pad(input_op, padded_dimensions_vec[i]));
    results.push_back(padded_op);
    for (int d = 0; d < dimensions_vec[i].size(); ++d) {
      if (dimensions_vec[i][d] != padded_dimensions_vec[i][d]) {
        results.push_back(mlir::stablehlo::GetDimensionSize(input_op, d));
      }
    }
  }
  mlir::func::Return(fb, results);
  auto module = mb.build();
  return module;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GetSliceModule(
    mlir::MLIRContext& mlir_context,
    absl::Span<const Dimensions> dimensions_vec,
    absl::Span<const Dimensions> padded_dimensions_vec,
    absl::Span<const mlir::ElementType> input_dtypes) {
  ABSL_VLOG(2) << "[GetSliceModule] creating a slice module for dynamism with "
               << dimensions_vec.size() << " shapes.";
  TT_RET_CHECK(!dimensions_vec.empty(), error::kInvalidArgument)
      << "[GetSliceModule] SliceModule requires at least one shape.";
  TT_RET_CHECK(dimensions_vec.size() == padded_dimensions_vec.size(),
               error::kInvalidArgument)
      << "[GetSliceModule] dimensions_vec and padded_dimensions_vec must have "
         "the same size.";
  TT_RET_CHECK(dimensions_vec.size() == input_dtypes.size(),
               error::kInvalidArgument)
      << "[GetSliceModule] dimensions_vec and input_dtypes must have the same "
         "size.";
  std::string module_name = "slice_module";
  mlir::ModuleBuilder mb(mlir_context, module_name);
  mlir::func::FunctionBuilder fb(mb, "main");
  DynamicMlirOpResults results;
  results.reserve(dimensions_vec.size());

  for (int i = 0; i < dimensions_vec.size(); ++i) {
    auto type =
        makeTensorType(mlir_context, padded_dimensions_vec[i], input_dtypes[i]);
    auto input_op = mlir::func::Argument(fb, type);
    if (dimensions_vec[i] == padded_dimensions_vec[i]) {
      results.push_back(input_op);
      continue;
    }
    TT_ASSIGN_OR_RETURN(auto slice_op, Slice(input_op, dimensions_vec[i]));
    results.push_back(slice_op);
  }
  mlir::func::Return(fb, results);
  auto module = mb.build();
  return module;
}
}  // namespace torch_tpu

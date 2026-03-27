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
#include <utility>

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
#include "torch_tpu/common/shape.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

absl::StatusOr<MlirUnaryOpBuilder> GetPaddingOpBuilder(
    Indices dimension_indices, Dimensions upper_bounds) {
  TT_RET_CHECK(dimension_indices.size() == upper_bounds.size(),
               error::kInvalidArgument)
      << "dimensions and upper_bounds must have the same size.";
  return [dimension_indices = std::move(dimension_indices),
          upper_bounds = std::move(upper_bounds)](
             mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
    const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
    TT_RET_CHECK(input_type.hasStaticShape(), error::kInvalidArgument)
        << "input type must have static shape, but has dynamic shape.";

    Dimensions edge_padding_low(input_type.getRank(), 0);
    Dimensions edge_padding_high(input_type.getRank(), 0);
    bool was_padded = false;
    for (int i = 0; i < dimension_indices.size(); ++i) {
      int64_t dimension_index = dimension_indices[i];
      int64_t upper_bound = upper_bounds[i];
      TT_RET_CHECK(
          dimension_index >= 0 && dimension_index < input_type.getRank(),
          error::kInvalidArgument)
          << "dimension index " << dimension_index
          << " is out of bounds for tensor " << input_type.getRank()
          << " dimensions";
      int padding_size = upper_bound - input_type.getDimSize(dimension_index);
      TT_RET_CHECK(padding_size >= 0, error::kInvalidArgument)
          << "padding size must be non-negative, but is " << padding_size;
      if (padding_size > 0) {
        was_padded = true;
      }
      edge_padding_high[dimension_index] = padding_size;
    }
    if (!was_padded) {
      return input_op;
    }
    auto padding_value = MakeScalarConstant(input_op.getBuilder(), 0,
                                            input_type.getElementType());
    auto pad_op =
        mlir::stablehlo::Pad(input_op, padding_value, edge_padding_low,
                             edge_padding_high, edge_padding_low);
    return pad_op;
  };
}

}  // namespace

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GetPadModule(
    mlir::MLIRContext& mlir_context, absl::Span<const Shape> shapes) {
  ABSL_VLOG(2) << "[GetPadModule] Creating a padding module for dynamism with "
               << shapes.size() << " shapes.";
  TT_RET_CHECK(!shapes.empty(), error::kInvalidArgument)
      << "PadModule requires at least one shape.";
  std::string module_name = "pad_module";
  mlir::ModuleBuilder mb(mlir_context, module_name);
  mlir::func::FunctionBuilder fb(mb, "main");
  DynamicMlirOpResults results;
  results.reserve(shapes.size());
  // Add a function parameter for each input shape.
  for (int i = 0; i < shapes.size(); ++i) {
    const Shape& shape = shapes[i];
    auto type = makeTensorType(mlir_context, shape.dimensions(), shape.dtype());
    // Zero-sized tensors are not passed to the executable.
    if (type.getNumElements() == 0) {
      continue;
    }
    auto input_op = mlir::func::Argument(fb, type);
    if (shape.dynamic_dimensions().empty()) {
      results.push_back(input_op);
      continue;
    }
    Indices dimension_indices;
    Dimensions upper_bounds;
    for (auto dynamic_dimension : shape.dynamic_dimensions()) {
      dimension_indices.push_back(dynamic_dimension.dimension);
      upper_bounds.push_back(dynamic_dimension.upper_bound);
    }
    ABSL_VLOG(2) << "[GetPadModule] Input " << i << " has "
                 << dimension_indices.size() << " dynamic dimensions";
    TT_ASSIGN_OR_RETURN(auto op_builder,
                        GetPaddingOpBuilder(std::move(dimension_indices),
                                            std::move(upper_bounds)));
    TT_ASSIGN_OR_RETURN(auto padded_op, op_builder(input_op));
    auto result = padded_op;
    for (auto dynamic_dimension : shape.dynamic_dimensions()) {
      auto dim_size = mlir::stablehlo::GetDimensionSize(
          input_op, dynamic_dimension.dimension);
      result = mlir::stablehlo::SetDimensionSize(result, dim_size,
                                                 dynamic_dimension.dimension);
    }
    results.push_back(result);
  }
  mlir::func::Return(fb, results);
  auto module = mb.build();
  return module;
}

}  // namespace torch_tpu

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
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

// Pads the input tensor to the target dimensions.
// The input tensor must have a static shape.
// The padding value is 0.
absl::StatusOr<mlir::MlirOp> Pad(mlir::MlirOp input_op,
                                 const Dimensions& target_dimensions) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  TT_RET_CHECK(input_type.hasStaticShape(), error::kInvalidArgument)
      << "expected static shape, got dynamic shape "
      << ToString(input_type.getShape());
  TT_RET_CHECK(input_type.getRank() == target_dimensions.size(),
               error::kInvalidArgument)
      << "expected input dimensions size to match target dimensions size, got "
         "input dimensions size="
      << input_type.getRank()
      << " target dimensions size=" << target_dimensions.size();
  Dimensions zeros(input_type.getRank(), 0);
  Dimensions edge_padding_high(input_type.getRank(), 0);
  bool was_padded = false;
  for (int i = 0; i < input_type.getRank(); ++i) {
    int64_t upper_bound = target_dimensions[i];
    int padding_size = upper_bound - input_type.getDimSize(i);
    TT_RET_CHECK(padding_size >= 0, error::kInvalidArgument)
        << "padding size must be non-negative, got " << padding_size;
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

// Slices the input tensor to the target dimensions.
// The input tensor must have the same rank as the target dimensions.
// If the input tensor has a static shape, it is returned as is.
// Otherwise, the input tensor converted to a statically shaped tensor and then
// sliced from offset 0 to the target size in each dimension.
absl::StatusOr<mlir::MlirOp> Slice(mlir::MlirOp input_op,
                                   const Dimensions& target_dimensions) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input_op);
  TT_RET_CHECK(input_type.getRank() == target_dimensions.size(),
               error::kInvalidArgument)
      << "expected input dimensions size to match target dimensions size, got "
         "input dimensions size="
      << input_type.getRank()
      << " target dimensions size=" << target_dimensions.size();

  if (input_type.hasStaticShape()) {
    TT_RET_CHECK(input_type.getShape().equals(target_dimensions),
                 error::kInvalidArgument)
        << "expected input shape to match target dimensions, got input shape="
        << ToString(input_type.getShape())
        << " target dimensions=" << ToString(target_dimensions);
    return input_op;
  }

  // Convert to static shape of the same size.
  mlir::RankedTensorType op_type = GetTensorTypeOrDie(input_op);
  mlir::stablehlo::Dimensions dims = GetDimensions(input_op);
  Dimensions static_dims;
  static_dims.reserve(dims.size());
  for (const auto& dim : dims) {
    static_dims.push_back(dim.size);
  }
  auto& fb = input_op.getBuilder();
  mlir::RankedTensorType current_type = op_type;
  for (int i = 0; i < op_type.getRank(); ++i) {
    if (op_type.isDynamicDim(i)) {
      Dimensions current_shape = CopyIntVector(current_type.getShape());
      current_shape[i] = static_dims[i];
      mlir::RankedTensorType next_type = makeTensorType(
          fb.getContext(), current_shape, current_type.getElementType());

      auto cst = MakeScalarConstant(fb, static_cast<int32_t>(static_dims[i]),
                                    mlir::ElementType::I32);

      auto loc = input_op.getBuilder().getLoc();
      auto set_dim_size_op = mlir::stablehlo::SetDimensionSizeOp::create(
          fb.getOpBuilder(), loc, next_type, input_op.getValue(),
          cst.getValue(), i);
      input_op =
          mlir::MlirOp(input_op.getBuilder(), set_dim_size_op.getResult());
      current_type = next_type;
    }
  }

  Dimensions start_indices(target_dimensions.size(), 0);
  Dimensions strides(target_dimensions.size(), 1);
  return mlir::stablehlo::Slice(input_op, start_indices, target_dimensions,
                                strides);
}

mlir::RankedTensorType MakeDynamicTensorType(
    mlir::MLIRContext& context, const Dimensions& dimensions,
    const Dimensions& padded_dimensions, mlir::ElementType element_type) {
  mlir::stablehlo::Dimensions dims;
  dims.reserve(dimensions.size());
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == padded_dimensions[i]) {
      dims.push_back(mlir::stablehlo::DimensionInfo{
          .size = dimensions[i],
      });
    } else {
      dims.push_back(mlir::stablehlo::DimensionInfo{
          .size = padded_dimensions[i],
          .boundOp = nullptr,  // Different from nullopt!
      });
    }
  }

  return mlir::stablehlo::getRankedTensorType(
      dims, mlir::getElementType(context, element_type));
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
    Dimensions padded_dimensions = shape.dimensions();
    for (const auto& dynamic_dim : shape.dynamic_dimensions()) {
      padded_dimensions[dynamic_dim.dimension] = dynamic_dim.upper_bound;
    }
    TT_ASSIGN_OR_RETURN(auto padded_op, Pad(input_op, padded_dimensions));
    results.push_back(padded_op);
    for (auto dynamic_dimension : shape.dynamic_dimensions()) {
      results.push_back(mlir::stablehlo::GetDimensionSize(
          input_op, dynamic_dimension.dimension));
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
      << "expected at least one dimensions vector, got none";
  TT_RET_CHECK(dimensions_vec.size() == padded_dimensions_vec.size(),
               error::kInvalidArgument)
      << "expected dimensions vector and padded dimensions vector to have the "
         "same size, got dimensions vector size="
      << dimensions_vec.size()
      << " padded dimensions vector size=" << padded_dimensions_vec.size();
  TT_RET_CHECK(dimensions_vec.size() == input_dtypes.size(),
               error::kInvalidArgument)
      << "expected dimensions vector and input dtypes to have the same size, "
         "got "
         "dimensions vector size="
      << dimensions_vec.size() << " input dtypes size=" << input_dtypes.size();
  std::string module_name = "slice_module";
  mlir::ModuleBuilder mb(mlir_context, module_name);
  mlir::func::FunctionBuilder fb(mb, "main");
  DynamicMlirOpResults results;
  results.reserve(dimensions_vec.size());

  for (int i = 0; i < dimensions_vec.size(); ++i) {
    if (dimensions_vec[i] == padded_dimensions_vec[i]) {
      results.push_back(mlir::func::Argument(
          fb,
          makeTensorType(mlir_context, dimensions_vec[i], input_dtypes[i])));
      continue;
    }
    // Create dynamic type for the input tensor.
    auto dynamic_type =
        MakeDynamicTensorType(mlir_context, dimensions_vec[i],
                              padded_dimensions_vec[i], input_dtypes[i]);
    auto input_op = mlir::func::Argument(fb, dynamic_type);
    TT_ASSIGN_OR_RETURN(auto slice_op, Slice(input_op, dimensions_vec[i]));
    results.push_back(slice_op);
  }
  mlir::func::Return(fb, results);
  auto module = mb.build();
  return module;
}
}  // namespace torch_tpu

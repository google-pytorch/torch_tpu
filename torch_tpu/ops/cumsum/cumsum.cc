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

#include "torch_tpu/ops/cumsum/cumsum.h"

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

namespace {

struct ReduceWindowAttributes {
  llvm::SmallVector<int64_t> window_dimensions;
  mlir::DenseI64ArrayAttr window_strides;
  mlir::DenseI64ArrayAttr base_dilations;
  mlir::DenseI64ArrayAttr window_dilations;
  mlir::DenseIntElementsAttr padding;
};

ReduceWindowAttributes GetReduceWindowAttributes(
    mlir::MlirBuilder& builder, mlir::RankedTensorType input_type,
    int64_t normalized_dim) {
  int64_t rank = input_type.getRank();

  llvm::SmallVector<int64_t> window_dimensions(rank, 1);
  window_dimensions[normalized_dim] = input_type.getDimSize(normalized_dim);

  llvm::SmallVector<int64_t> window_strides(rank, 1);
  llvm::SmallVector<int64_t> base_dilations(rank, 1);
  llvm::SmallVector<int64_t> window_dilations(rank, 1);

  // Padding: (dim - 1) values to low padding on `dim`.
  llvm::SmallVector<int64_t> pad_values(rank * 2, 0);
  pad_values[2 * normalized_dim] = input_type.getDimSize(normalized_dim) - 1;

  return ReduceWindowAttributes{
      .window_dimensions = window_dimensions,
      .window_strides =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), window_strides),
      .base_dilations =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), base_dilations),
      .window_dilations =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), window_dilations),
      .padding = mlir::DenseIntElementsAttr::get(
          makeTensorType(builder.getContext(), {rank, 2},
                         builder.getOpBuilder().getI64Type()),
          pad_values)};
}

}  // namespace

// Builds the cumulative sum operation using StableHLO's ReduceWindowOp.
//
// How it works:
// The cumulative sum along a dimension `dim` means that each element in the
// output is the sum of all elements up to that index in the input tensor
// along `dim`.
//
// We can achieve this using `stablehlo.reduce_window` with `stablehlo.add`
// as the reduction function. The key idea is to use padding.
//
// 1.  Window Size: The window size along the dimension `dim` is set to the
//     total size of that dimension in the input tensor (let's call it `N`).
//     For all other dimensions, the window size is 1.
//
// 2.  Padding: We add `N-1` padding elements (zeros, the identity for addition)
//     to the *beginning* (low side) of the input tensor along `dim`. No
//     padding is added to the end (high side).
//
// 3.  Sliding Window: The `reduce_window` operation slides this window of
//     size `N` across the padded tensor with a stride of 1. Since the window
//     size is equal to the original dimension size and we've prepended `N-1`
//     zeros, each position of the window captures a prefix of the original
//     tensor.
//
// Example: Input `[a, b, c]`, dim=0. Size N=3.
// -   Padded input: `[0, 0, a, b, c]` (N-1 = 2 zeros prepended).
// -   Window size: [3]. Stride: [1].
//
// -   Window 1: `[0, 0, a]` -> Reduce(Add) = a
// -   Window 2: `[0, a, b]` -> Reduce(Add) = a + b
// -   Window 3: `[a, b, c]` -> Reduce(Add) = a + b + c
//
// The result is `[a, a+b, a+b+c]`, which is the cumulative sum.
//
// For multi-dimensional tensors, this logic applies independently to each
// 1D slice along the specified `dim`.
absl::StatusOr<mlir::MlirOp> BuildCumsumShlo(
    const int64_t normalized_dim,
    const std::optional<mlir::ElementType> out_dtype, mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  mlir::Type element_type = input_type.getElementType();
  mlir::MlirBuilder& builder = input.getBuilder();

  // Respect the dtype if provided, otherwise always convert to int64 for
  // boolean and integer types for accumulation.
  if (out_dtype.has_value()) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=this check also exists in aten
                   // implementation of the op. The check is still here to
                   // ensure any direct use of this builder from any other op
                   // errors out
        !mlir::IsBoolean(out_dtype.value()), error::kUnimplemented)
        << "dtype bool is not yet supported";
    element_type = getElementType(builder.getContext(), out_dtype.value());
  } else if (input_type.getElementType().isInteger()) {
    element_type = builder.getOpBuilder().getI64Type();
  }

  // Convert should happen before the operation is performed to prevent data
  // overflows.
  if (input_type.getElementType() != element_type) {
    input = stablehlo::ConvertElementType(input, element_type);
  }

  mlir::MlirOp init_value = MakeScalarConstant(builder, 0, element_type);
  auto [window_dimensions, window_strides, base_dilations, window_dilations,
        padding] =
      GetReduceWindowAttributes(builder, input_type, normalized_dim);

  return stablehlo::ReduceWindow(
      builder, /*inputs=*/{input}, /*init_values=*/{init_value},
      /*body=*/
      [&](mlir::RegionBuilder& body) {
        stablehlo::buildReduceBody<stablehlo::AddOp>(
            element_type, body.getRegion(), body.getOpBuilder());
      },
      window_dimensions, window_strides, base_dilations, window_dilations,
      padding)[0];
}

}  // namespace torch_tpu

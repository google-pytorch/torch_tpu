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

#include "torch_tpu/ops/prod/prod.h"

#include <cstdint>
#include <optional>

#include "ATen/WrapDimUtils.h"
#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"

namespace torch_tpu {

namespace stablehlo = mlir::stablehlo;

// Helper to create the initial value for the product reduction (1 or True)
mlir::MlirOp GetProdInitValue(mlir::MlirBuilder& builder,
                              mlir::Type element_type) {
  if (element_type.isInteger(/*width=*/1)) {
    // For boolean, the multiplicative identity is True.
    return MakeScalarConstant(builder, true, element_type);
  }
  // For float, integer, complex, the multiplicative identity is 1.
  return MakeScalarConstant(builder, /*value=*/1, element_type);
}

mlir::MlirOp BuildProdShlo(std::optional<int64_t> dim, ReductionMode mode,
                           std::optional<mlir::ElementType> dtype,
                           mlir::MlirOp input) {
  ABSL_VLOG(1) << "BuildProdShlo input: " << input.ToString()
               << " dim: " << (dim ? absl::StrCat(*dim) : "None")
               << " keep_dim: " << (mode == ReductionMode::kKeepDims)
               << " dtype: " << (dtype ? absl::StrCat(*dtype) : "None");

  mlir::MlirBuilder& builder = input.getBuilder();

  const mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(input);

  mlir::Type element_type = input_tensor_type.getElementType();
  // Respect the dtype if provided, otherwise always convert to int64 for
  // boolean and integer types for accumulation.
  if (dtype.has_value()) {
    element_type = getElementType(builder.getContext(), dtype.value());
  } else if (input_tensor_type.getElementType().isInteger()) {
    element_type = builder.getOpBuilder().getI64Type();
  }

  mlir::MlirOp converted_input = input;
  if (input_tensor_type.getElementType() != element_type) {
    converted_input = stablehlo::ConvertElementType(input, element_type);
  }

  mlir::MlirOp init_value = GetProdInitValue(builder, element_type);

  auto reduce_fn = [element_type](mlir::RegionBuilder& rb) {
    stablehlo::buildReduceBody<stablehlo::MulOp>(element_type, rb.getRegion(),
                                                 rb.getOpBuilder());
  };

  int64_t rank = input_tensor_type.getRank();

  llvm::SmallVector<int64_t> dims;
  if (dim.has_value()) {
    // If dim is provided, then reduce only the specified dimension.
    int64_t normalized_dim =
        at::maybe_wrap_dim(*dim, rank);  // MAYBE_WRAP_DIM_OK=Verified in kernel
    dims.push_back(normalized_dim);
  } else {
    // If dim is not provided, then reduce all dimensions.
    // This is the default behavior of torch.prod.
    dims.resize(rank);
    absl::c_iota(dims, 0);
  }

  mlir::MlirOp result_op = stablehlo::Reduce(builder, {converted_input},
                                             {init_value}, reduce_fn, dims)[0];

  if (dim.has_value() && mode == ReductionMode::kKeepDims) {
    int64_t normalized_dim =
        at::maybe_wrap_dim(*dim, rank);  // MAYBE_WRAP_DIM_OK=Verified in kernel
    result_op = BuildKeepDimsShlo(converted_input, result_op, {normalized_dim});
  }
  return result_op;
}

}  // namespace torch_tpu

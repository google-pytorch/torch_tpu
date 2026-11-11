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

#include "csrc/ops/cumprod/cumprod_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/functional/bind_front.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/scan_builder.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

absl::StatusOr<mlir::MlirOp> BuildCumprodShlo(
    const int64_t normalized_dim,
    const std::optional<mlir::ElementType> out_dtype, mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  // PyTorch default promotion: when out_dtype is omitted, integral types
  // default to int64 (i64) to prevent overflow; float/complex preserve their
  // input dtype.
  mlir::ElementType target_element_type = out_dtype.value_or(
      input_type.getElementType().isInteger() ? mlir::ElementType::I64
                                              : GetElementTypeOrDie(input));

  mlir::MlirBuilder& builder = input.getBuilder();
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(target_element_type));
  if (GetElementTypeOrDie(input) != compute_dtype) {
    input = mlir::stablehlo::ConvertElementType(input, compute_dtype);
  }

  const mlir::Type compute_element_type =
      getElementType(builder.getContext(), compute_dtype);

  const mlir::RankedTensorType promoted_type = GetTensorTypeOrDie(input);
  // chlo.ScanOp carries are rank-reduced (the scan dimension is erased).
  Dimensions carry_shape = CopyIntVector(promoted_type.getShape());
  carry_shape.erase(carry_shape.begin() + normalized_dim);

  const mlir::MlirOp init_value =
      MakeScalarConstant(builder, 1, compute_element_type);
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp carry_init,
                      BroadcastIfNeeded(init_value, carry_shape));

  MultiInputScanBodyBuilder body_builder =
      [](mlir::OpBuilder& op_builder, mlir::Location loc,
         mlir::ValueRange input_slices, mlir::Value /*index*/,
         mlir::ValueRange carries) -> absl::StatusOr<ScanBodyResults> {
    llvm::SmallVector<mlir::Value> prod = {
        mlir::stablehlo::MulOp::create(op_builder, loc, input_slices[0],
                                       carries[0])
            .getResult()};
    // For a cumulative product the per-position output is the running carry.
    return ScanBodyResults{prod, prod};
  };

  // Associative scan -> chlo.ScanOp (native scan emitter). Results are
  // [carries..., outputs...]; the prefix-scan output is the single output.
  TT_ASSIGN_OR_RETURN(
      const DynamicMlirOpResults results,
      BuildScanShlo(
          builder, {input}, normalized_dim, /*num_scan_inputs=*/1,
          /*carry_inits=*/{carry_init}, /*output_inits=*/{input},
          std::move(body_builder),
          ScanOptions{.should_squeeze = true, .is_associative = true}));
  if (GetElementTypeOrDie(results[/*num_carries=*/1]) != target_element_type) {
    return mlir::stablehlo::ConvertElementType(results[/*num_carries=*/1],
                                               target_element_type);
  }
  return results[/*num_carries=*/1];
}

}  // namespace

at::Tensor& AtenCumprodOut(const at::Tensor& self, int64_t dim,
                           std::optional<at::ScalarType> dtype,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kCumprodOut, param_keys, (self, dim, dtype, out), {
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));

    // PyTorch ATen parity: when explicit dtype is provided, out tensor dtype
    // must strictly match dtype; safe-casting is only allowed when dtype is
    // omitted.
    if (dtype.has_value()) {
      TT_CHECK_THROW(out.scalar_type() == dtype.value(),
                     error::kInvalidArgument)
          << "expected the output dtype to be " << ToString(dtype.value())
          << ", got " << ToString(out.scalar_type());
    }

    // Dimension must be validated before early returns so that invalid dims
    // (e.g. dim > 0 on a 0-D scalar or empty tensor) throw IndexError.
    TT_ASSIGN_OR_THROW(const int64_t normalized_dim,
                       SafeWrapDim(dim, self.dim()));

    // Fast-path: empty tensors require no computation.
    if (out.numel() == 0) {
      return out;
    }

    // 0-D scalar tensor: cumulative scan along a scalar is an identity copy.
    // In PyTorch ATen, 0-D scalar tensors (including boolean) succeed via
    // fill_.
    if (self.dim() == 0) {
      out.copy_(self);
      return out;
    }

    // PyTorch does not support boolean cumulative scans for non-empty tensors
    // with dim > 0.
    TT_CHECK_THROW(!IsBool(out), error::kPythonNotImplementedError)
        << "not implemented for " << ToString(out.scalar_type());

    TT_ASSIGN_OR_THROW(const mlir::ElementType out_mlir_type,
                       ConvertTo<mlir::ElementType>(out.scalar_type()));

    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out,
        absl::bind_front(BuildCumprodShlo, normalized_dim, out_mlir_type),
        {.op_param_cache_keys = std::move(param_keys),
         .out_dtype = out_mlir_type,
         .allow_low_precision_float = !IsLowPrecisionFloat(out)}));
    return out;
  });
}

}  // namespace torch_tpu

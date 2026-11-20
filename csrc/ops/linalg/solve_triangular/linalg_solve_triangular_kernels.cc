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

#include "csrc/ops/linalg/solve_triangular/linalg_solve_triangular_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

absl::Status ValidateDimensions(const at::Tensor& a, const at::Tensor& b,
                                bool left) {
  // Solving A * X = B if left == true, X * A = B if left == false.
  TT_RET_CHECK(a.dim() >= 2, error::kInvalidArgument)
      << "expected the first argument to have at least 2 dimensions, got "
      << a.dim();
  TT_RET_CHECK(b.dim() == a.dim(), error::kInvalidArgument)
      << "expected the two inputs to have the same number of dimensions, got "
      << a.dim() << " and " << b.dim();
  TT_RET_CHECK(left || a.size(-1) == b.size(-1), error::kInvalidArgument)
      << "expected the two inputs to have matching last dimension (left == "
         "False means we are solving X * A = B), got "
      << a.size(-1) << " and " << b.size(-1);
  TT_RET_CHECK(!left || a.size(-2) == b.size(-2), error::kInvalidArgument)
      << "expected the two inputs to have matching second to last dimension "
         "(left == True means we are solving A * X = B), got "
      << a.size(-2) << " and " << b.size(-2);
  return absl::OkStatus();
}

}  // namespace

NAryMlirOpBuilder<2, 1> LinalgSolveTriangularBuilder(
    SolveTriangularOptions options) {
  return [options](FixedSizeSpan<mlir::MlirOp, 2> inputs)
             -> absl::StatusOr<mlir::MlirOp> {
    auto& [a_op, b_op] = inputs;
    ABSL_VLOG(1) << "triangular solve: " << "upper:" << options.upper
                 << " left:" << options.left
                 << " unitriangular:" << options.unitriangular;
    const mlir::stablehlo::Dimensions a_dims = GetDimensions(a_op);
    const mlir::stablehlo::Dimensions b_dims = GetDimensions(b_op);
    ABSL_CHECK_GE(a_dims.size(), 2)  // CRASH_OK=validated by caller.
        << "Expected rank >= 2, got " << a_dims.size()
        << ". This is a TorchTPU bug.";
    ABSL_CHECK_GE(b_dims.size(), 2)  // CRASH_OK=validated by caller.
        << "Expected rank >= 2, got " << b_dims.size()
        << ". This is a TorchTPU bug.";
    const absl::Span<const mlir::stablehlo::DimensionInfo> a_batch_dims =
        absl::MakeConstSpan(a_dims).subspan(0, a_dims.size() - 2);
    const absl::Span<const mlir::stablehlo::DimensionInfo> b_batch_dims =
        absl::MakeConstSpan(b_dims).subspan(0, b_dims.size() - 2);
    TT_ASSIGN_OR_RETURN(
        const mlir::stablehlo::Dimensions broadcasted_batch_dims,
        InferBroadcastDimensions(a_batch_dims, b_batch_dims));
    mlir::stablehlo::Dimensions a_broadcasted_dims = broadcasted_batch_dims;
    a_broadcasted_dims.append(a_dims.end() - 2, a_dims.end());
    mlir::stablehlo::Dimensions b_broadcasted_dims = broadcasted_batch_dims;
    b_broadcasted_dims.append(b_dims.end() - 2, b_dims.end());
    TT_ASSIGN_OR_RETURN(a_op,
                        BroadcastIfNeeded(a_op, std::move(a_broadcasted_dims)));
    TT_ASSIGN_OR_RETURN(b_op,
                        BroadcastIfNeeded(b_op, std::move(b_broadcasted_dims)));
    return TriangularSolveLike(
        a_op, b_op,
        TriangularSolveOptions{
            .side = options.left ? MatrixSide::kLeft : MatrixSide::kRight,
            .triangular = options.upper ? MatrixTriangular::kUpper
                                        : MatrixTriangular::kLower,
            .diagonal = options.unitriangular ? DiagonalKind::kUnit
                                              : DiagonalKind::kNonUnit,
            .transpose_a = options.adjoint ? MatrixTranspose::kAdjoint
                                           : MatrixTranspose::kNoTranspose,
        });
  };
}

at::Tensor& AtenLinalgSolveTriangularOut(const at::Tensor& a,
                                         const at::Tensor& b, bool upper,
                                         bool left, bool unitriangular,
                                         at::Tensor& out) {
  TT_KERNEL(
      OpName::kLinalgSolveTriangularOut, param_keys,
      (a, b, upper, left, unitriangular, out), {
        TT_THROW_IF_ERROR(ValidateDimensions(a, b, left));
        TT_CHECK_THROW(a.scalar_type() != c10::ScalarType::BFloat16 &&
                           a.scalar_type() != c10::ScalarType::Half &&
                           !IsIntegral(a),
                       error::kInvalidArgument)
            << "triangular solve not supported for dtype "
            << ToString(a.scalar_type());

        at::ScalarType out_scalar_type =
            c10::promoteTypes(a.scalar_type(), b.scalar_type());

        TT_THROW_IF_ERROR(ValidateOutDtype(out, out_scalar_type,
                                           /*allow_out_dtype_cast=*/false));

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out_scalar_type));

        TT_THROW_IF_ERROR(DispatchOpOut<2>(
            LinalgSolveTriangularBuilder(
                {.upper = upper, .left = left, .unitriangular = unitriangular}),
            {a, b}, out,
            {.out_dtype = out_dtype,
             .out_dims = CopyIntVector(b.sizes()),
             .op_param_cache_keys = std::move(param_keys)}));
        return out;
      });
}

at::Tensor AtenLinalgSolveTriangular(const at::Tensor& a, const at::Tensor& b,
                                     bool upper, bool left,
                                     bool unitriangular) {
  TT_KERNEL(
      OpName::kLinalgSolveTriangular, param_keys,
      (a, b, upper, left, unitriangular), {
        TT_THROW_IF_ERROR(ValidateDimensions(a, b, left));
        TT_CHECK_THROW(a.scalar_type() != c10::ScalarType::BFloat16 &&
                           a.scalar_type() != c10::ScalarType::Half &&
                           !IsIntegral(a),
                       error::kInvalidArgument)
            << "triangular solve not supported for dtype "
            << ToString(a.scalar_type());

        at::ScalarType out_scalar_type =
            c10::promoteTypes(a.scalar_type(), b.scalar_type());

        TT_ASSIGN_OR_THROW(mlir::ElementType out_dtype,
                           ConvertTo<mlir::ElementType>(out_scalar_type));

        TT_ASSIGN_OR_THROW(
            auto result_buffer,
            DispatchOp<2>(
                LinalgSolveTriangularBuilder({.upper = upper,
                                              .left = left,
                                              .unitriangular = unitriangular}),
                {a, b},
                {.out_dtype = out_dtype,
                 .out_dims = CopyIntVector(b.sizes()),
                 .op_param_cache_keys = std::move(param_keys)}));
        return MakeTensor(std::move(result_buffer));
      });
}

}  // namespace torch_tpu

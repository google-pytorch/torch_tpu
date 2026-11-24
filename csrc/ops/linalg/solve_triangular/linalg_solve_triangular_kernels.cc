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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/algorithm/container.h"
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
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
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

// Builds a `stablehlo.triangular_solve` that also supports bounded dynamic
// batch and matrix dimensions. Expects `a` and `b` to have equal rank >= 2 and
// matching batch dimensions; runs the triangular solve at static upper bounds
// when bounded (padding `a` past runtime sizes with an identity diagonal and
// zeroing `b` past runtime sizes), and restores bounded dynamic dimension
// sizes on the result.
absl::StatusOr<mlir::MlirOp> TriangularSolve(
    mlir::MlirOp a, mlir::MlirOp b, const SolveTriangularOptions& options) {
  const mlir::RankedTensorType a_type = GetTensorTypeOrDie(a);
  const mlir::RankedTensorType b_type = GetTensorTypeOrDie(b);
  const int64_t rank = a_type.getRank();
  ABSL_CHECK_GE(rank, 2)  // CRASH_OK=internal invariant.
      << "Expected rank >= 2, got " << rank << ". This is a TorchTPU bug.";
  ABSL_CHECK_EQ(b_type.getRank(), rank)  // CRASH_OK=internal invariant.
      << "Expected operand ranks to match, got " << rank << " and "
      << b_type.getRank() << ". This is a TorchTPU bug.";

  const bool left_side = options.left;
  const bool lower = !options.upper;
  const bool unit_diagonal = options.unitriangular;
  const enum mlir::stablehlo::Transpose transpose_a =
      options.adjoint ? mlir::stablehlo::Transpose::ADJOINT
                      : mlir::stablehlo::Transpose::NO_TRANSPOSE;

  const mlir::stablehlo::Dimensions a_dims = GetDimensions(a);
  const mlir::stablehlo::Dimensions b_dims = GetDimensions(b);
  for (int64_t d = 0; d < rank - 2; ++d) {
    ABSL_CHECK_EQ(a_dims[d].size,  // CRASH_OK=internal invariant.
                  b_dims[d].size)
        << "Expected matching batch dimensions at dimension " << d << ", got "
        << a_dims[d].size << " and " << b_dims[d].size
        << ". This is a TorchTPU bug.";
  }
  const int64_t a_rows = a_dims[rank - 2].size;
  const int64_t a_cols = a_dims[rank - 1].size;
  TT_RET_CHECK(a_rows == a_cols, error::kInvalidArgument)
      << "expected square "
      << (a_type.hasStaticShape() ? "matrix" : "upper-bound matrix")
      << " in triangular solve, got " << a_rows << " and " << a_cols;
  const int64_t m_dim = left_side ? rank - 2 : rank - 1;
  const int64_t b_shared = b_dims[m_dim].size;
  TT_RET_CHECK(a_cols == b_shared, error::kInvalidArgument)
      << "expected "
      << (a_type.hasStaticShape() && b_type.hasStaticShape()
              ? "the shared matrix dimensions"
              : "the upper bounds of the shared matrix dimension")
      << " in triangular solve to match, got " << a_cols << " and " << b_shared;

  mlir::MlirBuilder& builder = a.getBuilder();
  if (a_type.hasStaticShape() && b_type.hasStaticShape()) {
    return builder.create<mlir::stablehlo::TriangularSolveOp>(
        a.getValue(), b.getValue(), left_side, lower, unit_diagonal,
        transpose_a);
  }

  const StaticAtBounds static_a = MakeStaticAtBounds(a);
  const StaticAtBounds static_b = MakeStaticAtBounds(b);

  // When `a` has bounded dimensions, elements past its runtime sizes are
  // undefined. Fill them with the identity matrix (rather than zeros) so that:
  // 1) out-of-bounds diagonal entries are 1 instead of 0 (avoiding division by
  //    zero in non-unit solves), and
  // 2) out-of-bounds off-diagonal entries are 0, decoupling the padded block
  //    from the in-bounds system (e.g., during back-substitution from the last
  //    row/column) when paired with zero-padded `b`.
  // When `a` is static (`!a_has_dynamic`) and only `b` is bounded, `a` has no
  // out-of-bounds elements and can be used directly.
  mlir::MlirOp padded_a = static_a.op;
  const bool a_has_dynamic = absl::c_any_of(
      static_a.runtime_sizes,
      [](const std::optional<mlir::MlirOp>& sz) { return sz.has_value(); });
  if (a_has_dynamic) {
    mlir::MlirOp row_idx = mlir::stablehlo::IotaLike(static_a.op, rank - 2,
                                                     mlir::ElementType::I32);
    mlir::MlirOp col_idx = mlir::stablehlo::IotaLike(static_a.op, rank - 1,
                                                     mlir::ElementType::I32);
    mlir::MlirOp is_diag = mlir::stablehlo::Compare(
        row_idx, col_idx, mlir::stablehlo::ComparisonDirection::EQ);
    mlir::MlirOp one = MakeConstantLike(static_a.op, 1);
    mlir::MlirOp zero = MakeConstantLike(static_a.op, 0);
    mlir::MlirOp identity_a = mlir::stablehlo::Select(is_diag, one, zero);
    padded_a =
        FillPastRuntimeSizes(static_a.op, identity_a, static_a.runtime_sizes);
  }

  mlir::MlirOp zeroed_b = static_b.op;
  const bool b_has_dynamic = absl::c_any_of(
      static_b.runtime_sizes,
      [](const std::optional<mlir::MlirOp>& sz) { return sz.has_value(); });
  if (b_has_dynamic) {
    zeroed_b = FillPastRuntimeSizes(
        static_b.op, MakeConstantLike(static_b.op, 0), static_b.runtime_sizes);
  }
  mlir::MlirOp result = builder.create<mlir::stablehlo::TriangularSolveOp>(
      padded_a.getValue(), zeroed_b.getValue(), left_side, lower, unit_diagonal,
      transpose_a);

  std::vector<std::optional<mlir::MlirOp>> result_runtime_sizes =
      static_b.runtime_sizes;
  for (int64_t d = 0; d < rank - 2; ++d) {
    if (!result_runtime_sizes[d].has_value()) {
      result_runtime_sizes[d] = static_a.runtime_sizes[d];
    }
  }
  if (!result_runtime_sizes[m_dim].has_value()) {
    result_runtime_sizes[m_dim] = static_a.runtime_sizes[rank - 2].has_value()
                                      ? static_a.runtime_sizes[rank - 2]
                                      : static_a.runtime_sizes[rank - 1];
  }
  return RestoreDynamicDimSizes(result, result_runtime_sizes);
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
    return TriangularSolve(a_op, b_op, options);
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

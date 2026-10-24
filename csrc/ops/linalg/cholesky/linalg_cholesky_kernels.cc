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

#include "csrc/ops/linalg/cholesky/linalg_cholesky_kernels.h"

#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "ATen/ops/_linalg_check_errors.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

enum class CholeskyTriangularMode {
  kLower,
  kUpper,
};

absl::Status ValidateLinalgCholeskyInputs(const at::Tensor& self) {
  TT_RET_CHECK(self.dim() >= 2, error::kInvalidArgument)
      << "expected input tensor to have at least 2 dimensions, got "
      << self.dim();
  const int64_t m = self.size(-2);
  const int64_t n = self.size(-1);
  TT_RET_CHECK(m == n, error::kInvalidArgument)
      << "expected input tensor's last 2 dimensions to be equal, got [" << m
      << ", " << n << "]";
  const at::ScalarType dtype = self.scalar_type();
  // Match PyTorch's two-stage dtype check: TORCH_META_FUNC(linalg_cholesky_ex)
  // calls checkFloatingOrComplex() unconditionally, while
  // TORCH_IMPL_FUNC(linalg_cholesky_ex_out) early-returns on empty tensors
  // before cholesky_stub rejects low-precision floats.
  const bool is_supported_dtype =
      (dtype == at::kFloat || dtype == at::kDouble ||
       dtype == at::kComplexFloat || dtype == at::kComplexDouble) ||
      (self.numel() == 0 &&
       (c10::isFloatingType(dtype) || c10::isComplexType(dtype)));
  TT_RET_CHECK(is_supported_dtype, error::kInvalidArgument)
      << "expected the input dtype to be float32, float64, complex64, or "
         "complex128, got "
      << ToString(dtype);
  return absl::OkStatus();
}

// Zeroes out the strict opposite triangle of `factor` along its last two
// dimensions using precomputed `row_idx` and `col_idx` Iota tensors.
mlir::MlirOp ZeroOtherTriangle(mlir::MlirOp factor, mlir::MlirOp row_idx,
                               mlir::MlirOp col_idx,
                               CholeskyTriangularMode mode) {
  const mlir::stablehlo::ComparisonDirection dir =
      (mode == CholeskyTriangularMode::kUpper)
          ? mlir::stablehlo::ComparisonDirection::LE
          : mlir::stablehlo::ComparisonDirection::GE;
  mlir::MlirOp mask = mlir::stablehlo::Compare(row_idx, col_idx, dir);
  mlir::MlirOp zero = MakeConstantLike(factor, 0);
  return mlir::stablehlo::Select(mask, factor, zero);
}

// Computes the LAPACK-style `info` tensor of shape (...) with I32 dtype from
// the Cholesky factor `l_op` of shape (..., n, n).
// Returns 0 on success, or the 1-based index of the first non-finite or
// non-positive diagonal entry on failure.
mlir::MlirOp ComputeCholeskyInfo(mlir::MlirOp l_op, mlir::MlirOp row_idx,
                                 mlir::MlirOp col_idx) {
  mlir::MlirBuilder& builder = l_op.getBuilder();
  const mlir::RankedTensorType l_type = GetTensorTypeOrDie(l_op);
  const int64_t rank = l_type.getRank();

  mlir::MlirOp real_l =
      IsComplexType(l_type) ? mlir::stablehlo::Real(l_op) : l_op;
  mlir::MlirOp is_finite = mlir::stablehlo::IsFinite(real_l);
  if (IsComplexType(l_type)) {
    mlir::MlirOp imag_l = mlir::stablehlo::Imag(l_op);
    mlir::MlirOp imag_is_finite = mlir::stablehlo::IsFinite(imag_l);
    is_finite = mlir::stablehlo::And(is_finite, imag_is_finite);
  }
  mlir::MlirOp zero_real = MakeConstantLike(real_l, 0);
  mlir::MlirOp is_positive = mlir::stablehlo::Compare(
      real_l, zero_real, mlir::stablehlo::ComparisonDirection::GT);
  mlir::MlirOp is_valid = mlir::stablehlo::And(is_finite, is_positive);
  mlir::MlirOp is_invalid = mlir::stablehlo::Not(is_valid);
  mlir::MlirOp on_diag = mlir::stablehlo::Compare(
      row_idx, col_idx, mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp is_bad_diag = mlir::stablehlo::And(on_diag, is_invalid);

  constexpr int32_t kSentinel = std::numeric_limits<int32_t>::max();
  mlir::MlirOp one = MakeConstantLike(row_idx, 1);
  mlir::MlirOp one_based_row = mlir::stablehlo::Add(row_idx, one);
  mlir::MlirOp sentinel = MakeConstantLike(row_idx, kSentinel);
  mlir::MlirOp bad_idx =
      mlir::stablehlo::Select(is_bad_diag, one_based_row, sentinel);

  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  mlir::MlirOp init_val = MakeScalarConstant(builder, kSentinel, i32_type);
  const auto min_reduce_builder = [i32_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::MinOp>(
        i32_type, rb.getRegion(), rb.getOpBuilder());
  };
  mlir::MlirOp min_bad_idx = mlir::stablehlo::Reduce(
      builder, {bad_idx}, /*init_value=*/{init_val}, min_reduce_builder,
      /*dimensions=*/{rank - 2, rank - 1})[0];

  mlir::MlirOp sentinel_batch = MakeConstantLike(min_bad_idx, kSentinel);
  mlir::MlirOp has_error = mlir::stablehlo::Compare(
      min_bad_idx, sentinel_batch, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp zero_batch = MakeConstantLike(min_bad_idx, 0);
  return mlir::stablehlo::Select(has_error, min_bad_idx, zero_batch);
}

absl::StatusOr<MlirOpResults<2>> BuildCholeskyExShlo(
    mlir::MlirOp a, CholeskyTriangularMode mode) {
  mlir::MlirBuilder& builder = a.getBuilder();
  const mlir::RankedTensorType a_type = GetTensorTypeOrDie(a);
  const int64_t rank = a_type.getRank();

  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp row_idx,
                      Iota(a, /*iota_dimension=*/rank - 2, i32_type));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp col_idx,
                      Iota(a, /*iota_dimension=*/rank - 1, i32_type));

  const bool lower = (mode == CholeskyTriangularMode::kLower);
  mlir::MlirOp factor =
      (a_type.hasStaticShape() && a_type.getNumElements() == 0)
          ? MakeConstantLike(a, 0)
          : ZeroOtherTriangle(mlir::stablehlo::Cholesky(a, lower), row_idx,
                              col_idx, mode);
  mlir::MlirOp info = ComputeCholeskyInfo(factor, row_idx, col_idx);
  return MlirOpResults<2>{factor, info};
}

absl::Status LinalgCholeskyExOut(const at::Tensor& self,
                                 CholeskyTriangularMode mode,
                                 OpParamCacheKeys param_keys, at::Tensor& l,
                                 at::Tensor& info) {
  TT_RETURN_IF_ERROR(ValidateLinalgCholeskyInputs(self));
  TT_RETURN_IF_ERROR(
      ValidateOutDtype(l, self.scalar_type(), /*allow_out_dtype_cast=*/false));
  TT_RETURN_IF_ERROR(
      ValidateOutDtype(info, at::kInt, /*allow_out_dtype_cast=*/false));

  const Dimensions batch_dims(self.sizes().begin(), self.sizes().end() - 2);
  // Reset `l` and `info` to contiguous strides
  // (TORCH_META_FUNC(linalg_cholesky_ex) pre-allocates `l` with
  // Fortran-contiguous column-major strides, which otherwise causes
  // AssignBufferToAtTensor to insert an inverse-view TransposePrimitive that
  // corrupts COMPLEXF64 on XLA:TPU).
  TT_RETURN_IF_ERROR(ResizeTensor(l, self.sizes()));
  TT_RETURN_IF_ERROR(ResizeTensor(info, batch_dims));

  auto op_builder =
      [mode](mlir::MlirOp self_op) -> absl::StatusOr<MlirOpResults<2>> {
    return BuildCholeskyExShlo(self_op, mode);
  };

  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(self.scalar_type()));

  return DispatchOpOut<1, 2>(std::move(op_builder), self, {l, info},
                             {.out_dtypes = {out_dtype, mlir::ElementType::I32},
                              .out_dims_list = {self.sizes(), batch_dims},
                              .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

std::tuple<at::Tensor&, at::Tensor&> AtenLinalgCholeskyExL(
    const at::Tensor& self, bool upper, bool check_errors, at::Tensor& l,
    at::Tensor& info) {
  TT_KERNEL(OpName::kLinalgCholeskyExL, param_keys,
            (self, upper, IgnoreInCacheKey(check_errors, "Doesn't affect SHLO"),
             l, info),
            {
              const CholeskyTriangularMode mode =
                  upper ? CholeskyTriangularMode::kUpper
                        : CholeskyTriangularMode::kLower;
              TT_THROW_IF_ERROR(LinalgCholeskyExOut(
                  self, mode, std::move(param_keys), l, info));
            });
  if (check_errors) {
    at::_linalg_check_errors(info, "linalg.cholesky_ex",
                             /*is_matrix=*/self.dim() == 2);
  }
  return {l, info};
}

}  // namespace torch_tpu

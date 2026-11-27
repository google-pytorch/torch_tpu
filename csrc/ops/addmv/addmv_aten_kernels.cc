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

#include "csrc/ops/addmv/addmv_aten_kernels.h"

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
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
#include "csrc/ops/precision_context.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

namespace stablehlo = mlir::stablehlo;

// Matrix-vector dot product helper, accumulating in `compute_dtype`.
[[nodiscard]] mlir::MlirOp MatVecDot(mlir::MlirOp mat_op, mlir::MlirOp vec_op,
                                     mlir::ElementType compute_dtype,
                                     mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType mat_type = GetTensorTypeOrDie(mat_op);
  const mlir::RankedTensorType vec_type = GetTensorTypeOrDie(vec_op);

  // XLA doesn't support matmuls with i64, so we convert them to f64.
  const bool is_any_i64 = mat_type.getElementType().isInteger(64) ||
                          vec_type.getElementType().isInteger(64);
  if (is_any_i64) {
    mat_op = stablehlo::ConvertElementType(mat_op, mlir::ElementType::F64);
    vec_op = stablehlo::ConvertElementType(vec_op, mlir::ElementType::F64);
  }

  mlir::MlirOp mv_op = stablehlo::DotGeneral(
      mat_op, vec_op,
      /*dotDimsAttr=*/
      stablehlo::DotDimensionNumbersAttr::get(
          &mat_op.getContext(), /*lhs_batch_dimensions=*/{},
          /*rhs_batch_dimensions=*/{}, /*lhs_contracting_dimensions=*/{1},
          /*rhs_contracting_dimensions=*/{0}),
      /*precisionConfig=*/
      stablehlo::PrecisionConfigAttr::get(&mat_op.getContext(),
                                          {precision, precision}),
      /*preferredElementType=*/
      is_any_i64 ? std::nullopt
                 : std::optional<mlir::ElementType>(compute_dtype));

  if (is_any_i64) {
    mv_op = stablehlo::ConvertElementType(mv_op, compute_dtype);
  }
  return mv_op;
}

absl::StatusOr<mlir::MlirOp> BuildAddmvShlo(
    std::optional<mlir::MlirOp> self_op, mlir::MlirOp mat_op,
    mlir::MlirOp vec_op, std::optional<mlir::MlirOp> beta_op,
    std::optional<mlir::MlirOp> alpha_op, const MaybePromotedScalar& beta,
    const MaybePromotedScalar& alpha, mlir::ElementType out_dtype,
    mlir::stablehlo::Precision precision) {
  // Like CUDA and baddbmm, accumulate, scale, and add in the computation dtype
  // (float32 for float16 and bfloat16) and convert to out_dtype only at the
  // end. Cast scalar ops with CastIfNeeded because MakeBuffer() may downgrade
  // complex128 scalars to complex64 in eager mode.
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(out_dtype));
  const mlir::stablehlo::Dimensions out_dims = {GetDimensions(mat_op)[0]};

  // Case 1: beta is zero. Result is alpha * (mat @ vec); self is ignored.
  if (beta.IsZero()) {
    if (alpha.IsZero()) {
      TT_ASSIGN_OR_RETURN(
          mlir::MlirOp zero,
          MakeConstant(mat_op.getBuilder(), at::Scalar(0), out_dtype));
      return BroadcastIfNeeded(zero, out_dims);
    }
    mlir::MlirOp mv_op = MatVecDot(mat_op, vec_op, compute_dtype, precision);
    if (!alpha.IsOne()) {
      TT_ASSIGN_OR_RETURN(mlir::MlirOp alpha_cast,
                          CastIfNeeded(*alpha_op, compute_dtype));
      TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_alpha,
                          BroadcastIfNeeded(alpha_cast, mv_op));
      mv_op = stablehlo::Mul(broadcasted_alpha, mv_op);
    }
    return CastIfNeeded(mv_op, out_dtype);
  }

  // Case 2: beta is non-zero. self_op is guaranteed to have a value.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp self_cast,
                      CastIfNeeded(*self_op, compute_dtype));
  if (!beta.IsOne()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp beta_cast,
                        CastIfNeeded(*beta_op, compute_dtype));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_beta,
                        BroadcastIfNeeded(beta_cast, self_cast));
    self_cast = stablehlo::Mul(broadcasted_beta, self_cast);
  }

  // Case 2a: alpha is zero. Result is beta * self; mat @ vec is ignored.
  if (alpha.IsZero()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_bias,
                        BroadcastIfNeeded(self_cast, out_dims));
    return CastIfNeeded(broadcasted_bias, out_dtype);
  }

  // Case 2b: both beta and alpha are non-zero.
  mlir::MlirOp mv_op = MatVecDot(mat_op, vec_op, compute_dtype, precision);
  if (!alpha.IsOne()) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp alpha_cast,
                        CastIfNeeded(*alpha_op, compute_dtype));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_alpha,
                        BroadcastIfNeeded(alpha_cast, mv_op));
    mv_op = stablehlo::Mul(broadcasted_alpha, mv_op);
  }
  TT_ASSIGN_OR_RETURN(mlir::MlirOp broadcasted_bias,
                      BroadcastIfNeeded(self_cast, mv_op));
  return CastIfNeeded(stablehlo::Add(broadcasted_bias, mv_op), out_dtype);
}

absl::Status Addmv(const at::Tensor& self, const at::Tensor& mat,
                   const at::Tensor& vec, MaybePromotedScalar beta,
                   MaybePromotedScalar alpha, at::Tensor& out,
                   OpParamCacheKeys& param_keys) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));

  // BuildAddmvShlo() scales in the computation dtype, so materialize beta and
  // alpha in that dtype instead of rounding them to out_dtype on the host.
  const at::ScalarType compute_scalar_type =
      InferComputationDtype(out.scalar_type());
  const bool beta_is_zero = beta.IsZero();

  // If beta is zero, the content of self is not used, so we skip passing it.
  // Because beta is zero, PyTorch passes an uninitialized tensor to self as an
  // optimization (e.g. torch.mv), and we do not allow reading from
  // uninitialized tensors.
  std::vector<at::Tensor> inputs;
  inputs.reserve(5);
  if (!beta_is_zero) {
    inputs.push_back(self);
  }
  inputs.push_back(mat);
  inputs.push_back(vec);

  const bool has_beta_tensor = !beta.ValueMatchesExclude();
  if (has_beta_tensor) {
    TT_ASSIGN_OR_RETURN(at::Tensor beta_tensor,
                        beta.GetTensor(compute_scalar_type));
    inputs.push_back(std::move(beta_tensor));
  }
  const bool has_alpha_tensor = !alpha.ValueMatchesExclude();
  if (has_alpha_tensor) {
    TT_ASSIGN_OR_RETURN(at::Tensor alpha_tensor,
                        alpha.GetTensor(compute_scalar_type));
    inputs.push_back(std::move(alpha_tensor));
  }

  const mlir::stablehlo::Precision precision = GetAndAddPrecisionTo(param_keys);
  Dimensions result_shape = {mat.size(0)};
  DispatchOpOptions<1> options = {.out_dtype = out_dtype,
                                  .out_dims = std::move(result_shape),
                                  .op_param_cache_keys = std::move(param_keys)};

  auto op_builder =
      [beta = std::move(beta), alpha = std::move(alpha), beta_is_zero,
       has_beta_tensor, has_alpha_tensor, out_dtype, precision](
          absl::Span<mlir::MlirOp> inputs_op,
          mlir::MlirBuilder& /*builder*/) -> absl::StatusOr<mlir::MlirOp> {
    size_t idx = 0;
    std::optional<mlir::MlirOp> self_op;
    if (!beta_is_zero) {
      self_op = inputs_op[idx++];
    }
    mlir::MlirOp mat_op = inputs_op[idx++];
    mlir::MlirOp vec_op = inputs_op[idx++];
    std::optional<mlir::MlirOp> beta_op;
    if (has_beta_tensor) {
      beta_op = inputs_op[idx++];
    }
    std::optional<mlir::MlirOp> alpha_op;
    if (has_alpha_tensor) {
      alpha_op = inputs_op[idx++];
    }
    return BuildAddmvShlo(self_op, mat_op, vec_op, beta_op, alpha_op, beta,
                          alpha, out_dtype, precision);
  };
  return DispatchOpOut<kDynamicSize>(std::move(op_builder), inputs, out,
                                     std::move(options));
}

absl::Status ValidateAddmvInputs(const at::Tensor& self, const at::Tensor& mat,
                                 const at::Tensor& vec, const at::Scalar& beta,
                                 const at::Scalar& alpha) {
  TT_RET_CHECK(!IsBool(self), error::kInvalidArgument)
      << "the dtype of the first argument cannot be bool";
  TT_RET_CHECK(self.dim() <= 1, error::kInvalidArgument)
      << "expected the first argument to be a scalar or vector (0D or 1D "
         "tensor), got "
      << self.dim() << "D tensor";

  TT_RET_CHECK(mat.numel() == 0 || !c10::isIntegralType(mat.scalar_type(),
                                                        /*includeBool=*/false),
               error::kPythonNotImplementedError)
      << "not implemented for " << ToString(mat.scalar_type());

  TT_RET_CHECK(!IsBool(mat), error::kInvalidArgument)
      << "expected non-bool dtype for the second argument, got bool";
  TT_RET_CHECK(mat.dim() == 2, error::kInvalidArgument)
      << "expected the second argument to be a matrix (2D tensor), got "
      << mat.dim() << "D tensor";

  TT_RET_CHECK(!IsBool(vec), error::kInvalidArgument)
      << "expected non-bool dtype for the third argument, got bool";
  TT_RET_CHECK(vec.dim() == 1, error::kInvalidArgument)
      << "expected the third argument to be a vector (1D tensor), got "
      << vec.dim() << "D tensor";

  TT_RET_CHECK(IsComplex(self) || !IsComplex(alpha), error::kInvalidArgument)
      << "expected non-complex dtype for alpha with non-complex inputs, got "
      << ToString(alpha.type());

  TT_RET_CHECK(IsComplex(self) || !IsComplex(beta), error::kInvalidArgument)
      << "expected non-complex dtype for beta with non-complex inputs, got "
      << ToString(beta.type());

  if (self.dim() == 1) {
    TT_RET_CHECK(self.size(0) == 1 || self.size(0) == mat.size(0),
                 error::kInvalidArgument)
        << "expected argument shapes " << ToString(self.sizes()) << ", "
        << ToString(mat.sizes()) << ", " << ToString(vec.sizes())
        << " to be compatible, got " << self.size(0) << " vs " << mat.size(0);
  }

  TT_RET_CHECK(mat.size(1) == vec.size(0), error::kInvalidArgument)
      << "expected the last dimension of the second argument (matrix of size ["
      << absl::StrJoin(mat.sizes(), /* separator= */ ", ")
      << "]) to match the first dimension of the third argument (vector of "
         "size ["
      << absl::StrJoin(vec.sizes(), /* separator= */ ", ") << "]), got "
      << mat.size(1) << " vs " << vec.size(0);

  return absl::OkStatus();
}

}  // namespace

at::Tensor& AtenAddmvOut(const at::Tensor& self, const at::Tensor& mat,
                         const at::Tensor& vec, const at::Scalar& beta,
                         const at::Scalar& alpha, at::Tensor& out) {
  MaybePromotedScalar promoted_beta =
      PromoteScalar(beta).AvoidPromoting(ScalarValue::kZero, ScalarValue::kOne);
  MaybePromotedScalar promoted_alpha = PromoteScalar(alpha).AvoidPromoting(
      ScalarValue::kZero, ScalarValue::kOne);
  TT_KERNEL(
      OpName::kAddmvOut, param_keys,
      (self, mat, vec, promoted_beta, promoted_alpha, out), {
        TT_THROW_IF_ERROR(ValidateAddmvInputs(self, mat, vec, beta, alpha));
        TT_THROW_IF_ERROR(Addmv(self, mat, vec, std::move(promoted_beta),
                                std::move(promoted_alpha), out, param_keys));
        return out;
      });
}

}  // namespace torch_tpu

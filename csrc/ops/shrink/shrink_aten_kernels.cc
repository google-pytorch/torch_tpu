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

#include "csrc/ops/shrink/shrink_aten_kernels.h"

#include <limits>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary.h"
#include "csrc/ops/where/where.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

enum class ShrinkKind { kHard, kSoft };

// Whether `input` lies in [-lambd, lambd]. NaN is outside the range.
absl::StatusOr<mlir::MlirOp> BuildInRangeShlo(mlir::MlirOp input,
                                              mlir::MlirOp lambd) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp abs_input, BuildAbsShlo(input));
  return BuildLeShlo(abs_input, lambd);
}

absl::StatusOr<mlir::MlirOp> BuildHardshrinkShlo(mlir::MlirOp input,
                                                 mlir::MlirOp lambd) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType dtype, GetElementType(input));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp in_range, BuildInRangeShlo(input, lambd));
  mlir::MlirOp zero = MakeConstantLike(input, 0.0, dtype);
  return BuildWhereShlo(in_range, zero, input, dtype);
}

absl::StatusOr<mlir::MlirOp> BuildSoftshrinkShlo(mlir::MlirOp input,
                                                 mlir::MlirOp lambd) {
  // softshrink(x) = x - clamp(x, -lambd, lambd); lambd >= 0 is validated.
  // The clamp is written as maximum + minimum, whose binary broadcasts preserve
  // bounded dynamic dimensions.
  TT_ASSIGN_OR_RETURN(mlir::MlirOp neg_lambd, BuildNegShlo(lambd));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp lower, BuildMaximumShlo(input, neg_lambd));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp clamped, BuildMinimumShlo(lower, lambd));
  return BuildSubShlo(input, clamped);
}

// The gradient is zero wherever `self` is in [-lambd, lambd], for both
// hardshrink and softshrink.
absl::StatusOr<mlir::MlirOp> BuildShrinkBackwardShlo(mlir::MlirOp grad_output,
                                                     mlir::MlirOp self,
                                                     mlir::MlirOp lambd) {
  TT_ASSIGN_OR_RETURN(mlir::ElementType dtype, GetElementType(grad_output));
  TT_ASSIGN_OR_RETURN((auto [grad_bcast, self_bcast]),
                      ApplyBroadcastIfNeeded(grad_output, self));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp in_range,
                      BuildInRangeShlo(self_bcast, lambd));
  mlir::MlirOp zero = MakeConstantLike(grad_bcast, 0.0, dtype);
  return BuildWhereShlo(in_range, zero, grad_bcast, dtype);
}

// The largest finite value of `dtype`. PyTorch bounds softshrink's lambd by it.
double MaxFiniteValue(const at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Half:
      return static_cast<double>(std::numeric_limits<c10::Half>::max());
    case at::ScalarType::BFloat16:
      return static_cast<double>(std::numeric_limits<c10::BFloat16>::max());
    case at::ScalarType::Float:
      return std::numeric_limits<float>::max();
    default:  // Double; ValidateShrinkInput rejects every other dtype.
      return std::numeric_limits<double>::max();
  }
}

absl::Status ValidateShrinkInput(const at::ScalarType dtype) {
  TT_RET_CHECK(c10::isFloatingType(dtype), error::kInvalidArgument)
      << "expected the input dtype to be floating point, got "
      << ToString(dtype);
  return ValidateNotLowPrecisionFloat(dtype);
}

// PyTorch converts lambd to the input dtype and rejects a finite value that
// overflows it.
absl::Status ValidateShrinkLambd(const at::ScalarType dtype,
                                 const at::Scalar& lambd) {
  TT_ASSIGN_OR_RETURN(const bool can_cast,
                      CanCastScalarWithoutOverflow(lambd, dtype));
  TT_RET_CHECK(can_cast, error::kInvalidArgument)
      << "expected lambd to be representable as " << ToString(dtype) << ", got "
      << ToString(lambd);
  return absl::OkStatus();
}

// Like PyTorch's softshrink meta, bounds lambd to [0, max] of the input dtype.
absl::Status ValidateSoftshrinkLambd(const at::ScalarType dtype,
                                     const at::Scalar& lambd) {
  const double value = lambd.toDouble();
  const double max_lambd = MaxFiniteValue(dtype);
  // Written so that a NaN lambd is rejected too.
  TT_RET_CHECK(value >= 0 && value <= max_lambd, error::kInvalidArgument)
      << "expected lambd to be in the range [0, " << ToString(max_lambd)
      << "] for input dtype " << ToString(dtype) << ", got " << ToString(lambd);
  return absl::OkStatus();
}

absl::Status ShrinkOut(const at::Tensor& self, PromotedScalar lambd,
                       const ShrinkKind kind, at::Tensor& out,
                       OpParamCacheKeys param_keys) {
  TT_RETURN_IF_ERROR(ValidateShrinkInput(self.scalar_type()));
  TT_RETURN_IF_ERROR(
      kind == ShrinkKind::kSoft
          ? ValidateSoftshrinkLambd(self.scalar_type(), lambd.scalar())
          : ValidateShrinkLambd(self.scalar_type(), lambd.scalar()));
  TT_RETURN_IF_ERROR(ValidateOutDtype(out, self.scalar_type(),
                                      /*allow_out_dtype_cast=*/false));
  TT_ASSIGN_OR_RETURN(const at::Tensor lambd_tensor,
                      lambd.GetTensor(self.scalar_type()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  // Only softshrink does arithmetic; compare/select is exact in bf16/fp16.
  std::optional<mlir::ElementType> computation_dtype;
  if (kind == ShrinkKind::kSoft) {
    TT_ASSIGN_OR_RETURN(computation_dtype, InferComputationDtype(out_dtype));
  }

  auto op_builder = [kind](FixedSizeSpan<mlir::MlirOp, 2> inputs) {
    auto& [input, lambd] = inputs;
    return kind == ShrinkKind::kSoft ? BuildSoftshrinkShlo(input, lambd)
                                     : BuildHardshrinkShlo(input, lambd);
  };
  return DispatchOpOut<2>(std::move(op_builder), {self, lambd_tensor}, out,
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(self.sizes()),
                           .computation_dtype = computation_dtype,
                           .op_param_cache_keys = std::move(param_keys)});
}

absl::Status ShrinkBackwardGradInput(const at::Tensor& grad_output,
                                     const at::Tensor& self,
                                     PromotedScalar lambd,
                                     at::Tensor& grad_input,
                                     OpParamCacheKeys param_keys) {
  // Like CUDA's build_borrowing_binary_op, compute in the promoted dtype.
  const at::ScalarType common_dtype =
      c10::promoteTypes(grad_output.scalar_type(), self.scalar_type());
  TT_RETURN_IF_ERROR(ValidateShrinkInput(common_dtype));
  TT_RETURN_IF_ERROR(ValidateShrinkLambd(common_dtype, lambd.scalar()));
  // Default allow_out_dtype_cast=true matches enforce_safe_casting_to_output.
  TT_RETURN_IF_ERROR(ValidateOutDtype(grad_input, common_dtype));
  TT_ASSIGN_OR_RETURN(const at::Tensor lambd_tensor,
                      lambd.GetTensor(common_dtype));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(grad_input.scalar_type()));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType computation_dtype,
                      InferComputationDtype(out_dtype));

  auto op_builder = [](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
    auto& [grad_output, self, lambd] = inputs;
    return BuildShrinkBackwardShlo(grad_output, self, lambd);
  };
  return DispatchOpOut<3>(std::move(op_builder),
                          {grad_output, self, lambd_tensor}, grad_input,
                          {.out_dtype = out_dtype,
                           .out_dims = CopyIntVector(grad_input.sizes()),
                           .computation_dtype = computation_dtype,
                           .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor& AtenHardshrinkOut(const at::Tensor& self, const at::Scalar& lambd,
                              at::Tensor& out) {
  PromotedScalar promoted_lambd = PromoteScalar(lambd);
  TT_KERNEL(OpName::kHardshrinkOut, param_keys, (self, promoted_lambd, out), {
    TT_THROW_IF_ERROR(ShrinkOut(self, std::move(promoted_lambd),
                                ShrinkKind::kHard, out, std::move(param_keys)));
    return out;
  });
}

at::Tensor& AtenHardshrinkBackwardGradInput(const at::Tensor& grad_output,
                                            const at::Tensor& self,
                                            const at::Scalar& lambd,
                                            at::Tensor& grad_input) {
  PromotedScalar promoted_lambd = PromoteScalar(lambd);
  TT_KERNEL(OpName::kHardshrinkBackwardGradInput, param_keys,
            (grad_output, self, promoted_lambd, grad_input), {
              TT_THROW_IF_ERROR(ShrinkBackwardGradInput(
                  grad_output, self, std::move(promoted_lambd), grad_input,
                  std::move(param_keys)));
              return grad_input;
            });
}

at::Tensor& AtenSoftshrinkOut(const at::Tensor& self, const at::Scalar& lambd,
                              at::Tensor& out) {
  PromotedScalar promoted_lambd = PromoteScalar(lambd);
  TT_KERNEL(OpName::kSoftshrinkOut, param_keys, (self, promoted_lambd, out), {
    TT_THROW_IF_ERROR(ShrinkOut(self, std::move(promoted_lambd),
                                ShrinkKind::kSoft, out, std::move(param_keys)));
    return out;
  });
}

at::Tensor& AtenSoftshrinkBackwardGradInput(const at::Tensor& grad_output,
                                            const at::Tensor& self,
                                            const at::Scalar& lambd,
                                            at::Tensor& grad_input) {
  PromotedScalar promoted_lambd = PromoteScalar(lambd);
  TT_KERNEL(OpName::kSoftshrinkBackwardGradInput, param_keys,
            (grad_output, self, promoted_lambd, grad_input), {
              TT_THROW_IF_ERROR(ShrinkBackwardGradInput(
                  grad_output, self, std::move(promoted_lambd), grad_input,
                  std::move(param_keys)));
              return grad_input;
            });
}

}  // namespace torch_tpu

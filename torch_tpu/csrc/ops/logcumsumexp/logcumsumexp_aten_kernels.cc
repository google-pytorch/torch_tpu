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

#include "torch_tpu/csrc/ops/logcumsumexp/logcumsumexp_aten_kernels.h"

#include <cstdint>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/functional/bind_front.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/scan_builder.h"
#include "torch_tpu/csrc/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

// Associative scan combiner for logcumsumexp: a numerically stable logaddexp.
// logaddexp(a, b) = m + log(exp(a - m) + exp(b - m)) with m = max(a, b). When m
// is non-finite (both operands are -inf, or an input is +/-inf) the closed form
// degenerates to m, matching torch.logaddexp.
absl::StatusOr<ScanBodyResults> LogaddexpScanBody(mlir::OpBuilder& op_builder,
                                                  mlir::Location loc,
                                                  mlir::ValueRange input_slices,
                                                  mlir::Value /*index*/,
                                                  mlir::ValueRange carries) {
  const mlir::Value a = input_slices[0];
  const mlir::Value b = carries[0];
  const mlir::Value m =
      mlir::stablehlo::MaxOp::create(op_builder, loc, a, b).getResult();
  const mlir::Value exp_a =
      mlir::stablehlo::ExpOp::create(
          op_builder, loc,
          mlir::stablehlo::SubtractOp::create(op_builder, loc, a, m)
              .getResult())
          .getResult();
  const mlir::Value exp_b =
      mlir::stablehlo::ExpOp::create(
          op_builder, loc,
          mlir::stablehlo::SubtractOp::create(op_builder, loc, b, m)
              .getResult())
          .getResult();
  const mlir::Value log_sum =
      mlir::stablehlo::LogOp::create(
          op_builder, loc,
          mlir::stablehlo::AddOp::create(op_builder, loc, exp_a, exp_b)
              .getResult())
          .getResult();
  const mlir::Value safe =
      mlir::stablehlo::AddOp::create(op_builder, loc, m, log_sum).getResult();
  const mlir::Value finite =
      mlir::stablehlo::IsFiniteOp::create(op_builder, loc, m).getResult();
  llvm::SmallVector<mlir::Value> out = {
      mlir::stablehlo::SelectOp::create(op_builder, loc, finite, safe, m)
          .getResult()};
  // The per-position output is the running logsumexp carry.
  return ScanBodyResults{out, out};
}

// Lowers logcumsumexp along 'normalized_dim' to an associative chlo.ScanOp
// (native scan emitter). The combiner is a numerically stable logaddexp and the
// scan identity is -inf.
absl::StatusOr<mlir::MlirOp> BuildLogcumsumexpShlo(const int64_t normalized_dim,
                                                   mlir::MlirOp input) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  // logcumsumexp of a 0-dim (scalar) tensor is the scalar itself. Return the
  // input directly so the lowering stays pure SHLO; routing the scalar case
  // through an aten clone/copy would trip the composite-op check for this
  // non-whitelisted op.
  if (input_type.getRank() == 0) {
    return input;
  }
  const mlir::ElementType input_elem_type = GetElementTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(input_elem_type));
  TT_ASSIGN_OR_RETURN(input, CastIfNeeded(input, compute_dtype));

  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::RankedTensorType compute_input_type = GetTensorTypeOrDie(input);
  const mlir::Type compute_element_type = compute_input_type.getElementType();

  // chlo.ScanOp carries are rank-reduced (the scan dimension is erased).
  llvm::SmallVector<int64_t> carry_shape(compute_input_type.getShape().begin(),
                                         compute_input_type.getShape().end());
  carry_shape.erase(carry_shape.begin() + normalized_dim);

  // The identity of logaddexp is -inf.
  const llvm::APFloat neg_inf = llvm::APFloat::getInf(
      llvm::cast<mlir::FloatType>(compute_element_type).getFloatSemantics(),
      /*Negative=*/true);
  const mlir::Attribute neg_inf_attr =
      mlir::FloatAttr::get(compute_element_type, neg_inf);
  const mlir::MlirOp neg_inf_scalar = mlir::stablehlo::Constant(
      builder,
      mlir::DenseElementsAttr::get(
          mlir::RankedTensorType::get({}, compute_element_type), neg_inf_attr));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp carry_init,
                      BroadcastIfNeeded(neg_inf_scalar, carry_shape));

  // Associative scan -> chlo.ScanOp (native scan emitter). Results are
  // [carries..., outputs...]; the prefix-scan output is the single output. The
  // combiner is the numerically stable logaddexp in LogaddexpScanBody.
  TT_ASSIGN_OR_RETURN(
      const DynamicMlirOpResults results,
      BuildScanShlo(
          builder, {input}, normalized_dim, /*num_scan_inputs=*/1,
          /*carry_inits=*/{carry_init}, /*output_inits=*/{input},
          LogaddexpScanBody,
          ScanOptions{.should_squeeze = true, .is_associative = true}));
  return CastIfNeeded(results[/*num_carries=*/1], input_elem_type);
}

}  // namespace

at::Tensor AtenLogcumsumexp(const at::Tensor& self, const int64_t dim) {
  TT_KERNEL(OpName::kLogcumsumexp, param_keys, (self, dim), {
    TT_CHECK_THROW(self.is_floating_point(), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << ToString(self.scalar_type());
    // SafeWrapDim is only valid for rank >= 1; for a 0-dim scalar the builder
    // ignores the dim and returns the identity.
    int64_t normalized_dim = dim;
    if (self.dim() > 0) {
      TT_ASSIGN_OR_THROW(normalized_dim, SafeWrapDim(dim, self.dim()));
    }
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        UnaryOp(self, absl::bind_front(BuildLogcumsumexpShlo, normalized_dim),
                {.op_param_cache_keys = std::move(param_keys)}));
    return result;
  });
}

at::Tensor& AtenLogcumsumexpOut(const at::Tensor& self, const int64_t dim,
                                at::Tensor& out) {
  TT_KERNEL(OpName::kLogcumsumexpOut, param_keys, (self, dim, out), {
    // The out variant is unreachable in eager mode: functionalization rewrites
    // _logcumsumexp.out into the functional _logcumsumexp plus a copy, so the
    // identical check in AtenLogcumsumexp fires first. Kept as a defensive
    // check for non-functionalized paths.
    TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=functionalization rewrites .out to
                     // the functional _logcumsumexp, whose identical check
                     // fires first
        self.is_floating_point(), error::kInvalidArgument)
        << "expected the input dtype to be floating point, got "
        << ToString(self.scalar_type());
    int64_t normalized_dim = dim;
    if (self.dim() > 0) {
      TT_ASSIGN_OR_THROW(normalized_dim, SafeWrapDim(dim, self.dim()));
    }
    TT_THROW_IF_ERROR(UnaryOpOut(
        self, out, absl::bind_front(BuildLogcumsumexpShlo, normalized_dim),
        {.op_param_cache_keys = std::move(param_keys)}));
    return out;
  });
}

}  // namespace torch_tpu

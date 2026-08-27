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

#include "torch_tpu/csrc/ops/nan_to_num/nan_to_num_aten_kernels.h"

#include <limits>
#include <optional>
#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch_tpu/csrc/common/aten_utils.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "torch_tpu/csrc/common/to_string.h"
#include "torch_tpu/csrc/ops/macros/kernel.h"
#include "torch_tpu/csrc/ops/op_builder_utils.h"
#include "torch_tpu/csrc/ops/op_names.h"
#include "torch_tpu/csrc/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

// Lower torch.nan_to_num to StableHLO select and compare chains.
// If the input is not a floating-point or complex type, it is returned
// unmodified. Otherwise, NaNs are replaced with 'nan', positive infinities are
// replaced with 'posinf', and negative infinities are replaced with 'neginf'.
// If 'posinf' or 'neginf' are not specified, they default to the maximum and
// minimum representable finite values for the tensor's specific float type.
absl::StatusOr<mlir::MlirOp> BuildNanToNumShlo(mlir::MlirOp input,
                                               std::optional<double> nan,
                                               std::optional<double> posinf,
                                               std::optional<double> neginf) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);

  if (IsComplexType(input_type)) {
    // Process complex inputs component-wise (real and imaginary parts).
    auto real = mlir::stablehlo::Real(input);
    auto imag = mlir::stablehlo::Imag(input);

    TT_ASSIGN_OR_RETURN(auto real_processed,
                        BuildNanToNumShlo(real, nan, posinf, neginf));
    TT_ASSIGN_OR_RETURN(auto imag_processed,
                        BuildNanToNumShlo(imag, nan, posinf, neginf));

    return mlir::stablehlo::Complex(real_processed, imag_processed);
  }

  if (!IsFloatType(input_type)) {
    // Non-floating point, non-complex inputs are returned as a safe no-op.
    return input;
  }

  auto element_type = input_type.getElementType();

  // Determine target replacement values.
  // We use LLVM's APFloat (via op_builder_utils) to determine exact float
  // bounds dynamically to prevent type-specific overflow that occurs when
  // casting a generic double max (1.79e308) down to a lower-precision float.
  double nan_val = nan.value_or(0.0);
  double posinf_val =
      posinf.has_value() ? posinf.value() : GetMaxFiniteDouble(element_type);
  double neginf_val =
      neginf.has_value() ? neginf.value() : GetMinFiniteDouble(element_type);

  auto nan_cst = MakeConstantLike(input, nan_val);
  auto posinf_cst = MakeConstantLike(input, posinf_val);
  auto neginf_cst = MakeConstantLike(input, neginf_val);

  // Constants for infinite comparisons
  auto inf_cst =
      MakeConstantLike(input, std::numeric_limits<double>::infinity());
  auto neginf_cst_comp =
      MakeConstantLike(input, -std::numeric_limits<double>::infinity());

  // Detect NaNs: according to IEEE-754, NaN != NaN is the only state where
  // a value is not equal to itself.
  auto is_nan = mlir::stablehlo::Compare(
      input, input, mlir::stablehlo::ComparisonDirection::NE);

  // Detect PosInf: x == +infinity
  auto is_posinf = mlir::stablehlo::Compare(
      input, inf_cst, mlir::stablehlo::ComparisonDirection::EQ);

  // Detect NegInf: x == -infinity
  auto is_neginf = mlir::stablehlo::Compare(
      input, neginf_cst_comp, mlir::stablehlo::ComparisonDirection::EQ);

  // Apply the replacements element-wise via nested selects.
  // Note: Since these three states (NaN, +Inf, -Inf) are mutually exclusive,
  // the nesting order has no impact on correctness.
  auto result = mlir::stablehlo::Select(is_neginf, neginf_cst, input);
  result = mlir::stablehlo::Select(is_posinf, posinf_cst, result);
  result = mlir::stablehlo::Select(is_nan, nan_cst, result);

  return result;
}

}  // namespace

at::Tensor& AtenNanToNumOut(const at::Tensor& self, std::optional<double> nan,
                            std::optional<double> posinf,
                            std::optional<double> neginf, at::Tensor& out) {
  TT_KERNEL(OpName::kNanToNumOut, param_keys, (self, nan, posinf, neginf, out),
            {
              auto op_builder = [nan, posinf, neginf](mlir::MlirOp input) {
                return BuildNanToNumShlo(input, nan, posinf, neginf);
              };

              TT_THROW_IF_ERROR(
                  UnaryOpOut(self, out, op_builder,
                             {.op_param_cache_keys = std::move(param_keys)}));
              return out;
            });
}

}  // namespace torch_tpu

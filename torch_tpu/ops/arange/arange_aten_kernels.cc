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

#include "torch_tpu/ops/arange/arange_aten_kernels.h"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/native/RangeUtils.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/arange/arange.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/nullary_aten_kernels.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "xla/xla_data.pb.h"

namespace torch_tpu {

namespace {

// Check that the inputs are valid for the arange operation.
//
// It makes sure that:
// - step is non-zero
// - The bounds are consistent with the step sign
// - If the bounds are floating-point, they are finite
//
// We use need to use a template here because RangeBound can be either a double
// or a 64-bit integer. Using only one of them might result in a check failure
// on valid inputs.
//
// We fix step as double precision since we just check whether its is larger or
// smaller than 0, i.e. int64_t to double conversion issue is not a concern
// here.
template <typename RangeBound>
absl::Status CheckArangeInputs(const RangeBound start, const RangeBound end,
                               const double step) {
  TT_RET_CHECK(step != 0, error::kInvalidArgument) << "step must be non-zero";

  if constexpr (std::is_same_v<RangeBound, double>) {
    TT_RET_CHECK(std::isfinite(start) && std::isfinite(end),
                 error::kInvalidArgument)
        << "expected [start, end) interval to have finite bounds, got ["
        << start << ", " << end << ")";
  }

  if (end > start) {
    TT_RET_CHECK(step > 0, error::kInvalidArgument)
        << "expected step to be positive since start (" << start << ") < end ("
        << end << "), got step=" << step;
  } else if (end < start) {
    TT_RET_CHECK(step < 0, error::kInvalidArgument)
        << "expected step to be negative since start (" << start << ") > end ("
        << end << "), got step=" << step;
  }

  return absl::OkStatus();
}

// This implementation is based on the PyTorch compute_arange_size() function.
// Ref: aten/src/ATen/native/RangeUtils.h
//
// Instead of calling compute_arange_size() directly, we inline its content
// here. The reasons for doing so are:
//
// - It re-checks the inputs validity (already checked above)
// - It might throw an error, instead of cleanly propagating a status value
// - It is small and simple enough, we can replicate it here without adding
//   much complexity
//
// Known limitations:
// - Large integer bounds for an output of at::kDouble dtype results in an
//   incorrect result due to precision loss
//
// Note: PyTorch CPU/GPU behaves the same way.
int64_t ComputeArangeNumElements(const at::Scalar& start, const at::Scalar& end,
                                 const at::Scalar& step,
                                 const at::ScalarType output_dtype) {
  if (output_dtype == at::kLong) {
    // Special case for kLong (64-bit integer).
    //
    // Due to the higher precision of kLong, we need to handle this case
    // separately. When the output dtype is kLong, we might end up mapping
    // different values to the same floating point value.
    const int64_t step_i64 = step.toLong();
    const int64_t sgn = (step_i64 > 0) - (step_i64 < 0);
    return (end.toLong() - start.toLong() + step_i64 - sgn) / step_i64;
  } else {
    // For all other cases, we can use the double precision, since there is no
    // conversion issue.
    return std::ceil((end.toDouble() - start.toDouble()) / step.toDouble());
  }
}

absl::StatusOr<int64_t> GetArangeNumElements(
    const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
    const at::ScalarType output_dtype) {
  ABSL_VLOG(1) << "[GetArangeNumElements] start_val=" << start.toDouble()
               << ", end_val=" << end.toDouble()
               << ", step_val=" << step.toDouble()
               << ", dtype=" << output_dtype;

  const bool bounds_are_integral =
      start.isIntegral(true) && end.isIntegral(true);
  const double step_f64 = step.toDouble();

  // Watch out for int64_t to double conversion issue.
  //
  // In order to avoid issues such as 2^53 and 2^53+1 being mapped to the same
  // double value, we need to check whether both bounds are integer numbers. If
  // they are, we run the checks with their integer values. Otherwise, we
  // fallback to their double value.
  if (bounds_are_integral) {
    TT_RETURN_IF_ERROR(
        CheckArangeInputs(start.toLong(), end.toLong(), step_f64));
  } else {
    TT_RETURN_IF_ERROR(
        CheckArangeInputs(start.toDouble(), end.toDouble(), step_f64));
  }

  // From this point on, we know that the inputs are valid. In other words:
  // - step is non-zero
  // - step value is consistent with start and end bounds

  const int64_t num_elements =
      ComputeArangeNumElements(start, end, step, output_dtype);

  ABSL_CHECK_GE(num_elements, 0)  // CRASH_OK
      << "expected number of elements to be non-negative with inputs start="
      << start << " end=" << end << " step=" << step << ", got "
      << num_elements;

  ABSL_VLOG(1) << "[GetArangeNumElements] Calculated num_elements: "
               << num_elements;
  return num_elements;
}

absl::StatusOr<MlirNullaryOpBuilder> GetArangeOpBuilder(
    const int64_t num_elements, const at::Scalar start, const at::Scalar step,
    const at::ScalarType output_dtype) {
  TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=conversion error of
                        // output_dtype is first surfaced by the empty()
                        // operation.
      const auto output_mlir_type, ConvertTo<mlir::ElementType>(output_dtype));

  return [num_elements, output_mlir_type, start,
          step](mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=output_mlir_type comes from a
                          // successful conversion of output_dtype to
                          // ElementType.
        const mlir::Type mlir_type,
        GetMlirType(builder.getContext(), output_mlir_type));

    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=dtype of at::Scalar that comes
                          // from Python can always be converted to an
                          // ElementType.
        const auto start_constant, MakeConstant(builder, start));
    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=dtype of at::Scalar that comes
                          // from Python can always be converted to an
                          // ElementType.
        const auto step_constant, MakeConstant(builder, step));

    return torch_tpu::BuildArangeOp(num_elements, mlir_type, start_constant,
                                    step_constant);
  };
}

}  // namespace

at::Tensor& AtenArangeStartOut(const at::Scalar& start, const at::Scalar& end,
                               const at::Scalar& step, at::Tensor& out) {
  TT_KERNEL(OpName::kArangeStartOut, param_keys, (start, end, step, out), {
    const at::ScalarType output_dtype = out.scalar_type();

    TT_ASSIGN_OR_THROW(const auto num_elements,
                       GetArangeNumElements(start, end, step, output_dtype));

    TT_ASSIGN_OR_THROW(  // ERROR_COV_INFEASIBLE=inner checks are all
                         // infeasible.
        auto op_builder,
        GetArangeOpBuilder(num_elements, start, step, output_dtype));

    TT_THROW_IF_ERROR(  // ERROR_COV_INFEASIBLE=inner checks are all
                        // infeasible.
        ApplyNullaryOpOut(out, std::move(op_builder), output_dtype,
                          {num_elements}, std::move(param_keys)));
  });
  return out;
}

}  // namespace torch_tpu

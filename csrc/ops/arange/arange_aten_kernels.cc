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

#include "csrc/ops/arange/arange_aten_kernels.h"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/native/RangeUtils.h"
#include "ATen/native/Resize.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/arange/arange.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "mlir/IR/Types.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
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
absl::Status ValidateArangeInputs(const RangeBound start, const RangeBound end,
                                  const double step) {
  TT_RET_CHECK(step != 0, error::kInvalidArgument)
      << "expected step to be non-zero, got " << step;

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

  const bool bounds_are_integral = start.isIntegral(/*includeBool=*/true) &&
                                   end.isIntegral(/*includeBool=*/true);
  const double step_f64 = step.toDouble();

  // Watch out for int64_t to double conversion issue.
  //
  // In order to avoid issues such as 2^53 and 2^53+1 being mapped to the same
  // double value, we need to check whether both bounds are integer numbers. If
  // they are, we run the checks with their integer values. Otherwise, we
  // fallback to their double value.
  if (bounds_are_integral) {
    TT_RETURN_IF_ERROR(
        ValidateArangeInputs(start.toLong(), end.toLong(), step_f64));
  } else {
    TT_RETURN_IF_ERROR(
        ValidateArangeInputs(start.toDouble(), end.toDouble(), step_f64));
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

}  // namespace

at::Tensor& AtenArangeStartOut(const at::Scalar& start, const at::Scalar& end,
                               const at::Scalar& step, at::Tensor& out) {
  auto start_promoted = PromoteScalar(start);
  auto end_promoted = PromoteScalar(end);
  auto step_promoted = PromoteScalar(step);

  TT_KERNEL(
      OpName::kArangeStartOut, param_keys,
      (start_promoted, end_promoted, step_promoted, out), {
        const at::ScalarType output_dtype = out.scalar_type();
        const at::ScalarType acc_dtype = ToAccumulateType(output_dtype);

        TT_ASSIGN_OR_THROW(
            const auto num_elements,
            GetArangeNumElements(start_promoted.scalar(), end_promoted.scalar(),
                                 step_promoted.scalar(), output_dtype));

        TT_ASSIGN_OR_THROW(at::Tensor start_tensor,
                           start_promoted.GetTensor(acc_dtype));
        TT_ASSIGN_OR_THROW(at::Tensor end_tensor,
                           end_promoted.GetTensor(acc_dtype));
        TT_ASSIGN_OR_THROW(at::Tensor step_tensor,
                           step_promoted.GetTensor(acc_dtype));

        at::native::resize_output(out, {num_elements});

        TT_ASSIGN_OR_THROW(const auto output_mlir_type,
                           ConvertTo<mlir::ElementType>(output_dtype));

        auto op_builder = [num_elements, output_mlir_type](
                              FixedSizeSpan<mlir::MlirOp, 3> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          auto& [start_op, end_op, step_op] = inputs;
          mlir::MlirBuilder& builder = start_op.getBuilder();
          TT_ASSIGN_OR_RETURN(
              const mlir::Type mlir_type,
              GetMlirType(builder.getContext(), output_mlir_type));
          return torch_tpu::BuildArangeOp(num_elements, mlir_type, start_op,
                                          step_op);
        };

        TT_ASSIGN_OR_THROW(
            auto result_buf,
            DispatchOp<3>(std::move(op_builder),
                          {start_tensor, end_tensor, step_tensor},
                          {.out_dtype = output_mlir_type,
                           .out_dims = {num_elements},
                           .op_param_cache_keys = std::move(param_keys)}));

        TT_THROW_IF_ERROR(AssignBufferToAtTensor(std::move(result_buf), out));
      });
  return out;
}

}  // namespace torch_tpu
